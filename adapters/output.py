import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# IMAGE OUTPUT ADAPTER
# -----------------------------------------------------------------------------

class ImageOutputAdapter(nn.Module):
    """
    Reconstructs an image from a sequence of patch embeddings.

    This adapter performs the inverse operation of the ImageAdapter (patchify):
    1. Linearly projects the decoder output to pixel values for each patch.
    2. Rearranges (un-patchifies) the patch vectors back into a 2D image grid.

    Attributes:
        projection (nn.Linear): Projects output_dim -> channels * patch_size^2.
    """
    def __init__(self, output_dim: int, image_shape: tuple[int, int, int], patch_size: int):
        """
        Args:
            output_dim (int): Dimension of the vectors coming from the Perceiver Decoder.
            image_shape (tuple): Target output shape (Channels, Height, Width).
            patch_size (int): The patch size used during encoding (e.g., 4 or 16).
        """
        super().__init__()
        self.channels, self.height, self.width = image_shape
        self.patch_size = patch_size

        # Validate that image dimensions are divisible by patch size
        if not (self.height % patch_size == 0 and self.width % patch_size == 0):
            raise ValueError(f"Image dimensions ({self.height}, {self.width}) must be divisible by patch size {patch_size}")

        self.num_patches_h = self.height // patch_size
        self.num_patches_w = self.width // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Projection: from latent dim to all pixels in a single patch
        # Output size per patch: C * P * P
        self.patch_dim = self.channels * patch_size * patch_size
        self.projection = nn.Linear(output_dim, self.patch_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Decoder output [Batch, Num_Patches, Output_Dim]

        Returns:
            torch.Tensor: Reconstructed image [Batch, Channels, Height, Width]
        """
        b, n, _ = x.shape
        if not (n == self.num_patches):
            raise ValueError(f"Expected {self.num_patches} patches, got {n}")

        # 1. Project latents to patch pixels
        # [B, N, Output_Dim] -> [B, N, C * P * P]
        x = self.projection(x)

        # 2. Un-patchify (Reshape and Permute to restore grid)
        # Reshape to: [B, H_patches, W_patches, C, P, P]
        x = x.view(b, self.num_patches_h, self.num_patches_w, self.channels, self.patch_size, self.patch_size)

        # Permute to: [B, C, H_patches, P, W_patches, P]
        # This aligns the patches spatially
        x = x.permute(0, 3, 1, 4, 2, 5)

        # Final collapse to image: [B, C, H, W]
        x = x.reshape(b, self.channels, self.height, self.width)

        return x

# -----------------------------------------------------------------------------
# TEXT OUTPUT ADAPTER
# -----------------------------------------------------------------------------

class TextOutputAdapter(nn.Module):
    """
    Projects decoder outputs to vocabulary logits for text generation.

    Attributes:
        projection (nn.Linear): Projects output_dim -> vocab_size.
    """
    def __init__(self, output_dim: int, vocab_size: int):
        """
        Args:
            output_dim (int): Dimension of the vectors coming from the Perceiver Decoder.
            vocab_size (int): Size of the tokenizer vocabulary.
        """
        super().__init__()
        self.projection = nn.Linear(output_dim, vocab_size)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Decoder output [Batch, Seq_Len, Output_Dim]

        Returns:
            torch.Tensor: Logits [Batch, Seq_Len, Vocab_Size]
        """
        # Simple linear projection to logits
        return self.projection(x)

def decode_logits_to_text(logits, tokenizer, skip_special_tokens=True):
    """
    Converts raw logits from the TextOutputAdapter into human-readable text.

    Args:
        logits (torch.Tensor): Output from TextOutputAdapter.
                               Shape: [Batch, Seq_Len, Vocab_Size]
        tokenizer (PreTrainedTokenizer): The Hugging Face tokenizer used for encoding.
        skip_special_tokens (bool): Whether to remove [PAD], [CLS], [SEP] etc.
                                    Recommended True for readability.

    Returns:
        list[str]: A list of decoded strings, one for each item in the batch.
    """
    # 1. Convert Logits to Probabilities (Optional for visualization, unused for argmax)
    # probs = torch.softmax(logits, dim=-1)

    # 2. Greedy Decoding: Select the most likely token ID at each step
    # Shape changes from [Batch, Seq_Len, Vocab_Size] -> [Batch, Seq_Len]
    token_ids = torch.argmax(logits, dim=-1)

    # 3. Decode IDs to Strings
    # batch_decode handles the loop over the batch dimension efficiently
    decoded_text = tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)

    return decoded_text

# -----------------------------------------------------------------------------
# AUDIO OUTPUT ADAPTER
# -----------------------------------------------------------------------------

class AudioOutputAdapter(nn.Module):
    """
    Reconstructs audio spectrograms from patch embeddings.
    Essentially an ImageOutputAdapter specialized for single-channel spectrograms.

    NOTE: The output of this adapter is a Spectrogram (magnitude or mel).
    To hear audio, you must pass this output through a Vocoder (e.g., Griffin-Lim, HiFiGAN).

    Attributes:
        image_reconstruction (ImageOutputAdapter): Handles the un-patchifying logic.
    """
    def __init__(self, output_dim: int, spectro_shape: tuple[int, int], patch_size: int):
        """
        Args:
            output_dim (int): Dimension of the vectors coming from the Perceiver Decoder.
            spectro_shape (tuple): Target dimensions (Freq_Bins, Time_Frames).
                                   Note: Channels=1 is assumed internally.
            patch_size (int): The patch size used during encoding.
        """
        super().__init__()
        # Treat spectrogram as a 1-channel image: (1, Freq, Time)
        self.image_shape = (1, spectro_shape[0], spectro_shape[1])

        self.image_reconstruction = ImageOutputAdapter(
            output_dim=output_dim,
            image_shape=self.image_shape,
            patch_size=patch_size
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Decoder output [Batch, Num_Patches, Output_Dim]

        Returns:
            torch.Tensor: Reconstructed Spectrogram [Batch, 1, Freq, Time]
        """
        # Delegate to the image reconstruction logic
        return self.image_reconstruction(x)

    def recover_waveform_griffin_lim(self, spectrogram, n_fft=400, hop_length=160, win_length=400, power=2.0):
        """
        Helper utility to convert the predicted spectrogram back to audio using Griffin-Lim.
        This is a deterministic, phase-estimation algorithm (lower quality than HiFiGAN but requires no training).

        Args:
            spectrogram (torch.Tensor): [1, Freq, Time] (Single item, not batch)
            n_fft, hop_length, etc.: Must match the parameters used in MyAudioDataset.

        Returns:
            torch.Tensor: Waveform [1, Time_Samples]
        """
        import torchaudio.transforms as T

        # Ensure we are on CPU for torchaudio transforms usually, or match device
        griffin_lim = T.GriffinLim(
            n_fft=n_fft,
            n_iter=32,
            win_length=win_length,
            hop_length=hop_length,
            power=power
        ).to(spectrogram.device)

        # GriffinLim expects [..., Freq, Time]
        # Our input is [1, Freq, Time] which is correct
        return griffin_lim(spectrogram)

# -----------------------------------------------------------------------------
# VIDEO OUTPUT ADAPTER
# -----------------------------------------------------------------------------

class VideoOutputAdapter(nn.Module):
    """
    Reconstructs video from a sequence of spatiotemporal patch embeddings.

    This performs the inverse of the VideoAdapter (3D patchify).
    It projects decoder outputs back to pixels and rearranges them into (Time, Height, Width).

    Attributes:
        projection (nn.Linear): Projects output_dim -> channels * patch_t * patch_h * patch_w
    """
    def __init__(self, output_dim: int, video_shape: tuple[int, int, int, int],
                 patch_size_t: int, patch_size_hw: int):
        """
        Args:
            output_dim (int): Dimension of vectors from Perceiver Decoder.
            video_shape (tuple): Target shape (Channels, Time, Height, Width).
            patch_size_t (int): Temporal patch size.
            patch_size_hw (int): Spatial patch size (Height/Width).
        """
        super().__init__()
        self.channels, self.time, self.height, self.width = video_shape
        self.patch_size_t = patch_size_t
        self.patch_size_hw = patch_size_hw

        # Validate dimensions
        if not (self.time % patch_size_t == 0):
            raise ValueError(f"Time dim {self.time} not divisible by patch_t {patch_size_t}")
        if not (self.height % patch_size_hw == 0):
            raise ValueError(f"Height {self.height} not divisible by patch_hw {patch_size_hw}")
        if not (self.width % patch_size_hw == 0):
            raise ValueError(f"Width {self.width} not divisible by patch_hw {patch_size_hw}")

        self.num_patches_t = self.time // patch_size_t
        self.num_patches_h = self.height // patch_size_hw
        self.num_patches_w = self.width // patch_size_hw
        self.num_patches = self.num_patches_t * self.num_patches_h * self.num_patches_w

        # Projection size: C * Pt * Ph * Pw
        self.patch_dim = self.channels * patch_size_t * patch_size_hw * patch_size_hw
        self.projection = nn.Linear(output_dim, self.patch_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Decoder output [Batch, Num_Patches, Output_Dim]

        Returns:
            torch.Tensor: Reconstructed Video [Batch, Channels, Time, Height, Width]
        """
        b, n, _ = x.shape
        if not (n == self.num_patches):
            raise ValueError(f"Expected {self.num_patches} patches, got {n}")

        # 1. Project latents to tubelet pixels
        # [B, N, Output_Dim] -> [B, N, C * Pt * Ph * Pw]
        x = self.projection(x)

        # 2. Un-patchify (Reshape and Permute to restore spatiotemporal grid)
        # We assume the sequence order is (Time, Height, Width) based on standard flattening
        # Reshape to: [B, Nt, Nh, Nw, C, Pt, Ph, Pw]
        x = x.view(b, self.num_patches_t, self.num_patches_h, self.num_patches_w,
                   self.channels, self.patch_size_t, self.patch_size_hw, self.patch_size_hw)

        # Permute to align dimensions: [B, C, Nt, Pt, Nh, Ph, Nw, Pw]
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)

        # Final collapse: [B, C, T, H, W]
        x = x.reshape(b, self.channels, self.time, self.height, self.width)

        return x

# -----------------------------------------------------------------------------
# 3D OUTPUT ADAPTERS
# -----------------------------------------------------------------------------

class GaussianSplattingOutputAdapter(nn.Module):
    """
    Reconstructs 3D Gaussian Splatting parameters from latent embeddings.

    This adapter projects the decoder output into the specific attributes required
    for 3DGS rendering: Position, Rotation, Scale, Opacity, and Color (SH).
    It applies specific activation functions to each attribute to ensure valid ranges
    (e.g., positive scale, unit quaternions).

    Attributes:
        projection (nn.Linear): Projects output_dim -> total_gs_dim (approx 59).
    """
    def __init__(self, output_dim: int, num_gaussians: int, sh_degree: int = 3):
        """
        Args:
            output_dim (int): Dimension of vectors from Perceiver Decoder.
            num_gaussians (int): The number of Gaussians to generate.
            sh_degree (int): Degree of Spherical Harmonics (default 3).
        """
        super().__init__()
        self.num_gaussians = num_gaussians

        # --- Dimension Calculation ---
        # Position (XYZ) = 3
        # Rotation (Quaternion WXYZ) = 4
        # Scale (Log-Scale XYZ) = 3
        # Opacity (Logit) = 1
        # Spherical Harmonics = (Degree + 1)^2 * 3 Channels
        self.sh_dim = ((sh_degree + 1) ** 2) * 3
        self.total_dim = 3 + 4 + 3 + 1 + self.sh_dim

        self.projection = nn.Linear(output_dim, self.total_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Decoder output [Batch, Num_Gaussians, Output_Dim]

        Returns:
            torch.Tensor: Parameters [Batch, Num_Gaussians, Total_Dim]
                          Formatted as: [Pos(3), Rot(4), Scale(3), Opacity(1), SH(N)]
        """
        # 1. Linear Projection
        # [Batch, N, Output_Dim] -> [Batch, N, Total_Dim]
        x = self.projection(x)

        # 2. Split into attributes for specific activations
        # We assume the layout: Pos(0:3), Rot(3:7), Scale(7:10), Opacity(10:11), SH(11:)
        pos = x[..., 0:3]
        rot = x[..., 3:7]
        scale = x[..., 7:10]
        opacity = x[..., 10:11]
        sh = x[..., 11:]

        # 3. Apply Constraints/Activations

        # Position: Tanh to constrain to [-1, 1] (Assuming normalized data)
        # If your scene is unbounded, remove this activation.
        pos = torch.tanh(pos)

        # Rotation: Quaternions must be unit length (Norm = 1)
        rot = F.normalize(rot, p=2, dim=-1)

        # Scale: Must be positive. Softplus is safer than Exp (prevents infinity).
        # We add a small epsilon (1e-6) to avoid exact zero.
        scale = F.softplus(scale) + 1e-6

        # Opacity: Must be in [0, 1]. Sigmoid.
        opacity = torch.sigmoid(opacity)

        # SH Coeffs: Unbounded (Identity).

        # 4. Concatenate back
        return torch.cat([pos, rot, scale, opacity, sh], dim=-1)


class SkeletalAnimationOutputAdapter(nn.Module):
    """
    Reconstructs skeletal motion sequences from latent embeddings.

    It decodes the latent queries into a sequence of poses. Each pose consists of
    joint positions, rotations (quaternions), and velocities.

    Attributes:
        projection (nn.Linear): Projects output_dim -> channels.
    """
    def __init__(self, output_dim: int, num_joints: int, sequence_length: int, channels: int = 10):
        """
        Args:
            output_dim (int): Dimension of vectors from Perceiver Decoder.
            num_joints (int): Number of bones/joints in the rig.
            sequence_length (int): Number of frames to generate.
            channels (int): Features per joint (e.g., 3 Pos + 4 Rot + 3 Vel = 10).
        """
        super().__init__()
        self.num_joints = num_joints
        self.sequence_length = sequence_length
        self.channels = channels
        self.num_outputs = sequence_length * num_joints

        self.projection = nn.Linear(output_dim, channels)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Decoder output [Batch, Time * Num_Joints, Output_Dim]

        Returns:
            torch.Tensor: Animation Sequence [Batch, Time, Num_Joints, Channels]
        """
        b, n, _ = x.shape
        expected_n = self.sequence_length * self.num_joints

        if expected_n != n:
            raise ValueError(f"Expected {expected_n} queries (Time*Joints), got {n}")


        # 1. Project to Joint Features
        # [Batch, T*J, Output_Dim] -> [Batch, T*J, Channels]
        x = self.projection(x)

        # 2. Reshape to Spatiotemporal Grid
        # [Batch, Time, Num_Joints, Channels]
        x = x.view(b, self.sequence_length, self.num_joints, self.channels)

        # 3. Apply Constraints
        # Assuming channel layout: Pos(0:3), Rot(3:7), Vel(7:10)

        # We need to normalize the Quaternion part (indices 3 to 7)
        if self.channels >= 7:
            # Clone is necessary to avoid in-place modification errors in autograd
            rot = x[..., 3:7].clone()
            rot = F.normalize(rot, p=2, dim=-1)

            # Re-assemble
            # Note: We use torch.cat instead of slice assignment for safety
            pos = x[..., 0:3]
            rest = x[..., 7:] # Velocity or other features
            x = torch.cat([pos, rot, rest], dim=-1)

        return x