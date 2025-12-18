import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi, log

# -----------------------------------------------------------------------------
# TEXT ADAPTER
# -----------------------------------------------------------------------------

class TextAdapter(nn.Module):
    """
    A neural network module that converts integer token indices into dense vector embeddings
    augmented with learnable positional information.

    This adapter is typically used as the input layer for Transformer-based architectures,
    combining standard dictionary embeddings with absolute, learnable positional embeddings.

    Attributes:
        token_emb (nn.Embedding): The lookup table for token embeddings.
        pos_emb (nn.Parameter): Learnable positional embeddings initialized with random noise.
    """

    def __init__(self, vocab_size, model_dim, max_seq_len=512):
        """
        Initializes the TextAdapter parameters.

        Args:
            vocab_size (int): The size of the vocabulary (total number of unique tokens).
            model_dim (int): The dimensionality of the embedding vectors (hidden size).
            max_seq_len (int, optional): The maximum sequence length the model can handle.
                Used to determine the size of the positional embedding matrix. Defaults to 512.
        """
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, model_dim)

        # 1D Positional Encoding: [1, max_seq_len, model_dim]
        # Initialized with a small scale (0.02) to maintain training stability at start
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, model_dim) * 0.02)

    def forward(self, x):
        """
        Computes the forward pass, combining token and positional embeddings.

        Args:
            x (torch.Tensor): Input tensor containing batch of token indices.
                Expected shape: (Batch, SeqLen)

        Returns:
            torch.Tensor: The combined embeddings suitable for Transformer processing.
                Shape: (Batch, SeqLen, model_dim)
        """
        # x shape: [Batch, SeqLen]
        seq_len = x.shape[1]

        # Retrieve embeddings for tokens and add positional embeddings up to current sequence length
        # Broadcasting happens here: (Batch, SeqLen, Dim) + (1, SeqLen, Dim)
        return self.token_emb(x) + self.pos_emb[:, :seq_len, :]

# -----------------------------------------------------------------------------
# IMAGE ADAPTER
# -----------------------------------------------------------------------------

class ImageAdapter(nn.Module):
    """
    Transforms 2D images into a sequence of patch embeddings with positional information.

    This module performs the initial "patchify" step common in Vision Transformers (ViT).
    It breaks an image into non-overlapping patches using a strided convolution,
    projects them to a specific embedding dimension, and adds learnable positional
    embeddings to retain spatial information.
    """

    def __init__(self, model_dim: int, patch_size: int, img_size: tuple[int,int], in_channels: int = 3):
        """
        Initializes the ImageAdapter with validation for dimensions and patch alignment.

        Args:

            model_dim (int): The target embedding dimension size for each patch.
            patch_size (int): The height and width of each square patch.
            img_size (tuple[int, int]): The dimensions (Height, Width) of the input images.
            in_channels (int, optional): Number of input image channels (e.g., 3 for RGB).
                Defaults to 3.
        """
        super().__init__()

        # --- Parameter Validation ---
        if not isinstance(in_channels, int) or in_channels <= 0:
            raise TypeError(f"in_channels must be a positive integer, got {in_channels}")

        if not isinstance(model_dim, int) or model_dim <= 0:
            raise TypeError(f"model_dim must be a positive integer, got {model_dim}")

        if not isinstance(patch_size, int) or patch_size <= 0:
            raise TypeError(f"patch_size must be a positive integer, got {patch_size}")

        if not isinstance(img_size, (tuple, list)) or len(img_size) != 2:
            raise TypeError(f"img_size must be a tuple/list of two integers (H, W), got {img_size}")

        if not isinstance(img_size[0], int) or img_size[0] <= 0:
            raise ValueError(f"Image height img_size[0] must be a positive integer, got {img_size[0]}")

        if not isinstance(img_size[1], int) or img_size[1] <= 0:
            raise ValueError(f"Image width img_size[1] must be a positive integer, got {img_size[1]}")

        if img_size[0] % patch_size != 0 or img_size[1] % patch_size != 0:
            raise ValueError(f"Image dimensions {img_size} must be divisible by patch_size {patch_size}")
        # --- End Validation ---

        self._patch_size = patch_size
        # "Patch and embed" in one shot. It doesn't process the image like a CNN does, just divide it.
        # The perceiver will be the part that process the image
        self._patch_embed = nn.Conv2d(in_channels, model_dim, kernel_size=patch_size, stride=patch_size)

        # Calculate how many patches fit along height and width
        num_patches_h = img_size[0] // patch_size
        num_patches_w = img_size[1] // patch_size
        num_patches = num_patches_h * num_patches_w

        self._mask_h_in_grid = num_patches_h # height of the mask in in the patch grid
        self._mask_w_in_grid = num_patches_w # width of the mask in the patch grid

        # 2D Positional Encoding (learnable vector for every patch location)
        self._pos_emb = nn.Parameter(torch.randn(1, num_patches, model_dim) * 0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Processes the input image batch into a sequence of embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape `(Batch, Channels, Height, Width)`.
            mask (torch.Tensor): (Optional) Mask tensor of shape (Batch, Height, Width) - True = Keep, False = Ignore/Pad
        Returns:
            (torch.Tensor, torch.Tensor):
                - The flattened and embedded sequence of shape `(Batch, Num_Patches, Model_Dim)`, with positional embeddings added.
                - The mask of shape (Batch, Height, Width) or None
        """

        # 1. Embed the image

        # x shape: [Batch, Channels, Height, Width]
        x = self._patch_embed(x)  # [Batch, model_dim, h', w']
        # Flatten spatial dimensions into a sequence: [Batch, model_dim, num_patches]
        x = x.flatten(2)
        # Swap dimensions to match standard format: [Batch, num_patches, model_dim]
        x = x.transpose(1, 2)
        # Add the positional embedding
        x = x + self._pos_emb

        # 2. Process the mask, if provided
        patch_mask = None
        if mask is not None:
            # We need to add a channel dim for interpolate: [B, 1, H, W]
            m = mask.unsqueeze(1).float()

            # Resize mask to match the patch grid (e.g., 256x256 -> 16x16)
            # 'nearest' ensures we keep strict True/False values (0.0 or 1.0)
            m = F.interpolate(m, size=(self._mask_h_in_grid, self._mask_w_in_grid), mode='nearest')

            # Flatten to match the sequence: [B, num_patches]
            patch_mask = m.flatten(2).squeeze(1).bool()

        # Return BOTH the vector sequence and the mask
        return x, patch_mask

# -----------------------------------------------------------------------------
# AUDIO ADAPTER
# -----------------------------------------------------------------------------

class AudioAdapter(nn.Module):
    """
    A specific adapter for processing audio spectrograms as visual inputs.

    This module treats audio spectrograms as single-channel (grayscale) images and delegates
    processing to an underlying `ImageAdapter`. It handles the conversion of spectrogram
    dimensions into a sequence of patch embeddings suitable for Transformer inputs.

    Args:
        model_dim (int): The dimension of the output embeddings.
        patch_size (int): The size of the square patches used to partition the spectrogram.
        spectro_size (tuple[int, int]): The expected dimensions (Height, Width) of the
            input spectrograms.

    Attributes:
        _imageAdapter (ImageAdapter): The internal module used to patchify and embed the
            spectrogram data. Initialized with `in_channels=1`.
    """

    def __init__(self, model_dim: int, patch_size: int, spectro_size: tuple[int, int]):
        super().__init__()
        # Re-using ImageAdapter is perfect here.
        # We hardcode in_channels=1 because spectrograms are single-channel.
        self._imageAdapter = ImageAdapter(
            in_channels=1,
            model_dim=model_dim,
            patch_size=patch_size,
            img_size=spectro_size
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Processes the input audio as a spectrogram batch into a sequence of embeddings.

        Args:
            x (torch.Tensor): Input spectrogram tensor of shape `(Batch, Channels, Height, Width)`.
                Channels is expected to be 1.
            mask (torch.Tensor, optional): Mask tensor of shape `(Batch, Height, Width)`.
                Indicates valid regions (True) vs padded regions (False). Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - **embeddings**: The flattened sequence of shape `(Batch, Num_Patches, Model_Dim)`,
                  with positional embeddings added.
                - **mask**: The corresponding mask of shape `(Batch, Height, Width)` or None if not provided.
        """
        # Call the module directly, not .forward()
        return self._imageAdapter(x, mask=mask)

# -----------------------------------------------------------------------------
# VIDEO ADAPTER
# -----------------------------------------------------------------------------

class VideoAdapter(nn.Module):
    """
    A video adapter module that processes spatiotemporal video data into a sequence of embeddings
    for Transformer-based architectures.

    This module performs tubelet embedding (using 3D convolution) to partition the video into
    non-overlapping spatiotemporal patches. It then flattens these patches into a sequence and
    adds a learnable positional embedding.

    Args:
        in_channels (int): The number of input channels (e.g., 3 for RGB).
        model_dim (int): The dimension of the output embeddings.
        patch_size_t (int): The size of the patch along the temporal dimension.
        patch_size_hw (int): The size of the patch along the height and width dimensions.
        video_shape (tuple[int, int, int]): The input video shape as (frames, height, width).
            Used to calculate the static positional embedding size.

    Attributes:
        patch_embed (nn.Conv3d): 3D convolution layer used to project raw video patches into the embedding dimension.
        pos_emb (nn.Parameter): Learnable position embeddings added to the sequence. Shape: (1, num_patches, model_dim).
    """

    def __init__(self, in_channels, model_dim, patch_size_t, patch_size_hw, video_shape):
        super().__init__()
        self.patch_embed = nn.Conv3d(in_channels, model_dim,
                                     kernel_size=(patch_size_t, patch_size_hw, patch_size_hw),
                                     stride=(patch_size_t, patch_size_hw, patch_size_hw))

        # Calculate total number of spacetime patches
        num_t = video_shape[0] // patch_size_t
        num_h = video_shape[1] // patch_size_hw
        num_w = video_shape[2] // patch_size_hw
        num_patches = num_t * num_h * num_w

        self.pos_emb = nn.Parameter(torch.randn(1, num_patches, model_dim) * 0.02)

    def forward(self, x):
        """
        Forward pass of the VideoAdapter.

        The input video is passed through the patch embedding layer, flattened into a sequence
        of vectors, and summed with the positional embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Time, Height, Width).

        Returns:
            torch.Tensor: The output embedding sequence of shape (Batch, Num_Patches, Model_Dim).
        """
        # x shape: [Batch, Channels, Time, Height, Width]
        x = self.patch_embed(x)
        # Flatten all separate spatial/temporal dims into one sequence
        x = x.flatten(2).transpose(1, 2)
        return x + self.pos_emb

# -----------------------------------------------------------------------------
# 3D ADAPTERS
# -----------------------------------------------------------------------------

class FourierPositionEncoding3D(nn.Module):
    """
    Computes Fourier positional encodings for 3D coordinates.

    As described in the Perceiver paper, coordinate-based modalities (like Point Clouds)
    benefit from High-Fidelity Fourier Features. This maps low-dimensional coordinates
    (x, y, z) to a higher-dimensional space using sine and cosine functions at
    various frequencies.

    Formally:
        PE(x) = [sin(2*pi*f_1*x), cos(2*pi*f_1*x), ..., sin(2*pi*f_N*x), cos(2*pi*f_N*x)]
    """
    def __init__(self, num_bands: int = 64, max_freq: float = 10.0, include_original: bool = True):
        """
        Args:
            num_bands (int): Number of frequency bands.
            max_freq (float): The maximum frequency scale.
            include_original (bool): Whether to concatenate the original coordinates (x,y,z)
                                     to the output encoding.
        """
        super().__init__()
        self.num_bands = num_bands
        self.max_freq = max_freq
        self.include_original = include_original

        # Create frequencies linearly or log-spaced.
        # Perceiver often uses log-spaced frequencies for better coverage.
        if num_bands > 0:
            # frequencies shape: [num_bands]
            self.freq_bands = nn.Parameter(
                torch.logspace(0., log(max_freq), num_bands, base=2),
                requires_grad=False
            )

    def forward(self, coords):
        """
        Args:
            coords (torch.Tensor): Shape [Batch, Num_Points, 3] assuming (x, y, z) in [-1, 1] range.

        Returns:
            torch.Tensor: Shape [Batch, Num_Points, Output_Dim]
        """
        if self.num_bands == 0:
            return coords if self.include_original else torch.empty_like(coords)

        # 1. Prepare frequencies
        # coords: [B, N, 3] -> unsqueeze to [B, N, 3, 1]
        # freq: [F] -> reshape to [1, 1, 1, F]

        # We want to multiply every coord by every freq.
        # Efficient way:
        # coords shape: [..., 3]
        # freqs shape: [num_bands]

        # Result should be [..., 3 * num_bands * 2] (sin and cos)

        # coords[..., None]: [B, N, 3, 1]
        # freq_bands[None, None, None, :]: [1, 1, 1, num_bands]
        # product: [B, N, 3, num_bands]

        raw_proj = coords.unsqueeze(-1) * self.freq_bands[None, None, None, :] * pi

        # Flatten the last two dimensions to get all sin/cos terms in one feature vector
        # [B, N, 3, num_bands] -> [B, N, 3 * num_bands]
        raw_proj = raw_proj.view(coords.shape[0], coords.shape[1], -1)

        # Apply Sin and Cos
        # [B, N, 3 * num_bands] -> [B, N, 3 * num_bands * 2]
        fourier_feats = torch.cat([raw_proj.sin(), raw_proj.cos()], dim=-1)

        if self.include_original:
            return torch.cat([coords, fourier_feats], dim=-1)
        else:
            return fourier_feats

class GaussianSplattingAdapter(nn.Module):
    """
    Adapter for 3D Gaussian Splatting data.

    3DGS is essentially a Point Cloud with rich attributes.
    We treat it as a set of vectors.

    Structure per Gaussian:
    - Position: 3 floats (x, y, z)
    - Rotation: 4 floats (Quaternion: w, x, y, z)
    - Scale:    3 floats (log scale x, y, z)
    - Opacity:  1 float (logit)
    - Color/SH: 48 floats (Spherical Harmonics deg 3: 16 coeff * 3 channels)
    Total Input Dim ~ 59 floats per point.
    """
    def __init__(self, model_dim, num_sh_degree=3):
        super().__init__()

        # Calculate input dimension based on SH degree
        # (Degree + 1)^2 coefficients per color channel (3)
        num_coeffs = (num_sh_degree + 1) ** 2
        sh_dim = num_coeffs * 3

        # Pos(3) + Rot(4) + Scale(3) + Opacity(1) + SH(sh_dim)
        self.input_dim = 3 + 4 + 3 + 1 + sh_dim

        # Use Fourier Encoding for the Position (XYZ) part only?
        # Standard Perceiver practice: Encode position, concat features.
        self.pos_encoder = FourierPositionEncoding3D(num_bands=64, max_freq=10., include_original=True)

        # Fourier adds features to XYZ.
        # Output of Fourier is: 3 + (3 * 64 * 2) = 387 dimensions (if include_original=True)
        fourier_dim = 387

        # The rest of the attributes (Rot, Scale, Opacity, SH) are concatenated *after* encoding
        feature_dim = 4 + 3 + 1 + sh_dim

        total_dim = fourier_dim + feature_dim

        self.proj = nn.Linear(total_dim, model_dim)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        """
        Args:
            x: [Batch, Num_Gaussians, Input_Dim]
               We assume x[..., :3] is XYZ position.
        """
        # 1. Split Position from Attributes
        coords = x[..., :3]
        attributes = x[..., 3:]

        # 2. Fourier Encode Positions
        pos_enc = self.pos_encoder(coords) # [Batch, N, Fourier_Dim]

        # 3. Concatenate Encoded Pos + Original Attributes
        # [Batch, N, Fourier_Dim + Attribute_Dim]
        out = torch.cat([pos_enc, attributes], dim=-1)

        # 4. Project to Model Dim
        out = self.proj(out)
        return self.norm(out)

class SkeletalAnimationAdapter(nn.Module):
    """
    Adapter for 3D Animation data (Skeletal Motion).

    We treat this similarly to the VideoAdapter, but instead of spatial pixels (H*W),
    we have Joints (J).

    The 'Tubelet' embedding here combines Time + Joint features.
    """
    def __init__(self, model_dim, num_joints, sequence_length, input_channels=10):
        """
        Args:
            model_dim: Output dimension.
            num_joints: Number of bones/joints in the skeleton.
            sequence_length: Number of frames.
            input_channels: 3 (Pos) + 4 (Rot) + 3 (Vel) = 10 typically.
        """
        super().__init__()
        self.input_channels = input_channels
        self.num_joints = num_joints
        self.sequence_length = sequence_length

        # Temporal convolution to capture motion continuity
        # Kernel size (3, 1) means we look at 3 frames at a time for each joint independently
        self.motion_embed = nn.Conv2d(
            in_channels=input_channels,
            out_channels=model_dim,
            kernel_size=(3, 1), # (Time, Joint) - Mix time, keep joints separate initially
            padding=(1, 0)
        )

        # Positional Embedding:
        # We need to distinguish:
        # 1. Which Joint is this? (Spatial/Structural identity)
        # 2. Which Frame is this? (Temporal identity)

        self.joint_emb = nn.Parameter(torch.randn(1, 1, num_joints, model_dim) * 0.02)
        self.time_emb = nn.Parameter(torch.randn(1, sequence_length, 1, model_dim) * 0.02)

    def forward(self, x):
        """
        Args:
            x: [Batch, Time, Num_Joints, Input_Channels]
        """
        b, t, j, c = x.shape

        # Permute for Conv2d: [Batch, Channels, Time, Joints]
        x = x.permute(0, 3, 1, 2)

        # 1. Embed Motion
        x = self.motion_embed(x) # [Batch, Model_Dim, Time, Joints]

        # Back to [Batch, Time, Joints, Model_Dim]
        x = x.permute(0, 2, 3, 1)

        # 2. Add Positional Embeddings
        # Broadcasting handles the addition
        x = x + self.joint_emb + self.time_emb

        # 3. Flatten to Sequence
        # Output: [Batch, Time * Joints, Model_Dim]
        x = x.flatten(1, 2)

        return x