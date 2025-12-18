import torch
import torch.nn as nn
import math
from perceiver.attention import MultiHeadLatentAttention
from perceiver.layers import FeedForward

class PerceiverDecoder(nn.Module):
    """
    A Perceiver IO-style Decoder that utilizes Multi-Head Latent Attention (MLA).

    This decoder attends to a compressed latent representation to generate
    structured outputs. It integrates DeepSeek-V2's MLA mechanism to reduce
    Key-Value cache overhead and computational complexity, replacing standard
    Multi-Head Attention.

    The architecture follows a Pre-Norm sequence in each layer:
      1. Self-Attention (MLA): Refines the target queries (output tokens) to ensure
         internal coherence (e.g., grammatical structure in text).
      2. Cross-Attention (MLA): Retrieves information from the latent bottleneck
         (context) to populate the queries with content.
      3. Feed-Forward (SwiGLU): Processes the attended features.

    Attributes:
        layers (nn.ModuleList): A stack of decoder layers containing MLA and FFN blocks.
    """

    def __init__(self, latent_dim=2560, num_outputs=0, output_dim=2560, heads=32, num_layers=24, dropout=0.1):
        """
        Initializes the PerceiverDecoder.

        Args:
            latent_dim (int): The dimension of the input latent vectors (the compressed
                representation from the Encoder). Default: 2560.
            num_outputs (int): The number of output queries/tokens. (Note: In this implementation,
                queries are passed via `forward`, so this argument is currently unused
                for static initialization). Default: 0.
            output_dim (int): The hidden dimension of the output queries and the
                decoder's internal state. Default: 2560.
            heads (int): The number of attention heads for the MLA mechanism. Default: 32.
            num_layers (int): The depth of the decoder (number of sequential blocks).
                Default: 24 (typically adjusted for ~3B parameter scale).
            dropout (float): The dropout probability applied in the Feed-Forward network.
                Default: 0.1.
        """
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(num_layers):
            # 1. Self-Attention (MLA)
            # Queries attend to themselves. Essential for text grammar & image coherence.
            # Input: [Batch, Num_Queries, Dim] -> Output: [Batch, Num_Queries, Dim]
            self_mla = MultiHeadLatentAttention(
                d_model=output_dim, num_heads=heads,
                q_lora_rank=1536, kv_lora_rank=512, qk_rope_dim=64
            )

            # 2. Cross-Attention (MLA)
            # Queries attend to Latents. Essential for retrieving content from the Encoder.
            # Input: Queries + Context(Latents) -> Output: Queries
            cross_mla = MultiHeadLatentAttention(
                d_model=output_dim, num_heads=heads,
                q_lora_rank=1536, kv_lora_rank=512, qk_rope_dim=64
            )
            # FIX: Patch kv_down to accept 'latent_dim' instead of 'output_dim'
            # This allows the MLA mechanism to project the Context (latents) correctly
            # into the compressed KV space.
            cross_mla.kv_down = nn.Linear(latent_dim, 512, bias=False)

            # 3. Feed-Forward Network (SwiGLU style)
            ff = FeedForward(output_dim, output_dim * 4, dropout=dropout)

            # 4. Normalization Layers (Pre-Norm architecture)
            self.layers.append(nn.ModuleDict({
                'self_attn': self_mla,
                'cross_attn': cross_mla,
                'ff': ff,
                'norm1': nn.LayerNorm(output_dim),
                'norm2': nn.LayerNorm(output_dim),
                'norm3': nn.LayerNorm(output_dim)
            }))

    def forward(self, latents, target_queries):
        """
        Processes target queries by attending to the provided latent representation.

        This method iterates through the decoder layers, applying self-attention
        to the queries (for structure) and cross-attention to the latents (for content).

        Args:
            latents (torch.Tensor): The compressed context representation from the
                Encoder.
                Shape: `[Batch, Num_Latents, Latent_Dim]` (e.g., `[B, 512, 2560]`).
            target_queries (torch.Tensor): The initial state of the output tokens
                (questions/slots to be filled).
                Shape: `[Batch, Num_Queries, Output_Dim]` (e.g., `[B, Output_Size, 2560]`).

        Returns:
            torch.Tensor: The processed output embeddings, ready for projection to logits.
                Shape: `[Batch, Num_Queries, Output_Dim]`.
        """
        x = target_queries

        for layer in self.layers:
            # --- Block A: Self-Attention ---
            # "Look at other parts of the output being generated"
            residual = x
            x = layer['norm1'](x)
            # self_attn(x) implies context=None, so it uses x for both Q and K/V
            x = residual + layer['self_attn'](x)

            # --- Block B: Cross-Attention ---
            # "Look at the Latents to find the answer"
            residual = x
            x = layer['norm2'](x)
            # context=latents -> Q comes from x, K/V comes from latents
            x = residual + layer['cross_attn'](x, context=latents)

            # --- Block C: Feed-Forward ---
            # "Process and reason about the information"
            residual = x
            x = layer['norm3'](x)
            x = residual + layer['ff'](x)

        return x

class FourierQueryEncoding(nn.Module):
    """
    Generates Fourier feature encodings for continuous coordinates.
    Used for grid-based modalities like Image, Audio (Spectrogram), and Video.
    """
    def __init__(self, num_bands: int, max_freq: float, d_model: int, input_dim: int):
        """
        Args:
            num_bands (int): Number of frequency bands.
            max_freq (float): Maximum frequency.
            d_model (int): The target output dimension (decoder query dimension).
            input_dim (int): Dimensionality of the coordinates (2 for image/audio, 3 for video).
        """
        super().__init__()
        self.num_bands = num_bands
        self.input_dim = input_dim

        # Create log-spaced frequency bands
        # Shape: [num_bands]
        self.freq_bands = nn.Parameter(
            torch.logspace(0., math.log(max_freq), num_bands, base=2),
            requires_grad=False
        )

        # Projection from Fourier features to Decoder Dimension
        # Fourier dim = input_dim * num_bands * 2 (sin+cos) + input_dim (original coords)
        fourier_dim = (input_dim * num_bands * 2) + input_dim
        self.out_proj = nn.Linear(fourier_dim, d_model)

    def forward(self, coords):
        """
        Args:
            coords (torch.Tensor): [Batch, Num_Points, Input_Dim]
                                   Values should be normalized to [-1, 1].
        """
        # 1. Expand Frequencies
        # coords: [B, N, D] -> [B, N, D, 1]
        # freqs: [F] -> [1, 1, 1, F]
        # Product: [B, N, D, F]
        device = coords.device

        # Calculate encodings
        # [B, N, D, 1] * [1, 1, 1, F] = [B, N, D, F]
        raw_proj = coords.unsqueeze(-1) * self.freq_bands[None, None, None, :] * math.pi

        # Flatten D and F: [B, N, D * F]
        raw_proj = raw_proj.view(coords.shape[0], coords.shape[1], -1)

        # Apply Sin/Cos: [B, N, D * F * 2]
        fourier_feats = torch.cat([raw_proj.sin(), raw_proj.cos()], dim=-1)

        # Concatenate original coordinates: [B, N, Fourier_Dim]
        out = torch.cat([coords, fourier_feats], dim=-1)

        return self.out_proj(out)


class MultimodalQueryGenerator(nn.Module):
    """
    Central hub for generating queries for different modalities on-the-fly.

    Supports:
    - Grid Queries (Image, Audio, Video) via Fourier Features.
    - Sequence Queries (Text, 3D Primitives, Skeletal) via Learned Embeddings.
    """
    def __init__(self, decoder_dim: int, max_seq_len: int = 2048, max_set_size: int = 2048):
        super().__init__()
        self.decoder_dim = decoder_dim

        # --- 1. Coordinate-based Encoders ---
        self.fourier_2d = FourierQueryEncoding(num_bands=32, max_freq=10.0, d_model=decoder_dim, input_dim=2)
        self.fourier_3d = FourierQueryEncoding(num_bands=32, max_freq=10.0, d_model=decoder_dim, input_dim=3)

        # --- 2. Learned Query Banks (Separated) ---

        # A. Text Bank: Learns grammatical/sequential positions (e.g., "First word", "Second word")
        self.text_pos_emb = nn.Embedding(max_seq_len, decoder_dim)

        # B. Set Bank: Learns abstract feature slots (e.g., "Gaussian A", "Gaussian B")
        # We use a Parameter directly so we can easily slice it, though Embedding works too.
        # Initialized with random normal to break symmetry.
        self.set_query_emb = nn.Parameter(torch.randn(1, max_set_size, decoder_dim) * 0.02)

    def get_grid_queries_2d(self, batch_size, dim1, dim2, device):
        """Generates queries for Image (H, W) or Audio (Freq, Time)."""
        # Create normalized grid [-1, 1]
        # Note: 'indexing="ij"' ensures dim1 varies along rows, dim2 along cols
        d1 = torch.linspace(-1, 1, dim1, device=device)
        d2 = torch.linspace(-1, 1, dim2, device=device)
        mesh_d1, mesh_d2 = torch.meshgrid(d1, d2, indexing='ij')

        # Stack coordinates: [dim1, dim2, 2]
        grid = torch.stack((mesh_d1, mesh_d2), dim=-1)

        # Flatten: [Batch, dim1*dim2, 2]
        flat_grid = grid.reshape(1, -1, 2).repeat(batch_size, 1, 1)

        return self.fourier_2d(flat_grid)

    def get_grid_queries_3d(self, batch_size, dim1, dim2, dim3, device):
        """Generates queries for Video (Time, H, W)."""
        d1 = torch.linspace(-1, 1, dim1, device=device)
        d2 = torch.linspace(-1, 1, dim2, device=device)
        d3 = torch.linspace(-1, 1, dim3, device=device)
        mesh_d1, mesh_d2, mesh_d3 = torch.meshgrid(d1, d2, d3, indexing='ij')

        # Stack: [dim1, dim2, dim3, 3]
        grid = torch.stack((mesh_d1, mesh_d2, mesh_d3), dim=-1)

        # Flatten: [Batch, dim1*dim2*dim3, 3]
        flat_grid = grid.reshape(1, -1, 3).repeat(batch_size, 1, 1)

        return self.fourier_3d(flat_grid)

    def get_sequence_queries(self, batch_size, seq_len, device, query_type):
        """Generates queries for Text, 3D Gaussians, or Skeletal Animation."""
        # Generate indices [0, 1, ..., seq_len-1]
        if query_type == 'text':
            # Text: Use Embedding lookup with indices (0, 1, 2...)
            indices = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
            return self.text_pos_emb(indices)

        elif query_type == 'set':
            # Sets: Slice the learned parameter bank directly
            # [1, max_set_size, dim] -> [1, seq_len, dim] -> [Batch, seq_len, dim]
            if seq_len > self.set_query_emb.shape[1]:
                raise ValueError(f"Requested set size {seq_len} exceeds max_set_size {self.set_query_emb.shape[1]}")

            queries = self.set_query_emb[:, :seq_len, :]
            return queries.repeat(batch_size, 1, 1)

        else:
            raise ValueError(f"Unknown query_type: {query_type}")


class MultimodalDecoder(nn.Module):
    """
    The second half of the Perceiver IO architecture.
    It wraps the PerceiverDecoder (cross-attention engine) and the Modality Output Adapters.

    Functionality:
    1. Accepts the unified 'latents' from the encoder.
    2. Uses a QueryGenerator to create dynamic queries based on the requested output specifications.
    3. Concatenates all queries into a single stream.
    4. Passes them through the PerceiverDecoder to get output embeddings.
    5. Splits the output embeddings back into their respective modality chunks.
    6. Passes each chunk through its corresponding OutputAdapter to generate final data (pixels, audio, etc.).

    This version uses a configurable handler registry to map modalities to
    specific query generation logic.
    """

    def __init__(self, decoder: nn.Module, query_generator: MultimodalQueryGenerator, modality_map: dict = None, **adapters):
        """
        Args:
            decoder (nn.Module): The core PerceiverDecoder instance.
            query_generator (MultimodalQueryGenerator): Instance to generate queries.
            modality_map (dict, optional): Custom mapping of {modality_name: handler_callable}
                                           to override default logic.
            **adapters: Output adapters (e.g., image=ImageOutputAdapter(...)).
        """
        super().__init__()
        self.decoder = decoder
        self.query_generator = query_generator
        self.adapters = nn.ModuleDict(adapters)

        # --- Handler Registry ---
        # Explicit mapping of modality keys to handler methods.
        # This makes it easy to add new modalities or change behavior without touching forward().
        self.handler_registry = {
            'image': self._handle_image_query,
            'audio': self._handle_audio_query,
            'video': self._handle_video_query,

            # Text uses the text bank
            'text': lambda b, d, s, a: self._handle_sequence_query(b, d, s, a, q_type='text'),

            # 3D/Animation uses the set bank
            '3d': lambda b, d, s, a: self._handle_sequence_query(b, d, s, a, q_type='set'),
            'gaussian_splatting': lambda b, d, s, a: self._handle_sequence_query(b, d, s, a, q_type='set'),
            'animation': lambda b, d, s, a: self._handle_sequence_query(b, d, s, a, q_type='set'),
        }


        # Update with user-provided config
        if modality_map:
            self.handler_registry.update(modality_map)

    # --- Query Handler Methods ---

    def _handle_image_query(self, batch_size, device, spec, adapter):
        # Spec: (Height, Width)
        h, w = spec
        # Retrieve patch_size from adapter (defaulting to 16 if not found)
        patch_size = getattr(adapter, 'patch_size', 16)
        grid_h, grid_w = h // patch_size, w // patch_size
        return self.query_generator.get_grid_queries_2d(batch_size, grid_h, grid_w, device)

    def _handle_audio_query(self, batch_size, device, spec, adapter):
        # Spec: (Freq_Bins, Time_Frames)
        f, t = spec
        # Audio adapter often wraps an image reconstruction module
        if hasattr(adapter, 'image_reconstruction'):
            patch_size = adapter.image_reconstruction.patch_size
        else:
            patch_size = getattr(adapter, 'patch_size', 16)

        grid_f, grid_t = f // patch_size, t // patch_size
        return self.query_generator.get_grid_queries_2d(batch_size, grid_f, grid_t, device)

    def _handle_video_query(self, batch_size, device, spec, adapter):
        # Spec: (Time, Height, Width)
        t, h, w = spec
        pt = getattr(adapter, 'patch_size_t', 1)
        ph = getattr(adapter, 'patch_size_hw', 16)

        grid_t, grid_h, grid_w = t // pt, h // ph, w // ph
        return self.query_generator.get_grid_queries_3d(batch_size, grid_t, grid_h, grid_w, device)

    def _handle_sequence_query(self, batch_size, device, spec, adapter, q_type='text'):
        # Spec: Length (int) OR (Dim1, Dim2...) -> Flattened Length
        if isinstance(spec, int):
            length = spec
        elif isinstance(spec, (tuple, list)):
            length = 1
            for dim in spec:
                length *= dim
        else:
            raise ValueError(f"Invalid sequence spec: {spec}")

        # Pass the q_type (text/set) to the generator
        return self.query_generator.get_sequence_queries(batch_size, length, device, query_type=q_type)

    def forward(self, latents: torch.Tensor, output_specs: dict):
        """
        Args:
            latents (torch.Tensor): Unified representation [Batch, N, Dim].
            output_specs (dict): Defines what to generate. Keys must match adapter names.

                                 Examples:
                                 {
                                    'image': (64, 64),         # Image: (H, W) -> Uses 2D Grid
                                    'audio': (128, 128),       # Audio: (Freq, Time) -> Uses 2D Grid
                                    'video': (16, 64, 64),     # Video: (Time, H, W) -> Uses 3D Grid
                                    'text': 20,                # Text: Length -> Uses Sequence Indices
                                    '3d_gaussians': 1024       # 3D: Num Points -> Uses Sequence Indices
                                 }

        Returns:
            dict: Reconstructed outputs for the requested modalities.
            Example:
            {
                      'image': [Batch, 3, H, W],
                      'audio': [Batch, 1, Freq, Time],
                      ...
                  }
        """
        device = latents.device
        batch_size = latents.shape[0]

        queries_list = []
        query_lengths = []
        ordered_keys = []

        for key, spec in output_specs.items():
            if key not in self.adapters:
                continue

            ordered_keys.append(key)
            adapter = self.adapters[key]

            # Use the registry to find the appropriate handler
            if key in self.handler_registry:
                handler = self.handler_registry[key]
                q = handler(batch_size, device, spec, adapter)
            else:
                # Fallback for unknown keys (e.g., generic sequence)
                if isinstance(spec, int):
                    q = self.query_generator.get_sequence_queries(batch_size, spec, device)
                else:
                    raise ValueError(f"No handler found for modality '{key}' and spec is not a simple integer length.")

            queries_list.append(q)
            query_lengths.append(q.shape[1])

        if not queries_list:
            return {}

        # 2. Concatenate all queries
        # [Batch, Total_Queries, D_Model]
        full_queries = torch.cat(queries_list, dim=1)

        # 3. Pass through Perceiver Decoder
        # The decoder cross-attends to the latents using these dynamically generated queries
        full_outputs = self.decoder(latents, target_queries=full_queries)

        # 4. Split outputs back to modality chunks
        split_outputs = torch.split(full_outputs, query_lengths, dim=1)

        # 5. Route to Adapters
        results = {}
        for key, output_chunk in zip(ordered_keys, split_outputs):
            # Pass the embedding chunk to the specific adapter
            results[key] = self.adapters[key](output_chunk)

        return results