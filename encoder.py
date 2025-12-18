import torch
import torch.nn as nn
from attention import MultiHeadLatentAttention, NativeSparseCrossAttention
from layers import FeedForward, DropPath

class PerceiverBlock(nn.Module):
    def __init__(self, latent_dim, input_dim, num_self_attn_layers, heads, dropout, drop_path_rates):
        super().__init__()

        # --- Cross Attention: DeepSeek NSA + MLA ---
        # Note: Input dimension from adapters is 768, Latent is 2560.
        # We need an adapter linear layer inside MLA or before it.
        # Our MLA class handles input projection via `kv_down` (d_model -> rank).
        # But `kv_down` expects `d_model` input. We must handle the mismatch (768 vs 2560).

        self.cross_norm = nn.LayerNorm(latent_dim)

        # Native Sparse Cross Attention with MLA
        # We define a specialized MLA for cross-attn that accepts input_dim=768
        cross_mla = MultiHeadLatentAttention(
            d_model=latent_dim,
            num_heads=heads,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_rope_dim=64
        )
        # Monkey-patch or adjust the kv_down layer for the input dimension
        # (A clean implementation would perform this in __init__)
        cross_mla.kv_down = nn.Linear(input_dim, cross_mla.kv_lora_rank, bias=False)
        cross_mla.router_k = nn.Linear(input_dim, 64) # Adjust router too

        self.cross_attn = NativeSparseCrossAttention(
            mla_module=cross_mla,
            input_dim=input_dim,
            block_size=64,    # Block size for sparsity
            top_k_blocks=16   # How many blocks to attend to
        )

        self.cross_ff_norm = nn.LayerNorm(latent_dim)
        self.cross_ff = FeedForward(latent_dim, latent_dim * 4, dropout=dropout)
        self.cross_drop_path = DropPath(0.0)

        # --- Self Attention Stack: Pure MLA ---
        self.layers = nn.ModuleList([])
        for i in range(num_self_attn_layers):
            # MLA for Self Attention
            self_mla = MultiHeadLatentAttention(
                d_model=latent_dim,
                num_heads=heads,
                q_lora_rank=1536,
                kv_lora_rank=512
            )

            # Wrap in our SelfAttentionLayer structure
            layer = nn.ModuleDict({
                'norm1': nn.LayerNorm(latent_dim),
                'attn': self_mla,
                'norm2': nn.LayerNorm(latent_dim),
                'ff': FeedForward(latent_dim, latent_dim * 4, dropout=dropout),
                'drop_path': DropPath(drop_path_rates[i])
            })
            self.layers.append(layer)

    def forward(self, latents, byte_array):
        # 1. Native Sparse Cross-Attention
        # latents: [B, N, D], byte_array: [B, T, 768]
        generated = self.cross_attn(self.cross_norm(latents), byte_array)
        latents = latents + self.cross_drop_path(generated)

        latents = latents + self.cross_drop_path(self.cross_ff(self.cross_ff_norm(latents)))

        # 2. Self-Attention Stack (MLA)
        for layer in self.layers:
            # MLA forward(x, context=None) -> Self Attention
            attn_out = layer['attn'](layer['norm1'](latents))
            latents = latents + layer['drop_path'](attn_out)

            ff_out = layer['ff'](layer['norm2'](latents))
            latents = latents + layer['drop_path'](ff_out)

        return latents

class PerceiverEncoder(nn.Module):
    def __init__(
        self,
        num_latents=512,
        latent_dim=2560,    # Increased width due to MLA efficiency
        input_dim=768,
        num_blocks=6,
        num_self_layers_per_block=6,
        self_attn_heads=32, # Increased heads
        dropout=0.1,
        stochastic_depth_prob=0.2
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        total_layers = num_blocks * num_self_layers_per_block
        dpr = torch.linspace(0, stochastic_depth_prob, total_layers).tolist()

        self.blocks = nn.ModuleList([])
        layer_idx_counter = 0

        for _ in range(num_blocks):
            block_dpr = dpr[layer_idx_counter : layer_idx_counter + num_self_layers_per_block]
            self.blocks.append(
                PerceiverBlock(
                    latent_dim=latent_dim,
                    input_dim=input_dim,
                    num_self_attn_layers=num_self_layers_per_block,
                    heads=self_attn_heads,
                    dropout=dropout,
                    drop_path_rates=block_dpr
                )
            )
            layer_idx_counter += num_self_layers_per_block

        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, byte_array):
        b = byte_array.shape[0]
        x = self.latents.repeat(b, 1, 1)
        for block in self.blocks:
            x = block(x, byte_array)
        return self.norm(x)

class MultimodalEncoder(nn.Module):
    def __init__(self, encoder, **adapters):
        super().__init__()
        self.encoder = encoder
        # Register adapters as sub-modules so they are trained
        self.adapters = nn.ModuleDict(adapters)

    def forward(self, input_dict):
        """
        input_dict: A dictionary containing raw data, e.g.:
        {
            'image': torch.Tensor(...),
            'text': torch.Tensor(...)
        }
        """
        processed_inputs = []

        # Iterate through available inputs and their corresponding adapters
        for modality, data in input_dict.items():
            if modality in self.adapters:
                # 1. Pass raw data through specific adapter
                out = self.adapters[modality](data)
                processed_inputs.append(out)
            else:
                print(f"Warning: No adapter found for modality '{modality}'")

        # 2. Concatenate all modalities into one long sequence (the "Byte Array")
        # [Batch, (tokens_img + tokens_text + ...), model_dim]
        byte_array = torch.cat(processed_inputs, dim=1)

        # 3. Pass to the unified encoder
        latents = self.encoder(byte_array)
        return latents