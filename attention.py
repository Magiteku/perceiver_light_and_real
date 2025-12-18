import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadLatentAttention(nn.Module):
    """
    DeepSeek-V2/V3 Multi-Head Latent Attention (MLA).
    Features:
    1. Low-Rank Key-Value Joint Compression (reduces params & KV cache).
    2. Decoupled RoPE (rotary embeddings applied to a separate 'pe' vector).
    """
    def __init__(self, d_model, num_heads, q_lora_rank=1536, kv_lora_rank=512, qk_rope_dim=64, dropout=0.):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qk_rope_dim = qk_rope_dim
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank

        # --- 1. Query Compression (W_DQ -> W_UQ) ---
        # Projects Query to low-rank latent c_Q
        self.q_down = nn.Linear(d_model, q_lora_rank, bias=False)
        self.q_norm = nn.LayerNorm(q_lora_rank)
        # Up-projects to generate "Content" Queries (d_model = num_heads * head_dim)
        self.q_up = nn.Linear(q_lora_rank, self.num_heads * self.head_dim, bias=False)
        # Separate head for RoPE (Decoupled)
        self.q_rope = nn.Linear(q_lora_rank, self.num_heads * qk_rope_dim, bias=False)

        # --- 2. Key-Value Compression (W_DKV -> W_UK, W_UV) ---
        # Projects Input (K/V source) to low-rank latent c_KV
        # Note: In self-attn, input_dim=d_model. In cross-attn, this changes (handled in forward).
        self.kv_down = nn.Linear(d_model, kv_lora_rank, bias=False)
        self.kv_norm = nn.LayerNorm(kv_lora_rank)

        # Up-projects to "Content" Keys and Values
        self.k_up = nn.Linear(kv_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.v_up = nn.Linear(kv_lora_rank, self.num_heads * self.head_dim, bias=False)
        # Separate head for Key RoPE
        self.k_rope = nn.Linear(kv_lora_rank, self.num_heads * qk_rope_dim, bias=False)

        # Final Output Projection
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Precompute RoPE frequencies (simple implementation)
        self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, qk_rope_dim, 2).float() / qk_rope_dim)))

    def apply_rope(self, x, seq_len):
        # x: [Batch, Heads, Seq, RoPE_Dim]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # [Seq, RoPE_Dim]
        return x * emb.cos() + (torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), dim=-1)) * emb.sin()

    def forward(self, x, context=None):
        # x: [Batch, Seq_Q, Dim] (Latents)
        # context: [Batch, Seq_KV, Dim] (Byte Array or Latents)

        b, seq_q, _ = x.shape
        kv_input = context if context is not None else x
        seq_kv = kv_input.shape[1]

        # --- 1. Query Processing ---
        # Compress
        c_q = self.q_norm(self.q_down(x))
        # Generate Content & RoPE parts
        q_content = self.q_up(c_q).view(b, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        q_rope = self.q_rope(c_q).view(b, seq_q, self.num_heads, self.qk_rope_dim).transpose(1, 2)
        # Apply RoPE
        q_rope = self.apply_rope(q_rope, seq_q)
        # Concatenate (MLA strategy: [Content, RoPE]) for Dot Product
        # Q becomes: [Batch, Heads, Seq, Head_Dim + RoPE_Dim]
        q = torch.cat([q_content, q_rope], dim=-1)

        # --- 2. Key/Value Processing ---
        # Compress
        c_kv = self.kv_norm(self.kv_down(kv_input))
        # Generate Content & RoPE parts
        k_content = self.k_up(c_kv).view(b, seq_kv, self.num_heads, self.head_dim).transpose(1, 2)
        k_rope = self.k_rope(c_kv).view(b, seq_kv, self.num_heads, self.qk_rope_dim).transpose(1, 2)
        v = self.v_up(c_kv).view(b, seq_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Keys
        k_rope = self.apply_rope(k_rope, seq_kv)
        k = torch.cat([k_content, k_rope], dim=-1)

        # --- 3. Attention ---
        # Scaled Dot Product
        # q: [B, H, Sq, D+R], k: [B, H, Sk, D+R]
        scale = (self.head_dim + self.qk_rope_dim) ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # --- 4. Output ---
        out = (attn @ v).transpose(1, 2).reshape(b, seq_q, -1)
        return self.out_proj(out)


class NativeSparseCrossAttention(nn.Module):
    """
    Implements DeepSeek's Native Sparse Attention (NSA) for Cross-Attention.
    Strategy: Coarse-grained Top-K Block Selection.

    1. Divide Input (Byte Array) into fixed-size blocks.
    2. Use a lightweight router to score (Latents vs Blocks).
    3. Select Top-K blocks.
    4. Gather High-Res features for those blocks.
    5. Perform MLA only on selected blocks.
    """
    def __init__(self, mla_module, input_dim, block_size=64, top_k_blocks=16):
        super().__init__()
        self.mla = mla_module
        self.block_size = block_size
        self.top_k = top_k_blocks

        # Lightweight Router (Coarse Attention)
        # Projects Latents and Block-Summaries to a tiny space to calculate relevance scores
        router_dim = 64
        self.router_q = nn.Linear(mla_module.d_model, router_dim)
        # We assume the router reads the same "compressed" latent from the input adapter
        self.router_k = nn.Linear(input_dim, router_dim)

    def forward(self, latents, byte_array):
        """
        latents: [Batch, Num_Latents, Dim]
        byte_array: [Batch, Length, Input_Dim] (Huge sequence)
        """
        B, N_L, D = latents.shape
        B, N_T, D_in = byte_array.shape

        # 1. Padding (Ensure length is divisible by block_size)
        pad_len = (self.block_size - (N_T % self.block_size)) % self.block_size
        if pad_len > 0:
            byte_array = F.pad(byte_array, (0, 0, 0, pad_len))
            N_T += pad_len

        num_blocks = N_T // self.block_size

        # 2. Reshape into Blocks: [Batch, Num_Blocks, Block_Size, Input_Dim]
        byte_blocks = byte_array.view(B, num_blocks, self.block_size, D_in)

        # 3. Create Block Summaries (Mean Pooling) for Routing
        # [Batch, Num_Blocks, Input_Dim]
        block_summaries = byte_blocks.mean(dim=2)

        # 4. Routing (Coarse-Grained Attention Scores)
        # r_q: [B, N_L, 64], r_k: [B, Num_Blocks, 64]
        r_q = self.router_q(latents)
        r_k = self.router_k(block_summaries)

        # Scores: [B, N_L, Num_Blocks]
        router_logits = torch.matmul(r_q, r_k.transpose(-1, -2))

        # 5. Top-K Selection
        # indices: [B, N_L, Top_K]
        topk_scores, topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1)

        # 6. Gather Selected Blocks for MLA
        # We need to construct the specific Key/Value context for *each* latent based on its top-k blocks.
        # However, standard Attention expects a shared K/V sequence or a complex mask.
        # DeepSeek-V3 Optimization: Gather the actual data.

        # Expansion for gathering: [B, N_L, Top_K, Block_Size, Input_Dim]
        # This can be memory intensive if implemented naively.
        # Efficient PyTorch: Flatten indices to gather blocks.

        # Gather logic:
        # We want to form a "Selected Context" of size [Batch, N_L, Top_K * Block_Size, Input_Dim]
        # But this makes the context size dependent on N_L (Num Latents).
        # Standard MLA cross-attention does Q(N_L) x K(Context).
        # If every latent has a DIFFERENT context, we cannot use standard matmul (batching breaks).

        # Solution (Perceiver-specific):
        # Since Perceiver Latents (512) are relatively small, we can treat the "Batch" dimension
        # as (Batch * Num_Latents) for the purpose of this attention step if we want perfect per-latent sparsity.
        # OR: We enforce that we select the Top-K blocks *globally* for the whole image/audio?
        # NO, DeepSeek NSA is fine-grained. Each token (latent) picks its own blocks.

        # Re-batching for Attention:
        # New Batch Size = B * N_L
        # Query = latents.view(B * N_L, 1, D)

        # Gather indices: [B, N_L, Top_K] -> expand to [B, N_L, Top_K, Block_Size, D_in]
        # For efficiency in Python, we simply gather the block summaries or full blocks.

        # index selection:
        # batch_indices: [B, 1, 1]
        batch_indices = torch.arange(B, device=latents.device).view(B, 1, 1)
        # final indices: [B, N_L, Top_K]

        # Gather the blocks:
        # byte_blocks: [B, Num_Blocks, Block_Size, D_in]
        selected_blocks = byte_blocks[batch_indices, topk_indices] # [B, N_L, Top_K, Block_Size, D_in]

        # Flatten the selected context: [B * N_L, Top_K * Block_Size, D_in]
        sparse_context = selected_blocks.view(B * N_L, self.top_k * self.block_size, D_in)

        # Flatten Queries: [B * N_L, 1, D]
        flat_latents = latents.view(B * N_L, 1, D)

        # 7. Run MLA on Sparse Context
        # Note: We must ensure MLA handles this reshaped batch size
        # MLA forward expects [New_Batch, Seq, Dim]
        out_flat = self.mla(flat_latents, context=sparse_context)

        # 8. Reshape back
        # [B * N_L, 1, D] -> [B, N_L, D]
        return out_flat.view(B, N_L, D)