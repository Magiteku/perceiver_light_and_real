import torch
import torch.nn.functional as F
import math
import numpy as np

def mesh_to_gaussian_splatting(mesh_path: str, num_points: int = 10000, sh_degree: int = 3, device: str = 'cpu') -> torch.Tensor:
    """
    Utility function to convert a 3D mesh into Gaussian Splatting parameters.
    This is useful for initializing 'static 3d' scenes or creating training data.

    Args:
        mesh_path (str): Path to the mesh file (.obj, .ply, .stl, etc.)
        num_points (int): Number of Gaussians to sample from the mesh surface.
        sh_degree (int): Degree of Spherical Harmonics (default 3).
        device (str): Torch device to store the output tensor.

    Returns:
        torch.Tensor: Tensor of shape [num_points, 3 + 4 + 3 + 1 + sh_dim]
                      containing [Position(3), Rotation(4), Scale(3), Opacity(1), SH(N)].
    """
    try:
        import trimesh
        import numpy as np
    except ImportError:
        raise ImportError("This function requires 'trimesh' and 'numpy'. Please install them: pip install trimesh numpy")

    # 1. Load the Mesh
    try:
        mesh = trimesh.load(mesh_path, force='mesh')
    except Exception as e:
        raise ValueError(f"Failed to load mesh from {mesh_path}: {e}")

    # 2. Sample Points from Surface
    # trimesh.sample.sample_surface returns (points, face_indices)
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)

    # 3. Initialize Attributes

    # A. Position (XYZ)
    xyz = torch.tensor(points, dtype=torch.float32, device=device)

    # B. Scale
    # Heuristic: Use average distance between points or a fraction of bounding box
    # to ensure Gaussians cover the surface without too much overlap.
    # Volume of bounding box / num_points gives approx volume per point.
    if hasattr(mesh, 'bounding_box') and hasattr(mesh.bounding_box, 'volume'):
        vol = mesh.bounding_box.volume
    else:
        # Fallback if volume calc fails
        vol = (mesh.extents ** 3).prod() if hasattr(mesh, 'extents') else 1.0

    avg_vol_per_point = vol / max(num_points, 1)
    radius = (avg_vol_per_point ** (1/3))
    # Use log-scale as is common in 3DGS optimization
    log_scale = math.log(max(radius, 1e-6))
    scale = torch.full((num_points, 3), log_scale, dtype=torch.float32, device=device)

    # C. Rotation (Quaternion W, X, Y, Z)
    # Initialize to Identity (1, 0, 0, 0) - No rotation
    rot = torch.zeros((num_points, 4), dtype=torch.float32, device=device)
    rot[:, 0] = 1.0

    # D. Opacity (Logit)
    # Initialize to opaque. sigmoid(x) -> 1.
    # Inverse sigmoid(0.99) approx 4.6.
    opacity = torch.full((num_points, 1), 4.6, dtype=torch.float32, device=device)

    # E. Spherical Harmonics (Color)
    # Extract colors from mesh if available
    colors = None

    # Try fetching face colors using indices
    if hasattr(mesh.visual, 'face_colors') and len(mesh.visual.face_colors) > 0:
        # face_colors is usually [N_faces, 4] (RGBA) (uint8)
        sampled_colors = mesh.visual.face_colors[face_indices]
        rgb = sampled_colors[:, :3].astype(np.float32) / 255.0
    else:
        # Default Grey
        rgb = np.ones((num_points, 3), dtype=np.float32) * 0.5

    rgb_tensor = torch.tensor(rgb, dtype=torch.float32, device=device)

    # Convert RGB to SH DC component (0th order)
    # Standard 3DGS formula: RGB = 0.5 + C0 * SH_0
    # SH_0 = (RGB - 0.5) / C0
    C0 = 0.28209479177387814
    sh_dc = (rgb_tensor - 0.5) / C0

    # Create zero-filled tensor for higher order SH coefficients
    # Total coeffs per channel = (degree + 1)^2
    # We used 1 (DC), need ((degree+1)^2 - 1) more
    num_coeffs = (sh_degree + 1) ** 2
    num_extra = num_coeffs - 1

    if num_extra > 0:
        extra_sh = torch.zeros((num_points, num_extra * 3), dtype=torch.float32, device=device)
        # Note: We need to be careful about interleaving if the adapter expects it.
        # Assuming simple concatenation of DC + Rest for now.
        sh_features = torch.cat([sh_dc, extra_sh], dim=1)
    else:
        sh_features = sh_dc

    # 4. Concatenate All Features
    # Order: [Pos(3), Rot(4), Scale(3), Opacity(1), SH(N)]
    gaussians = torch.cat([xyz, rot, scale, opacity, sh_features], dim=-1)

    return gaussians


def extract_specs_from_batch(batch: dict) -> dict:
    """
    Inspects a batch dictionary (from TaskDatasetAdapter) and generates
    the output_specs required by the MultimodalDecoder.

    Args:
        batch (dict): Batch containing 'pixel_values' (images/audio), 'input_ids' (text), etc.
                      Keys should map to specific modalities.

    Returns:
        dict: output_specs dictionary (e.g., {'image': (H, W), 'text': SeqLen})
    """
    specs = {}

    # 1. Image / Video / Audio (Grid-based)
    # Assumes keys match the adapter names defined in your Decoder
    if 'image' in batch:
        # shape: [Batch, C, H, W] -> Spec: (H, W)
        shape = batch['image'].shape
        specs['image'] = (shape[2], shape[3])

    if 'video' in batch:
        # shape: [Batch, C, T, H, W] -> Spec: (T, H, W)
        shape = batch['video'].shape
        specs['video'] = (shape[2], shape[3], shape[4])

    if 'audio' in batch:
        # shape: [Batch, 1, Freq, Time] -> Spec: (Freq, Time)
        shape = batch['audio'].shape
        specs['audio'] = (shape[2], shape[3])

    # 2. Text / Sequences (Index-based)
    if 'text' in batch:
        # shape: [Batch, Seq_Len] -> Spec: Seq_Len (int)
        specs['text'] = batch['text'].shape[1]

    if '3d' in batch:
        # shape: [Batch, Num_Points, 3] -> Spec: Num_Points (int)
        specs['3d'] = batch['3d'].shape[1]

    return specs


def calculate_multimodal_loss(predictions: dict, targets: dict, masks: dict = None) -> dict:
    """
    Calculates reconstruction loss for all available modalities.
    Handles masking for variable-length sequences.

    Args:
        predictions (dict): Output from MultimodalDecoder.
        targets (dict): Original inputs (Ground Truth).
        masks (dict, optional): Boolean masks for sequences (True=Valid, False=Pad).

    Returns:
        dict: {'total_loss': tensor, 'image_loss': tensor, ...}
    """
    total_loss = 0.0
    losses = {}

    # --- 1. Image / Video / Audio (MSE Loss) ---
    # Usually fixed size within a batch, or handled via reshaping.
    for mod in ['image', 'video', 'audio', '3d']:
        if mod in predictions and mod in targets:
            pred = predictions[mod]
            tgt = targets[mod]

            # Standard MSE
            loss = F.mse_loss(pred, tgt)
            losses[f'{mod}_loss'] = loss
            total_loss += loss

    # --- 2. Text (Cross Entropy) ---
    if 'text' in predictions and 'text' in targets:
        pred = predictions['text'] # [Batch, Seq, Vocab_Size]
        tgt = targets['text']      # [Batch, Seq] (Indices)

        # Flatten for CrossEntropy: [Batch*Seq, Vocab] vs [Batch*Seq]
        pred_flat = pred.reshape(-1, pred.size(-1))
        tgt_flat = tgt.reshape(-1)

        if masks is not None and 'text' in masks:
            # Apply Mask: Only compute loss on valid tokens
            mask_flat = masks['text'].reshape(-1).bool()

            # Select valid indices
            pred_active = pred_flat[mask_flat]
            tgt_active = tgt_flat[mask_flat]

            if len(tgt_active) > 0:
                loss = F.cross_entropy(pred_active, tgt_active)
            else:
                loss = torch.tensor(0.0, device=pred.device)
        else:
            # Standard Loss (assumes padding index is handled or minimal)
            # ignore_index=-100 is standard in PyTorch/HuggingFace
            loss = F.cross_entropy(pred_flat, tgt_flat, ignore_index=-100)

        losses['text_loss'] = loss
        total_loss += loss

    losses['total_loss'] = total_loss
    return losses