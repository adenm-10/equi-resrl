import torch
import math

def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle (rotation vector) to 3x3 rotation matrix using Rodrigues' formula."""
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / (angle + 1e-8)
    
    cos_a = torch.cos(angle).unsqueeze(-1)
    sin_a = torch.sin(angle).unsqueeze(-1)
    
    K = torch.zeros(*axis.shape[:-1], 3, 3, device=axis.device, dtype=axis.dtype)
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]
    
    I = torch.eye(3, device=axis.device, dtype=axis.dtype).expand_as(K)
    return I + sin_a * K + (1 - cos_a) * (K @ K)


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """Convert 3x3 rotation matrix to axis-angle."""
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}")
    
    batch_shape = matrix.shape[:-2]
    m = matrix.reshape(-1, 3, 3)
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    
    quat = torch.zeros(m.shape[0], 4, device=matrix.device, dtype=matrix.dtype)
    
    # Case 1: trace > 0
    s = torch.sqrt(torch.clamp(trace + 1, min=1e-10)) * 2
    mask = trace > 0
    quat[mask, 0] = 0.25 * s[mask]
    quat[mask, 1] = (m[mask, 2, 1] - m[mask, 1, 2]) / s[mask]
    quat[mask, 2] = (m[mask, 0, 2] - m[mask, 2, 0]) / s[mask]
    quat[mask, 3] = (m[mask, 1, 0] - m[mask, 0, 1]) / s[mask]
    
    # Case 2-4: trace <= 0
    for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        cond = (~mask) & (m[:, i, i] >= m[:, j, j]) & (m[:, i, i] >= m[:, k, k])
        s2 = torch.sqrt(torch.clamp(1.0 + m[:, i, i] - m[:, j, j] - m[:, k, k], min=1e-10)) * 2
        quat[cond, i + 1] = 0.25 * s2[cond]
        quat[cond, 0] = (m[cond, k, j] - m[cond, j, k]) / s2[cond]
        quat[cond, j + 1] = (m[cond, j, i] + m[cond, i, j]) / s2[cond]
        quat[cond, k + 1] = (m[cond, k, i] + m[cond, i, k]) / s2[cond]
        mask = mask | cond
    
    return quat.reshape(*batch_shape, 4)


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    q = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    mat = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
        2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return mat


def quaternion_to_axis_angle(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) to axis-angle."""
    q = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)
    # Ensure w > 0 for consistent axis-angle
    q = q * torch.sign(q[..., :1])
    xyz = q[..., 1:]
    sin_half = torch.norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, q[..., :1])
    axis = xyz / (sin_half + 1e-8)
    return axis * angle


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrix (Zhou et al.)."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Convert 3x3 rotation matrix to 6D representation (first two columns)."""
    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)