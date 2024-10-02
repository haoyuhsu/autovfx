import os
import torch
import torch.nn.functional as F
import einops
from e3nn import o3


"""
Some functions are borrowed from PhysDreamer: https://github.com/a1600012888/PhysDreamer/blob/main/physdreamer/gaussian_3d/utils/rigid_body_utils.py
"""


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    from pytorch3d. Based on trace_method like: https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L205
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def quternion_to_matrix(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    From pytorch3d
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    ret = torch.stack((ow, ox, oy, oz), -1)
    ret = standardize_quaternion(ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    from Pytorch3d
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def transform_shs(shs_feat, rot_rotation_matrix):
    """
    Transform spherical harmonics features with rotation matrix
    Borrowed from: https://github.com/graphdeco-inria/gaussian-splatting/issues/176#issuecomment-2060513169
    TODO: this function has not been tested
    """
    #degree 1 transformation for now 
    # frist_degree_shs = shs_feat[:, 0:1]
    # permuting the last rgb to brg
    mat = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).to(device=shs_feat.device).float()

    rot_angles = o3._rotation.matrix_to_angles(rot_rotation_matrix.cpu())
    #Construction coefficient
    D_0 = o3.wigner_D(0, rot_angles[0], rot_angles[1], rot_angles[2]).cuda()
    D_1 = o3.wigner_D(1, rot_angles[0], rot_angles[1], rot_angles[2]).cuda()
    D_2 = o3.wigner_D(2, rot_angles[0], rot_angles[1], rot_angles[2]).cuda()
    D_3 = o3.wigner_D(3, rot_angles[0], rot_angles[1], rot_angles[2]).cuda()

    #rotation of the shs features
    two_degree_shs = shs_feat[:, 0:3]
    two_degree_shs = torch.matmul(two_degree_shs, mat)
    two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    two_degree_shs = torch.matmul(two_degree_shs, D_1)
    # print(D_1.shape)
    # print(two_degree_shs.shape)
    # two_degree_shs = torch.einsum(
    #         D_1,
    #         two_degree_shs,
    #         "... i j, ... j -> ... i",
    #     )
    two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    two_degree_shs = torch.matmul(two_degree_shs, torch.inverse(mat))
    shs_feat[:, 0:3] = two_degree_shs

    three_degree_shs = shs_feat[:, 3:8]
    three_degree_shs = torch.matmul(three_degree_shs, mat)
    three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    three_degree_shs = torch.matmul(three_degree_shs, D_2)
    # three_degree_shs = torch.einsum(
    #         D_2,
    #         three_degree_shs,
    #         "... i j, ... j -> ... i",
    #     )
    three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    three_degree_shs = torch.matmul(three_degree_shs, torch.inverse(mat))
    shs_feat[:, 3:8] = three_degree_shs

    four_degree_shs = shs_feat[:, 8:15]
    four_degree_shs = torch.matmul(four_degree_shs, mat)
    four_degree_shs = einops.rearrange(four_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    four_degree_shs = torch.matmul(four_degree_shs, D_3)
    # four_degree_shs = torch.einsum(
    #         D_3,
    #         four_degree_shs,
    #         "... i j, ... j -> ... i",
    #     )
    four_degree_shs = einops.rearrange(four_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    four_degree_shs = torch.matmul(four_degree_shs, torch.inverse(mat))
    shs_feat[:, 8:15] = four_degree_shs

    return shs_feat