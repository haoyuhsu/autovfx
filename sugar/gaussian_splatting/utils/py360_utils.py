# This code is borrowed from https://github.com/sunset1995/py360convert/

import numpy as np
from scipy.ndimage import map_coordinates


def c2e(cubemap, h, w, mode='bilinear', cube_format='dice'):
    if mode == 'bilinear':
        order = 1
    elif mode == 'nearest':
        order = 0
    else:
        raise NotImplementedError('unknown mode')

    if cube_format == 'horizon':
        pass
    elif cube_format == 'list':
        cubemap = cube_list2h(cubemap)
    elif cube_format == 'dict':
        cubemap = cube_dict2h(cubemap)
    elif cube_format == 'dice':
        cubemap = cube_dice2h(cubemap)
    else:
        raise NotImplementedError('unknown cube_format')
    assert len(cubemap.shape) == 3
    assert cubemap.shape[0] * 6 == cubemap.shape[1]
    assert w % 8 == 0
    face_w = cubemap.shape[0]

    uv = equirect_uvgrid(h, w)
    u, v = np.split(uv, 2, axis=-1)
    u = u[..., 0]
    v = v[..., 0]
    cube_faces = np.stack(np.split(cubemap, 6, 1), 0)

    # Get face id to each pixel: 0F 1R 2B 3L 4U 5D
    tp = equirect_facetype(h, w)
    coor_x = np.zeros((h, w))
    coor_y = np.zeros((h, w))

    for i in range(4):
        mask = (tp == i)
        coor_x[mask] = 0.5 * np.tan(u[mask] - np.pi * i / 2)
        coor_y[mask] = -0.5 * np.tan(v[mask]) / np.cos(u[mask] - np.pi * i / 2)

    mask = (tp == 4)
    c = 0.5 * np.tan(np.pi / 2 - v[mask])
    coor_x[mask] = c * np.sin(u[mask])
    coor_y[mask] = c * np.cos(u[mask])

    mask = (tp == 5)
    c = 0.5 * np.tan(np.pi / 2 - np.abs(v[mask]))
    coor_x[mask] = c * np.sin(u[mask])
    coor_y[mask] = -c * np.cos(u[mask])

    # Final renormalize
    coor_x = (np.clip(coor_x, -0.5, 0.5) + 0.5) * face_w
    coor_y = (np.clip(coor_y, -0.5, 0.5) + 0.5) * face_w

    equirec = np.stack([
        sample_cubefaces(cube_faces[..., i], tp, coor_y, coor_x, order=order)
        for i in range(cube_faces.shape[3])
    ], axis=-1)

    return equirec


def xyzcube(face_w):
    '''
    Return the xyz cordinates of the unit cube in [F R B L U D] format.
    '''
    out = np.zeros((face_w, face_w * 6, 3), np.float32)
    rng = np.linspace(-0.5, 0.5, num=face_w, dtype=np.float32)
    grid = np.stack(np.meshgrid(rng, -rng), -1)

    # Front face (z = 0.5)
    out[:, 0*face_w:1*face_w, [0, 1]] = grid
    out[:, 0*face_w:1*face_w, 2] = 0.5

    # Right face (x = 0.5)
    grid_r = np.flip(grid, axis=1)
    out[:, 1*face_w:2*face_w, [2, 1]] = grid_r
    out[:, 1*face_w:2*face_w, 0] = 0.5

    # Back face (z = -0.5)
    grid_b = np.flip(grid, axis=1)
    out[:, 2*face_w:3*face_w, [0, 1]] = grid_b
    out[:, 2*face_w:3*face_w, 2] = -0.5

    # Left face (x = -0.5)
    out[:, 3*face_w:4*face_w, [2, 1]] = grid
    out[:, 3*face_w:4*face_w, 0] = -0.5

    # Up face (y = 0.5)
    grid_u = np.flip(grid, axis=0)
    out[:, 4*face_w:5*face_w, [0, 2]] = grid_u
    out[:, 4*face_w:5*face_w, 1] = 0.5

    # Down face (y = -0.5)
    out[:, 5*face_w:6*face_w, [0, 2]] = grid
    out[:, 5*face_w:6*face_w, 1] = -0.5

    return out


def equirect_uvgrid(h, w):
    u = np.linspace(-np.pi, np.pi, num=w, dtype=np.float32)
    v = np.linspace(np.pi, -np.pi, num=h, dtype=np.float32) / 2

    return np.stack(np.meshgrid(u, v), axis=-1)


def equirect_facetype(h, w):
    '''
    0F 1R 2B 3L 4U 5D
    '''
    tp = np.roll(np.arange(4).repeat(w // 4)[None, :].repeat(h, 0), 3 * w // 8, 1)

    # Prepare ceil mask
    mask = np.zeros((h, w // 4))
    idx = np.linspace(-np.pi, np.pi, w // 4) / 4
    idx = h // 2 - np.round(np.arctan(np.cos(idx)) * h / np.pi).astype(int)
    for i, j in enumerate(idx):
        mask[:j, i] = 1
    mask = np.roll(np.concatenate([mask] * 4, 1), 3 * w // 8, 1)

    tp[mask.astype(bool)] = 4
    tp[np.flip(mask, 0).astype(bool)] = 5

    return tp.astype(np.int32)


def xyzpers(h_fov, v_fov, u, v, out_hw, in_rot):
    out = np.ones((*out_hw, 3), np.float32)

    x_max = np.tan(h_fov / 2)
    y_max = np.tan(v_fov / 2)
    x_rng = np.linspace(-x_max, x_max, num=out_hw[1], dtype=np.float32)
    y_rng = np.linspace(-y_max, y_max, num=out_hw[0], dtype=np.float32)
    out[..., :2] = np.stack(np.meshgrid(x_rng, -y_rng), -1)
    Rx = rotation_matrix(v, [1, 0, 0])
    Ry = rotation_matrix(u, [0, 1, 0])
    Ri = rotation_matrix(in_rot, np.array([0, 0, 1.0]).dot(Rx).dot(Ry))

    return out.dot(Rx).dot(Ry).dot(Ri)


def xyz2uv(xyz):
    '''
    xyz: ndarray in shape of [..., 3]
    '''
    x, y, z = np.split(xyz, 3, axis=-1)
    u = np.arctan2(x, z)
    c = np.sqrt(x**2 + z**2)
    v = np.arctan2(y, c)

    return np.concatenate([u, v], axis=-1)


def uv2unitxyz(uv):
    u, v = np.split(uv, 2, axis=-1)
    y = np.sin(v)
    c = np.cos(v)
    x = c * np.sin(u)
    z = c * np.cos(u)

    return np.concatenate([x, y, z], axis=-1)


def uv2coor(uv, h, w):
    '''
    uv: ndarray in shape of [..., 2]
    h: int, height of the equirectangular image
    w: int, width of the equirectangular image
    '''
    u, v = np.split(uv, 2, axis=-1)
    coor_x = (u / (2 * np.pi) + 0.5) * w - 0.5
    coor_y = (-v / np.pi + 0.5) * h - 0.5

    return np.concatenate([coor_x, coor_y], axis=-1)


def coor2uv(coorxy, h, w):
    coor_x, coor_y = np.split(coorxy, 2, axis=-1)
    u = ((coor_x + 0.5) / w - 0.5) * 2 * np.pi
    v = -((coor_y + 0.5) / h - 0.5) * np.pi

    return np.concatenate([u, v], axis=-1)


def sample_equirec(e_img, coor_xy, order):
    w = e_img.shape[1]
    coor_x, coor_y = np.split(coor_xy, 2, axis=-1)
    pad_u = np.roll(e_img[[0]], w // 2, 1)
    pad_d = np.roll(e_img[[-1]], w // 2, 1)
    e_img = np.concatenate([e_img, pad_d, pad_u], 0)
    return map_coordinates(e_img, [coor_y, coor_x],
                           order=order, mode='wrap')[..., 0]


def sample_cubefaces(cube_faces, tp, coor_y, coor_x, order):
    cube_faces = cube_faces.copy()
    # cube_faces[1] = np.flip(cube_faces[1], 1)
    # cube_faces[2] = np.flip(cube_faces[2], 1)
    # cube_faces[4] = np.flip(cube_faces[4], 0)

    # Pad up down
    pad_ud = np.zeros((6, 2, cube_faces.shape[2]))
    pad_ud[0, 0] = cube_faces[5, 0, :]
    pad_ud[0, 1] = cube_faces[4, -1, :]
    pad_ud[1, 0] = cube_faces[5, :, -1]
    pad_ud[1, 1] = cube_faces[4, ::-1, -1]
    pad_ud[2, 0] = cube_faces[5, -1, ::-1]
    pad_ud[2, 1] = cube_faces[4, 0, ::-1]
    pad_ud[3, 0] = cube_faces[5, ::-1, 0]
    pad_ud[3, 1] = cube_faces[4, :, 0]
    pad_ud[4, 0] = cube_faces[0, 0, :]
    pad_ud[4, 1] = cube_faces[2, 0, ::-1]
    pad_ud[5, 0] = cube_faces[2, -1, ::-1]
    pad_ud[5, 1] = cube_faces[0, -1, :]
    cube_faces = np.concatenate([cube_faces, pad_ud], 1)

    # Pad left right
    pad_lr = np.zeros((6, cube_faces.shape[1], 2))
    pad_lr[0, :, 0] = cube_faces[1, :, 0]
    pad_lr[0, :, 1] = cube_faces[3, :, -1]
    pad_lr[1, :, 0] = cube_faces[2, :, 0]
    pad_lr[1, :, 1] = cube_faces[0, :, -1]
    pad_lr[2, :, 0] = cube_faces[3, :, 0]
    pad_lr[2, :, 1] = cube_faces[1, :, -1]
    pad_lr[3, :, 0] = cube_faces[0, :, 0]
    pad_lr[3, :, 1] = cube_faces[2, :, -1]
    pad_lr[4, 1:-1, 0] = cube_faces[1, 0, ::-1]
    pad_lr[4, 1:-1, 1] = cube_faces[3, 0, :]
    pad_lr[5, 1:-1, 0] = cube_faces[1, -2, :]
    pad_lr[5, 1:-1, 1] = cube_faces[3, -2, ::-1]
    cube_faces = np.concatenate([cube_faces, pad_lr], 2)

    return map_coordinates(cube_faces, [tp, coor_y, coor_x], order=order, mode='wrap')


def cube_h2list(cube_h):
    assert cube_h.shape[0] * 6 == cube_h.shape[1]
    return np.split(cube_h, 6, axis=1)


def cube_list2h(cube_list):
    assert len(cube_list) == 6
    assert sum(face.shape == cube_list[0].shape for face in cube_list) == 6
    return np.concatenate(cube_list, axis=1)


def cube_h2dict(cube_h):
    cube_list = cube_h2list(cube_h)
    return dict([(k, cube_list[i])
                 for i, k in enumerate(['front', 'right', 'back', 'left', 'up', 'down'])])


def cube_dict2h(cube_dict, face_k=['front', 'right', 'back', 'left', 'up', 'down']):
    assert len(face_k) == 6
    return cube_list2h([cube_dict[k] for k in face_k])


def cube_h2dice(cube_h):
    assert cube_h.shape[0] * 6 == cube_h.shape[1]
    w = cube_h.shape[0]
    cube_dice = np.zeros((w * 3, w * 4, cube_h.shape[2]), dtype=cube_h.dtype)
    cube_list = cube_h2list(cube_h)
    # Order: F R B L U D
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]
    for i, (sx, sy) in enumerate(sxy):
        face = cube_list[i]
        if i in [1, 2]:
            face = np.flip(face, axis=1)
        if i == 4:
            face = np.flip(face, axis=0)
        cube_dice[sy*w:(sy+1)*w, sx*w:(sx+1)*w] = face
    return cube_dice


def cube_dice2h(cube_dice):
    w = cube_dice.shape[0] // 3
    assert cube_dice.shape[0] == w * 3 and cube_dice.shape[1] == w * 4
    cube_h = np.zeros((w, w * 6, cube_dice.shape[2]), dtype=cube_dice.dtype)
    # Order: F R B L U D
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]
    for i, (sx, sy) in enumerate(sxy):
        face = cube_dice[sy*w:(sy+1)*w, sx*w:(sx+1)*w]
        if i in [1, 2]:
            face = np.flip(face, axis=1)
        if i == 4:
            face = np.flip(face, axis=0)
        cube_h[:, i*w:(i+1)*w] = face
    return cube_h


def rotation_matrix(rad, ax):
    ax = np.array(ax)
    assert len(ax.shape) == 1 and ax.shape[0] == 3
    ax = ax / np.sqrt((ax**2).sum())
    R = np.diag([np.cos(rad)] * 3)
    R = R + np.outer(ax, ax) * (1.0 - np.cos(rad))

    ax = ax * np.sin(rad)
    R = R + np.array([[0, -ax[2], ax[1]],
                      [ax[2], 0, -ax[0]],
                      [-ax[1], ax[0], 0]])

    return R


if __name__ == '__main__':
    import argparse
    import os
    from PIL import Image
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help='The path to the input directory of 6 cube maps.')
    parser.add_argument('--width', type=int, default=2048,
                        help='The width of the equirectangular image.')
    parser.add_argument('--height', type=int, default=1024,
                        help='The height of the equirectangular image.')
    parser.add_argument('--blend', action='store_true', default=False,
                        help='Whether to alpha blend the inpainted panorama.')
    args = parser.parse_args()

    up = Image.open(os.path.join(args.input_dir, 'up.png'))
    down = Image.open(os.path.join(args.input_dir, 'down.png'))
    left = Image.open(os.path.join(args.input_dir, 'left.png'))
    right = Image.open(os.path.join(args.input_dir, 'right.png'))
    front = Image.open(os.path.join(args.input_dir, 'front.png'))
    back = Image.open(os.path.join(args.input_dir, 'back.png'))

    up = np.array(up) / 255.
    down = np.array(down) / 255.
    left = np.array(left) / 255.
    right = np.array(right) / 255.
    front = np.array(front) / 255.
    back = np.array(back) / 255.

    cube_map_dict = {
        'up': np.array(up),
        'down': np.array(down),
        'left': np.array(left),
        'right': np.array(right),
        'front': np.array(front),
        'back': np.array(back)
    }

    # convert (h, w, 3) cube map into (h, w, 4) cube map by adding alpha channel to 1.0
    for face in cube_map_dict:
        if cube_map_dict[face].shape[-1] == 3:
            cube_map_dict[face] = np.concatenate([cube_map_dict[face], np.ones_like(cube_map_dict[face][..., :1])], axis=-1)

    equirectangular = c2e(cube_map_dict, args.height, args.width, mode='bilinear', cube_format='dict')

    if args.blend:
        # check if inpainted panorama exists
        if os.path.exists(os.path.join(args.input_dir, 'equirectangular.png')):
            inpainted_pano = Image.open(os.path.join(args.input_dir, 'equirectangular.png'))
            inpainted_pano = np.array(inpainted_pano) / 255.
            if inpainted_pano.shape[-1] == 3:
                inpainted_pano = np.concatenate([inpainted_pano, np.ones_like(inpainted_pano[..., :1])], axis=-1)
            # alpha composite
            alpha = equirectangular[..., 3]
            equirectangular = alpha[:, :, None] * equirectangular + (1 - alpha[:, :, None]) * inpainted_pano

    equirectangular = Image.fromarray(np.clip(equirectangular * 255, 0, 255).astype(np.uint8))
    equirectangular.save(os.path.join(args.input_dir, 'equirectangular_composite.png'))