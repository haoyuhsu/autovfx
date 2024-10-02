from .utils import *


class EquirectRotate:
  """
    @:param height: height of image to rotate
    @:param width: widht of image to rotate
    @:param rotation: x, y, z degree to rotate
  """

  def __init__(self, height: int, width: int, rotate_angle: tuple, rotation_matrix: np.ndarray = None):
    assert height * 2 == width
    self.height = height
    self.width = width

    out_img = np.zeros((height, width))  # (H, W)

    # mapping equirect coordinate into LatLon coordinate system
    self.out_LonLat = Pixel2LatLon(out_img)  # (H, W, (lat, lon))

    # mapping LatLon coordinate into xyz(sphere) coordinate system
    self.out_xyz = LatLon2Sphere(self.out_LonLat)  # (H, W, (x, y, z))

    self.src_xyz = np.zeros_like(self.out_xyz)  # (H, W, (x, y, z))

    # make pair of xyz coordinate between src_xyz and out_xyz.
    # src_xyz @ R = out_xyz
    # src_xyz     = out_xyz @ R^t
    if rotation_matrix is not None:
      self.R = rotation_matrix
    else:
      self.R = getRotMatrix(np.array(rotate_angle))
    Rt = np.transpose(self.R)  # we should fill out the output image, so use R^t.
    for i in range(self.height):
      for j in range(self.width):
        self.src_xyz[i][j] = self.out_xyz[i][j] @ Rt

    # mapping xyz(sphere) coordinate into LatLon coordinate system
    self.src_LonLat = Sphere2LatLon(self.src_xyz)  # (H, W, 2)

    # mapping LatLon coordinate into equirect coordinate system
    self.src_Pixel = LatLon2Pixel(self.src_LonLat)  # (H, W, 2)

  def rotate(self, image):
    """
    :param image: (H, W, C)
    :return: rotated (H, W, C)
    """
    assert image.shape[:2] == (self.height, self.width)

    rotated_img = np.zeros_like(image)  # (H, W, C)
    for i in range(self.height):
      for j in range(self.width):
        pixel = self.src_Pixel[i][j]
        rotated_img[i][j] = image[pixel[0]][pixel[1]]
    return rotated_img

  def setInverse(self, isInverse=False):
    if not isInverse:
      return

    # re-generate coordinate pairing
    self.R = np.transpose(self.R)
    Rt = np.transpose(self.R)  # we should fill out the output image, so use R^t.
    for i in range(self.height):
      for j in range(self.width):
        self.src_xyz[i][j] = self.out_xyz[i][j] @ Rt

    # mapping xyz(sphere) coordinate into LatLon coordinate system
    self.src_LonLat = Sphere2LatLon(self.src_xyz)  # (H, W, 2)

    # mapping LatLon coordinate into equirect coordinate system
    self.src_Pixel = LatLon2Pixel(self.src_LonLat)  # (H, W, 2)


def pointRotate(h, w, index, rotation):
  """
  :param i, j: index of pixel in equirectangular
  :param rotation: (yaw, pitch, roll) in degree
  :return: rotated index of pixel
  """
  i, j = index
  assert (0 <= i < h) and (0 <= j < w)

  # convert pixel index to LatLon
  Lat = (0.5 - i / h) * np.pi
  Lon = (j / w - 0.5) * 2 * np.pi

  # convert LatLon to xyz
  x = np.cos(Lat) * np.cos(Lon)
  y = np.cos(Lat) * np.sin(Lon)
  z = np.sin(Lat)
  xyz = np.array([x, y, z])
  R = getRotMatrix(np.array(rotation))
  rotated_xyz = xyz @ R

  _Lat = np.pi / 2 - np.arccos(rotated_xyz[2])
  _Lon = np.arctan2(rotated_xyz[1], rotated_xyz[0])

  _i = h * (0.5 - _Lat / np.pi) % h
  _j = w * (0.5 + _Lon / (2 * np.pi)) % w

  return _i, _j
