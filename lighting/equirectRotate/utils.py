import numpy as np

def getRotMatrix(rotation):
  """
  :param rotation: (yaw, pitch, roll) in degree
  :return: general rotational matrix
  refer this: https://en.wikipedia.org/wiki/Rotation_matrix#General_rotations
  """
  yaw, pitch, roll = (rotation / 180) * np.pi

  Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                 [np.sin(yaw), np.cos(yaw), 0],
                 [0, 0, 1]])
  Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                 [0, 1, 0],
                 [-np.sin(pitch), 0, np.cos(pitch)]])
  Rx = np.array([[1, 0, 0],
                 [0, np.cos(roll), -np.sin(roll)],
                 [0, np.sin(roll), np.cos(roll)]])

  return Rz @ Ry @ Rx

def Pixel2LatLon(equirect):
  # LatLon (H, W, (lat, lon))
  h, w = equirect.shape

  Lat = (0.5 - np.arange(0, h)/h) * np.pi
  Lon = (np.arange(0, w) / w - 0.5) * 2 * np.pi

  Lat = np.tile(Lat[:, np.newaxis], w)
  Lon = np.tile(Lon, (h, 1))

  return np.dstack((Lat, Lon))

def LatLon2Sphere(LatLon):
  Lat = LatLon[:, :, 0]
  Lon = LatLon[:, :, 1]
  x = np.cos(Lat) * np.cos(Lon)
  y = np.cos(Lat) * np.sin(Lon)
  z = np.sin(Lat)

  return np.dstack((x, y, z))

def Sphere2LatLon(xyz):
  Lat = np.pi / 2 - np.arccos(xyz[:, :, 2])
  Lon = np.arctan2(xyz[:, :, 1], xyz[:, :, 0])

  return np.dstack((Lat, Lon))


def LatLon2Pixel(LatLon):
  h, w, _ = LatLon.shape
  Lat = LatLon[:, :, 0]
  Lon = LatLon[:, :, 1]
  i = (h * (0.5 - Lat / np.pi)) % h
  j = (w * (0.5 + Lon / (2 * np.pi))) % w

  return np.dstack((i, j)).astype('int')