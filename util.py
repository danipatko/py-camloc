import numpy as np


def save(camera: str, cameraMatrix, distCoeffs, newCameraMatrix):
    np.save(camera + "/cameramatrix", cameraMatrix)
    np.save(camera + "/distcoeffs", distCoeffs)
    np.save(camera + "/newcameramatrix", newCameraMatrix)


def save_remap(camera: str, map1, map2):
    np.save(camera + "/map1", map1)
    np.save(camera + "/map2", map2)


def load(camera: str):
    cameraMatrix = np.load(camera + "/cameramatrix.npy")
    distCoeffs = np.load(camera + "/distcoeffs.npy")
    newCameraMatrix = np.load(camera + "/newcameramatrix.npy")

    return (cameraMatrix, distCoeffs, newCameraMatrix)


def load_remap(camera: str):
    map1 = np.load(camera + "/map1.npy")
    map2 = np.load(camera + "/map2.npy")

    return (map1, map2)
