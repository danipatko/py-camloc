import numpy as np


def save(filename: str, cameraMatrix, distCoeffs, newCameraMatrix):
    np.save(filename + "_cameramatrix", cameraMatrix)
    np.save(filename + "_distcoeffs", distCoeffs)
    np.save(filename + "_newcameramatrix", newCameraMatrix)


def load(filename: str):
    cameraMatrix = np.load(filename + "_cameramatrix.npy")
    distCoeffs = np.load(filename + "_distcoeffs.npy")
    newCameraMatrix = np.load(filename + "_newcameramatrix.npy")

    return (cameraMatrix, distCoeffs, newCameraMatrix)
