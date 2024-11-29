import numpy as np


def applyKernels(kernels: np.ndarray, input: np.ndarray, stride: int) -> np.ndarray:
    size: int = int((input.shape[2] - kernels.shape[3]) / stride) + 1
    appliedKernels = np.zeros(shape=(input.shape[0], kernels.shape[0], size, size))
    for img in range(input.shape[0]):
        for ker in range(kernels.shape[0]):
            for x in range(size):
                for y in range(size):
                    for ch in range(input.shape[1]):
                        appliedKernels[img][ker][x][y] += np.sum(input[img, ch, x:x+3, y:y+3] * kernels[ker][ch])
    return appliedKernels


def applyRelu(input: np.ndarray) -> np.ndarray:
    appliedRelu = np.zeros(shape=input.shape)
    for img in range(input.shape[0]):
        for ker in range(input.shape[1]):
            for x in range(input.shape[2]):
                for y in range(input.shape[3]):
                    appliedRelu[img][ker][x][y] = 0 if input[img][ker][x][y] < 0 else input[img][ker][x][y]
    return appliedRelu


def applyMaxPool(input: np.ndarray, stride: int) -> np.ndarray:
    size = int(input.shape[2] / stride)
    appliedMaxPool = np.zeros(shape=(input.shape[0], input.shape[1], size, size))
    for img in range(input.shape[0]):
        for ker in range(input.shape[1]):
            for x in range(size):
                for y in range(size):
                    patch = input[img, ker, x*stride:x*stride+stride, y*stride:y*stride+stride]
                    appliedMaxPool[img][ker][x][y] = np.max(patch)
    return appliedMaxPool


def backwardMaxPool(prevGradients: np.ndarray, prePool: np.ndarray, stride: int) -> np.ndarray:
    gradients = np.zeros(shape=prePool.shape)
    for img in range(prevGradients.shape[0]):
        for ker in range(prevGradients.shape[1]):
            for x in range(prevGradients.shape[2]):
                for y in range(prevGradients.shape[3]):
                    X = x * stride
                    Y = y * stride
                    maxValue = prePool[img][ker][X][Y]
                    for i in range(x * 2, x * 2 + stride):
                        for j in range(y * 2, y * 2 + stride):
                            if prePool[img][ker][i][j] > maxValue:
                                maxValue = prePool[img][ker][i][j]
                                X = i
                                Y = j
                    gradients[img][ker][X][Y] = prevGradients[img][ker][x][y]
    return gradients


def backwardRelu(prevGradients: np.ndarray, preRelu: np.ndarray) -> None:
    for img in range(prevGradients.shape[0]):
        for ker in range(prevGradients.shape[1]):
            for x in range(prevGradients.shape[2]):
                for y in range(prevGradients.shape[3]):
                    if preRelu[img][ker][x][y] < 0:
                        prevGradients[img][ker][x][y] = 0


def backwardKernelGradients(prevGradients: np.ndarray, input: np.ndarray, kernels: np.ndarray) -> np.ndarray:
    gradients = np.zeros(shape=kernels.shape)
    for img in range(input.shape[0]):
        for ker in range(prevGradients.shape[1]):
            for x in range(kernels.shape[2]):
                for y in range(kernels.shape[3]):
                    for ch in range(input.shape[1]):
                        gradients[ker][ch][x][y] += np.sum(input[img, ch, x:x+prevGradients.shape[3],
                                                                y:y+prevGradients.shape[3]] * prevGradients[img][ker])
    return gradients
