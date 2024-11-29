import numpy as np

# Larger input image (5x5)
X = np.random.randn(10, 10)

# Kernel (2x2) remains the same
K = np.random.randn(16, 2, 2)

# Output gradient (dL/d_out) (4x4)
dOut = np.random.randn(16, 8, 8)


# Function to calculate gradients for the kernel dK
def conv2d_gradient(Images, d_out):
    dK = np.zeros((16, 2, 2))
    for f in range(16):
        for i in range(8):
            for j in range(8):
                X_patch = Images[i:i+2, j:j+2]
                dK[f] += X_patch * d_out[f, i, j]
    return dK


def conv2d_gradientTwo(Images, prevGradients):
    dK = np.zeros((16, 2, 2))
    for f in range(16):
        for i in range(2):
            for j in range(2):
                dK[f, i, j] = np.sum(Images[i:i+8, j:j+8] * prevGradients[f])

    return dK


# Calculate the gradient of the kernel
dKr = conv2d_gradient(X, dOut)

print("Input Image (X):\n", X)
print("Kernel (K):\n", K[0])
print("Output Gradient (d_out):\n", dOut[0])
print("Gradient of Kernel (dK):\n", dKr[0])

dKT = conv2d_gradientTwo(X, dOut)
print("Gradient of Kernel (dKT):\n", dKT[0])
