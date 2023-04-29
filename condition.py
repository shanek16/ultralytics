import numpy as np

# Define the size of the array
n_rows = 4
n_cols = 10

def L2(p1,p2):
    p1x,p1y = p1.flatten()
    p2x,p2y = p2.flatten()
    return np.sqrt((p2x-p1x)**2 + (p2y-p1y)**2)

# Define the scalar and vector values for the rows
A = np.zeros((4,10), dtype=object)
A[0] = np.random.randint(low=1, high=4, size=(1, n_cols))
A[1][1] = np.random.rand(1, 2)
A[2][1] = np.random.rand(1, 2)
for i in range(10):
    A[3][i] = np.random.rand(1, 2) # Each row in the 4th row has a vector of length 2
print("Array A:\n", A)

# Define the threshold epsilon
epsilon = 0.1

# Find columns that satisfy the conditions
valid_cols = np.where(
    ((A[0] == 1) | (A[0] == 2))
    & (A[1] != 0)
    & (A[2] != 0)
    )[0]

# valid_cols = np.where(
#     & ~np.any([
#         L2(A[3, i], A[3, j]) < epsilon
#         for i in range(A.shape[1])
#         for j in range(i + 1, A.shape[1])
#     ])
# )[0]

print("Valid columns:", valid_cols)