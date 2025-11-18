import numpy as np
from scipy.sparse import diags
import numpy
import time
import matplotlib.pyplot as plt

def generate_safe_system(n):
    """
    Generate a linear system A x = b where A is strictly diagonally dominant,
    ensuring LU factorization without pivoting will work.

    Parameters:
        n (int): Size of the system (n x n)

    Returns:
        A (ndarray): n x n strictly diagonally dominant matrix
        b (ndarray): RHS vector
        x_true (ndarray): The true solution vector
    """

    k = [np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)]
    offset = [-1, 0, 1]
    A = diags(k, offset).toarray()

    # Solution is always all ones
    x_true = np.ones((n, 1))

    # Compute b = A @ x_true
    b = A @ x_true

    return A, b, x_true


def lu_factorisation(A):

    rows = A.shape[0]

    n, m = A.shape
    if n != m:
        raise ValueError(f"Matrix A is not square {A.shape=}")

    # construct arrays of zeros
    L, U = np.zeros_like(A), np.zeros_like(A)

    #filling all diagonal values to 1 in L matrix
    np.fill_diagonal(L, 1)

    for j in range(rows): #for the coloumns iterating through
        for i in range(j+1): #iterating for the U
      # Compute factors u_{ij}
            
            mysumU = sum(L[i,k] * U[k,j] for k in range(i)) #calculating the dot product of the current elemets of L and U so we know what to subtract
            U[i,j] = A[i,j] - mysumU #subtracting the calculated sum from the matrix A to get that element of U 
            

        for i in range(j+1, n): #iterating for the L
      # Compute factors l_{ij}
            
            mysumL = sum(L[i,k] * U[k,j] for k in range(i)) #calculating dot product again for getting elements of L
            L[i,j] = (A[i,j] - mysumL)/U[j,j] #this is us placing the multiplier in the current positin for L

    return(L, U)


def determinant(A):
    n = A.shape[0]
    L, U = lu_factorisation(A)

    det_L = 1.0
    det_U = 1.0

    for i in range(n):
        det_L *= L[i, i]
        det_U *= U[i, i]

    return det_L * det_U

def system_size(A, b):
    if A.ndim != 2:
        raise ValueError(f"Matrix A must be 2D, but got {A.ndim}D array")

    n, m = A.shape
    if n != m:
        raise ValueError(f"Matrix A must be square, but got A.shape={A.shape}")

    if b.shape[0] != n:
        raise ValueError(
            f"System shapes are not compatible: A.shape={A.shape}, "
            f"b.shape={b.shape}"
        )

    return n

def row_swap(A, b, p, q):
    n = system_size(A, b)
    # swap rows of A
    for j in range(n):
        A[p, j], A[q, j] = A[q, j], A[p, j]
    # swap rows of b
    b[p, 0], b[q, 0] = b[q, 0], b[p, 0]

def row_scale(A, b, p, k):
    n = system_size(A, b)

    # scale row p of A
    for j in range(n):
        A[p, j] = k * A[p, j]
    # scale row p of b
    b[p, 0] = b[p, 0] * k

def row_add(A, b, p, k, q):
    n = system_size(A, b)

    # Perform the row operation
    for j in range(n):
        A[p, j] = A[p, j] + k * A[q, j]

    # Update the corresponding value in b
    b[p, 0] = b[p, 0] + k * b[q, 0]

def gaussian_elimination(A, b, verbose=False):
     # find shape of system
    n = system_size(A, b)

    # perform forwards elimination
    for i in range(n - 1):
        # eliminate column i
        if verbose:
            print(f"eliminating column {i}")
        for j in range(i + 1, n):
            # row j
            factor = A[j, i] / A[i, i]
            if verbose:
                print(f"  row {j} |-> row {j} - {factor} * row {i}")
            row_add(A, b, j, -factor, i)


def forward_substitution(A, b):
    # get size of system
    n = system_size(A, b)

    # check is lower triangular
    if not np.allclose(A, np.tril(A)):
        raise ValueError("Matrix A is not lower triangular")

    # create solution variable
    x = np.empty_like(b)

    # perform forwards solve
    for i in range(n):
        partial_sum = 0.0
        for j in range(0, i):
            partial_sum += A[i, j] * x[j]
        x[i] = 1.0 / A[i, i] * (b[i] - partial_sum)

    return x

def backward_substitution(A, b):
    n = system_size(A, b)

    # check is upper triangular
    assert np.allclose(A, np.triu(A))

    # create solution variable
    x = np.empty_like(b)

    # perform backwards solve
    for i in range(n - 1, -1, -1):  # iterate over rows backwards
        partial_sum = 0.0
        for j in range(i + 1, n):
            partial_sum += A[i, j] * x[j]
        x[i] = 1.0 / A[i, i] * (b[i] - partial_sum)

    return x

def lu_factorization_graph(A, b):
    L, U = lu_factorisation(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)

    return x

Atry, btry, x = generate_safe_system(3)

#graceA = np.array([[4,2,0],[2,3,1],[0,1,2.5]])
'''
print(Atry)


print("This is my Matrix U (Upper triangular)")
print(U)
print("This is my matrix L (Lower triangular)")
print(L)
print("\n")

print("This shows that L x U = A")
if (np.allclose(L @ U, Atry)):
    print(L ,"x", U ,"=", L@U)
    print("This is my origianl Matrix: ", Atry)



'''
L, U = lu_factorisation(Atry)
result = determinant(Atry)
print(result)



'''

sizes = [2**j for j in range(1, 6)]
times_lu = []
times_gauss = []

for n in sizes:
    # generate a random system of linear equations of size n
    A, b, x = generate_safe_system(n)

    # do the solve
    start_time = time.time()
    lu_factorization_graph(A, b)
    end_time = time.time()
    times_lu.append(end_time - start_time)

    start_time2 = time.time()
    gaussian_elimination(A, b)
    end_time2 = time.time()
    times_gauss.append(end_time2 - start_time2)


plt.figure(figsize=(8, 5))

plt.plot(sizes, times_lu, marker='o', label='LU Factorization Version', linewidth=2)
plt.plot(sizes, times_gauss, marker='s', label='Gaussian Elimination Version', linewidth=2)

plt.title("LU Factorization VS Gaussian Elimination")
plt.xlabel("Matrix Size: ")
plt.ylabel("Execution Time (in seconds):")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.savefig("luVSguassgraph.png")

'''

