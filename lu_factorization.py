import numpy as np
from scipy.sparse import diags
import numpy


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

    
    n, m = A.shape
    if n != m:
        raise ValueError(f"Matrix A is not square {A.shape=}")

    # construct arrays of zeros
    L, U = np.zeros_like(A), np.zeros_like(A)
    #filling all diagonal values to 1 in L matrix
    np.fill_diagonal(L, 1)

    #creating an augmented matrix
    #solving for upper trianglular

    rows = A.shape[0]
    i = 0
    secondA = A.copy()
    while i < rows: # iterating through columns
        if secondA[i][i] == 0.0: #checking if the diagonal entry is zero
           print ("Diagonal Entry is Zero")
           return
     
        for j in range (i+1, rows): #iterate rows

            multiplier = secondA[j][i] / secondA[i][i]


            secondA[j] = secondA[j] - (multiplier * secondA[i]) #this makes a particular val under the diagonal zero
            print("This is iteration ", j) #keeping track of iter
            print(secondA)

            #this section I am making L , putting the multipler values
             
            L[j][i] = multiplier
            
        i += 1

        U = secondA.copy()
    
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

Atry, btry, x = generate_safe_system(3)
determinantL, determinantU = determinant(Atry)

'''
    
    print(Atry)
    L, U = lu_factorisation(Atry)

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



