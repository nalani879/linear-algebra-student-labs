def lu_factorisation(A):

    n, m = A.shape
    if n != m:
        raise ValueError(f"Matrix A is not square {A.shape=}")
    
    rows = A.shape[0] #number of rows

    # construct arrays of zeros
    L, U = np.zeros_like(A), np.zeros_like(A)

    #filling all diagonal values to 1 in L matrix
    np.fill_diagonal(L, 1)

    for j in range(rows): #for the coloumns iterating through
        for i in range(j+1): #iterating for the U
        # Compute factors u_{ij}
            mysumU = sum(L[i,k] * U[k,j] for k in range(rows)) #calculating the dot product of the current elemets of L and U so we know what to subtract
            if L[i,j] != 0: #handling the zero
                U[i,j] = (A[i,j] - mysumU)/L[i,j] #subtracting the calculated sum from the matrix A to get that element of U 
            else:
                U[i,j] = (A[i,j] - mysumU)


        for i in range(j+1, rows): #iterating for the L
        # Compute factors l_{ij}
            mysumL = sum(L[i,k] * U[k,j] for k in range(rows)) #calculating dot product again for getting elements of L
            L[i,j] = (A[i,j] - mysumL)/U[j,j] #this is us placing the multiplier in the current positin for L

    return(L, U)