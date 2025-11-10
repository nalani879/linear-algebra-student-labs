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