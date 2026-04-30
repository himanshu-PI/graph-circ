from utils import *
from graphs import * 




def gf2_rank(mat):
    """Compute rank of a binary matrix over GF(2) using Gaussian elimination."""
    mat = mat.copy()
    rank = 0
    n_rows, n_cols = mat.shape
    
    for col in range(n_cols):
        pivot = -1
        # Find pivot row
        for row in range(rank, n_rows):
            if mat[row, col]:
                pivot = row
                break
        if pivot == -1:
            continue
            
        # Swap current row with pivot row
        mat[[rank, pivot]] = mat[[pivot, rank]]
        
        # Eliminate this column in other rows
        for row in range(n_rows):
            if row != rank and mat[row, col]:
                mat[row] = (mat[row] + mat[rank]) % 2
        rank += 1
    
    return rank


def binaryMatrix(zStabilizers):
    
    N = len(zStabilizers)
    Na = len(zStabilizers[0])
    binaryMatrix = np.zeros((N,2*Na))

    r = 0 # Row number
    
    for row in zStabilizers:
        c = 0 # Column number
        for i in row:
            if i == 3: # Pauli Z
                binaryMatrix[r,Na + c] = 1
            if i == 2: # Pauli Y
                binaryMatrix[r,Na + c] = 1
                binaryMatrix[r,c] = 1
            if i == 1: # Pauli X
                binaryMatrix[r,c] = 1
            c += 1
        r += 1

    return binaryMatrix


def ent_state(tab_forward, sysA , sysB) -> float:
        
    stabilizers = tab_forward.to_stabilizers()

    gA = [stim.PauliString([s[q] for q in sysA]) for s in stabilizers]
    # gB = [stim.PauliString([s[q] for q in sysB]) for s in stabilizers]

    na = len(sysA); nb = len(sysB)


    binary_matrix = binaryMatrix(gA)

    rank = gf2_rank(binary_matrix)
    
    return rank - na