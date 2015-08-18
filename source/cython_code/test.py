import SGFModel as SGFM

import numpy as np

a = np.array([[2.,4.,6.],[1.,3.,5.]])

sgfM = SGFM.sgfModel()

sgfM.print_array_2d(a)

# set the tree structure in C model
sgfM.add_node(0,-1, np.array([1,6]))
sgfM.add_node(1,0, np.array([2,5]))
sgfM.add_node(2,1, np.array([3,4]))
sgfM.add_node(3,2, np.array([-1], dtype=int))
sgfM.add_node(4,2, np.array([-1], dtype=int))
sgfM.add_node(5,1, np.array([-1], dtype=int))
sgfM.add_node(6,0, np.array([-1], dtype=int))

# add a connection data
P1 = np.array([[.9,0.],[.9,-1.],[.9,1.]])
P2 = np.array([[.9,0.],[.9,-1.],[.9,1.]])
P3 = np.array([[.9,0.],[.9,-1.],[.9,1.]])
P4 = np.array([[.9,0.],[.9,-1.],[.9,1.]])
H1_K = np.array([5.,4.,3.,2.,1.])
H0 = 5.
sgfM.add_connection_data(0, 1,
            P1, P2, P3, P4,
            H0, H1_K)

# define a hines matrix
# - sparse
hm_indices = np.array([ [0,0],[0,1],[0,6],
                        [1,0],[1,1],[1,2],[1,5],
                        [2,1],[2,2],[2,3],[2,4],
                        [3,2],[3,3],
                        [4,2],[4,4],
                        [5,1],[5,5],
                        [6,0],[6,6] ])
hm_vals = np.arange(hm_indices.shape[0])+1.
# hm_vals = np.zeros(hm_indices.shape[0])
# -dense
N = np.max(hm_indices) + 1
hm_dense = np.zeros((N,N))
for i, inds in enumerate(hm_indices):
    hm_dense[tuple(inds)] = hm_vals[i]

# define a Y array
# Y = np.zeros(N, dtype=float)
Y = np.arange(N, dtype=float) + 5.

# set the hines matrix in C model
sgfM.set_hines_matrix(hm_indices, hm_vals)
# solve the hines matrix in C model
sgfM.solve_hines_matrix_equation(Y)
sgfM.solve_hines_matrix_equation(Y)

print ""
print "hm = "
print hm_dense
print ""
print "Y = "
print Y
print ""
print "X = "
# solve the matrix with the standard gaussian elimination in numpy
print np.linalg.solve(hm_dense, Y)