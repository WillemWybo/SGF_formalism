"""
Author: Willem Wybo
Date: 18/08/2015
Place: BBP, Geneva
"""

cimport numpy as np
import numpy as np

from libcpp.string cimport string
from libc.stdint cimport int16_t, int32_t, int64_t
from libcpp cimport bool
from libcpp.vector cimport vector

import posix

cdef extern from "sgfCode.h":
    cdef cppclass sgfModel_c:

        sgfModel_c()

        void set_dt_from_python(double dt_)
        void add_node_from_python(int node_index, int parent_index, int64_t *child_indices, int n_children)
        void set_hines_matrix_from_python(int64_t *indices, double *values, int n_val)
        void add_connenction_data_from_python(int i0, int i1, 
                    double *P1, double *P2, double *P3, double *P4, int n_mem,
                    double H0, double *H1_K, int n_K,
                    bool memory_flag, bool H1_K_flag)
        int add_synapse_from_python(int i, double tau1, double tau2, double E_r, double weight)
        void add_spiketrain_from_python(int n, double *spiketimes, int n_spikes)
        # void add_recorder_from_python(int i)
        void solve_hines_matrix_equation_from_python(double *Y, int n_val)
        void run_from_python(double tmax, int64_t *rec_inds, int Nind, double *recordings)
        
        void print_array_2d(double *arr, int dimX, int dimY)

    # cdef cppclass buffer:

    #     buffer()


cdef class sgfModel:
    cdef double dt
    cdef list rec_inds
    cdef sgfModel_c *ptr
    # cdef list rec_inds
    # cdef int dt

    def __cinit__(self):
        dt = .025
        self.dt = dt
        rec_inds = []
        self.rec_inds = rec_inds
        self.ptr = new sgfModel_c()

    def set_dt(self, dt):
        self.dt = dt
        self.ptr.set_dt_from_python(self.dt)

    def add_node(self, node_index, parent_index, np.ndarray[np.int64_t, ndim=1] child_indices):
        self.ptr.add_node_from_python(node_index, parent_index, &child_indices[0], child_indices.shape[0])

    def set_hines_matrix(self, np.ndarray[np.int64_t, ndim=2] indices, np.ndarray[np.float64_t, ndim=1] values):
        assert indices.shape[0] == len(values)
        assert indices.shape[1] == 2
        self.ptr.set_hines_matrix_from_python(&indices[0,0], &values[0], len(values))

    def add_connection_data(self, i0, i1, 
                    np.ndarray[np.float64_t, ndim=2] P1_, np.ndarray[np.float64_t, ndim=2] P2_,
                    np.ndarray[np.float64_t, ndim=2] P3_, np.ndarray[np.float64_t, ndim=2] P4_,
                    H0, np.ndarray[np.float64_t, ndim=1] H1_K_):
        assert P1_.shape[0] == P2_.shape[0]; assert P1_.shape[0] == P3_.shape[0]; assert P1_.shape[0] == P4_.shape[0]
        assert P1_.shape[1] == 2; assert P2_.shape[1] == 2; assert P3_.shape[1] == 2; assert P4_.shape[1] == 2
        cdef np.ndarray[np.float64_t, ndim=2, mode="c"] P1 
        cdef np.ndarray[np.float64_t, ndim=2, mode="c"] P2
        cdef np.ndarray[np.float64_t, ndim=2, mode="c"] P3
        cdef np.ndarray[np.float64_t, ndim=2, mode="c"] P4
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] H1_K
        if P1_.shape[0] == 0:
            P1 = np.array([[-1.,-1.]])
            P2 = np.array([[-1.,-1.]])
            P3 = np.array([[-1.,-1.]])
            P4 = np.array([[-1.,-1.]])
            memory_flag = False
        else:
            P1 = P1_
            P2 = P2_
            P3 = P3_
            P4 = P4_
            memory_flag = True
        if H1_K_.shape[0] == 0:
            H1_K = np.array([-1.])
            H1_K_flag = False
        else:
            H1_K = H1_K_
            H1_K_flag = True
        self.ptr.add_connenction_data_from_python(i0, i1,
                    &P1[0,0], &P2[0,0], &P3[0,0], &P4[0,0], P1.shape[0],
                    H0, &H1_K[0], H1_K.shape[0], 
                    memory_flag, H1_K_flag)

    def add_synapse(self, i, tau1, tau2, E_r, weight):
        n = self.ptr.add_synapse_from_python(i, tau1, tau2, E_r, weight)
        return n

    def add_spiketrain(self, n, np.ndarray[np.float64_t, ndim=1] spikes):
        if spikes.shape[0] > 0:
            self.ptr.add_spiketrain_from_python(n, &spikes[0], spikes.shape[0])

    def add_recorder(self, i):
        self.rec_inds.append(i)

    def solve_hines_matrix_equation(self, np.ndarray[np.float64_t, ndim=1] Y):
        self.ptr.solve_hines_matrix_equation_from_python(&Y[0], Y.shape[0])

    def run(self, tmax):
        cdef np.ndarray[np.double_t, ndim=1, mode="c"] t_recordings = np.linspace(0., tmax, int(tmax/self.dt))
        cdef np.ndarray[np.double_t, ndim=2, mode="c"] recordings = np.zeros((len(self.rec_inds), int(tmax/self.dt)), dtype=float)
        cdef np.ndarray[np.int64_t, ndim=1] rec_ind_arr = np.array(self.rec_inds, dtype=int)
        t0 = posix.times()[0]
        print ">>> Integrating the C sparse GCM for " + str(tmax) + " ms. <<<"
        self.ptr.run_from_python(tmax, &rec_ind_arr[0], rec_ind_arr.shape[0], &recordings[0,0])
        t1 = posix.times()[0]
        print ">>> Integration done, took " + str(t1-t0) + " s <<<"
        return {'t': t_recordings, 'Vm': recordings, 't_exec': t1-t0}

    def print_array_2d(self, np.ndarray[np.float64_t, ndim=2] arr):
        self.ptr.print_array_2d(&arr[0,0], arr.shape[0], arr.shape[1])
