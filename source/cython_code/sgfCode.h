/*
Author: Willem Wybo
Date: 18/08/2015
Place: BBP, Geneva
*/

#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <map>
#include <complex>
#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h>
#include <time.h>

using namespace std;


inline int modulo(int a, int b){
    const int result = a % b;
    return result >= 0 ? result : result + b;
}

inline void print_double_vector(vector< double > &V){
    // cout << "V = ";
    for(int i=0; i<V.size(); i++){
        cout << V[i] << " ";
    }
    cout << endl;
}

inline void print_int_vector(vector< int > &V){
    // cout << "V = ";
    for(int i=0; i<V.size(); i++){
        cout << V[i] << " ";
    }
    cout << endl;
}

inline void arr2vec_int(vector< int64_t > &vec, int64_t *arr, int dim){
    vec.clear();
    vec = vector< int64_t > (dim, 0);
    for(int i=0; i<dim; i++){
        vec[i] = arr[i];
    }
}

inline void arr2vec_double(vector< double > &vec, double *arr, int dim){
    vec.clear();
    vec = vector< double > (dim, 0.);
    for(int i=0; i<dim; i++){
        vec[i] = arr[i];
    }
}

inline void arr2vec_complex(vector< complex< double > > &vec, double *arr, int dim){
    vec.clear();
    for(int i=0; i<dim; i++){
        complex< double > c(arr[2*i], arr[2*i+1]);
        vec.push_back(c);
    }
}
// inline void arr2vec_complex(vector< double > &vec, double *arr, int dim){
//     vec.clear();
//     for(int i=0; i<dim; i++){
//         vec.push_back(arr[2*i]);
//     }
// }

inline void arr2hines_double(map< pair< int64_t, int64_t >, double > &hmat, int64_t *indices, double *values, int n_val){
    hmat.clear();
    for(int i=0; i<n_val; i++){
        hmat[make_pair(indices[2*i], indices[2*i+1])] = values[i];
    }
}


struct node{
    // tree structure indices
    int index;
    int parent_index;
    vector< int64_t > child_indices;
    // auxiliary index to indicate passing in Hines algorithm
    int passed = 0;
};


template< typename T >
class buffer {
private:
    // data vector for buffer
    vector< T > data;
    // size of the buffer
    int size;
    // reference index
    int ref;

public:
    // does not work with cython for some reason
    // buffer();
    // ~buffer();

    void set_data(vector< T > dat){
        size = dat.size();
        ref = dat.size()-1;
        data = dat;
    }

    void set(T val){
        ref += 1;
        if(ref == size) ref = 0;
        data[ref] = val;
    }

    T get(int n){
        return data[modulo(ref-n, size)];
    }
};


class double_exp_synapse {
private:
    double dt;
    // vector containing spiketimes
    vector< int > spike_inds;
    // index indicating the current time position
    int index;
    // weight associated with this synapse
    double weight;
    // reversal potential of the synapse
    double E_r;
    // propagators for the synapse dynamics
    double p1, p2;
    // state variables for the synapse
    double s1, s2;

    // take care of input spikes
    void _add_spikes(int k){
        // cout << index << endl;
        // cout << spike_inds.size() << endl;
        if( index < spike_inds.size() ){
            if( spike_inds[index] == k ){
                // print_int_vector(spike_inds);
                index += 1;
                int N_spikes = 1;
                while( (spike_inds[index] == k) && (index < spike_inds.size()) ){
                    index += 1;
                    N_spikes += 1;
                }
                s1 += N_spikes*weight;
                s2 += N_spikes*weight;
            }
        }
    }

public:
    void set_params(double dt_, double tau1, double tau2, double E_r_,
                    double weight_){
        dt = dt_;
        p1 = exp(-dt/tau1); p2 = exp(-dt/tau2);
        double tp = (tau1*tau2)/(tau2 - tau1) * log(tau2/tau1);
        double factor = 1./(-exp(-tp/tau1) + exp(-tp/tau2));
        weight = weight_ * factor;
        E_r = E_r_;
        // initialize spike_inds to be safe when no spikes are set
        spike_inds.clear();
        spike_inds.push_back(1);
        index = 10;
    }

    void newsim(double *spktms, int n_spikes){
        s1 = 0.; s2 = 0.;
        index = 0;
        vector< double > spikes;
        arr2vec_double(spikes, spktms, n_spikes);
        spike_inds.clear();
        for(int i=0; i< spikes.size(); i++){
            spike_inds.push_back(int(spikes[i]/dt));
        }
    }

    void advance(int k){
        s1 *= p1;
        s2 *= p2;
        _add_spikes(k);
    }

    pair< double, double > get_conductance(){
        double g_syn = - s2 + s1;
        double c_syn = - g_syn * E_r;
        return make_pair(g_syn, c_syn);
    }
};


struct connection_data{
    int K;
    int n_mem;
    // flags
    bool memory_flag;
    bool H1_K_flag;
    // memory term
    vector< complex< double > > P1;
    vector< complex< double > > P2;
    vector< complex< double > > P3;
    vector< complex< double > > P4;
    // vector< double > P1;
    // vector< double > P2;
    // vector< double > P3;
    // vector< double > P4;
    // convolution term
    vector< double > H1_K;
    // first convolution term
    double H0;
    // buffer to store convolution data
    buffer< double > V_store;
    // vector to store state vairables memory term:
    vector< complex< double > > ys;
    // vector< double > ys;
};


class sgfModel_c {
private:
    double dt = 0.025;
    /*
    structural data for the model
    */
    // list of nodes in the tree graph
    vector< node > tree;
    // list of nodes that are leafs
    vector< node > leafs;
    // list of connection datasets for each connection
    vector< connection_data > conn_data;
    // map of connection (index indicates data position 
    // in connection_data)
    vector< pair< int64_t, int64_t > > connections;
    map< pair< int64_t, int64_t >, int > connection_vec_indices;

    /*
    synapse data for the model
    */
    vector< double_exp_synapse > synapses;
    vector< int > synapse_positions;

    // hines matrix, is constructed and inverted each timestep
    map< pair< int64_t, int64_t >, double > hines_matrix;

    // functions associated with the hines algorithm
    void _down_sweep(const node &n, int leaf_index, vector< double > &Y);
    void _up_sweep(const node &n, vector< double > &X, vector< double > &Y);

public:
    // constructor, destructor
    sgfModel_c();
    ~sgfModel_c();

    // initialization functions
    void set_dt_from_python(double dt_);
    void add_node_from_python(int node_index, int parent_index, int64_t* child_indices, int n_children);
    void set_hines_matrix_from_python(int64_t *indices, double *values, int n_val);
    void solve_hines_matrix_equation_from_python(double *Y, int n_val);
    void add_connenction_data_from_python(int i0, int i1, 
            double *P1, double *P2, double *P3, double *P4, int n_mem,
            double H0, double *H1_K, int n_K,
            bool memory_flag, bool H1_K_flag);
    int add_synapse_from_python(int i, double tau1, double tau2, double E_r, double weight);
    void add_spiketrain_from_python(int n, double *spiketimes, int n_spikes);
    void run_from_python(double tmax, int64_t *rec_inds, int Nind, double *recordings);

    vector<node> get_leafs();
    void solve_hines_matrix_equation(vector< double > &X, vector< double > &Y);  
    double convolve(connection_data &cdat);

    void print_node(const node &pnode);
    void print_connection_data(connection_data &cdat);
    void print_hines_matrix();
    void print_connection_vector();
    void print_tree();
    // void print_double_vector(vector< double > &V);
    void print_complex_vector(vector< complex< double > > &V);
    void print_array_2d(double *arr, int dimX, int dimY);
};