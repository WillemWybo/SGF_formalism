/*
Author: Willem Wybo
Date: 18/08/2015
Place: BBP, Geneva
*/
#include "sgfCode.h"

// constructor
sgfModel_c::sgfModel_c(){}
// destructor
sgfModel_c::~sgfModel_c(){}

void sgfModel_c::set_dt_from_python(double dt_){
    dt = dt_;
}

void sgfModel_c::add_node_from_python(int node_index, int parent_index, int64_t* child_indices, int n_children){
    /*
    Add a node to the tree structure via the pyhthon interface

    leafs should have [-1] as child indices
    root shoud have -1 as parent index
    */
    node n;
    n.index = node_index;
    n.parent_index = parent_index;
    arr2vec_int(n.child_indices, child_indices, n_children);
    tree.push_back(n);
    // print_node(n);
}

void sgfModel_c::set_hines_matrix_from_python(int64_t *indices, double *values, int n_val){
    arr2hines_double(hines_matrix, indices, values, n_val);
}

void sgfModel_c::solve_hines_matrix_equation_from_python(double *Y, int n_val){
    leafs = get_leafs();
    vector< double > X_ = vector< double > (n_val, 0.0);
    vector< double > Y_ = vector< double > (n_val, 0.0);
    for(int i=0; i<n_val; i++){
        Y_[i] = Y[i];
    }
    solve_hines_matrix_equation(X_, Y_);
}

void sgfModel_c::add_connenction_data_from_python(int i0, int i1, 
            double *P1, double *P2, double *P3, double *P4, int n_mem,
            double H0, double *H1_K, int n_K,
            bool memory_flag, bool H1_K_flag){
    /*
    add a set of connection data from python

    i0,i1: indices of nodes between which the connection is
    P1, P2, P3, P4: complex vectors needed to implement the memory term
    H0: real number, first entry of quadrature term
    H1_k: real vector to implement the quadrature term
    */
    // add the connection
    int pos = conn_data.size();
    connections.push_back(make_pair(i0, i1));
    connection_vec_indices[make_pair(i0, i1)] = pos;
    // add the connection data
    connection_data cdat;
    cdat.n_mem = n_mem;
    arr2vec_complex(cdat.P1, P1, n_mem);
    arr2vec_complex(cdat.P2, P2, n_mem);
    arr2vec_complex(cdat.P3, P3, n_mem);
    arr2vec_complex(cdat.P4, P4, n_mem);
    arr2vec_double(cdat.H1_K, H1_K, n_K);
    cdat.H0 = H0;
    cdat.K = cdat.H1_K.size();
    cdat.memory_flag = memory_flag;
    cdat.H1_K_flag = H1_K_flag;
    // initialize the voltages needed for convolution to 0
    cdat.V_store.set_data(vector< double > (cdat.K+1, 0.));
    // initialize state variables memory term
    cdat.ys = vector< complex< double > > (n_mem, (0., 0.));
    // cdat.ys = vector< double > (n_mem, 0.);
    // append to connection data
    conn_data.push_back(cdat);
    // cout << ">>> connection " << i0 << "<->" << i1 << " <<<" << endl;
    // print_connection_data(conn_data[connection_vec_indices[make_pair(i0,i1)]]);
}

int sgfModel_c::add_synapse_from_python(int i,
        double tau1, double tau2, double E_r, double weight){
    synapse_positions.push_back(i);
    double_exp_synapse syn;
    syn.set_params(dt, tau1, tau2, E_r, weight);
    synapses.push_back(syn);
    return synapses.size() - 1;
}

void sgfModel_c::add_spiketrain_from_python(int n, 
        double *spiketimes, int n_spikes){
    synapses[n].newsim(spiketimes, n_spikes);
}

vector< node > sgfModel_c::get_leafs(){
    vector< node > leafs;
    for(int i=0; i<tree.size(); i++){
        if(tree[i].child_indices[0] == -1){
            leafs.push_back(tree[i]);
            // cout << ">>> leaf <<<" << endl;
            // print_node(&tree[i]);
        }
    }
    return leafs;
}

void sgfModel_c::run_from_python(double tmax, int64_t *rec_inds, int Nind, double *recordings){
    // set leafs for hines algorithm
    leafs = get_leafs();
    // count integers
    int Nloc = tree.size();
    int Nmax = int(tmax/dt);
    // data vectors
    vector< double > V = vector< double > (Nloc, 0.); // voltage
    vector< double > I = vector< double > (Nloc, 0.); // input current full
    vector< double > G = vector< double > (Nloc, 0.); // conductance
    vector< double > C = vector< double > (Nloc, 0.); // input current part
    vector< double > F = vector< double > (Nloc, 0.); // to solve the hines matrix

    // time measurement
    clock_t daux;
    clock_t dsyn;
    clock_t dmat;
    clock_t dsolve;

    for(int j=0; j<Nmax; j++){
        // record the relevant voltages
        for(int k=0; k<Nind; k++){
            recordings[k*Nmax+j] = V[rec_inds[k]];
        }
        // compute synaptic current
        // daux = clock();
        fill(G.begin(), G.end(), 0.);
        fill(C.begin(), C.end(), 0.);
        for(int i=0; i<synapse_positions.size(); i++){
            int ind = synapse_positions[i];
            // cout << "synapse " << ind << endl;
            synapses[i].advance(j);
            pair< double, double > gc = synapses[i].get_conductance();
            G[ind] += gc.first;
            C[ind] += gc.second;
        }
        // daux = clock() - daux;
        // dsyn += daux;

        // compute convolutions and construct hines matrix
        // daux = clock();
        hines_matrix.clear();
        fill(F.begin(), F.end(), 0.);
        for(int i=0; i<connections.size(); i++){
            pair< int64_t, int64_t> &conn = connections[i];
            connection_data &cdat = conn_data[i];
            if(conn.first == conn.second){
                cdat.V_store.set(I[conn.second]);
                hines_matrix[conn] = 1. - cdat.H0 * G[conn.first] ;
                F[conn.first] += cdat.H0 * C[conn.first];
            } else {
                cdat.V_store.set(V[conn.second]);
                hines_matrix[conn] = -cdat.H0;
            }
            F[conn.first] += convolve(cdat);
        }
        // clear old voltage
        fill(V.begin(), V.end(), 0.);
        // daux = clock() - daux;
        // dmat += daux;

        // solve hines matrix
        // daux = clock();
        solve_hines_matrix_equation(V, F);
        // daux = clock() - daux;
        // dsolve += daux;

        // set the current
        for(int i=0; i<Nloc; i++){
            I[i] = G[i]*V[i] + C[i];
        }
    }

    // cout << "time synapse computation (s) = " << ((float)dsyn) / CLOCKS_PER_SEC << endl;
    // cout << "time matrix construction (s) = " << ((float)dmat) / CLOCKS_PER_SEC << endl;
    // cout << "time matrix solution (s)     = " << ((float)dsolve) / CLOCKS_PER_SEC << endl;
}

double sgfModel_c::convolve(connection_data &cdat){
    // memory term
    complex< double > val(0., 0.);
    if (cdat.memory_flag){
    // double val = 0.;
        for(int i=0; i<cdat.n_mem; i++){
            cdat.ys[i] = cdat.P1[i] * cdat.ys[i] + 
                        cdat.P2[i] * cdat.V_store.get(cdat.K-1) + cdat.P3[i] * cdat.V_store.get(cdat.K);
            val += cdat.ys[i] * cdat.P4[i];
        }
    }
    double val_ = real(val);
    // quadrature term
    if (cdat.H1_K_flag){
        for(int i=0; i<cdat.K; i++){
            val_ += cdat.V_store.get(i) * cdat.H1_K[i];
        }
    }
    return val_;
}

void sgfModel_c::solve_hines_matrix_equation(vector< double > &X, vector< double > &Y){
    int leaf_index = 0;

    // solve the hines matrix equation
    _down_sweep(leafs[leaf_index], leaf_index, Y);
    _up_sweep(tree[0], X, Y);

    // set the passed indices to 0 again
    for(int i=0; i<tree.size(); i++){
        tree[i].passed = 0;
    }
    // cout << ">>> hines matrix solution <<<" << endl;
    // print_double_vector(X);
}

void sgfModel_c::_down_sweep(const node &n, int leaf_index, vector< double > &Y){
    int index   = n.index;
    int p_index = n.parent_index;
    if(p_index != -1){
        tree[p_index].passed += 1;

        hines_matrix[make_pair(p_index, p_index)] -= hines_matrix[make_pair(p_index, index)] / 
                                                     hines_matrix[make_pair(index, index)] *
                                                     hines_matrix[make_pair(index, p_index)];
        Y[p_index] -= hines_matrix[make_pair(p_index, index)] / 
                      hines_matrix[make_pair(index, index)] * Y[index];

        if(tree[p_index].passed == tree[p_index].child_indices.size()){
            // cout << "go further down" << endl;
            _down_sweep(tree[p_index], leaf_index, Y);
        } else if(leaf_index < leafs.size()) {
            leaf_index += 1;
            // cout << "go to leaf " << leafs[leaf_index].index << endl;
            _down_sweep(leafs[leaf_index], leaf_index, Y);
        }
    }
}

void sgfModel_c::_up_sweep(const node &n, vector< double > &X, vector< double > &Y){
    int index = n.index;
    int p_index = n.parent_index;
    if(index != 0){
        Y[index] -= hines_matrix[make_pair(index, p_index)] /
                    hines_matrix[make_pair(p_index, p_index)] * Y[p_index];
    }
    X[index] = Y[index] / hines_matrix[make_pair(index, index)];
    if(n.child_indices[0] != -1){
        for(vector< int64_t >::const_iterator i=n.child_indices.begin(); i != n.child_indices.end(); i++){
            _up_sweep(tree[*i], X, Y);
        }
    }
}

void sgfModel_c::print_node(const node &pnode){
    cout << ">>> node data <<<" << endl;
    cout << "index = " << pnode.index << endl;
    cout << "parent index = " << pnode.parent_index << endl;
    cout << "child indices = ";
    for(vector< int64_t >::const_iterator i=pnode.child_indices.begin(); i != pnode.child_indices.end(); i++){
        cout << *i << " ";
    }
    cout << endl;
    cout << endl;
}

void sgfModel_c::print_connection_data(connection_data &cdat){
    cout << ">>> connection data <<<" << endl;
    cout << "P1 = "; print_complex_vector(cdat.P1);
    cout << "P2 = "; print_complex_vector(cdat.P2);
    cout << "P3 = "; print_complex_vector(cdat.P3);
    cout << "P4 = "; print_complex_vector(cdat.P4);
    cout << "K = " << cdat.K << endl;
    cout << "H1_K = "; print_double_vector(cdat.H1_K);
    cout << "H0 = " << cdat.H0 << endl;
    cout << endl;
}

void sgfModel_c::print_hines_matrix(){
    int Nloc = tree.size();
    cout << ">>> hines matrix <<<" << endl;
    for(int64_t i=0; i<Nloc; i++){
        for(int64_t j=0; j<Nloc; j++){
            pair< int64_t, int64_t > inds = make_pair(i, j);
            if(find(connections.begin(), connections.end(), inds) != connections.end()){
                cout << " " << hines_matrix[inds];
            } else {
                cout << " " << 0.;
            }
        }
        cout << endl;
    }
}

void sgfModel_c::print_connection_vector(){
    cout << ">>> connection vector <<<" << endl;
    for(int i=0; i<connections.size(); i++){
        cout << connections[i].first << ", " << connections[i].second << endl;
    }
}

void sgfModel_c::print_tree(){
    for(int i=0; i<tree.size(); i++){
        print_node(tree[i]);
    }
}

void sgfModel_c::print_complex_vector(vector< complex< double > > &V){
    cout << "V = ";
    for(int i=0; i<V.size(); i++){
        cout << V[i] << " ";
    }
    cout << endl;
}

void sgfModel_c::print_array_2d(double *arr, int dimX, int dimY){
    cout << "arr 2d =" << endl;
    for(int i=0; i<dimX; i=i+1){
    	for(int j=0; j<dimY; j=j+1){
    		cout << arr[i*dimY + j] << " ";
    	}
		cout << endl;
    }
    cout << endl;
}