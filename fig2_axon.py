"""
Author: Willem Wybo
Date: 18/08/2015
Place: BBP, Geneva
"""

import matplotlib.pyplot as pl
import numpy as np

import copy
import pickle
from multiprocessing import Pool

from mpl_toolkits.axes_grid1 import ImageGrid

import morphologyReader as morphR
import neuronModels as neurM

save = True
overwrite = True

def get_myelin_nodes(greenstree, node_inds_of_Ranvier, ind_ais):
    nodes = greenstree.tree.get_nodes(somanodes=False)
    node_inds = [n._index for n in nodes]
    myelin_node_inds = list(set(node_inds) - set([1]) - set([ind_ais]) - set(node_inds_of_Ranvier))
    return myelin_node_inds

def threshold_crossing_time(Vm, dt, Vth=-20.):
    ind1 = np.where(Vm > Vth)[0][0]
    ind0 = ind1-1
    return ind0*dt + (Vth - Vm[ind0]) / (Vm[ind1] - Vm[ind0]) * dt


def measure_velocity(greenstree, greenstree_pas, node_inds_of_Ranvier, ind_ais, Vth=-20, run_NEURON=False, pprint=False, pplot=False, temp=18.5):
    # parameters
    dt = .025
    tmax = 70.
    V0 = -65.
    inds = [n for n in range(6,11)]
    number_of_nodes = 8
    # initialize a greens function calculator
    gfcalc = morphR.greensFunctionCalculator(greenstree)
    gfcalc.set_impedances_logscale(fmax=7, base=10, num=200)
    gfcalc_pas = morphR.greensFunctionCalculator(greenstree_pas)
    gfcalc_pas.set_impedances_logscale(fmax=7, base=10, num=200)
    # input locations
    inlocs = [{'node': n, 'x': 0.5, 'ID': ind} for ind, n in enumerate([1] + [ind_ais] + node_inds_of_Ranvier)]
    # integration point conductances
    gs_point, es_point = morphR.get_axon_node_conductances(greenstree, node_inds_of_Ranvier, ind_ais)
    # input
    Iclamps = [{'ID': 0, 'x': inlocs[0]['x'], 'node':1, 'delay': 15. , 'dur': 2., 'amp': .5}]
    # compute SGF
    alphas, gammas, pairs, Ms = gfcalc_pas.kernelSet_sparse(inlocs, FFT=False, kernelconstants=True, pprint=False)
    # preprocessor test
    prep = neurM.preprocessor()
    mat_dict_On = prep.construct_volterra_matrices_On(dt, alphas, gammas, pprint=False)
    sv_dict = prep.construct_ionchannel_matrices(inlocs, gs_point, es_point, temp=temp)
    I_in = prep.construct_current_input_matrix(dt, tmax, inlocs, Iclamps)
    # backwards integration
    axon1 = neurM.axon_vectorized(len(inlocs), sv_dict, mat_dict_On, E_eq=V0)
    result = axon1.run_volterra_back_On(tmax, dt, I_in=I_in)
    if run_NEURON:
        # run neuron neuron
        HHneuron = neurM.NeuronNeuron(greenstree, dt=dt, truemorph=False, factorlambda=10)
        HHneuron.add_Iclamp(Iclamps)
        HHneuron.add_recorder(inlocs)
        Vm = HHneuron.run(tdur=tmax, pprint=True)

    if pplot:
        if run_NEURON:
            pl.plot(Vm['t'], Vm[inds[0]], 'r')
            pl.plot(Vm['t'], Vm[inds[0]+number_of_nodes], 'b')
            pl.plot(Vm['t'], Vm[len(result['Vm'])-1], 'g')
        pl.plot(result['t'], result['Vm'][inds[0]], 'r--', lw=2)
        pl.plot(result['t'], result['Vm'][inds[0]+number_of_nodes], 'b--', lw=2)
        pl.plot(result['t'], result['Vm'][-1], 'g--', lw=2)
        pl.show()

    # compute velocity
    v_list = []
    if run_NEURON: v_list_NEURON = []
    for j in inds:
        # i_rv = node_inds_of_Ranvier[j]
        t1 = threshold_crossing_time(result['Vm'][j], dt, Vth=Vth)
        t2 = threshold_crossing_time(result['Vm'][j+number_of_nodes], dt, Vth=Vth)
        node_ind_ranvier = node_inds_of_Ranvier[-1]
        node_ind_myelin = node_ind_ranvier - 1
        node_ranvier = greenstree.tree.get_node_with_index(node_ind_ranvier)
        node_myelin = greenstree.tree.get_node_with_index(node_ind_myelin)
        Dx = number_of_nodes*(node_ranvier.get_content()['impedance'].length + \
                node_myelin.get_content()['impedance'].length) * 1e-2 # m
        Dt = (t2 - t1) * 1e-3 # s
        v_list.append(Dx/Dt) # m/s
        if run_NEURON:
            t1_ = threshold_crossing_time(Vm[j], dt, Vth=Vth)
            t2_ = threshold_crossing_time(Vm[j+number_of_nodes], dt, Vth=Vth)
            Dt_ = (t2_ - t1_) * 1e-3 # s
            v_list_NEURON.append(Dx/Dt_)

    v_avg = np.mean(np.array(v_list))
    if pprint: print 'velocity= ', v_avg, ' m/s'
    if run_NEURON:
        v_avg_NEURON = np.mean(np.array(v_list_NEURON))
        if pprint: print v_avg_NEURON

    if run_NEURON:
        return v_avg, result, Vm
    else:
        return v_avg


def scan_parameter_range(greenstree, greenstree_pas, variable, prange, node_inds_of_Ranvier, ind_ais):
    gts = []; gts_pas = []
    if variable == 'cm_myelin':
        myelin_node_inds = get_myelin_nodes(greenstree, node_inds_of_Ranvier, ind_ais)
        for factor in prange:
            gts.append(copy.deepcopy(greenstree))
            gts_pas.append(copy.deepcopy(greenstree_pas))
            ns = gts[-1].tree.get_nodes(somanodes=False)
            ns_pas = gts_pas[-1].tree.get_nodes(somanodes=False)
            for ind, n in enumerate(ns):
                if n._index in myelin_node_inds:
                    n.get_content()['physiology'].cm = n.get_content()['physiology'].cm * factor
                    ns_pas[ind].get_content()['physiology'].cm = ns_pas[ind].get_content()['physiology'].cm * factor

    elif variable == 'cm_node':
        node_inds = [ind_ais] + node_inds_of_Ranvier
        for factor in prange:
            gts.append(copy.deepcopy(greenstree))
            gts_pas.append(copy.deepcopy(greenstree_pas))
            ns = gts[-1].tree.get_nodes(somanodes=False)
            ns_pas = gts_pas[-1].tree.get_nodes(somanodes=False)
            for ind, n in enumerate(ns):
                if n._index in node_inds:
                    n.get_content()['physiology'].cm = n.get_content()['physiology'].cm * factor
                    ns_pas[ind].get_content()['physiology'].cm = ns_pas[ind].get_content()['physiology'].cm * factor

    elif variable == 'L_internode':
        myelin_node_inds = get_myelin_nodes(greenstree, node_inds_of_Ranvier, ind_ais)
        for factor in prange:
            gts.append(copy.deepcopy(greenstree))
            gts_pas.append(copy.deepcopy(greenstree_pas))
            ns = gts[-1].tree.get_nodes(somanodes=False)
            L_myelin = (ns[2].get_content()['p3d'].x - ns[1].get_content()['p3d'].x) * factor
            L_node = ns[1].get_content()['p3d'].x
            xprev = 0.
            ns_pas = gts_pas[-1].tree.get_nodes(somanodes=False)
            for ind, n in enumerate(ns[1:]):
                if n._index in myelin_node_inds:
                    xprev += L_myelin
                else:
                    xprev += L_node
                n.get_content()['p3d'].x = xprev
                ns_pas[ind+1].get_content()['p3d'].x = xprev

    elif variable == 'A_node':
        myelin_node_inds = get_myelin_nodes(greenstree, node_inds_of_Ranvier, ind_ais)
        for factor in prange:
            gts.append(copy.deepcopy(greenstree))
            gts_pas.append(copy.deepcopy(greenstree_pas))
            ns = gts[-1].tree.get_nodes(somanodes=False)
            L_myelin = (ns[2].get_content()['p3d'].x - ns[1].get_content()['p3d'].x)
            L_node = ns[1].get_content()['p3d'].x * factor
            xprev = 0.
            ns_pas = gts_pas[-1].tree.get_nodes(somanodes=False)
            for ind, n in enumerate(ns[1:]):
                if n._index in myelin_node_inds:
                    xprev += L_myelin
                else:
                    xprev += L_node
                n.get_content()['p3d'].x = xprev
                ns_pas[ind+1].get_content()['p3d'].x = xprev

    elif variable == 'g_bar':
        node_inds = [ind_ais] + node_inds_of_Ranvier
        for factor in prange:
            gts.append(copy.deepcopy(greenstree))
            gts_pas.append(copy.deepcopy(greenstree_pas))
            ns = gts[-1].tree.get_nodes(somanodes=False)
            ns_pas = gts_pas[-1].tree.get_nodes(somanodes=False)
            for ind, n in enumerate(ns):
                if n._index in node_inds:
                    gs = n.get_content()['physiology'].gs
                    for key in gs.keys():
                        gs[key] *= factor

    elif variable == 'd_axon':
        axon_nodes = greenstree.tree.get_nodes(somanodes=False)[1:]
        axon_node_inds = [n._index for n in axon_nodes]
        myelin_node_inds = get_myelin_nodes(greenstree, node_inds_of_Ranvier, ind_ais)
        for factor in prange:
            gts.append(copy.deepcopy(greenstree))
            gts_pas.append(copy.deepcopy(greenstree_pas))
            ns = gts[-1].tree.get_nodes(somanodes=False)
            ns_pas = gts_pas[-1].tree.get_nodes(somanodes=False)
            for ind, n in enumerate(ns):
                if n._index in axon_node_inds:
                    n.get_content()['p3d'].radius = n.get_content()['p3d'].radius * factor
                    ns_pas[ind].get_content()['p3d'].radius = ns_pas[ind].get_content()['p3d'].radius * factor
                if n._index in myelin_node_inds:
                    n.get_content()['physiology'].cm = n.get_content()['physiology'].cm / factor
                    gs =  n.get_content()['physiology'].gs
                    gs['L'] = gs['L'] / factor
                    ns_pas[ind].get_content()['physiology'].cm = ns_pas[ind].get_content()['physiology'].cm / factor
                    gs =  ns_pas[ind].get_content()['physiology'].gs
                    gs['L'] = gs['L'] / factor


    elif variable == 'r_a':
        nodes = greenstree.tree.get_nodes(somanodes=False)
        node_inds = [n._index for n in nodes]
        for factor in prange:
            gts.append(copy.deepcopy(greenstree))
            gts_pas.append(copy.deepcopy(greenstree_pas))
            ns = gts[-1].tree.get_nodes(somanodes=False)
            ns_pas = gts_pas[-1].tree.get_nodes(somanodes=False)
            for ind, n in enumerate(ns):
                if n._index in node_inds:
                    n.get_content()['physiology'].r_a = n.get_content()['physiology'].r_a / factor
                    ns_pas[ind].get_content()['physiology'].r_a = ns_pas[ind].get_content()['physiology'].r_a / factor


    velocity = np.zeros(len(gts))
    for ind, gt in enumerate(gts):
        velocity[ind] = measure_velocity(gt, gts_pas[ind], node_inds_of_Ranvier, ind_ais, Vth=-40.)

    return velocity

def scan_parameter_range_star(arglist):
    return scan_parameter_range(*arglist)

# parameter ranges
param = ['cm_myelin', 'cm_node', 'L_internode', 'A_node', 'g_bar', 'd_axon', 'r_a']
texnames = [r'$C_m$ myelin', r'$C_m$ node', r'$L$ internode', r'$A$ node', r'$\overline{g}$', r'd', r'$R_a$']
param_range = {}
param_range['cm_myelin']    = np.linspace(0.9,1.6,10)
param_range['cm_node']      = np.linspace(0.4,1.1,10)
param_range['L_internode']  = np.linspace(0.5,1.1,10)
param_range['A_node']       = np.linspace(0.4,1.1,10)
param_range['g_bar']        = np.linspace(0.4,1.1,10)
param_range['d_axon']       = np.linspace(0.9,1.55,10)
param_range['r_a']          = np.linspace(0.9,1.55,10)

# initialize greenstree's
node_inds_of_Ranvier = [n for n in range(45) if n%2 == 0 and n > 5]
ind_ais = 4
greenstree, greenstree_pas = morphR.make_axon_trees('morphologies/Moore1978_axon.swc', nodes_of_ranvier=node_inds_of_Ranvier, ais_node_index=ind_ais)

v_0, result, Vm = measure_velocity(copy.deepcopy(greenstree), copy.deepcopy(greenstree_pas), node_inds_of_Ranvier=node_inds_of_Ranvier, ind_ais=ind_ais, run_NEURON=True, temp=6.3)

v_standard =  measure_velocity(copy.deepcopy(greenstree), copy.deepcopy(greenstree_pas), node_inds_of_Ranvier=node_inds_of_Ranvier, ind_ais=ind_ais)

# velocity = scan_parameter_range(greenstree, greenstree_pas, 'cm_myelin', param_range['cm_myelin'], node_inds_of_Ranvier, ind_ais)
# velocity = scan_parameter_range(greenstree, greenstree_pas, 'cm_node', param_range['cm_node'], node_inds_of_Ranvier, ind_ais)
# velocity = scan_parameter_range(greenstree, greenstree_pas, 'L_internode', param_range['L_internode'], node_inds_of_Ranvier, ind_ais)
# velocity = scan_parameter_range(greenstree, greenstree_pas, 'A_node', param_range['A_node'], node_inds_of_Ranvier, ind_ais)
# velocity = scan_parameter_range(greenstree, greenstree_pas, 'g_bar', param_range['g_bar'], node_inds_of_Ranvier, ind_ais)
# velocity = scan_parameter_range(greenstree, greenstree_pas, 'd_axon', param_range['d_axon'], node_inds_of_Ranvier, ind_ais)
# velocities = scan_parameter_range(greenstree, greenstree_pas, 'r_a', param_range['r_a'], node_inds_of_Ranvier, ind_ais)

pool = Pool(processes=7)
velocities = pool.map(scan_parameter_range_star, [ \
                [greenstree, greenstree_pas, 'cm_myelin', param_range['cm_myelin'], node_inds_of_Ranvier, ind_ais],
                [greenstree, greenstree_pas, 'cm_node', param_range['cm_node'], node_inds_of_Ranvier, ind_ais],
                [greenstree, greenstree_pas, 'L_internode', param_range['L_internode'], node_inds_of_Ranvier, ind_ais],
                [greenstree, greenstree_pas, 'A_node', param_range['A_node'], node_inds_of_Ranvier, ind_ais],
                [greenstree, greenstree_pas, 'g_bar', param_range['g_bar'], node_inds_of_Ranvier, ind_ais],
                [greenstree, greenstree_pas, 'd_axon', param_range['d_axon'], node_inds_of_Ranvier, ind_ais],
                [greenstree, greenstree_pas, 'r_a', param_range['r_a'], node_inds_of_Ranvier, ind_ais] \
                ] )
## results #############################################################

from matplotlib import rc, rcParams
legendsize = 18
labelsize = 20
ticksize = 18
lwidth = 1.5
fontsize = 18
rc('font',**{'family':'serif','serif':['Palatino'], 'size': 15.0})
rc('text', usetex=True)
rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
rc('legend',**{'fontsize': 'medium'})
rc('xtick',**{'labelsize': 'small'})
rc('ytick',**{'labelsize': 'small'})
rc('axes',**{'labelsize': 'large', 'labelweight': 'normal'})

colours = ['DeepPink', 'Purple', 'Blue', 'cyan', 'Green', 'DarkOrange', 'Sienna']

F = pl.figure('axonfig', figsize=(8,9))

from matplotlib.gridspec import GridSpec
gs2 = GridSpec(1, 1)
gs2.update(left=0.12, right=0.94, top=0.95, bottom=0.75, hspace=0.2)
ax2 = pl.subplot(gs2[0,0])

labels = [r'AIS', r'NoR']
for j,i in enumerate([1,10]):
    ax2.plot(result['t']-10., result['Vm'][i], lw=lwidth, c=colours[4*j], label=labels[j])
    if i==10:
        # ind = len(node_inds_of_Ranvier) + 1
        ax2.plot(Vm['t']-10., Vm[i], ls='--', lw=lwidth*1.2, c='k', label=r'NEURON')
    else:
        ax2.plot(Vm['t']-10., Vm[i], ls='--', lw=lwidth*1.2, c='k')
        # limits
ax2.set_xlim((0.,25.))
ax2.set_ylim((-80.,60.))
# labels
ax2.set_xlabel(r'$t$ (ms)')
ax2.set_ylabel(r'$V_m$ (mV)')
#legend
leg = ax2.legend(loc=1)
leg.draw_frame(False)
# axes
ax2.spines['top'].set_color('none')
ax2.spines['right'].set_color('none')
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')

gs1 = GridSpec(1, 1)
gs1.update(left=0.12, right=0.95, top=0.65, bottom=0.08, hspace=0.2)
ax1 = pl.subplot(gs1[0,0])
# plot
for ind, key in enumerate(param):
    ax1.plot(param_range[key], velocities[ind] / v_standard, c=colours[ind], lw=lwidth, label=texnames[ind])
# limits
ax1.set_xlim((0.,1.62))
ax1.set_ylim((0.,1.62))
# labels
ax1.set_xlabel(r'Relative parameter')
ax1.set_ylabel(r'Relative velocity')
#legend
leg = ax1.legend(loc=3)
leg.draw_frame(False)
# axes
ax1.spines['top'].set_color('none')
ax1.spines['right'].set_color('none')
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')

if save:
    import os.path
    if os.path.exists('fig_paper_sparsegf/fig_axon_Moore1978.svg'):
        if overwrite:
            pl.savefig('fig_paper_sparsegf/fig_axon_Moore1978.svg')
            pl.savefig('fig_paper_sparsegf/fig_axon_Moore1978.eps')
            pl.savefig('fig_paper_sparsegf/fig_axon_Moore1978.pdf')
            pl.savefig('fig_paper_sparsegf/fig_axon_Moore1978.png')
        else:
            pl.savefig('fig_paper_sparsegf/fig_axon_Moore1978_.svg')
            pl.savefig('fig_paper_sparsegf/fig_axon_Moore1978_.eps')
            pl.savefig('fig_paper_sparsegf/fig_axon_Moore1978_.pdf')
            pl.savefig('fig_paper_sparsegf/fig_axon_Moore1978_.png')
    else:
        pl.savefig('fig_paper_sparsegf/fig_axon_Moore1978.svg')
        pl.savefig('fig_paper_sparsegf/fig_axon_Moore1978.eps')
        pl.savefig('fig_paper_sparsegf/fig_axon_Moore1978.pdf')
        pl.savefig('fig_paper_sparsegf/fig_axon_Moore1978.png')
    import pickle
    if os.path.exists('fig_paper_sparsegf/fig_axon_Moore1978.pkl'):
        if overwrite:
            outfile = open('fig_paper_sparsegf/fig_axon_Moore1978', 'wb')
        else:
            outfile = open('fig_paper_sparsegf/fig_axon_Moore1978_.pkl', 'wb')
    else:
        outfile = open('fig_paper_sparsegf/fig_axon_Moore1978.pkl', 'wb')
    pickle.dump(param_range, outfile)
    pickle.dump(param, outfile)
    pickle.dump(velocities, outfile)
    pickle.dump(v_standard, outfile)
    pickle.dump(result, outfile)
    pickle.dump(Vm, outfile)
    outfile.close()

pl.show()