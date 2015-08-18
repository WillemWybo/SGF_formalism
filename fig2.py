"""
Author: Willem Wybo
Date: 18/08/2015
Place: BBP, Geneva
"""

import matplotlib.pyplot as pl
import numpy as np
import math

import copy
import pickle
import sys
sys.setrecursionlimit(2000)

import morphologyReader as morphR
import neuronModels as neurM

import btstats


save = False
overwrite = False
plot_trace_comparison = False

## parameters ##########################################################
# general
tmax = 1000.
dt = 0.025
rate = 0.003
K = 4
# synapse
w = 0.0008#0.005
t1 = 0.2
t2 = 5.
# plot
tpmax = 1000.
########################################################################

## membrane params #####################################################
# real default channel distribution 
distr_sim = {'L': {'type': 'fit', 'calctype': 'pas', 'param': [-65., 20.], 'E': -65.}
                }
# real soma channel distribution 
s_distr_sim = {'L': {'type': 'fit', 'calctype': 'pas', 'param': [-65., 20.], 'E': -65.},
                'Kv3_1': {'type': 'flat', 'calctype': 'pas', 'param': [0.766*1e6], 'E': -80.},
                'Na_Ta': {'type': 'flat', 'calctype': 'pas', 'param': [1.71 *1e6], 'E':  50.}
                }
########################################################################

## initialization ######################################################
morphfile = 'morphologies/stellate_v2.swc'#ball_and_stick_taper.swc'#ball_and_stick_taper.swc'#N19ttwt.CNG.swc'#3y_tree.swc'#neocortical_pyramidv2.swc'#
# greenstree
greenstree_sim = morphR.greensTree(morphfile, soma_distr=s_distr_sim, ionc_distr=distr_sim, pprint=False)
# greens tree
# greenstree_calc = morphR.greensTree(morphfile, soma_distr=s_distr_calc, ionc_distr=distr_calc, pprint=False)
greenstree_calc = copy.deepcopy(greenstree_sim)
snode = greenstree_calc.tree.get_node_with_index(1)
print 'number of dendrites: ', len(snode.get_child_nodes()[2:])
gs_soma = snode.get_content()['physiology'].gs
print gs_soma
print snode.get_content()['physiology'].es
for key in gs_soma.keys():
    gs_soma[key] = 0.
gfcalc = morphR.greensFunctionCalculator(greenstree_calc)
gfcalc.set_impedances_logscale(fmax=7, base=10, num=200)
inlocs = greenstree_calc.distribute_inlocs(num=50, distrtype='random', radius=0.0070)
# inlocs = [{'node': 1, 'x': 0.5, 'ID': 0}, {'node': 18, 'x': 0.6, 'ID': 1}]
# (inlocs, inlocs_2) = greenstree_calc.distribute_inlocs(num=15, distrtype='fromleaf', radius=0.0120, split_radius=0.0050)
# print inlocs
# print inlocs_2
print '\n>>> number of input locations     =    ', len(inlocs)
# print '\n>>> number of input locations avg =    ', len(inlocs_2)
# calculate dendritic length
btst = btstats.BTStats(greenstree_calc.tree)
Ltot = btst.total_length()
print '\n>>> total length                  =    ', Ltot, 'um'
# simulation gfcalc
gfcalc_sim = morphR.greensFunctionCalculator(greenstree_sim)
gfcalc_sim.set_impedances_logscale(fmax=7, base=10, num=2)
# synapses
synparams = []
for inloc in inlocs:
    synparams.append(copy.deepcopy(inloc))
    synparams[-1]['weight'] = w; synparams[-1]['E_r'] = 0.
    synparams[-1]['tau1'] = t1; synparams[-1]['tau2'] = t2
# synparams_2 = []
# for inloc in inlocs_2:
#     synparams_2.append(copy.deepcopy(inloc))
#     synparams_2[-1]['weight'] = w; synparams_2[-1]['E_r'] = 0.
#     synparams_2[-1]['tau1'] = t1; synparams_2[-1]['tau2'] = t2
# poisson spiketrains
import spikeTrainGenerator as s
spiketimes = []
for i in range(len(synparams)):
    spiketimes.append({'ID': synparams[i]['ID'], 'spks': s.poissonTrain(rate,tmax)})
    # spiketimes.append({'ID': synparams[i]['ID'], 'spks': []})
inlocs_IDs = np.array([inloc['ID'] for inloc in inlocs])
# spiketimes_2 = [copy.deepcopy(spiketimes[0])]
# for i, synpar in enumerate(synparams_2[1:]):
#     spks = []
#     for ID in synpar['IDs']:
#         ind = np.where(inlocs_IDs == ID)[0][0]
#         spks += spiketimes[ind]['spks']
#     spiketimes_2.append({'ID': synpar['ID'], 'spks': np.sort(spks)})
# spiketimes[0]['spks'] = [0.125] + spiketimes[0]['spks']
# for i in range(len(synparams)): 
    # spiketimes.append({'ID': synparams[i]['ID'], 'spks': [i*100.+2.]})
# calc inloc conductances 
# gs_point, es_point, gcalctype_point = greenstree_sim.calc_IP_conductances(inlocs)
# for key in gs_point.keys():
#     if key != 0:
#         gs_point[key]['L'] = 0.
# print gs_point
# print es_point
snode = greenstree_sim.tree.get_node_with_index(1)
gs_soma = snode.get_content()['physiology'].gs
es_soma = snode.get_content()['physiology'].es
somaA = snode.get_content()['impedance'].somaA
gs_point = {inloc['ID']: {'L': 0.} for inloc in inlocs}
es_point = {inloc['ID']: {'L': -65.} for inloc in inlocs}
gcalctype_point = {inloc['ID']: {'L': 'pas'} for inloc in inlocs}
for key in gs_soma.keys():
    gs_point[inlocs[0]['ID']][key] = gs_soma[key]*somaA
    es_point[inlocs[0]['ID']][key] = es_soma[key]
    gcalctype_point[inlocs[0]['ID']][key] = 'pas'
# print gs_point, es_point
# compute SGF
alphas, gammas, pairs, Ms = gfcalc.kernelSet_sparse(inlocs, FFT=False, kernelconstants=True, pprint=False)
for key in alphas.keys():
    print key
    print alphas[key]
# alphas_sparse, v2y_sparse, y2v_sparse, pairs_sparse, indices = gfcalc.kernelSet_sparse(inlocs, FFT=False)
nK = 0; nExp = 0
for key in alphas.keys(): 
    nK += 1; nExp += len(alphas[key])
nExpavg = float(nExp) / float(nK)
print '\n>>> number of kernels =                ', nK
print '\n>>> O_n approach <<<'
print 'number of exponentials / kernel =        ', nExpavg
# preprocessor test
prep = neurM.preprocessor()
# alphas_sparse, prop1, prop2, v_inv_partial, y2v_v_prod, magnitudes, pairs_sparse = prep.construct_matrices_from_kernels(v2y_sparse, y2v_sparse, alphas_sparse, pairs_sparse, 1e-3, dt, full=False)
mat_dict_On2 = prep.construct_volterra_matrices_On2(dt, alphas, gammas, pprint=False)
nStep = 0
for ind, Nconv in np.ndenumerate(mat_dict_On2['N_conv']):
    nStep += Nconv
nStepavg = float(nStep) / float(nK)
print '\n>>> O_n^2 approach <<<'
print 'number of steps / kernel =               ', nStepavg
mat_dict_On = prep.construct_volterra_matrices_On(dt, alphas, gammas, pprint=False)
mat_dict_hybrid = prep.construct_volterra_matrices_hybrid(dt, alphas, gammas, K, pprint=False)
nExp_hybrid = 0;
for key in mat_dict_hybrid['v2y']:
    nExp_hybrid += len(mat_dict_hybrid['v2y'][key])
nExpavg_hybrid = float(nExp_hybrid) / float(nK)
print '\n>>> Hybrid approach <<<'
print 'number of exponentials / kernel =        ', nExpavg_hybrid 
print 'number of convolution steps / kernel =   ', float(K)
print 'computations / kernel =                  ', nExpavg_hybrid  + K
print '\n'
spiketimes_mat, multiplicity_mat = prep.construct_spiketimes_matrix(dt, tmax, spiketimes)
########################################################################


## run simulations #####################################################
# SGF
neuron_SGF_ = neurM.lightweight_integrator_neuron(len(inlocs))
if plot_trace_comparison:
    result_SGF = neuron_SGF_.run_volterra_On(tmax, dt, spiketimes_mat, multiplicity_mat, weight=w, tau1=t1, tau2=t2,
                                                gs_soma=gs_point[inlocs[0]['ID']], es_soma=es_point[inlocs[0]['ID']], mat_dict=mat_dict_On)
    # neuron_SGF_1 = neurM.integratorneuron(inlocs, synparams, [], gs_point, es_point, gcalctype_point)
    # result_SGF = neuron_SGF_1.run_volterra_On(tmax, dt, spiketimes, mat_dict=mat_dict_On)
    result_SGF_On2 = neuron_SGF_.run_volterra_On2(tmax, dt, spiketimes_mat, multiplicity_mat, weight=w, tau1=t1, tau2=t2, 
                                                gs_soma=gs_point[inlocs[0]['ID']], es_soma=es_point[inlocs[0]['ID']], mat_dict=mat_dict_On2)
    # neuron_SGF_2 = neurM.integratorneuron(inlocs, synparams, [], gs_point, es_point, gcalctype_point)
    # result_SGF_On2 = neuron_SGF_2.run_volterra_On2(tmax, dt, spiketimes, mat_dict=mat_dict_On2)
result_SGF_hybrid = neuron_SGF_.run_volterra_hybrid(tmax, dt, spiketimes_mat, multiplicity_mat, weight=w, tau1=t1, tau2=t2, 
                                                    gs_soma=gs_point[inlocs[0]['ID']], es_soma=es_point[inlocs[0]['ID']], mat_dict=mat_dict_hybrid)
# neuron_SGF = neurM.integratorneuron(inlocs, synparams, [], gs_point, es_point, gcalctype_point)
# result_SGF_hybrid = neuron_SGF.run_volterra_hybrid(tmax, dt, spiketimes, mat_dict=mat_dict_hybrid)
# NEURON
# HHneuron_2 = neurM.NeuronNeuron(greenstree_sim, dt=dt, truemorph=False, factorlambda=1.)
# HHneuron_2.add_double_exp_synapses(copy.deepcopy(synparams_2))
# HHneuron.add_double_exp_current(copy.deepcopy(synparams))
# HHneuron_2.set_spiketrains(spiketimes_2)
# Vm_2 = HHneuron_2.run(tdur=tmax, pprint=True)

HHneuron = neurM.NeuronNeuron(greenstree_sim, dt=dt, truemorph=False, factorlambda=1.)
HHneuron.add_double_exp_synapses(copy.deepcopy(synparams))
# HHneuron.add_double_exp_current(copy.deepcopy(synparams))
HHneuron.set_spiketrains(spiketimes)
Vm = HHneuron.run(tdur=tmax, pprint=True)
########################################################################

from matplotlib import rc, rcParams
legendsize = 10
labelsize = 15
ticksize = 15
lwidth = 1.5
fontsize = 16
#~ font = {'family' : 'serif',
        #~ 'weight' : 'normal',
        #~ 'size'   : fontsize} 
        #'sans-serif':'Helvetica'}
#'family':'serif','serif':['Palatino']}
#~ rc('font', **font)
rc('font',**{'family':'serif','serif':['Palatino'], 'size': 15.0})
rc('text', usetex=True)
rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
rc('legend',**{'fontsize': 'medium'})
rc('xtick',**{'labelsize': 'small'})
rc('ytick',**{'labelsize': 'small'})
rc('axes',**{'labelsize': 'large', 'labelweight': 'normal'})

import matplotlib.pyplot as pl
F = pl.figure(figsize=(14,9))

from matplotlib.offsetbox import AnchoredText
size = dict(size=fontsize+3)
A = AnchoredText('A', loc=2, prop=size, pad=0., borderpad=-1.5, frameon=False)
B = AnchoredText('B', loc=2, prop=size, pad=0., borderpad=-1.5, frameon=False)

# plot neuron
import btviz
from matplotlib.gridspec import GridSpec
gs1 = GridSpec(1, 1)
gs1.update(left=0.08, right=0.38, top=0.95, bottom=0.6)

# syncolors
# cm = pl.get_cmap('jet')
# IDs = np.array([inl['ID'] for inl in inlocs])
# syncolors = {1: 'r'}
# for i, inloc in enumerate(inlocs_2[1:]):
#     inds = []
#     fcolor = float(i) / float(len(inlocs_2[1:]))
#     for ID in inloc['IDs']:
#         ind = np.where(ID == IDs)[0][0]
#         syncolors[inlocs[ind]['node']] = cm(fcolor)
# print syncolors

# synapse 0
synnodes = [synpar['node'] for synpar in synparams]
ax1 = pl.subplot(gs1[0,0])
# ax1.add_artist(A)
btviz.plot_2D_SWC(tree=greenstree_calc.tree, synapses=synnodes, syn_labels=None)

# plot temporal integration
gs2 = GridSpec(1, 1)
gs2.update(left=0.08, right=0.95, bottom=0.08, top=0.5, hspace=0.2)


ax2 = pl.subplot(gs2[0,0])
# plot soma trace
i_end = int(tpmax/dt)
# ax2.add_artist(AnchoredText(r'Soma', loc=2, prop=size, pad=0.1, frameon=False))
ax2.plot(result_SGF_hybrid['t'][0:i_end], result_SGF_hybrid['Vm'][0][0:i_end], 'r', lw=lwidth, label=r'GF sparse')
# ax2.plot(result_SGF['t'], result_SGF['Vm'][0], 'b-.', lw=lwidth, label=r'GF sparse allexp')
# ax2.plot(result_SGF_On2['t'], result_SGF_On2['Vm'][0], 'c-.', lw=lwidth, label=r'GF sparse allconv')
ax2.plot(Vm['t'][0:i_end], Vm['vmsoma'][0:i_end], 'k--', lw=lwidth*1.2, label=r'NEURON')
# ax2.plot(Vm_2['t'][0:i_end], Vm_2['vmsoma'][0:i_end], 'c-.', lw=lwidth*1.2, label=r'NEURON avg')
#labels
ax2.set_xlabel(r'$t$ (ms)', fontsize=labelsize)
ax2.set_ylabel(r'$V_m$ (mV)', fontsize=labelsize)
#legend
leg = ax2.legend(loc=1, ncol=1, markerscale=lwidth)
leg.draw_frame(False)
# limits
# ax2.set_ylim((-66,-50))
# axes
ax2.spines['top'].set_color('none')
ax2.spines['right'].set_color('none')
ax2.yaxis.set_ticks_position('left')
ax2.xaxis.set_ticks_position('bottom')
# ax2.axes.get_xaxis().set_visible(False)

gs4 = GridSpec(1, 2)
gs4.update(left=0.42, right=0.95, top=0.95, bottom=0.6, hspace=0.2, wspace=0.3)
ax4 = pl.subplot(gs4[0,0])
# make barplot
error_config = {'ecolor': '0.1'}
opacity = 0.8
number_of_kernels = len(alphas.keys())
ax4.bar([1./7.], [len(inlocs)**2], 2./7., alpha=opacity, color='b', error_kw=error_config, label=r'GF')
ax4.bar([4./7.], [number_of_kernels], 2./7., alpha=opacity, color='r', error_kw=error_config, label=r'SGF')
ax4.set_xlim((0,1))
ax4.set_ylim(bottom=0)
ax4.set_ylabel(r'No. of kernels')
ax4.set_xlabel(r'Formalism')
#axes
ax4.spines['top'].set_color('none')
ax4.spines['right'].set_color('none')
ax4.yaxis.set_ticks_position('left')
ax4.xaxis.set_ticks_position('bottom')
ax4.xaxis.set_ticks((2./7., 5./7.))
ax4.xaxis.set_ticklabels([r'GF', r'SGF'])

ax5 = pl.subplot(gs4[0,1])
# make barplot
error_config = {'ecolor': '0.1'}
opacity = 0.8
number_of_kernels = len(alphas.keys())
ax5.bar([1./10.], [nExpavg], 2./10., alpha=opacity, color='b', error_kw=error_config)
ax5.bar([4./10.], [nStepavg], 2./10., alpha=opacity, color='g', error_kw=error_config)
ax5.bar([7./10.], [nExpavg_hybrid + K], 2./10., alpha=opacity, color='r', error_kw=error_config)
ax5.set_xlim((0,1))
ax5.set_ylim(bottom=0)
ax5.set_ylabel(r'$n_K$')
ax5.set_xlabel(r'Convolution method')
#axes
ax5.spines['top'].set_color('none')
ax5.spines['right'].set_color('none')
ax5.yaxis.set_ticks_position('left')
ax5.xaxis.set_ticks_position('bottom')
ax5.xaxis.set_ticks((2./10., 5./10., 8./10.))
ax5.xaxis.set_ticklabels([r'Exp', r'Quad', r'Mix'])

if save:
    import os.path
    if os.path.exists('fig_paper_sparsegf/fig2math.svg'):
        if overwrite:
            pl.savefig('fig_paper_sparsegf/fig2math.svg')
            pl.savefig('fig_paper_sparsegf/fig2math.eps')
            pl.savefig('fig_paper_sparsegf/fig2math.pdf')
            pl.savefig('fig_paper_sparsegf/fig2math.png')
        else:
            pl.savefig('fig_paper_sparsegf/fig2math_.svg')
            pl.savefig('fig_paper_sparsegf/fig2math_.eps')
            pl.savefig('fig_paper_sparsegf/fig2math_.pdf')
            pl.savefig('fig_paper_sparsegf/fig2math_.png')
    else:
        pl.savefig('fig_paper_sparsegf/fig2math.svg')
        pl.savefig('fig_paper_sparsegf/fig2math.eps')
        pl.savefig('fig_paper_sparsegf/fig2math.pdf')
        pl.savefig('fig_paper_sparsegf/fig2math.png')

    if os.path.exists('fig_paper_sparsegf/fig2math.pkl'):
        if overwrite:
            outfile = open('fig_paper_sparsegf/fig2math.pkl', 'wb')
        else:
            outfile = open('fig_paper_sparsegf/fig2math_.pkl', 'wb')
    else:
        outfile = open('fig_paper_sparsegf/fig2math.pkl', 'wb')
    pickle.dump(inlocs, outfile)
    pickle.dump(mat_dict_On, outfile)
    pickle.dump(number_of_kernels, outfile)
    pickle.dump(result_SGF_hybrid, outfile)
    pickle.dump(Vm, outfile)
    # pickle.dump(Vm_2, outfile)
    pickle.dump(nExpavg, outfile)
    pickle.dump(nStepavg, outfile)
    pickle.dump(nExpavg_hybrid, outfile)
    pickle.dump(nK, outfile)
    # pickle.dump(coinc, outfile)
    # pickle.dump(coinc_2, outfile)
    outfile.close()

if plot_trace_comparison:
    pl.figure('trace comparison', figsize=(7,4))
    pl.plot(result_SGF_hybrid['t'], result_SGF_hybrid['Vm'][0], 'r', lw=lwidth, label=r'GF sparse')
    pl.plot(result_SGF['t'], result_SGF['Vm'][0], 'b-.', lw=lwidth, label=r'GF sparse allexp')
    pl.plot(result_SGF_On2['t'], result_SGF_On2['Vm'][0], 'c-.', lw=lwidth, label=r'GF sparse allconv')
    pl.plot(Vm['t'], Vm['vmsoma'], 'k--', lw=lwidth*1.2, label=r'NEURON')
    #labels
    pl.xlabel(r'$t$ (ms)', fontsize=labelsize)
    pl.ylabel(r'$V_m$ (mV)', fontsize=labelsize)
    #legend
    leg = pl.legend(loc=1, ncol=1, markerscale=lwidth)
    leg.draw_frame(False)

pl.show()
########################################################################
