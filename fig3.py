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

## parameters ##########################################################
# general
tmax = 10000.
dt = .1
rate = 1.
K = 10
# synapse
w = 0.0002#0.005
t1 = 0.2
t2 = 5.
# plot
tpmax = 1000.
imax = int(tpmax/dt)
# sims
Nmax = 3
ind_recloc = 0 # soma
########################################################################

## membrane params #####################################################
# real default channel distribution 
distr_sim = {'L': {'type': 'fit', 'calctype': 'pas', 'param': [-65., 20.], 'E': -65.}
                }
# real soma channel distribution 
s_distr_sim = {'L': {'type': 'fit', 'calctype': 'pas', 'param': [-65., 20.], 'E': -65.}
                }
########################################################################

## initialization and simulation #######################################
morphfile = 'morphologies/stellate_v2.swc'#ball_and_stick_taper.swc'#y_tree.swc'#neocortical_pyramidv2.swc'#3y_tree.swc'#
# greenstree
greenstree = morphR.greensTree(morphfile, soma_distr=s_distr_sim, ionc_distr=distr_sim, pprint=False)
# greens tree
gfcalc = morphR.greensFunctionCalculator(greenstree)
gfcalc.set_impedances_logscale(fmax=7, base=10, num=200)
# inlocs = greenstree.distribute_inlocs(num=50, distrtype='uniform', radius=0.0070)
inlocs_all = greenstree.distribute_inlocs(num=Nmax, distrtype='hines', radius=0.0050)
# NNs,_ = greenstree.get_nearest_neighbours(inlocs, add_leaves=False, separate=False, test=False)
# print '>>> inlocs '
# print inlocs
# print '>>> nearest neighbours '
# print NNs
IDs_all = [inloc['ID'] for inloc in inlocs_all]
# equilibrium potential
Es_eq_all = {ID: -65. for ID in IDs_all}
# calculate dendritic length
btst = btstats.BTStats(greenstree.tree)
Ltot = btst.total_length()
print '\n>>> total length                  =    ', Ltot, 'um'
# lists to save the results
# res_C_ns_list = []; res_C_s_list = []
# Vm_ns_list = []; Vm_s_list = []
t_exec1 = []
t_exec2 = []
t_exec3 = []
t_exec4 = []
nK_avg_list = []
for N in range(2,len(inlocs_all)):
    print '\n>>> simulation with ' + str(N) + ' inlocs'
    inlocs = inlocs_all[:N]
    IDs = IDs_all[:N]
    NNs,_ = greenstree.get_nearest_neighbours(inlocs, add_leaves=False, separate=False, test=False)
    # synapses
    synparams = []
    for inloc in inlocs:
        synparams.append(copy.deepcopy(inloc))
        synparams[-1]['weight'] = w; synparams[-1]['E_r'] = 0.
        synparams[-1]['tau1'] = t1; synparams[-1]['tau2'] = t2
    # poisson spiketrains
    import spikeTrainGenerator as s
    spiketimes = []
    for i in range(len(synparams)):
        spiketimes.append({'ID': synparams[i]['ID'], 'spks': np.array(s.poissonTrain(rate/N,tmax))})
        # spiketimes.append({'ID': synparams[i]['ID'], 'spks': np.array([100.+100.*i])})
        # spiketimes.append({'ID': synparams[i]['ID'], 'spks': np.array([])})
        # spiketimes.append({'ID': synparams[i]['ID'], 'spks': np.linspace(0.,tmax, int(tmax/dt))})
    inlocs_IDs = np.array([inloc['ID'] for inloc in inlocs])
    # compute SGF
    alphas, gammas, pairs, Ms = gfcalc.kernelSet_sparse(inlocs, FFT=False, kernelconstants=True, pprint=False)
    # preprocessor 
    prep = neurM.preprocessor()
    # C model no spikes
    sgfM_ns, nK = prep.construct_C_model_hybrid(dt, inlocs, NNs, alphas, gammas, K)
    sgfM_ns.add_recorder(ind_recloc)
    print "<n_K> =", nK
    nK_avg_list.append(nK)
    # C model spikes
    sgfM_s, _ = prep.construct_C_model_hybrid(dt, inlocs, NNs, alphas, gammas, K)
    sgfM_s.add_recorder(ind_recloc)
    for i, synpar in enumerate(synparams):
        locind = IDs.index(synpar['ID'])
        n = sgfM_s.add_synapse(locind, synpar['tau1'], synpar['tau2'], synpar['E_r']-Es_eq_all[synpar['ID']], synpar['weight'])
        sgfM_s.add_spiketrain(n, spiketimes[i]['spks'])
    # run SGF C models
    res_C_ns = sgfM_ns.run(tmax)
    res_C_s = sgfM_s.run(tmax)
    # add equilibrium potential
    for i, V in enumerate(res_C_ns['Vm']):
        res_C_ns['Vm'][i] += Es_eq_all[inlocs[ind_recloc]['ID']]
    for i, V in enumerate(res_C_s['Vm']):
        res_C_s['Vm'][i] += Es_eq_all[inlocs[ind_recloc]['ID']]
    # res_C_ns_list.append(res_C_ns)
    # res_C_s_list.append(res_C_s)
    t_exec1.append(res_C_ns['t_exec'])
    t_exec2.append(res_C_s['t_exec'])
    # construtc NEURON model no spikes
    HHneuron_ns = neurM.NeuronNeuron(greenstree, dt=dt, truemorph=False, dx=15., printtopology=False)
    print 'Nsec NEURON = ', HHneuron_ns.count_nsec()
    # construct NEURON spikes
    HHneuron_s = neurM.NeuronNeuron(greenstree, dt=dt, truemorph=False, dx=15., printtopology=False)
    HHneuron_s.add_double_exp_synapses(copy.deepcopy(synparams))
    HHneuron_s.set_spiketrains(spiketimes)
    # run NEURON models
    Vm_ns = HHneuron_ns.run(tdur=tmax, pprint=True, record_from_locs=False)
    Vm_s = HHneuron_s.run(tdur=tmax, pprint=True, record_from_locs=False)
    # Vm_ns_list.append(Vm_ns)
    # Vm_s_list.append(Vm_s)
    t_exec3.append(Vm_ns['t_exec'])
    t_exec4.append(Vm_s['t_exec'])
########################################################################


## plotting ############################################################
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

F = pl.figure(figsize=(14,9))

import btviz
from matplotlib.gridspec import GridSpec
gs1 = GridSpec(1, 1)
gs1.update(left=0.04, right=0.34, top=0.75, bottom=0.25)

# plot morphology
ax1 = pl.subplot(gs1[0,0])
synnodes = [synpar['node'] for synpar in synparams]
btviz.plot_2D_SWC(tree=greenstree.tree, synapses=synnodes, syn_labels=None)

gs2 = GridSpec(3, 1)
gs2.update(left=0.42, right=0.95, top=0.95, bottom=0.08, hspace=0.3 )

# pl.figure('voltage')
# ax = pl.gca()
ax = pl.subplot(gs2[0,0])
ax.plot(res_C_s['t'][:imax], res_C_s['Vm'][0][:imax], 'r', lw=lwidth, label=r'SGF')
ax.plot(Vm_s['t'][:imax], Vm_s['vmsoma'][:imax], 'k--', lw=lwidth*1.2, label=r'NEURON')
ax.set_xlabel(r'$t$ (ms)')
ax.set_ylabel(r'$V_m$ (mV)')
ax.set_ylim((-70., -30.))
leg = ax.legend(loc=0, ncol=1, markerscale=lwidth)
leg.draw_frame(False)
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

# pl.figure('runtime')
# ax = pl.gca()
ax = pl.subplot(gs2[1,0])
x_arr = np.arange(2, len(inlocs_all))
t_exec1 = np.array(t_exec1)
t_exec2 = np.array(t_exec2)
t_exec3 = np.array(t_exec3)
t_exec4 = np.array(t_exec4)
ax.plot(x_arr, t_exec1, 'rD--', ms=5*lwidth, lw=lwidth, label=r'SGF stim')
ax.plot(x_arr, t_exec2, 'ro--', ms=5*lwidth, lw=lwidth, label=r'SGF no stim')
ax.plot(x_arr, t_exec3, 'kD--', ms=5*lwidth, lw=lwidth, label=r'NEURON stim')
ax.plot(x_arr, t_exec4, 'ko--', ms=5*lwidth, lw=lwidth, label=r'NEURON no stim')
ax.set_xlabel(r'$N$')
ax.set_ylabel(r'$t_{exec}$ (s)')
ax.set_ylim((-.01, 6.))
leg = ax.legend(loc=0, ncol=1)
leg.draw_frame(False)
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

# pl.figure('<nK>')
# ax = pl.gca()
ax = pl.subplot(gs2[2,0])
ax.plot(x_arr, nK_avg_list, 'go--', ms=5*lwidth, lw=lwidth)
ax.set_xlabel(r'$N$')
ax.set_ylabel(r'$\langle n_K \rangle$')
ax.set_ylim((0,17))
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

if save:
    import os.path
    if os.path.exists('fig_paper_sparsegf/fig3math.svg'):
        if overwrite:
            pl.savefig('fig_paper_sparsegf/fig3math.svg')
            pl.savefig('fig_paper_sparsegf/fig3math.eps')
            pl.savefig('fig_paper_sparsegf/fig3math.pdf')
            pl.savefig('fig_paper_sparsegf/fig3math.png')
        else:
            pl.savefig('fig_paper_sparsegf/fig3math_.svg')
            pl.savefig('fig_paper_sparsegf/fig3math_.eps')
            pl.savefig('fig_paper_sparsegf/fig3math_.pdf')
            pl.savefig('fig_paper_sparsegf/fig3math_.png')
    else:
        pl.savefig('fig_paper_sparsegf/fig3math.svg')
        pl.savefig('fig_paper_sparsegf/fig3math.eps')
        pl.savefig('fig_paper_sparsegf/fig3math.pdf')
        pl.savefig('fig_paper_sparsegf/fig3math.png')

pl.show()

########################################################################