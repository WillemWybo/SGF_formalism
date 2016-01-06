import matplotlib.pyplot as pl
import numpy as np

import copy
import pickle
import sys
sys.setrecursionlimit(2000)

import morphologyReader as morphR
import neuronModels as neurM
import functionFitter as funF

## parameters
Veq = -65. # mV
tmax = 300. # ms
dt = .1 # ms
K = 4 

## initialization #####################################################################
## Step 0: initialize the morphology
# Specify the path to an '.swc' file.
morphfile = 'morphologies/ball_and_stick_taper.swc'
# Define the ion channel distributions for dendrites and soma. Here the neuron model is 
# passive.
d_distr = {'L': {'type': 'fit', 'param': [Veq, 50.], 'E': Veq, 'calctype': 'pas'}}
s_distr = {'L': {'type': 'fit', 'param': [Veq, 50.], 'E': Veq, 'calctype': 'pas'}}
# initialize a greensTree. Here, all the quantities are stored to compute the GF in the
# frequency domain (algorithm of Koch and Poggio, 1985).
greenstree = morphR.greensTree(morphfile, soma_distr=s_distr, ionc_distr=d_distr, cnodesdistr='all')
# initialize a greensFunctionCalculator using the previously created greensTree. This class
# stores all variables necessary to compute the GF in a format fit for simulation, either 
# the plain time domain or with the partial fraction decomposition.
gfcalc = morphR.greensFunctionCalculator(greenstree)
gfcalc.set_impedances_logscale(fmax=7, base=10, num=200)
# Now a list of input locations needs to be defined. For the sparse reformulation, the 
# first location needs to be the soma
inlocs = [  {'node': 1, 'x': .5, 'ID': 0}, {'node': 4, 'x': .5, 'ID': 1}, {'node': 5, 'x': .5, 'ID': 2}, 
			{'node': 6, 'x': .5, 'ID': 3}, {'node': 7, 'x': .5, 'ID': 4}, {'node': 8, 'x': .5, 'ID': 5}, 
			{'node': 9, 'x': .5, 'ID': 6}]
## Steps 1,2,3 and 4:
# find sets of nearest neighbours, computes the necessary GF kernels, then computes the
# sparse kernels and then fits the partial fraction decomposition using the VF algorithm.
alphas, gammas, pairs, Ms = gfcalc.kernelSet_sparse(inlocs, FFT=False, kernelconstants=True)
## Step 4 bis: compute the vectors that will be used in the simulation
prep = neurM.preprocessor()
mat_dict_hybrid = prep.construct_volterra_matrices_hybrid(dt, alphas, gammas, K, pprint=False)
## Examples of steps that happen within the kernelSet_sparse function
## Step 1: example to find the nearest neighbours
NNs, _ = gfcalc.greenstree.get_nearest_neighbours(inlocs, add_leaves=False, reduced=False)
## Step 2: example of finding a kernel
g_example = gfcalc.greenstree.calc_greensfunction(inlocs[0], inlocs[1], voltage=True)
## Step 4: example of computing a partial fraction decomposition
FEF = funF.fExpFitter()
alpha_example, gamma_example, pair_example, rms = FEF.fitFExp_increment(gfcalc.s, g_example, \
                           rtol=1e-8, maxiter=50, realpoles=False, constrained=True, zerostart=False)
# # plot the kernel example
# pl.figure('kernel example')
# pl.plot(gfcalc.s.imag, g_example.real, 'b')
# pl.plot(gfcalc.s.imag, g_example.real, 'r')
#######################################################################################


## Simulation #########################################################################
# define a synapse and a spiketime
synapseparams = [{'node': 9, 'x': .5, 'ID': 0, 'tau1': .2, 'tau2': 3., 'E_r': 0., 'weight': 5.*1e-3}]
spiketimes = [{'ID': 0, 'spks': [10.]}]
# ion channel conductances at integration points, in this example, there is only leak which is 
# already incorporated in the GF
gs_point = {inloc['ID']: {'L': 0.} for inloc in inlocs}
es_point = {inloc['ID']: {'L': -65.} for inloc in inlocs}
gcalctype_point = {inloc['ID']: {'L': 'pas'} for inloc in inlocs}
# create an SGF neuron 
SGFneuron = neurM.integratorneuron(inlocs, synapseparams, [], gs_point, es_point, gcalctype_point,
																E_eq=Veq, nonlinear=False)
# run the simulation
SGFres = SGFneuron.run_volterra_hybrid(tmax, dt, spiketimes, mat_dict=mat_dict_hybrid)
# run a neuron simulation for comparison
NEURONneuron = neurM.NeuronNeuron(greenstree, dt=dt, truemorph=True, factorlambda=10.)
NEURONneuron.add_double_exp_synapses(copy.deepcopy(synapseparams))
NEURONneuron.set_spiketrains(spiketimes)
NEURres = NEURONneuron.run(tdur=tmax, pprint=False)
#######################################################################################


## plot trace
pl.figure('simulation')
pl.plot(NEURres['t'], NEURres['vmsoma'],  'r-', label=r'NEURON soma')
pl.plot(NEURres['t'], NEURres[0],         'b-', label=r'NEURON syn')
pl.plot(SGFres['t'], SGFres['Vm'][0,:],  'r--', lw=1.7, label=r'SGF soma')
pl.plot(SGFres['t'], SGFres['Vm'][-1,:], 'b--', lw=1.7, label=r'SGF syn')
pl.xlabel(r'$t$ (ms)')
pl.ylabel(r'$V_m$ (mV)')
pl.legend(loc=0)
pl.show()
