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
import functionFitter as funF


save = False
overwrite = False

## membrane params #####################################################
# real default channel distribution 
distr_sim = {'L': {'type': 'flat', 'param': [1./20000.*1e6], 'E': -65.}}
# real soma channel distribution 
s_distr_sim = {'L': {'type': 'flat', 'param': [1./20000.*1e6], 'E': -65.}}
########################################################################


## initialization ######################################################
morphfile = 'morphologies/ball_and_stick_taper.swc'
# greens tree
greenstree = morphR.greensTree(morphfile, soma_distr=s_distr_sim, ionc_distr=distr_sim, pprint=False)
gfcalc = morphR.greensFunctionCalculator(greenstree)
gfcalc.set_impedances_logscale(fmax=7, base=10, num=200)
inlocs = greenstree.distribute_inlocs(num=50, distrtype='uniform', radius=0.0051)
# kernels
inpmat, transfmat, s = gfcalc.kernelSet_sparse(inlocs, freqdomain=True, pprint=False)
ydata = transfmat[1,2]
########################################################################


## approximation #######################################################
FEF = funF.fExpFitter()
alphalist = []; gammalist = []; rmselist = []; pairslist = []
yfitlist = []
for l in range(1,20):
    trialpoles, pairs = FEF._find_start_nodes(s, l, False, 'log10')
    alist, clist, rmslist, pairslist = FEF._run_fit(s, transfmat[1,2], trialpoles, pairs, 1e-50, 5, True, False)
    indmin = np.argmin(np.array(rmslist))
    alphalist.append(alist[indmin]); gammalist.append(clist[indmin]); rmselist.append(rmslist[indmin]); pairslist.append(pairslist[indmin])
    yfitlist.append(FEF.sumFExp(s, alist[indmin], clist[indmin]))
    print '\nnumber of poles =  ', l
    print 'RMSE =               ', rmselist[-1]
########################################################################


## figure ##############################################################
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
F = pl.figure(figsize=(20,7.5))

from matplotlib.offsetbox import AnchoredText
size = dict(size=fontsize+3)
A = AnchoredText('A', loc=2, prop=size, pad=0., borderpad=-1.5, frameon=False)
B = AnchoredText('B', loc=2, prop=size, pad=0., borderpad=-1.5, frameon=False)
C = AnchoredText('C', loc=2, prop=size, pad=0., borderpad=-1.5, frameon=False)

from matplotlib.gridspec import GridSpec
gs = GridSpec(1, 3)
gs.update(left=0.06, right=0.95, top=0.92, bottom=0.12, hspace=0.2, wspace=0.3)

ax1 = pl.subplot(gs[0,0])
ax1.add_artist(A)
#plot
ax1.plot(s.imag, ydata.real, 'b', lw=lwidth, label=r'calc real')
ax1.plot(s.imag, ydata.imag, 'r', lw=lwidth, label=r'calc imag')
ax1.plot(s.imag, yfitlist[4].real, 'b--', lw=lwidth*2, label=r'fit ($L=10$) real')
ax1.plot(s.imag, yfitlist[4].imag, 'r--', lw=lwidth*2, label=r'fit ($L=10$) imag')
# limits
ax1.set_xlim((-1e6, 1e6))
# labels
ax1.set_xlabel((r'$\omega$ (Hz)'))
ax1.set_ylabel((r'$h(\omega)$'))
# legend
leg = ax1.legend(loc=1, ncol=1, markerscale=lwidth)
leg.draw_frame(False)
# ticks
ax1.spines['top'].set_color('none')
ax1.spines['right'].set_color('none')
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')

ax2 = pl.subplot(gs[0,1])
ax2.add_artist(B)
# plot
ax2.plot(s.imag, np.abs((ydata - yfitlist[4]) / np.max(np.abs(ydata))), 'b', lw=lwidth, label=r'real')
# limits
ax2.set_xlim((-1e6, 1e6))
# labels
ax2.set_xlabel((r'$\omega$ (Hz)'))
ax2.set_ylabel((r'$E(\omega)$'))
# ticks
ax2.spines['top'].set_color('none')
ax2.spines['right'].set_color('none')
ax2.yaxis.set_ticks_position('left')
ax2.xaxis.set_ticks_position('bottom')

ax3 = pl.subplot(gs[0,2])
ax3.add_artist(C)
# plot
ax3.semilogy(2*np.arange(1,20), rmselist, 'b', lw=lwidth)
# limits
# labels
ax3.set_ylabel((r'$\langle E \rangle$'))
ax3.set_xlabel((r'L'))
# ticks
ax3.spines['top'].set_color('none')
ax3.spines['right'].set_color('none')
ax3.yaxis.set_ticks_position('left')
ax3.xaxis.set_ticks_position('bottom')

if save:
    import os.path
    if os.path.exists('fig_paper_sparsegf/fig4math.svg'):
        if overwrite:
            pl.savefig('fig_paper_sparsegf/fig4math.svg')
            pl.savefig('fig_paper_sparsegf/fig4math.eps')
            pl.savefig('fig_paper_sparsegf/fig4math.pdf')
            pl.savefig('fig_paper_sparsegf/fig4math.png')
        else:
            pl.savefig('fig_paper_sparsegf/fig4math_.svg')
            pl.savefig('fig_paper_sparsegf/fig4math_.eps')
            pl.savefig('fig_paper_sparsegf/fig4math_.pdf')
            pl.savefig('fig_paper_sparsegf/fig2math_.png')
    else:
        pl.savefig('fig_paper_sparsegf/fig4math.svg')
        pl.savefig('fig_paper_sparsegf/fig4math.eps')
        pl.savefig('fig_paper_sparsegf/fig4math.pdf')
        pl.savefig('fig_paper_sparsegf/fig4math.png')

pl.show()
