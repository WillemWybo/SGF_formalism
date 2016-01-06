"""
Author: Willem Wybo
Date: 18/08/2015
Place: BBP, Geneva
"""

import numpy as np
import sympy as sp
import scipy.optimize
import scipy.interpolate
import scipy.linalg as la
import scipy.optimize as so
import scipy.sparse.linalg as sla
import scipy.sparse as ss

import math
from itertools import product
from collections import deque
import copy
import posix

from sys import argv, stdout
import os
import time
from inspect import currentframe, getframeinfo

import btstructs

import SGFModel as SGFM

import ionchannels as ionc
import morphologyReader as morphR
import functionFitter as funF

## NEURON neuron with morphology #######################################
import neuron
from neuron import h
h.load_file("stdlib.hoc") # contains the lambda rule
# h.load_nrnmech
h.nrn_load_dll('../x86_64/.libs/libnrnmech.so')
# mechanism_name_translation NEURON
mechname = {'L': 'pas', 
            'h': 'Ih', 'h_HAY': 'Ih_HAY',
            'Na': 'INa', 'Na_p': 'INa_p', 'Na_Ta': 'INa_Ta', 'Na_Ta2': 'INa_Ta2',
            'K': 'IK', 'Klva': 'IKlva', 'KA': 'IKA', 'm': 'Im', 'Kpst': 'IKpst', 'Ktst': 'IKtst', 'Kv3_1': 'IKv3_1', 'KA_prox': 'IKA_prox', 'SK': 'ISK',
            'Ca_LVA': 'ICa_LVA', 'Ca_HVA': 'ICa_HVA',
            'ca': 'conc_ca'}

class NeuronNeuron:
    def __init__(self, greenstree, dt=0.025, truemorph=False, pprint=False, printtopology=False, factorlambda=1., dx=None):
        self.greenstree = greenstree
        nodes = self.greenstree.tree.get_nodes()
        changenodes = [node for node in nodes if morphR.is_changenode(node)]
        self.sections = {}
        if truemorph:
            for node in [nodes[0]] + nodes[3:]:
                self.sections.update({node.get_index(): \
                                       self._make_true_section(node, pprint, factorlambda, dx)})
        else:
            for node in changenodes:
                self.sections.update({node.get_index(): \
                                   self._make_section(node, pprint, factorlambda, dx)})
        self.truemorph = truemorph
        self.dt = dt
        self.VmRec = {}
        if printtopology:
            print h.topology()

    def count_nsec(self):
        k = 0
        for key in self.sections.keys():
            compartment = self.sections[key]
            for seg in compartment:
                k += 1
        return k
    
    def _make_section(self, node, pprint, factorlambda, dx):
        compartment = neuron.h.Section(name = str(node._index))
        cont_imp = node.get_content()['impedance']
        cont_phys = node.get_content()['physiology']
        if node._index != 1:
            compartment.push()
            compartment.diam = 2.*cont_imp.radius*1e4  # convert to um (NEURON takes diam = 2*r)
            compartment.L = cont_imp.length*1e4     # convert to um
            if pprint:
                print 'Current node: ', node
                print 'compartment Length = %.2f um' % compartment.L
                print 'compartment Diam = %.2f um' % compartment.diam          
            
            if dx == None:
                if type(factorlambda) == float: 
                    compartment.nseg = int(((compartment.L/(0.1*h.lambda_f(100))+0.9)/2)*2+1)*int(factorlambda)
                else:
                     compartment.nseg = factorlambda
            else:
                compartment.nseg = int(compartment.L/dx) + 1

            compartment.cm = cont_phys.cm # uF/cm^2
            compartment.Ra = cont_phys.r_a*1e6 # Ohm*cm 
            
            for key in cont_phys.gs.keys():
                if cont_phys.gs[key] > 0.:
                    compartment.insert(mechname[key])
                    for seg in compartment:
                        exec('seg.' + mechname[key] + '.g = ' + str(cont_phys.gs[key]) + '*1e-6') # uS/cm^2
                        exec('seg.' + mechname[key] + '.e = ' + str(cont_phys.es[key]))          # mV 

            for key in cont_phys.conc_mechs.keys():
                compartment.insert(mechname[key])
                for seg in compartment:
                    exec('seg.' + mechname[key] + '.tau = ' + str(cont_phys.conc_mechs[key]['tau']))
                    exec('seg.' + mechname[key] + '.gamma_frac = ' + str(cont_phys.conc_mechs[key]['gamma']))
                    exec('seg.' + mechname[key] + '.inf = ' + str(cont_phys.conc_mechs[key]['inf']))
                    # print 'gamma_orig_NEURON=', eval('seg.' + mechname[key] + '.gamma')
                    # print 'gamma_NEURON=', eval('seg.' + mechname[key] + '.gamma * seg.' + mechname[key] + '.gamma_frac')
                
            h.pop_section()
            compartment.connect(self.sections[morphR.find_next_changenode(node).get_index()],1,0)
        else:
            compartment.push()
            compartment.diam = 2.*cont_imp.radius*1e4  # convert to um (NEURON takes diam=2*r)
            compartment.L = 2.*cont_imp.radius*1e4     # convert to um (to get correct surface)
            if pprint:
                print 'In neuron model'
                print 'soma Radius = %.8f cm' % cont_imp.radius
                print 'soma Length = %.2f um' % compartment.L
                print 'soma Diam = %.2f um' % compartment.diam
            
            compartment.cm = cont_phys.cm
            compartment.Ra = cont_phys.r_a*1e6
            
            for key in cont_phys.gs.keys():
                if cont_phys.gs[key] > 0.:
                    compartment.insert(mechname[key])
                    for seg in compartment:
                        exec('seg.' + mechname[key] + '.g = ' + str(cont_phys.gs[key]) + '*1e-6')    # S/cm^2
                        exec('seg.' + mechname[key] + '.e = ' + str(cont_phys.es[key]))             # mV

            for key in cont_phys.conc_mechs.keys():
                compartment.insert(mechname[key])
                # print mechname[key]
                # print cont_phys.conc_mechs[key]
                for seg in compartment:
                    exec('seg.' + mechname[key] + '.tau = ' + str(cont_phys.conc_mechs[key]['tau']))
                    exec('seg.' + mechname[key] + '.gamma_frac = ' + str(cont_phys.conc_mechs[key]['gamma']))
                    exec('seg.' + mechname[key] + '.inf = ' + str(cont_phys.conc_mechs[key]['inf']))
                    # print 'gamma_orig_NEURON=', eval('seg.' + mechname[key] + '.gamma')
                    # print 'gamma_NEURON=', eval('seg.' + mechname[key] + '.gamma * seg.' + mechname[key] + '.gamma_frac')
                # print seg.conc_Ca.tau

            h.pop_section()
        return compartment
        
    def _make_true_section(self, node, pprint, factorlambda, dx):
        compartment = neuron.h.Section(name=str(node._index)) # NEW NRN SECTION
        cont_phys = node.get_content()['physiology']
        # assume three point soma
        if node.get_index() not in [1,2,3] :
            #~ pPos = node.get_parent_node().get_content()['p3d']
            #~ cPos = node.get_content()['p3d']
            pnode = node.get_parent_node()
            radius, length = morphR.get_cylinder_radius_length([node, pnode]) # radius, length in um
            #~ print node, pnode
            #~ print radius, length
            compartment.push()
            # morphology
            compartment.diam = 2.*radius  # um (NEURON takes diam = 2*r)
            compartment.L = length     # um
            #~ h.pt3dadd(float(pPos.x),float(pPos.y),float(pPos.z),float(pPos.radius))
            #~ h.pt3dadd(float(cPos.x),float(cPos.y),float(cPos.z),float(cPos.radius))
            #~ print 'pp3d Diam: ', pPos.radius
            #~ print 'cp3d Diam: ', cPos.radius
            if pprint:
                print 'Current node: ', node
                print 'compartment Length = %.2f um' % compartment.L
                print 'compartment Diam = %.2f um' % compartment.diam

            if dx == None:
                if type(factorlambda) == float: 
                    compartment.nseg = int(((compartment.L/(0.1*h.lambda_f(100))+0.9)/2)*2+1)*int(factorlambda)
                else:
                     compartment.nseg = factorlambda
            else:
                compartment.nseg = int(compartment.L/dx) + 1

            compartment.cm = cont_phys.cm
            compartment.Ra = cont_phys.r_a*1e6
        
            for key in cont_phys.gs.keys():
                if cont_phys.gs[key] > 0.:
                    compartment.insert(mechname[key])
                    for seg in compartment:
                        exec('seg.' + mechname[key] + '.g = ' + str(cont_phys.gs[key]) + '*1e-6')    # S/cm^2
                        exec('seg.' + mechname[key] + '.e = ' + str(cont_phys.es[key]))             # mV

            for key in cont_phys.conc_mechs.keys():
                compartment.insert(mechname[key])
                for seg in compartment:
                    exec('seg.' + mechname[key] + '.tau = ' + str(cont_phys.conc_mechs[key]['tau']))
                    exec('seg.' + mechname[key] + '.gamma_frac = ' + str(cont_phys.conc_mechs[key]['gamma']))
                    exec('seg.' + mechname[key] + '.inf = ' + str(cont_phys.conc_mechs[key]['inf']))
                    # print 'gamma_orig_NEURON=', eval('seg.' + mechname[key] + '.gamma')
                    # print 'gamma_NEURON=', eval('seg.' + mechname[key] + '.gamma * seg.' + mechname[key] + '.gamma_frac')
            
            h.pop_section()
            compartment.connect(self.sections.get(node.get_parent_node().get_index()),1,0)
        elif node.get_index() == 1 :
            # root of SWC tree = soma
            cPos = node.get_content()['p3d']
            compartment.push()
            compartment.diam = 2.*cPos.radius
            compartment.L= 2.*cPos.radius
            if pprint:
                print 'In neuron model'
                print 'soma Radius = %.8f cm' % r
                print 'soma Length = %.2f um' % compartment.L
                print 'soma Diam = %.2f um' % compartment.diam

            compartment.cm = cont_phys.cm
            compartment.Ra = cont_phys.r_a*1e6
        
            for key in cont_phys.gs.keys():
                if cont_phys.gs[key] > 0.:
                    compartment.insert(mechname[key])
                    for seg in compartment:
                        exec('seg.' + mechname[key] + '.g = ' + str(cont_phys.gs[key]) + '*1e-6')    # S/cm^2
                        exec('seg.' + mechname[key] + '.e = ' + str(cont_phys.es[key]))             # mV

            for key in cont_phys.conc_mechs.keys():
                compartment.insert(mechname[key])
                for seg in compartment:
                    exec('seg.' + mechname[key] + '.tau = ' + str(cont_phys.conc_mechs[key]['tau']))
                    exec('seg.' + mechname[key] + '.gamma_frac = ' + str(cont_phys.conc_mechs[key]['gamma']))
                    exec('seg.' + mechname[key] + '.inf = ' + str(cont_phys.conc_mechs[key]['inf']))
                    # print 'gamma_orig_NEURON=', eval('seg.' + mechname[key] + '.gamma')
                    # print 'gamma_NEURON=', eval('seg.' + mechname[key] + '.gamma * seg.' + mechname[key] + '.gamma_frac')
                
            h.pop_section()
        return compartment
        
    def _map_loc(self, locs):
        if self.truemorph:
            return locs
        else:
            if type(locs) is list:
                newlocs = []
                for loc in locs:
                    n1 = self.greenstree.tree.get_node_with_index(loc['node'])
                    if not morphR.is_changenode(n1):
                        n2 = morphR.find_previous_changenode(n1)[0]
                    else:
                        n2 = n1
                    path = self.greenstree.tree.path_between_nodes(n2, n1)
                    newlocs.append(morphR.get_reduced_loc(loc, path))
                return newlocs
            else:
                loc = locs
                n1 = self.greenstree.tree.get_node_with_index(loc['node'])
                if not morphR.is_changenode(n1):
                    n2 = morphR.find_previous_changenode(n1)[0]
                else:
                    n2 = n1
                path = self.greenstree.tree.path_between_nodes(n2, n1)
                newloc = morphR.get_reduced_loc(loc, path)
                return newloc

    def add_double_exp_current(self, synapseparams):
        synpars = self._map_loc(synapseparams)
        try:
            self.synparams
        except AttributeError:
            self.synparams = {}
        try:
            self.syns
        except AttributeError:
            self.syns = {}
        try:
            self.loclist
        except AttributeError:
            self.loclist = []
        for synparam in synpars:
            node = self.greenstree.tree.get_node_with_index(synparam['node'])
            # save synapse params
            self.synparams[synparam['ID']] = synparam
            # add synapse
            self.syns[synparam['ID']] = h.epsc_double_exp(self.sections[node._index](synparam['x'])) 
            self.syns[synparam['ID']].tau1 = synparam['tau1']
            self.syns[synparam['ID']].tau2 = synparam['tau2']
            self.loclist.append({'node': synparam['node'], 'x': synparam['x'], 'ID': synparam['ID']})

    
    def add_exp_synapses(self, synapseparams):
        synpars = self._map_loc(synapseparams)
        try:
            self.synparams
        except AttributeError:
            self.synparams = {}
        try:
            self.syns
        except AttributeError:
            self.syns = {}
        try:
            self.loclist
        except AttributeError:
            self.loclist = []
        for synparam in synpars:
            node = self.greenstree.tree.get_node_with_index(synparam['node'])
            # save synapse params
            self.synparams[synparam['ID']] = synparam
            # add synapse
            self.syns[synparam['ID']] = h.exp_AMPA_NMDA(self.sections[node._index](synparam['x'])) 
            self.syns[synparam['ID']].tau = synparam['tau']
            self.syns[synparam['ID']].e = synparam['E_r']
            if 'NMDA_ratio' in synparam.keys():
                self.syns[synparam['ID']].NMDA_ratio = synparam['NMDA_ratio']
                self.syns[synparam['ID']].tau_NMDA = synparam['tau_NMDA']
            self.loclist.append({'node': synparam['node'], 'x': synparam['x'], 'ID': synparam['ID']})
                
    def add_double_exp_synapses(self, synapseparams):
        synapseparams = copy.deepcopy(synapseparams)
        synpars = self._map_loc(synapseparams)
        try:
            self.synparams
        except AttributeError:
            self.synparams = {}
        try:
            self.syns
        except AttributeError:
            self.syns = {}
        try:
            self.loclist
        except AttributeError:
            self.loclist = []
        for synparam in synpars:
            node = self.greenstree.tree.get_node_with_index(synparam['node'])
            # save synapse params
            self.synparams[synparam['ID']] = synparam
            # add synapse
            self.syns[synparam['ID']] = h.Exp2Syn(self.sections[node._index](synparam['x'])) 
            self.syns[synparam['ID']].tau1 = synparam['tau1']
            self.syns[synparam['ID']].tau2 = synparam['tau2']
            self.syns[synparam['ID']].e = synparam['E_r']
            self.loclist.append({'node': synparam['node'], 'x': synparam['x'], 'ID': synparam['ID']})
            
    def add_NMDA_synapses_Branco(self, synapseparams):
        synpars = self._map_loc(synapseparams)
        try:
            self.synparams
        except AttributeError:
            self.synparams = {}
        try:
            self.syns
        except AttributeError:
            self.syns = {}
        try:
            self.loclist
        except AttributeError:
            self.loclist = []
        try: 
            self.Branco_syns
        except AttributeError:
            self.Branco_syns = {}
        for synpar in synapseparams:
            if 'tau' in synpar.keys():
                synpar['tau1'] = 2. # ms
                synpar['tau2'] = synpar['tau']
        for synparam in synpars:
            node = self.greenstree.tree.get_node_with_index(synparam['node'])
            # save synapse params
            self.synparams[synparam['ID']] = synparam
            # add synapse
            self.syns[synparam['ID']] = h.exp_AMPA_NMDA(self.sections[node._index](synparam['x']))
            self.syns[synparam['ID']].tau = synparam['tau']
            self.syns[synparam['ID']].tau_NMDA = synparam['tau_NMDA']
            self.syns[synparam['ID']].e = synparam['E_r']
            self.syns[synparam['ID']].NMDA_ratio = 0.
            # stuff to implement Branco synapse
            self.Branco_syns[synparam['ID']] = []
            self.Branco_syns[synparam['ID']].append(h.rel(self.sections[node._index](synparam['x'])))
            self.Branco_syns[synparam['ID']][0].tau = 1.#0.5
            self.Branco_syns[synparam['ID']][0].amp = 1.#2.0
            self.Branco_syns[synparam['ID']].append(h.NMDA_Mg_T(self.sections[node._index](synparam['x'])))
            self.Branco_syns[synparam['ID']][1].gmax = 5. * synparam['NMDA_ratio'] * synparam['weight'] * 1e6
            self.Branco_syns[synparam['ID']][1].Erev = synparam['E_r']
            h.setpointer(self.Branco_syns[synparam['ID']][0]._ref_T, 'C', self.Branco_syns[synparam['ID']][1])
            # append node to loclist
            self.loclist.append({'node': synparam['node'], 'x': synparam['x'], 'ID': synparam['ID']})

    def add_NMDA_synapses(self, synapseparams):
        synpars = self._map_loc(synapseparams)
        try:
            self.synparams
        except AttributeError:
            self.synparams = {}
        try:
            self.syns
        except AttributeError:
            self.syns = {}
        try:
            self.loclist
        except AttributeError:
            self.loclist = []
        for synpar in synapseparams:
            if 'tau' in synpar.keys():
                synpar['tau1'] = 2. # ms
                synpar['tau2'] = synpar['tau']
        for synparam in synpars:
            node = self.greenstree.tree.get_node_with_index(synparam['node'])
            # save synapse params
            self.synparams[synparam['ID']] = synparam
            # add synapse
            self.syns[synparam['ID']] = h.exp_AMPA_NMDA(self.sections[node._index](synparam['x'])) 
            self.syns[synparam['ID']].tau = synparam['tau']
            self.syns[synparam['ID']].tau_NMDA = synparam['tau_NMDA']
            self.syns[synparam['ID']].e = synparam['E_r']
            if 'NMDA_ratio' in synparam.keys():
                self.syns[synparam['ID']].NMDA_ratio = synparam['NMDA_ratio']
            else:
                self.syns[synparam['ID']].NMDA_ratio = 1.7
            self.loclist.append({'node': synparam['node'], 'x': synparam['x'], 'ID': synparam['ID']})
    
    def add_Iclamp(self, Iclamps=None):
        ''' overwrites previous Iclamp!!! '''
        Iclamps = self._map_loc(Iclamps)
        try:
            self.Iclamps
        except AttributeError:
            self.Iclamps = {}
        try:
            self.loclist
        except AttributeError:
            self.loclist = []
        for Iclamp in Iclamps:
            ID = Iclamp['ID']
            self.Iclamps[ID] = h.IClamp(self.sections[Iclamp['node']](Iclamp['x']))
            self.Iclamps[ID].delay = Iclamp['delay'] # ms
            self.Iclamps[ID].dur = Iclamp['dur'] # ms
            self.Iclamps[ID].amp = Iclamp['amp'] # nA
            self.loclist.append({'node': Iclamp['node'], 'x': Iclamp['x'], 'ID': Iclamp['ID']})
        
    def add_recorder(self, locs=[{'node': 1, 'x': 0.5, 'ID':0}]):
        locs = self._map_loc(locs)
        try:
            self.loclist
        except AttributeError:
            self.loclist = []
        for ind, loc in enumerate(locs):
            self.loclist.append(loc)
    
    def set_spiketrains(self, spiketimes):
        self.vecstims = {}
        self.netcons = {}
        try:
            self.Branco_syns
            self.Branco_cons = {}
        except AttributeError: pass
        self.vec = {}
        for spkstms in spiketimes:
            if not (spkstms['ID'] in self.synparams.keys()):
                print 'ID error: spiketimes ID has to be a synapse ID'
                exit(1)
            # add spiketrain
            self.vec[spkstms['ID']] = h.Vector(spkstms['spks'])
            self.vecstims[spkstms['ID']] = h.VecStim()
            self.vecstims[spkstms['ID']].play(self.vec[spkstms['ID']])
            self.netcons[spkstms['ID']] = h.NetCon(self.vecstims[spkstms['ID']], \
                                self.syns[spkstms['ID']], 0, self.dt, self.synparams[spkstms['ID']]['weight'])
            try:
                # for tm in spkstms['spks']:
                #     self.Branco_syns[spkstms['ID']][0].del_rel = tm
                self.Branco_cons[spkstms['ID']] = h.NetCon(self.vecstims[spkstms['ID']], \
                            self.Branco_syns[spkstms['ID']][0], 0, self.dt, self.synparams[spkstms['ID']]['weight'])
            except (AttributeError, KeyError): pass
    
    def run(self, tdur, Vinit=-65., pprint=False, record_concentrations=False, record_from_syns=False, record_from_locs=True):
        # recorders
        VmRec= self.VmRec
        for var in 't', 'vmsoma':
            VmRec[var] = h.Vector()
        VmRec['vmsoma'].record(self.sections[1](0.5)._ref_v)
        VmRec['t'].record(h._ref_t)
        if record_from_syns: 
            VmRec['Im'] = {}
            try:
                self.Branco_syns
                VmRec['Im_Branco'] = {}
            except AttributeError:
                if pprint: print 'no Branco synapses.'
        if record_from_locs:
            try:
                for loc in self.loclist:
                    if loc['ID'] not in VmRec.keys():
                        VmRec[loc['ID']] = h.Vector()
                        VmRec[loc['ID']].record(self.sections[loc['node']](loc['x'])._ref_v)
            except AttributeError:
                if pprint:
                    print 'No recording locs in NEURON-model other than soma'
        if record_from_syns:
            for syn in self.syns.keys():
                VmRec['Im'][syn] = h.Vector()
                VmRec['Im'][syn].record(self.syns[syn]._ref_i)
            try:
                for syn in self.Branco_syns.keys():
                    VmRec['Im_Branco'][syn] = h.Vector()
                    VmRec['Im_Branco'][syn].record(self.Branco_syns[syn][1]._ref_i)
            except AttributeError: 
                if pprint: print 'no Branco synapses.'
        if record_concentrations:
            ions = ['ca']
            VmRec['conc'] = {}
            for ind, loc in enumerate(self.loclist):
                VmRec['conc'][loc['ID']] = {}
                for ion in ions:
                    VmRec['conc'][loc['ID']][ion] = h.Vector()
                    exec('VmRec[\'conc\'][loc[\'ID\']][ion].record(self.sections[loc[\'node\']](loc[\'x\'])._ref_' + ion + 'i)')
        # CaiRecTest = h.Vector()
        # iCaRecTest = h.Vector()
        # CaiRecTest.record(self.sections[1](0.5)._ref_cai)
        # iCaRecTest.record(self.sections[1](0.5)._ref_ica)
        # initialize
        neuron.celsius=37.
        h.finitialize(Vinit)
        h.dt = self.dt
        # simulate
        if pprint: print '>>> Integrating the NEURON model for ' + str(tdur) + ' ms. <<<'
        t0 = posix.times()[0]

        neuron.run(tdur)

        t1 = posix.times()[0]
        if pprint: print ">>> Integration done, took " + str(t1-t0) + " s <<<"
        VmRec['t_exec'] = t1 - t0
        
        # process recordings
        for var in VmRec.iterkeys():
            if var != 'Im' and var != 'Im_Branco' and var != 'conc':
                VmRec[var] = np.array(VmRec[var])
            elif var == 'conc':
                for var2 in VmRec['conc'].keys():
                    for ion in VmRec['conc'][var2].keys():
                        VmRec['conc'][var2][ion] = np.array(VmRec['conc'][var2][ion])
            else:
                for var2 in VmRec['Im'].keys():
                    VmRec['Im'][var2] = np.array(VmRec['Im'][var2])
                for var2 in VmRec['Im_Branco'].keys():
                    VmRec['Im_Branco'][var2] = np.array(VmRec['Im_Branco'][var2])

        return VmRec
########################################################################


## integrates integration point models #################################
class integratorneuron:
    def __init__(self, integrationPoints, synParam, Iclamps, gsdict, esdict, gcalctypedict, concdict=None, E_eq=-65., nonlinear=True):
        '''
        bla
        '''
        self.synapseParam = self._assign_synapse_to_integrationPoint(integrationPoints, synParam)
        self.membraneCurrents = self._assign_memcurrents_to_integrationPoint(integrationPoints, gsdict, esdict, gcalctypedict)
        self.Iclamps = self._assign_Iclamps_to_integrationPoint(integrationPoints, Iclamps)
        if concdict != None: self.concmechs = self._assign_concmechs_to_integrationPoint(integrationPoints, concdict)
        else: self.concmechs = [None for _ in integrationPoints]
        numIntegrationPoints = len(integrationPoints)
        if type(E_eq) == float:
            self.E_eq = E_eq * np.ones((numIntegrationPoints,1))
        else:
            assert len(E_eq) == numIntegrationPoints
            self.E_eq = E_eq[:, np.newaxis]
        self.integrationPoints = [integrationPoint(self.synapseParam[i], self.membraneCurrents[i], \
                        self.Iclamps[i], self.concmechs[i], E0=self.E_eq[i,0], nonlinear=nonlinear) for i in range(numIntegrationPoints)]
        self.synIDs = [syn['ID'] for syn in synParam]#[self.synapseParam[i][j]['ID'] for i in range(len(self.synapseParam)) for j in range(len(self.synapseParam[i]))]
        self.numPoints = numIntegrationPoints
        
    #~ def add_Iclamps(self, inloc, delay=0., dur=0.1, amp=25.):
    
    def _assign_Iclamps_to_integrationPoint(self, integrationPoints, Iclamps, IPradius=0.0001):
        Iclamps_new = [[] for _ in integrationPoints]
        for Iclamp in Iclamps:
            for ind, integ in enumerate(integrationPoints):
                if (Iclamp['node'] == integ['node']) and \
                    (np.abs(Iclamp['x'] - integ['x']) < IPradius):
                        Iclamps_new[ind].append(Iclamp)
        return Iclamps_new
    
    def _assign_synapse_to_integrationPoint(self, integrationPoints, synapseParam, IPradius=0.0001):
        synparams = [[] for _ in integrationPoints]
        for synpar in synapseParam:
            for ind, integ in enumerate(integrationPoints):
                if (synpar['node'] == integ['node']) and \
                    (np.abs(synpar['x'] - integ['x']) < IPradius):
                        synparams[ind].append(synpar)
        return synparams
        
    #~ def _assign_memcurrents_to_integrationPoint(self, integrationPoints, greenstree):
        #~ # TO DO: implement full neuron capability
        #~ phys = greenstree.tree.get_node_with_index(1).get_content()['physiology']
        #~ imp = greenstree.tree.get_node_with_index(1).get_content()['impedance']
        #~ memcurrents = [{} for _ in integrationPoints]
        #~ #print phys.gs
        #~ for key in phys.gs.keys():
            #~ if phys.gs[key] > 0.:
                #~ memcurrents[0][key] = [imp.somaA * phys.gs[key], phys.es[key]]
        #~ #print memcurrents
        #~ return memcurrents
        
    def _assign_memcurrents_to_integrationPoint(self, integrationPoints, gsdict, esdict, gcalctypedict):
        '''
        Assigns memcurrents to integration points
        '''
        memcurrents = [{} for _ in integrationPoints]
        for ind, integ in enumerate(integrationPoints):
            ID = integ['ID']
            for key in gsdict[ID]:
                memcurrents[ind][key] = [gsdict[ID][key], esdict[ID][key], gcalctypedict[ID][key]]
        #~ print memcurrents
        return memcurrents

    def _assign_concmechs_to_integrationPoint(self, integrationPoints, concmechs):
        concmechlist = [{} for _ in integrationPoints]
        for ind, integ in enumerate(integrationPoints):
            ID = integ['ID']
            for key in concmechs[ID]:
                concmechlist[ind][key] = concmechs[ID][key]
        return concmechlist

    def run_volterra_On(self, tmax, dt, spiketimes, mat_dict=None):
        # model matrices
        P1 = mat_dict['P1']
        P2 = mat_dict['P2']
        P3 = mat_dict['P3']
        y2v = mat_dict['y2v']
        v2y = mat_dict['v2y']
        H0 = mat_dict['H0']
        H1 = mat_dict['H1']
        F0 = np.diag(H0)
        H0 = copy.copy(H0) - np.diag(F0)
        # simulation matrices
        timesim = np.arange(0.,tmax,dt)
        V_m = np.zeros((self.numPoints, int(tmax/dt)))  # membrane potentials
        I_m = np.zeros((self.numPoints, int(tmax/dt)))  # total current at integration point
        g_m = np.zeros((self.numPoints, int(tmax/dt)))  # total current at integration point
        c_m = np.zeros((self.numPoints, int(tmax/dt)))  # total current at integration point
        y_sv = {'V': np.zeros(P1['V'].shape), 'I': np.zeros(P1['I'].shape)} # state variables convolution memory terms
        # spikefeeder
        spikef = spikeFeeder(self.synIDs, spiketimes)

        stdout.write('>>> Integrating the Exp sparse GCM for ' + str(tmax) + ' ms. <<<\n')
        stdout.flush()

        for l in range(1,int(tmax/dt)):
            # loop over integration points
            spikeIDs = spikef.advance(dt)
            for m,x in enumerate(self.integrationPoints): 
                x.feedSpikes(spikeIDs)
                x.advance(V_m[m,l-1] + self.E_eq[m,0], dt)
                g_m[m,l], c_m[m,l] = x.get_current_part(V_m[m,l-1] + self.E_eq[m,0])        # main computational loop

            mat_aux = np.identity(self.numPoints) - H0 - np.diag(F0 * g_m[:,l])

            inp = {'V': V_m[v2y['V'], l-1], 'I': I_m[v2y['I'], l-1]} 
            inp_ = {'V': V_m[v2y['V'], l-2], 'I': I_m[v2y['I'], l-2]} 
            for key in y_sv.keys():
                y_sv[key] = P1[key] * y_sv[key] + P2[key] * inp[key] + P3[key] * inp_[key]

            K_vect = np.zeros(self.numPoints) # state variables convolutions
            for key in y2v.keys():
                if key[0] == key[1]:
                    K_vect[key[0]] += H1[key] * I_m[key[1], l-1] + np.sum(y_sv['I'][y2v[key]] * P1['I'][y2v[key]]).real
                else:
                    K_vect[key[0]] += H1[key] * V_m[key[1], l-1] + np.sum(y_sv['V'][y2v[key]] * P1['V'][y2v[key]]).real
            K_vect += F0 * c_m[:,l]

            V_m[:,l] = la.solve(mat_aux, K_vect)
            I_m[:,l] = g_m[:,l] * V_m[:,l] + c_m[:,l]

        V_m += self.E_eq

        return {'Vm': V_m, 'Im': I_m, 't': timesim}

    def run_volterra_On2(self, tmax, dt, spiketimes, mat_dict=None):
        # model matrices
        H_mat = mat_dict['H_mat']
        C_mat = mat_dict['C_mat']
        N_conv = mat_dict['N_conv']
        Kstep = np.max(N_conv)
        F_mat = np.diag(H_mat)
        H_mat = copy.copy(H_mat) - np.diag(F_mat)
        # simulation matrices
        timesim = np.arange(0.,tmax,dt)
        V_m = np.zeros((self.numPoints, int(tmax/dt)+Kstep))  # membrane potentials
        I_m = np.zeros((self.numPoints, int(tmax/dt)+Kstep))  # total current at integration point
        g_m = np.zeros((self.numPoints, int(tmax/dt)+Kstep))  # total current at integration point
        c_m = np.zeros((self.numPoints, int(tmax/dt)+Kstep))  # total current at integration point
        # spikefeeder
        spikef = spikeFeeder(self.synIDs, spiketimes)
        # main computational loop
        stdout.write('>>> Integrating the Quad sparse GCM for ' + str(tmax) + ' ms. <<<\n')
        stdout.flush()

        for k in range(1,int(tmax/dt)):
            l = k + Kstep
            # loop over integration points
            spikeIDs = spikef.advance(dt)
            for m,x in enumerate(self.integrationPoints): 
                x.feedSpikes(spikeIDs)
                x.advance(V_m[m,l-1] + self.E_eq[m,0], dt)
                g_m[m,l], c_m[m,l] = x.get_current_part(V_m[m,l-1] + self.E_eq[m,0])        # main computational loop

            mat_aux = np.identity(self.numPoints) - H_mat - np.diag(F_mat * g_m[:,l])

            K_vect = np.zeros(self.numPoints) # state variables convolutions
            for key in C_mat.keys():
                if key[0] == key[1]:
                    K_vect[key[0]] += np.sum(I_m[key[0], l-N_conv[key]:l] * C_mat[key][::-1])
                else:
                    K_vect[key[0]] += np.sum(V_m[key[1], l-N_conv[key]:l] * C_mat[key][::-1])
            K_vect += F_mat * c_m[:,l]

            V_m[:,l] = la.solve(mat_aux, K_vect)
            I_m[:,l] = g_m[:,l] * V_m[:,l] + c_m[:,l]

        V_m += self.E_eq

        return {'Vm': V_m[:,Kstep:], 'Im': I_m[:,Kstep:], 't': timesim}

    def run_volterra_hybrid(self, tmax, dt, spiketimes, mat_dict=None):
        # model matrices
        P1 = mat_dict['P1']
        P2 = mat_dict['P2']
        P3 = mat_dict['P3']
        P4 = mat_dict['P4']
        y2v = mat_dict['y2v']
        v2y = mat_dict['v2y']
        H0 = mat_dict['H0']
        H1_K = mat_dict['H1_K']
        K = mat_dict['K']
        F0 = np.diag(H0)
        H0 = copy.copy(H0) - np.diag(F0)
        # simulation matrices
        timesim = np.arange(0.,tmax,dt)
        V_m = np.zeros((self.numPoints, int(tmax/dt)+K))  # membrane potentials
        I_m = np.zeros((self.numPoints, int(tmax/dt)+K))  # total current at integration point
        g_m = np.zeros((self.numPoints, int(tmax/dt)+K))  # total current at integration point
        c_m = np.zeros((self.numPoints, int(tmax/dt)+K))  # total current at integration point
        y_sv = {'V': np.zeros(P1['V'].shape), 'I': np.zeros(P1['I'].shape)} # state variables convolution memory terms
        # spikefeeder
        spikef = spikeFeeder(self.synIDs, spiketimes)
        # main computational loop
        stdout.write('>>> Integrating the Mix sparse GCM for ' + str(tmax) + ' ms. <<<\n')
        stdout.flush()

        # main computational loop
        for k in range(1,int(tmax/dt)):
            l = k + K 
            # loop over integration points
            spikeIDs = spikef.advance(dt)
            for m,x in enumerate(self.integrationPoints): 
                x.feedSpikes(spikeIDs)
                x.advance(V_m[m,l-1] + self.E_eq[m,0], dt)
                g_m[m,l], c_m[m,l] = x.get_current_part(V_m[m,l-1] + self.E_eq[m,0])

            mat_aux = np.identity(self.numPoints) - H0 - np.diag(F0 * g_m[:,l])

            inp = {'V': V_m[v2y['V'], l-K], 'I': I_m[v2y['I'], l-K]} 
            inp_ = {'V': V_m[v2y['V'], l-K-1], 'I': I_m[v2y['I'], l-K-1]} 
            for key in y_sv.keys():
                y_sv[key] = P1[key] * y_sv[key] + P2[key] * inp[key] + P3[key] * inp_[key]

            K_vect = np.zeros(self.numPoints) # state variables convolutions
            for key in y2v.keys():
                if key[0] == key[1]:
                    K_vect[key[0]] +=   np.sum(I_m[key[1], l-K:l] * H1_K[key][::-1]) + \
                                        np.sum(y_sv['I'][y2v[key]] * P4['I'][y2v[key]]).real 
                else:
                    K_vect[key[0]] +=   np.sum(V_m[key[1], l-K:l] * H1_K[key][::-1]) + \
                                        np.sum(y_sv['V'][y2v[key]] * P4['V'][y2v[key]]).real
            K_vect += F0 * c_m[:,l]

            V_m[:,l] = la.solve(mat_aux, K_vect)
            I_m[:,l] = g_m[:,l] * V_m[:,l] + c_m[:,l]

        V_m += self.E_eq

        return {'Vm': V_m[:,K:], 'Im': I_m[:,K:], 't': timesim}


class spikeFeeder:
    def __init__(self, synIDs, spiketimes=[0]):
        self.spiketimes, self.synIDs = self._assign_to_synID(spiketimes, synIDs)
        self.k = np.zeros(len(synIDs),dtype=int)    # counter
        self.l = 0  #loop counter
        
    def _assign_to_synID(self, spiketimes, synIDs):
        spkstms = [[] for _ in range(len(synIDs))]
        IDs = [[] for _ in range(len(synIDs))]
        for ind, spks in enumerate(spiketimes):
            IDs[ind] = spks['ID'] 
            spkstms[ind] = spks['spks']
        return spkstms, IDs

    def advance(self, dt=0.025):
        spikeIDs = []
        for j in range(len(self.synIDs)):
            self.count = 0
            while self.k[j] < len(self.spiketimes[j]) and self.l*dt <= self.spiketimes[j][self.k[j]] and self.spiketimes[j][self.k[j]] < (self.l+1)*dt:
                if self.l == 0 or self.l ==1:
                    print 'Problem!'
                    print 'syn ID:', self.synIDs[j]
                    print 'spiketimes: ', self.spiketimes[j]
                self.k[j] += 1
                spikeIDs.append(self.synIDs[j])
                self.count += 1
            if self.count > 1:
                print 'time: ', self.spiketimes[j][self.k[j]-1]
        self.l += 1
        return spikeIDs


class exp2SynCurrent:
    def __init__(self, E0, **kwargs):
        self.weight = kwargs['weight'] # weight [nA]
        self.tau1 = kwargs['tau1'] # rise time [ms]
        self.tau2 = kwargs['tau2'] # decay time [ms]
        self.ID = kwargs['ID']

        self.I1 = 0
        self.I2 = 0

        tp = (self.tau1*self.tau2)/(self.tau2 - self.tau1) * np.log(self.tau2/self.tau1)
        self.factor = 1./(-np.exp(-tp/self.tau1) + np.exp(-tp/self.tau2))

    def feedSpike(self):
        self.I1 += self.weight*self.factor
        self.I2 += self.weight*self.factor

    def advance(self, V, dt):
        self.I1 = np.exp(-dt/self.tau1) * self.I1
        self.I2 = np.exp(-dt/self.tau2) * self.I2

    def getCurrent(self, V):
        return self.I2 - self.I1

    def get_current_part(self):
        return 0., self.I2 - self.I1
        
class expSyn:
    def __init__(self, E0, **kwargs):
        self.weight = kwargs['weight']   # weight [uS]
        self.E_r = kwargs['E_r']      # V_m [mV]
        self.tau = kwargs['tau']     # decay time [ms]
        self.ID = kwargs['ID']
        self.E0 = E0
        self.g = 0              # conductance [nS]
    
    def feedSpike(self):
        self.g += self.weight
    
    def advance(self, V, dt):
        self.g -= dt*self.g/self.tau
    
    def getCurrent(self, V):
        return self.g*(self.E_r - V) 

    def get_current_part(self):
        return -self.g, self.g*(self.E_r-self.E0)

class exp2Syn:
    def __init__(self, E0, **kwargs):
        self.weight = kwargs['weight']   # weight [uS]
        self.E_r = kwargs['E_r']      # V_m [mV]
        self.tau1 = kwargs['tau1']     # rise time [ms]
        self.tau2 = kwargs['tau2']     # decay time [ms]
        self.ID = kwargs['ID']
        self.E0 = E0
        self.g1 = 0              # conductance [nS]
        self.g2 = 0              # conductance [nS]
        tp = (self.tau1*self.tau2)/(self.tau2 - self.tau1) * np.log(self.tau2/self.tau1)
        self.factor = 1./(-np.exp(-tp/self.tau1) + np.exp(-tp/self.tau2))
        self.NMDA = False
        if 'NMDA' in kwargs.keys():
            self.NMDA = True
    
    def feedSpike(self):
        self.g1 += self.weight*self.factor
        self.g2 += self.weight*self.factor
    
    def advance(self, V, dt):
        # self.g1 -= dt*self.g1/self.tau1
        # self.g2 -= dt*self.g2/self.tau2
        self.g1 = np.exp(-dt/self.tau1) * self.g1
        self.g2 = np.exp(-dt/self.tau2) * self.g2
    
    def getCurrent(self, V):
        if self.NMDA:
            gNMDA = self.get_NMDA_conductance(V)
        else:
            gNMDA = 1.
        return (self.g2 - self.g1) * gNMDA * (self.E_r - V)

    def get_current_part(self):
        if self.NMDA:
            gNMDA = self.get_NMDA_conductance(V)
        else:
            gNMDA = 1.
        geff = (self.g2 - self.g1) * gNMDA
        return -geff, geff*(self.E_r-self.E0)
        
    def get_NMDA_conductance(self, V):
        return 1. / (1. + np.exp(-0.062  * V) / 3.57)
        
class Iclampcur:
    def __init__(self, delay, dur, amp):
        self.delay = delay
        self.dur = dur
        self.amp = amp
        self.timer = 0.
        
    def advance(self, V, dt):
        self.timer += dt
        
    def getCurrent(self, V):
        if self.timer > self.delay and self.timer < self.delay + self.dur:
            return self.amp
        else:
            return 0.

    def get_current_part(self):
        curr = self.getCurrent(0.)
        if self.timer > self.delay and self.timer < self.delay + self.dur:
            return 0., self.amp
        else:
            return 0., 0.

class Isincur:
    def __init__(self, delay, dur, amp, freq, phase):
        self.delay = delay
        self.dur = dur
        self.amp = amp
        self.freq = freq
        self.phase = phase
        self.timer = 0.

    def advance(self, V, dt):
        self.timer += dt

    def getCurrent(self, V):
        if self.timer > self.delay and self.timer < self.delay + self.dur:
            return self.amp * np.cos(self.freq*self.timer + self.phase)
        else:
            return 0.

    def get_current_part(self):
        if self.timer > self.delay and self.timer < self.delay + self.dur:
            return 0., self.amp * np.cos(self.freq*self.timer + self.phase)
        else:
            return 0., 0.

        
class integrationPoint:
    def __init__(self, synparams, memcurrents, Iclamps, concmechs, E0=-65., nonlinear=True):
        self.numSyn = range(len(synparams))
        self.syncurrents = []
        self.nonlinear = nonlinear
        if len(synparams):
            for i in range(len(synparams)):
                if 'E_r' not in synparams[i].keys():
                    self.syncurrents.append(exp2SynCurrent(E0, weight=synparams[i]['weight'], tau1=synparams[i]['tau1'], tau2=synparams[i]['tau2'], ID=synparams[i]['ID']))
                elif 'tau' in synparams[i].keys():
                    self.syncurrents.append(expSyn(E0, weight=synparams[i]['weight'], E_r=synparams[i]['E_r'], tau=synparams[i]['tau'], ID=synparams[i]['ID']))
                elif 'tau2' in synparams[i].keys():
                    self.syncurrents.append(exp2Syn(E0, weight=synparams[i]['weight'], E_r=synparams[i]['E_r'], tau1=synparams[i]['tau1'], tau2=synparams[i]['tau2'], ID=synparams[i]['ID']))
        else:
            self.syncurrents = []
        self.memcurrents = {}
        self.ions = []
        for key in memcurrents.keys():
            # if key == 'L':
            #     pass
            if memcurrents[key][0] > 1e-9:
                if memcurrents[key][2] == 'lin':
                    channel = eval('ionc.' + key + '(g=memcurrents[\'' + key + '\'][0], e=memcurrents[\'' + key + '\'][1], V0=' + str(E0) + ', nonlinear=True)')
                elif memcurrents[key][2] == 'pas':
                    channel = eval('ionc.' + key + '(g=memcurrents[\'' + key + '\'][0], e=memcurrents[\'' + key + '\'][1], V0=' + str(E0) + ', nonlinear=False)')
                else:
                    print 'Not a valid calctype'
                if channel.ion in self.memcurrents.keys():
                    self.memcurrents[channel.ion].append(channel)
                else:
                    self.memcurrents[channel.ion] = [channel]
                if len(channel.concentrations) > 0:
                    self.ions.extend(channel.concentrations)
        self.ions = list(set(self.ions))
        self.conc_mech = {}
        if concmechs != None:
            for ion in self.ions:
                gamma=0.05;tau=20.;inf=1e-4
                self.conc_mech[ion] = eval('ionc.conc_' + ion + '(gamma=' + str(concmechs[ion]['gamma']) + ', tau=' + \
                                             str(concmechs[ion]['tau']) + ', inf=' + str(concmechs[ion]['inf']) + \
                                             ', conc0=' + str(concmechs[ion]['conc0']) + ', V0=' + str(E0) + ')')
        self.conc_currents = {}
        self.concentrations = {}
        for ion in self.ions:
            self.conc_currents[ion] = 0.
            self.concentrations[ion] = 0.
        self.synIDs = [x.ID for x in self.syncurrents]
        self.Iclamps = []
        if len(Iclamps):
            for Iclamp in Iclamps:
                if 'freq' in Iclamp.keys():
                    self.Iclamps.append(Isincur(Iclamp['delay'], Iclamp['dur'], Iclamp['amp'], Iclamp['freq'], Iclamp['phase']))
                else:
                    self.Iclamps.append(Iclampcur(Iclamp['delay'], Iclamp['dur'], Iclamp['amp']))

    def advance(self, V, dt):
        for x in self.syncurrents + self.Iclamps: x.advance(V, dt)
        for key in self.conc_mech.keys():
            self.conc_mech[key].advance(self.conc_currents[key], dt)
            self.concentrations[key] = self.conc_mech[key].getConc(self.conc_currents[key])
        for key in self.memcurrents.keys():
            for current in self.memcurrents[key]:
                current.advance(V, dt, self.concentrations)
    
    def getCurrent(self,V):
        curr = np.sum(x.getCurrent(V) for x in self.syncurrents + self.Iclamps)
        for key in self.memcurrents.keys():
            if key in self.ions:
                I = 0.; Ilin = 0.
                for current in self.memcurrents[key]:
                    I += current.get_full_current(V)
                    if self.nonlinear: Ilin += current.get_linear_current(V)
                curr += I - Ilin
                self.conc_currents[key] = I
            else:
                for current in self.memcurrents[key]:
                    curr += current.getCurrent(V)
        return curr

    def get_current_part(self, V):
        if len(self.syncurrents + self.Iclamps) > 0:
            curr = np.sum(np.array([x.get_current_part() for x in self.syncurrents + self.Iclamps]), 0)
        else:
            curr = (0., 0.)
        for key in self.memcurrents.keys():
            if key in self.ions:
                g = 0.; c = 0. 
                glin = 0.; clin = 0.
                for current in self.memcurrents[key]:
                    g1, c1 = current.get_full_current_part()
                    g += g1; c += c1
                    if self.nonlinear: 
                        g1, c1= current.get_linear_current_part()
                        glin += g1; clin += c1
                g -= glin; c -= clin
                curr[0] += g; curr[1] += c
                self.conc_currents[key] = g*V + c
            else:
                for current in self.memcurrents[key]:
                    g1, c1 = current.get_current_part()
                    curr[0] += g1; curr[1] += c1
        return curr

    
    def feedSpikes(self, IDs):
        for i in IDs: 
            if i in self.synIDs: self.syncurrents[self.synIDs.index(i)].feedSpike()

class lightweight_integrator_neuron:
    def __init__(self, numintegrationPoints):
        '''
        bla
        '''
        self.numPoints = numintegrationPoints
        self.E_eq = -65.

    def run(self, tmax, dt, spiketimes_matrix, multiplicity, weight=0.001, tau1=0.2, tau2=5., full=False, prop1=None, prop2=None, **kwargs):
        if full:
            v2y = kwargs['v2y']
            y2v = kwargs['y2v']
        else:
            v_inv_partial = kwargs['v_inv_partial']
            y2v_v_prod = kwargs['y2v_v_prod']
        timesim = np.arange(0.,tmax,dt)
        V_m = np.zeros((self.numPoints, int(tmax/dt)))  # membrane potentials
        I_m = np.zeros((self.numPoints, int(tmax/dt)))  # total current at integration point
        g_s1 = np.zeros(self.numPoints)  # conductances of synapses at integration points
        g_s2 = np.zeros(self.numPoints)  # conductances of synapses at integration points
        prop1_syn1 = np.exp(-dt/tau1)
        prop1_syn2 = np.exp(-dt/tau2)
        tp = (tau1*tau2)/(tau2 - tau1) * np.log(tau2/tau1)
        factor = 1./(-np.exp(-tp/tau1) + np.exp(-tp/tau2))
        # set up auxiliary array for simulation
        ys = np.zeros(prop1.shape, dtype=complex)
        # main computational loop
        if full:
            stdout.write('>>> Integrating the lightweight full GCM for ' + str(tmax) + ' ms. <<<\n')
            stdout.flush()
        else:
            stdout.write('>>> Integrating the lightweight sparse GCM for ' + str(tmax) + ' ms. <<<\n')
            stdout.flush()
        for l in range(1,int(tmax/dt)):
            # loop over integration points
            g_s1 = prop1_syn1 * g_s1
            g_s2 = prop1_syn2 * g_s2
            if spiketimes_matrix[l] != None:
                g_s1[spiketimes_matrix[l]] += multiplicity[l]*factor*weight * np.ones(spiketimes_matrix[l].shape)
                g_s2[spiketimes_matrix[l]] += multiplicity[l]*factor*weight * np.ones(spiketimes_matrix[l].shape)
            I_m[:,l] = (g_s2 - g_s1) * (-(V_m[:,l-1] + self.E_eq))
            # perform integration
            inp = I_m[:,l]
            if full:
                ys = prop1 * ys + prop2 * inp[v2y]
                V_m[:,l] = np.dot(y2v, ys).real
            else:
                ys = prop1 * ys + prop2 * np.dot(v_inv_partial, inp)
                V_m[:,l] = np.dot(y2v_v_prod, ys).real

        V_m += self.E_eq

        return {'Vm': V_m, 'Im': I_m, 't': timesim, 'components': len(prop1)}

    def run_volterra_On(self, tmax, dt, spiketimes_matrix, multiplicity, weight=0.001, tau1=0.2, tau2=5., gs_soma={}, es_soma={}, mat_dict=None):
        # model matrices
        P1 = mat_dict['P1']
        P2 = mat_dict['P2']
        P3 = mat_dict['P3']
        y2v = mat_dict['y2v']
        v2y = mat_dict['v2y']
        H0 = mat_dict['H0']
        H1 = mat_dict['H1']
        F0 = np.diag(H0)
        H0 = copy.copy(H0) - np.diag(F0)
        # simulation matrices
        timesim = np.arange(0.,tmax,dt)
        V_m = np.zeros((self.numPoints, int(tmax/dt)))  # membrane potentials
        I_m = np.zeros((self.numPoints, int(tmax/dt)))  # total current at integration point
        g_s1 = np.zeros(self.numPoints)  # conductances of synapses at integration points
        g_s2 = np.zeros(self.numPoints)  # conductances of synapses at integration points
        prop1_syn1 = np.exp(-dt/tau1)
        prop1_syn2 = np.exp(-dt/tau2)
        tp = (tau1*tau2)/(tau2 - tau1) * np.log(tau2/tau1)
        factor = 1./(-np.exp(-tp/tau1) + np.exp(-tp/tau2))
        y_sv = {'V': np.zeros(P1['V'].shape), 'I': np.zeros(P1['I'].shape)} # state variables convolution memory terms
        # soma channels
        g_chan = np.zeros(self.numPoints)
        c_chan = np.zeros(self.numPoints)
        channels = []
        for key in gs_soma.keys():
            channels.append(eval('ionc.' + key + '(g=gs_soma[\'' + key + '\'], e=es_soma[\'' + key + '\'], V0=' + str(self.E_eq) + ', nonlinear=False)'))
        # main computational loop
        stdout.write('>>> Integrating the sparse exp GCM for ' + str(tmax) + ' ms. <<<\n')
        stdout.flush()

        for l in range(1,int(tmax/dt)):
            # loop over integration points
            g_s1 = prop1_syn1 * g_s1
            g_s2 = prop1_syn2 * g_s2
            if spiketimes_matrix[l] != None:
                g_s1[spiketimes_matrix[l]] += multiplicity[l]*factor*weight * np.ones(spiketimes_matrix[l].shape)
                g_s2[spiketimes_matrix[l]] += multiplicity[l]*factor*weight * np.ones(spiketimes_matrix[l].shape)

            g_syn = -(g_s2 - g_s1)
            c_syn = -(g_s2 - g_s1) * self.E_eq

            for chan in channels: chan.advance(V_m[0,l-1]+self.E_eq, dt)
            if len(channels) > 0:
                g_chan[0], c_chan[0] = np.sum([chan.get_current_part() for chan in channels], 0)
            else:
                g_chan[0] = 0.; c_chan[0] = 0.

            g_m = g_chan + g_syn
            c_m = c_chan + c_syn

            mat_aux = np.identity(self.numPoints) - H0 - np.diag(F0 * g_m)

            inp = {'V': V_m[v2y['V'], l-1], 'I': I_m[v2y['I'], l-1]} 
            inp_ = {'V': V_m[v2y['V'], l-2], 'I': I_m[v2y['I'], l-2]} 
            for key in y_sv.keys():
                y_sv[key] = P1[key] * y_sv[key] + P2[key] * inp[key] + P3[key] * inp_[key]

            K_vect = np.zeros(self.numPoints) # state variables convolutions
            for key in y2v.keys():
                if key[0] == key[1]:
                    K_vect[key[0]] += H1[key] * I_m[key[1], l-1] + np.sum(y_sv['I'][y2v[key]] * P1['I'][y2v[key]]).real
                else:
                    K_vect[key[0]] += H1[key] * V_m[key[1], l-1] + np.sum(y_sv['V'][y2v[key]] * P1['V'][y2v[key]]).real
            K_vect += F0 * c_m

            V_m[:,l] = la.solve(mat_aux, K_vect)
            I_m[:,l] = g_m * V_m[:,l] + c_m

        V_m += self.E_eq

        return {'Vm': V_m, 'Im': I_m, 't': timesim}

    def run_volterra_On2(self, tmax, dt, spiketimes_matrix, multiplicity, weight=0.001, tau1=0.2, tau2=5., gs_soma={}, es_soma={}, mat_dict=None):
        # model matrices
        H_mat = mat_dict['H_mat']
        C_mat = mat_dict['C_mat']
        N_conv = mat_dict['N_conv']
        Kstep = np.max(N_conv)
        F_mat = np.diag(H_mat)
        H_mat = copy.copy(H_mat) - np.diag(F_mat)
        # simulation matrices
        timesim = np.arange(0.,tmax,dt)
        V_m = np.zeros((self.numPoints, int(tmax/dt)+Kstep))  # membrane potentials
        I_m = np.zeros((self.numPoints, int(tmax/dt)+Kstep))  # total current at integration point
        g_s1 = np.zeros(self.numPoints)  # conductances of synapses at integration points
        g_s2 = np.zeros(self.numPoints)  # conductances of synapses at integration points
        prop1_syn1 = np.exp(-dt/tau1)
        prop1_syn2 = np.exp(-dt/tau2)
        tp = (tau1*tau2)/(tau2 - tau1) * np.log(tau2/tau1)
        factor = 1./(-np.exp(-tp/tau1) + np.exp(-tp/tau2))
        # soma channels
        g_chan = np.zeros(self.numPoints)
        c_chan = np.zeros(self.numPoints)
        channels = []
        for key in gs_soma.keys():
            channels.append(eval('ionc.' + key + '(g=gs_soma[\'' + key + '\'], e=es_soma[\'' + key + '\'], V0=' + str(self.E_eq) + ', nonlinear=False)'))
        # main computational loop
        stdout.write('>>> Integrating the sparse quad GCM for ' + str(tmax) + ' ms. <<<\n')
        stdout.flush()

        for k in range(1,int(tmax/dt)):
            l = k + Kstep
            # loop over integration points
            g_s1 = prop1_syn1 * g_s1
            g_s2 = prop1_syn2 * g_s2
            if spiketimes_matrix[k] != None:
                g_s1[spiketimes_matrix[k]] += multiplicity[k]*factor*weight * np.ones(spiketimes_matrix[k].shape)
                g_s2[spiketimes_matrix[k]] += multiplicity[k]*factor*weight * np.ones(spiketimes_matrix[k].shape)

            g_syn = -(g_s2 - g_s1)
            c_syn = -(g_s2 - g_s1) * self.E_eq

            for chan in channels: chan.advance(V_m[0,l-1]+self.E_eq, dt)
            if len(channels) > 0:
                g_chan[0], c_chan[0] = np.sum([chan.get_current_part() for chan in channels], 0)
            else:
                g_chan[0] = 0.; c_chan[0] = 0.

            g_m = g_chan + g_syn
            c_m = c_chan + c_syn

            mat_aux = np.identity(self.numPoints) - H_mat - np.diag(F_mat * g_m)

            K_vect = np.zeros(self.numPoints) # state variables convolutions
            for key in C_mat.keys():
                if key[0] == key[1]:
                    K_vect[key[0]] += np.sum(I_m[key[0], l-N_conv[key]:l] * C_mat[key][::-1])
                else:
                    K_vect[key[0]] += np.sum(V_m[key[1], l-N_conv[key]:l] * C_mat[key][::-1])
            K_vect += F_mat * c_m

            V_m[:,l] = la.solve(mat_aux, K_vect)
            I_m[:,l] = g_m * V_m[:,l] + c_m

        V_m += self.E_eq

        return {'Vm': V_m[:,Kstep:], 'Im': I_m[:,Kstep:], 't': timesim}

    def run_volterra_hybrid(self, tmax, dt, spiketimes_matrix, multiplicity, weight=0.001, tau1=0.2, tau2=5., gs_soma={}, es_soma={}, mat_dict=None):
        # model matrices
        P1 = mat_dict['P1']
        P2 = mat_dict['P2']
        P3 = mat_dict['P3']
        P4 = mat_dict['P4']
        y2v = mat_dict['y2v']
        v2y = mat_dict['v2y']
        H0 = mat_dict['H0']
        H1_K = mat_dict['H1_K']
        K = mat_dict['K']
        F0 = np.diag(H0)
        H0 = copy.copy(H0) - np.diag(F0)
        # simulation matrices
        timesim = np.arange(0.,tmax,dt)
        V_m = np.zeros((self.numPoints, int(tmax/dt)+K))  # membrane potentials
        I_m = np.zeros((self.numPoints, int(tmax/dt)+K))  # total current at integration point
        g_s1 = np.zeros(self.numPoints)  # conductances of synapses at integration points
        g_s2 = np.zeros(self.numPoints)  # conductances of synapses at integration points
        prop1_syn1 = np.exp(-dt/tau1)
        prop1_syn2 = np.exp(-dt/tau2)
        tp = (tau1*tau2)/(tau2 - tau1) * np.log(tau2/tau1)
        factor = 1./(-np.exp(-tp/tau1) + np.exp(-tp/tau2))
        y_sv = {'V': np.zeros(P1['V'].shape), 'I': np.zeros(P1['I'].shape)} # state variables convolution memory terms
        # soma channels
        g_chan = np.zeros(self.numPoints)
        c_chan = np.zeros(self.numPoints)
        channels = []
        for key in gs_soma.keys():
            channels.append(eval('ionc.' + key + '(g=gs_soma[\'' + key + '\'], e=es_soma[\'' + key + '\'], V0=' + str(self.E_eq) + ', nonlinear=False)'))
        # main computational loop
        stdout.write('>>> Integrating the sparse hybrid GCM for ' + str(tmax) + ' ms. <<<\n')
        stdout.flush()

        for k in range(1,int(tmax/dt)):
            l = k + K 
            # loop over integration points
            g_s1 = prop1_syn1 * g_s1
            g_s2 = prop1_syn2 * g_s2
            if spiketimes_matrix[k] != None:
                g_s1[spiketimes_matrix[k]] += multiplicity[k]*factor*weight * np.ones(spiketimes_matrix[k].shape)
                g_s2[spiketimes_matrix[k]] += multiplicity[k]*factor*weight * np.ones(spiketimes_matrix[k].shape)

            g_syn = -(g_s2 - g_s1)
            c_syn = -(g_s2 - g_s1) * self.E_eq

            for chan in channels: chan.advance(V_m[0,l-1]+self.E_eq, dt)
            if len(channels) > 0:
                g_chan[0], c_chan[0] = np.sum([chan.get_current_part() for chan in channels], 0)
            else:
                g_chan[0] = 0.; c_chan[0] = 0.

            g_m = g_chan + g_syn
            c_m = c_chan + c_syn

            mat_aux = np.identity(self.numPoints) - H0 - np.diag(F0 * g_m)

            inp = {'V': V_m[v2y['V'], l-K], 'I': I_m[v2y['I'], l-K]} 
            inp_ = {'V': V_m[v2y['V'], l-K-1], 'I': I_m[v2y['I'], l-K-1]} 
            for key in y_sv.keys():
                y_sv[key] = P1[key] * y_sv[key] + P2[key] * inp[key] + P3[key] * inp_[key]

            K_vect = np.zeros(self.numPoints) # state variables convolutions
            for key in y2v.keys():
                if key[0] == key[1]:
                    K_vect[key[0]] +=   np.sum(I_m[key[1], l-K:l] * H1_K[key][::-1]) + \
                                        np.sum(y_sv['I'][y2v[key]] * P4['I'][y2v[key]]).real 
                else:
                    K_vect[key[0]] +=   np.sum(V_m[key[1], l-K:l] * H1_K[key][::-1]) + \
                                        np.sum(y_sv['V'][y2v[key]] * P4['V'][y2v[key]]).real
            K_vect += F0 * c_m

            V_m[:,l] = la.solve(mat_aux, K_vect)
            I_m[:,l] = g_m * V_m[:,l] + c_m

        V_m += self.E_eq

        return {'Vm': V_m[:,K:], 'Im': I_m[:,K:], 't': timesim}

    
class axon_vectorized():
    def __init__(self, numintegrationPoints, sv_dict, mat_dict, E_eq=-65.):
        self.numPoints = numintegrationPoints
        self.E_eq = E_eq
        # ionchannel matrices
        self.gs = sv_dict['gs']
        self.es = sv_dict['es']
        self.v2x = sv_dict['v2x']
        self.v2c = sv_dict['v2c']
        self.c2i = sv_dict['c2i']
        self.x_sv = sv_dict['x_sv']
        self.pow_sv = sv_dict['pow_sv']
        self.g_sv = sv_dict['g_sv']
        self.prod_sv = sv_dict['prod_sv']
        self.sum_sv = sv_dict['sum_sv']
        self.names_sv = sv_dict['names_sv']
        self.Vlim = sv_dict['Vlim']
        self.dV = sv_dict['dV']
        self.V_range = sv_dict['V_range']
        self.gL = sv_dict['gL']
        self.eL = sv_dict['eL']
        if 'H_mat' in mat_dict.keys():
            # model can be used for volterra O(n^2)
            self.H_mat = mat_dict['H_mat']
            self.C_mat = mat_dict['C_mat']
            self.N_conv = mat_dict['N_conv']
        elif 'P1' in mat_dict.keys():
            # model can be used for volterra O(n)
            self.P1 = mat_dict['P1']
            self.P2 = mat_dict['P2']
            self.P3 = mat_dict['P3']
            self.y2v = mat_dict['y2v']
            self.v2y = mat_dict['v2y']
            self.H0 = mat_dict['H0']
            self.H1 = mat_dict['H1']

    def run_volterra_back_On2(self, tmax, dt, I_in=None):
        H_mat = self.H_mat
        C_mat = self.C_mat
        N_conv = self.N_conv
        # matrices
        F_mat = np.diag(H_mat)
        H_mat = copy.copy(H_mat) - np.diag(F_mat)

        # state variables
        Nstep = int(tmax/dt)
        Kstep = np.max(N_conv)
        timesim = np.linspace(0., tmax, Nstep)
        V_m = np.zeros((self.numPoints, Nstep + Kstep))  # membrane potentials
        I_m = np.zeros((self.numPoints, Nstep + Kstep))  # total current at integration point
        x_sv = copy.copy(self.x_sv)

        if I_in == None: I_in = np.zeros((self.numPoints, Nstep))

        print '>>> Integrating the backwards Volterra axon for ' + str(tmax) + ' ms. <<<'

        for k in range(1, Nstep):
            l = k + Kstep 
            x_sv += dt * self._fun_x(V_m[:,l-1], x_sv, dt=dt)

            g_chan, c_chan = self._fun_I_part(x_sv, I_in=I_in[:,k])

            mat_aux = np.identity(self.numPoints) - H_mat - np.diag(F_mat * g_chan)

            K_vect = np.zeros(self.numPoints) # state variables convolutions
            for key in C_mat.keys():
                if key[0] == key[1]:
                    K_vect[key[0]] += np.sum(I_m[key[0], l-N_conv[key]:l] * C_mat[key][::-1])
                else:
                    K_vect[key[0]] += np.sum(V_m[key[1], l-N_conv[key]:l] * C_mat[key][::-1])
            K_vect += F_mat * c_chan

            V_m[:,l] = la.solve(mat_aux, K_vect)
            I_m[:,l] = g_chan * V_m[:,l] + c_chan

        V_m += self.E_eq

        return {'Vm': V_m[:,Kstep:], 'Im': I_m[:,Kstep:], 't': timesim}

    def run_volterra_back_On(self, tmax, dt, I_in=None):
        P1 = self.P1
        P2 = self.P2
        P3 = self.P3
        y2v = self.y2v
        v2y = self.v2y
        H0 = self.H0
        H1 = self.H1
        # matrices
        F0 = np.diag(H0)
        H0 = copy.copy(H0) - np.diag(F0)

        # state variables
        Nstep = int(tmax/dt)
        timesim = np.linspace(0., tmax, Nstep)
        V_m = np.zeros((self.numPoints, Nstep))  # membrane potentials
        I_m = np.zeros((self.numPoints, Nstep))  # total current at integration point
        y_sv = {'V': np.zeros(P1['V'].shape), 'I': np.zeros(P1['I'].shape)} # state variables convolution memory terms
        x_sv = copy.copy(self.x_sv)

        # input current
        if I_in == None: I_in = np.zeros((self.numPoints, Nstep))

        print '>>> Integrating the backwards Volterra axon for ' + str(tmax) + ' ms. <<<'

        for l in range(1, Nstep):
            x_sv += dt * self._fun_x(V_m[:,l-1], x_sv, dt=dt)

            g_chan, c_chan = self._fun_I_part(x_sv, I_in=I_in[:,l])

            mat_aux = np.identity(self.numPoints) - H0 - np.diag(F0 * g_chan)

            inp = {'V': V_m[v2y['V'], l-1], 'I': I_m[v2y['I'], l-1]} 
            inp_ = {'V': V_m[v2y['V'], l-2], 'I': I_m[v2y['I'], l-2]} 
            for key in y_sv.keys():
                y_sv[key] = P1[key] * y_sv[key] + P2[key] * inp[key] + P3[key] * inp_[key]

            K_vect = np.zeros(self.numPoints) # state variables convolutions
            for key in y2v.keys():
                if key[0] == key[1]:
                    K_vect[key[0]] += H1[key] * I_m[key[1], l-1] + np.sum(y_sv['I'][y2v[key]] * P1['I'][y2v[key]]).real
                else:
                    K_vect[key[0]] += H1[key] * V_m[key[1], l-1] + np.sum(y_sv['V'][y2v[key]] * P1['V'][y2v[key]]).real
            K_vect += F0 * c_chan

            V_m[:,l] = la.solve(mat_aux, K_vect)
            I_m[:,l] = g_chan * V_m[:,l] + c_chan

        V_m += self.E_eq

        return {'Vm': V_m, 'Im': I_m, 't': timesim}

    def _fun_x(self, V, x, dt):
        V = copy.copy(V) + self.E_eq

        Vind = (V - self.Vlim[0]) / self.dV
        Vind_low = np.floor(Vind).astype(int)
        Vind_high = np.ceil(Vind).astype(int)
        
        ind_toohigh = np.where(Vind_high > len(self.V_range))[0]
        Vind_high[ind_toohigh] = len(self.V_range) - 1
        Vind_low[ind_toohigh] = len(self.V_range) - 2        
        ind_toolow = np.where(Vind_low < 0)[0]
        Vind_high[ind_toolow] = 1
        Vind_low[ind_toolow] = 0
        # if len(ind_toohigh) > 0: print 'unstable high'
        # if len(ind_toolow) > 0: print 'unstable low'

        indlist_low = [np.arange(len(x)), np.zeros(len(x), dtype=int), Vind_low[self.v2x]]
        indlist_high = [np.arange(len(x)), np.zeros(len(x), dtype=int), Vind_high[self.v2x]]

        x_inf = self.g_sv[indlist_low] + \
                        (self.g_sv[indlist_high] - self.g_sv[indlist_low]) / self.dV * \
                        (V[self.v2x] - self.V_range[Vind_low[self.v2x]])

        indlist_low[1] = np.ones(len(x), dtype=int)
        indlist_high[1] = np.ones(len(x), dtype=int)

        x_tau = self.g_sv[indlist_low] + \
                        (self.g_sv[indlist_high] - self.g_sv[indlist_low]) / self.dV * \
                        (V[self.v2x] - self.V_range[Vind_low[self.v2x]])

        return ((1. - np.exp(-dt/x_tau)) * (x_inf - x)) / dt

    def _fun_I(self, V, x, I_in=0.):
        V = copy.copy(V) + self.E_eq

        x_ = (x ** self.pow_sv)[self.prod_sv[0]] 
        for key in self.prod_sv.keys():
            if key > 0:
                x_[self.prod_sv[key][:,0]] *= x[self.prod_sv[key][:,1]]
        x__ = copy.copy(x_[self.sum_sv[0]])
        for key in self.sum_sv.keys():
            if key > 0:
                x__[self.sum_sv[key][:,0]] += x_[self.sum_sv[key][:,1]]

        channelcurrents = - self.gs * x__ * (V[self.v2c] - self.es)

        cc_ = copy.copy(channelcurrents[self.c2i[0]])
        for key in self.c2i.keys():
            if key > 0:
                cc_[self.c2i[key][:,0]] += channelcurrents[self.c2i[key][:,1]]

        return cc_ - self.gL * (V - self.eL) + I_in

    def _fun_I_part(self, x, I_in=0.):
        x_ = (x ** self.pow_sv)[self.prod_sv[0]] 
        for key in self.prod_sv.keys():
            if key > 0:
                x_[self.prod_sv[key][:,0]] *= x[self.prod_sv[key][:,1]]
        x__ = copy.copy(x_[self.sum_sv[0]])
        for key in self.sum_sv.keys():
            if key > 0:
                x__[self.sum_sv[key][:,0]] += x_[self.sum_sv[key][:,1]]

        channelconds = - self.gs * x__ 
        channelEs = - channelconds * self.es

        cc_ = copy.copy(channelconds[self.c2i[0]])
        cE_ = copy.copy(channelEs[self.c2i[0]])
        for key in self.c2i.keys():
            if key > 0:
                cc_[self.c2i[key][:,0]] += channelconds[self.c2i[key][:,1]]
                cE_[self.c2i[key][:,0]] += channelEs[self.c2i[key][:,1]]

        return cc_ - self.gL, (cc_ - self.gL) * self.E_eq + cE_ + self.gL * self.eL + I_in

class preprocessor:
    def __init__(self): pass

    def construct_volterra_matrices_On2(self, dt, alphas, gammas, pprint=False):
        # define functions
        fun = lambda t, a, c: np.sum(gammas[key] * np.exp(alphas[key]*t)).real
        funmin = lambda t, a, c: -np.sum(gammas[key] * np.exp(alphas[key]*t)).real
        funder = lambda t, a, c: np.sum(gammas[key] * alphas[key] * np.exp(alphas[key]*t)).real
        funint = lambda t, a, c: np.sum(gammas[key]/alphas[key] * (np.exp( alphas[key]*t) - 1.)).real
        funint_full = lambda a, c: np.sum(- c / a).real
        # matrices to store convolution coefficients
        convolution_coeff = morphR.objectArrayDict(shape=alphas.shape)
        convolution_coeff1 = np.zeros(alphas.shape)
        convolution_length = np.zeros((alphas.shape), dtype=int)
        for key in alphas.keys():
            # find endpoint of convolution
            t0 = 1000.; tp = 500.; condition = False
            while np.abs(tp-t0) > 0.1:
                Deltat = np.abs(tp-t0) / 2.
                t0 = tp
                if condition:
                    tp += Deltat
                else:
                    tp -= Deltat
                condition = np.abs(funint(tp, alphas[key], gammas[key]) - funint_full(alphas[key], gammas[key])) > 1e-8
            # numper of timesteps
            k = int(tp/dt) + 1
            if pprint:
                print 'Kstep: ', k
            # make lists to store stuff
            coeff_list = []
            tau = 20.*np.max(1./np.abs(alphas[key]))
            # compute the coefficients for each convolution step
            for n in range(1,k):
                coeff_list.append( np.sum( \
                            gammas[key] / (alphas[key]**2 * dt) * \
                            (np.exp( alphas[key]*(n+1.)*dt ) - 2. * np.exp( alphas[key]*n*dt ) + np.exp( alphas[key]*(n-1.)*dt) ) ).real )
            # store coefficients and convolution length
            convolution_coeff[key] = np.array(coeff_list)
            convolution_length[key] = k - 1
            # compute first coefficient
            convolution_coeff1[key] = np.sum( \
                                            - gammas[key] / alphas[key] + \
                                            gammas[key] / (alphas[key]**2 * dt) * \
                                            (np.exp( alphas[key]*dt ) - 1.) ).real
            if pprint:
                t = np.linspace(0.,50.,50000)
                EF = funF.ExpFitter()
                kkk = EF.sumExp(t, alphas[key], gammas[key])
                print 'surface estimate: ', (t[1]-t[0])*np.sum(kkk)
                print 'surface exact stored: ', convolution_coeff1[key] + np.sum(convolution_coeff[key])
                print 'surface exact tp: ', funint(tp, alphas[key], gammas[key])
                print 'surface exact infinity: ', funint_full(alphas[key], gammas[key])
        return {'H_mat': convolution_coeff1, 'C_mat': convolution_coeff, 'N_conv': convolution_length}

    def construct_volterra_matrices_On(self, dt, alphas, gammas, pprint=False):
        y2v = morphR.objectArrayDict(shape=alphas.shape)
        v2y_V = []; v2y_I = []
        P1_V = []; P1_I = []
        P2_V = []; P2_I = []
        P3_V = []; P3_I = []
        H0 = np.zeros(alphas.shape)
        H1 = np.zeros(alphas.shape)
        # loop over all kernels
        for key in alphas.keys():
            if pprint: print len(alphas[key])
            # compute coefficients
            # if key[0] == key[1]:
            #     H0[key] = np.sum( gammas[key] / alphas[key] * (np.exp(alphas[key] * dt) - 1.) ).real
            #     H1[key] = np.sum( gammas[key] / alphas[key] * (np.exp(2. * alphas[key] * dt) - np.exp(alphas[key] * dt) ) ).real
            # else:
            H0[key] = np.sum( \
                            - gammas[key] / alphas[key] + \
                            gammas[key] / (alphas[key]**2 * dt) * \
                            (np.exp( alphas[key]*dt ) - 1.) ).real
            H1[key] = np.sum( \
                            gammas[key] / alphas[key] * np.exp(alphas[key] * dt) - \
                            gammas[key] / (alphas[key]**2 * dt) * (np.exp(alphas[key] * dt) - 1.) ).real
            # convolutions memory terms
            if key[0] == key[1]:
                len0 = len(P1_I)
                P1_I.extend( (np.exp(alphas[key]*dt)).tolist() )
                P2_I.extend( ( - gammas[key] / alphas[key] + gammas[key] / (alphas[key]**2 * dt) * \
                                (np.exp(alphas[key]*dt) - 1.) ).tolist() )
                P3_I.extend( ( gammas[key] / alphas[key] * np.exp(alphas[key]*dt) + \
                                - gammas[key] / (alphas[key]**2 * dt) * \
                                (np.exp(alphas[key]*dt) - 1.) ).tolist() )
                len1 = len(P1_I)
                v2y_I.extend([key[1] for i in range(len1-len0)])
            else:
                len0 = len(P1_V)
                P1_V.extend( (np.exp(alphas[key]*dt)).tolist() )
                P2_V.extend( ( - gammas[key] / alphas[key] + gammas[key] / (alphas[key]**2 * dt) * \
                                (np.exp(alphas[key]*dt) - 1.) ).tolist() )
                P3_V.extend( ( gammas[key] / alphas[key] * np.exp(alphas[key]*dt) + \
                               - gammas[key] / (alphas[key]**2 * dt) * \
                                (np.exp(alphas[key]*dt) - 1.) ).tolist() )
                len1 = len(P1_V)
                v2y_V.extend([key[1] for i in range(len1-len0)])
            y2v[key] = np.arange(len0, len1)

        # cast to numpy matrices
        P1 = {'V': np.array(P1_V).real, 'I': np.array(P1_I).real}
        P2 = {'V': np.array(P2_V).real, 'I': np.array(P2_I).real}
        P3 = {'V': np.array(P3_V).real, 'I': np.array(P3_I).real}
        v2y = {'V': np.array(v2y_V), 'I': np.array(v2y_I)}

        return {'P1': P1, 'P2': P2, 'P3': P3, 'y2v': y2v, 'v2y': v2y, 'H0': H0, 'H1': H1}

    def construct_volterra_matrices_hybrid(self, dt, alphas, gammas, K, pprint):
        y2v = morphR.objectArrayDict(shape=alphas.shape)
        v2y_V = []; v2y_I = []
        P1_V = []; P1_I = []
        P2_V = []; P2_I = []
        P3_V = []; P3_I = []
        P4_V = []; P4_I = []
        H0 = np.zeros(alphas.shape)
        H1_K = morphR.objectArrayDict(shape=alphas.shape)
        # loop over all kernels
        for key in alphas.keys():
            # compute coefficients for inversion matrix
            H0[key] = np.sum( \
                            - gammas[key] / alphas[key] + \
                            gammas[key] / (alphas[key]**2 * dt) * \
                            (np.exp( alphas[key]*dt ) - 1.) ).real
            # compute the coefficients for convolution steps
            Hkey_list = []
            for n in range(1,K):
                Hkey_list.append( np.sum( \
                            gammas[key] / (alphas[key]**2 * dt) * \
                            (np.exp( alphas[key]*(n+1.)*dt ) - 2. * np.exp( alphas[key]*n*dt ) + np.exp( alphas[key]*(n-1.)*dt) ) ).real )
            Hkey_list.append( np.sum( \
                            gammas[key] / alphas[key] * np.exp(alphas[key]*K*dt) - \
                            gammas[key] / (alphas[key]**2 * dt) * ( np.exp(alphas[key]*K*dt) - np.exp(alphas[key]*(K-1.)*dt) ) ).real )
            H1_K[key] = np.array(Hkey_list)
            # compute coefficients for memory term
            if key[0] == key[1]:
                len0 = len(P1_I)
                # find the exponentials to include
                inds = np.where( np.logical_or( gammas[key]*np.exp(alphas[key]*K*dt) > 1e-8, \
                                                1./np.abs(alphas[key]) > K*dt/10.) )[0]
                # inds2 = np.arange(len(alphas[key]))
                # print inds, inds2
                # compute memory terms
                P1_I.extend( (np.exp(alphas[key][inds]*dt)).tolist() )
                P2_I.extend( ( - gammas[key][inds] / alphas[key][inds] + gammas[key][inds] / (alphas[key][inds]**2 * dt) * \
                                (np.exp(alphas[key][inds]*dt) - 1.) ).tolist() )
                P3_I.extend( ( gammas[key][inds] / alphas[key][inds] * np.exp(alphas[key][inds]*dt) + \
                                - gammas[key][inds] / (alphas[key][inds]**2 * dt) * \
                                (np.exp(alphas[key][inds]*dt) - 1.) ).tolist() )
                P4_I.extend( (np.exp(alphas[key][inds]*K*dt)).tolist() )
                len1 = len(P1_I)
                v2y_I.extend([key[1] for i in range(len1-len0)])
            else:
                len0 = len(P1_V)
                # find exponentials to compute
                arr = gammas[key]*np.exp(alphas[key]*K*dt)
                inds = np.where( np.logical_or( arr > np.max(np.abs(arr))*1e-10, \
                                                1./np.abs(alphas[key]) > K*dt/10.) )[0]
                # inds2 = np.arange(len(alphas[key]))
                # print inds, inds2
                # compute memory terms
                P1_V.extend( (np.exp(alphas[key][inds]*dt)).tolist() )
                P2_V.extend( ( - gammas[key][inds] / alphas[key][inds] + gammas[key][inds] / (alphas[key][inds]**2 * dt) * \
                                (np.exp(alphas[key][inds]*dt) - 1.) ).tolist() )
                P3_V.extend( ( gammas[key][inds] / alphas[key][inds] * np.exp(alphas[key][inds]*dt) + \
                               - gammas[key][inds] / (alphas[key][inds]**2 * dt) * \
                                (np.exp(alphas[key][inds]*dt) - 1.) ).tolist() )
                P4_V.extend( (np.exp(alphas[key][inds]*K*dt)).tolist() )
                len1 = len(P1_V)
                v2y_V.extend([key[1] for i in range(len1-len0)])
            y2v[key] = np.arange(len0, len1)

        # cast to numpy matrices
        P1 = {'V': np.array(P1_V).real, 'I': np.array(P1_I).real}
        P2 = {'V': np.array(P2_V).real, 'I': np.array(P2_I).real}
        P3 = {'V': np.array(P3_V).real, 'I': np.array(P3_I).real}
        P4 = {'V': np.array(P4_V).real, 'I': np.array(P4_I).real}
        v2y = {'V': np.array(v2y_V), 'I': np.array(v2y_I)}

        return {'P1': P1, 'P2': P2, 'P3': P3, 'P4': P4, 'y2v': y2v, 'v2y': v2y, 'H0': H0, 'H1_K': H1_K, 'K': K}

    def construct_C_model_hybrid(self, dt, inlocs, NNs, alphas, gammas, K, pprint=False):
        N = len(inlocs)
        IDs = np.array([inloc['ID'] for inloc in inlocs])
        # initialize C model
        sgfM = SGFM.sgfModel()
        # create tree structure
        nodes = []
        for i in range(N):
            nodes.append(btstructs.SNode(i))
            nodes[i]._child_nodes = []
            nodes[i]._parent_node = -1
        for i, inloc in enumerate(inlocs):
            inlNN = NNs[i]
            for j, setNN in enumerate(inlNN):
                for inl in setNN:
                    ind = np.where(IDs == inl['ID'])[0][0]
                    nodes[i]._child_nodes.append(ind)
                    nodes[ind]._parent_node = i
        connections = alphas.keys()
        # loop over connections
        nK = 0.
        for c in connections:
            # set parent and child nodes
            # if c[1] > c[0]:
            #     if nodes[c[0]]._child_nodes[0] == -1:
            #         nodes[c[0]]._child_nodes = []
            #     nodes[c[0]]._child_nodes.append(c[1])
            #     nodes[c[1]]._parent_node = c[0]
            # compute connection data
            # compute coefficients for memory term
            # find the exponentials to include and the optimal K at the same time
            inds_list = []
            N_op_list = []
            for k in range(1,30):
                inds = np.where( np.logical_or( gammas[c]*np.exp(alphas[c]*k*dt) > 1e-8, \
                                            1./np.abs(alphas[c]) > k*dt/10.) )[0]
                inds_list.append(inds)
                N_op_list.append(3*len(inds) + k)
            k = np.argmin(N_op_list)
            inds = inds_list[k]
            K = k+1
            # compute memory terms
            P1 = np.exp(alphas[c][inds]*dt)
            P2 = - gammas[c][inds] / alphas[c][inds] + gammas[c][inds] / (alphas[c][inds]**2 * dt) * \
                            (np.exp(alphas[c][inds]*dt) - 1.)
            P3 = gammas[c][inds] / alphas[c][inds] * np.exp(alphas[c][inds]*dt) + \
                            - gammas[c][inds] / (alphas[c][inds]**2 * dt) * \
                            (np.exp(alphas[c][inds]*dt) - 1.)
            P4 = np.exp(alphas[c][inds]*K*dt)
            P1 = self._c2r(P1)
            P2 = self._c2r(P2)
            P3 = self._c2r(P3)
            P4 = self._c2r(P4)
            # compute coefficients for inversion matrix
            H0 = np.sum( - gammas[c] / alphas[c] + \
                            gammas[c] / (alphas[c]**2 * dt) * \
                            (np.exp( alphas[c]*dt ) - 1.) ).real
            # compute the coefficients for convolution steps
            H1_K = []
            for n in range(1,K):
                H1_K.append( np.sum( \
                            gammas[c] / (alphas[c]**2 * dt) * \
                            (np.exp( alphas[c]*(n+1.)*dt ) - 2. * np.exp( alphas[c]*n*dt ) +\
                             np.exp( alphas[c]*(n-1.)*dt) ) ).real )
            val = np.sum( \
                            gammas[c] / alphas[c] * np.exp(alphas[c]*K*dt) - \
                            gammas[c] / (alphas[c]**2 * dt) * ( np.exp(alphas[c]*K*dt) - np.exp(alphas[c]*(K-1.)*dt) ) ).real 
            if val > 1e-6: H1_K.append( val )
            H1_K = np.array(H1_K)
            nK += len(H1_K) + len(inds) 
            # print '>>> connection ', c[0], c[1]
            # print 'K = ', len(H1_K)
            # print 'N_mem = ', len(inds)
            # print 'P1   =', P1
            # print 'P2   =', P2
            # print 'P3   =', P3
            # print 'P4   =', P4
            # print 'H0   =', H0
            # print 'H1_K =', H1_K
            # add connection data
            sgfM.add_connection_data(c[0], c[1],
                                        P1, P2, P3, P4,
                                        H0, H1_K)
        nK /= len(connections)
        # loop over nodes
        for node in nodes:
            if len(node._child_nodes) == 0:
                node._child_nodes = [-1]
            sgfM.add_node(node._index, node._parent_node, np.array(node._child_nodes))
        # set dt
        sgfM.set_dt(dt)

        return sgfM, nK

    def _c2r(self, arr_c):
        return np.concatenate((arr_c.real[:, np.newaxis], arr_c.imag[:, np.newaxis]), 1)

    def construct_ionchannel_matrices(self, inlocs, gs, es, Vlim=[-100., 100.], dV=.98, temp=6.3):
        gs_arr = [] # array containing all ion channels conductances
        es_arr = [] # array containing all ion channel reversals
        sv0_arr = [] # array containing equilibirum opening probabilities
        v2x = [] # index array to convert V into channel state variables
        v2c = [] # index array to convert V into channel currents
        c2i = {0: []} # index array to convert channel currents in inloc currents
        gL = [] # leak conductances
        eL = [] # leak reversals
        names_sv = [] # names of the state variables
        x_sv = [] # state variables
        pow_sv = [] # powers for state variables
        g_sv = [] # state variable functions [var, 0: inf / 1: tau, table]
        prod_sv = {0: []} # index arrays to know which state variables must be multiplied
        sum_sv = {0: []} # index arrays for which state variables to sum
        # V values for function table
        V_range = np.arange(Vlim[0], Vlim[1], dV)
        # loop over input locations
        for i, inloc in enumerate(inlocs):
            # assing local conductances
            gs_point = gs[inloc['ID']]
            es_point = es[inloc['ID']]
            # set first vector of c2i
            c2i[0].append(len(gs_arr))
            # loop over included ion channels
            for j0, key in enumerate(gs_point):
                # add leak
                if key == 'L':
                    gL.append(gs_point[key])
                    eL.append(es_point[key])
                # do not include leak here with other active channels
                elif gs_point[key] > -1.:
                    v2c.append(i)
                    # create ion channel object
                    channel = eval('ionc.' + key + '(g=gs_point[\'' + key + '\'], e=es_point[\'' + key + '\'], calc=True, temp=' + str(temp) + ')')
                    gs_arr.append(gs_point[key])
                    es_arr.append(es_point[key])
                    # set other vectors of c2i
                    if j0 > 0:
                        if j0 in c2i.keys():
                            c2i[j0].append([len(c2i[0])-1, len(gs_arr)-1])
                        else:
                            c2i[j0] = [[len(c2i[0])-1, len(gs_arr)-1]]
                    # append the correct index for the first terms of the sum
                    sum_sv[0].append(len(prod_sv[0]))
                    # create index arrays for multiplication
                    for i0, sv_horizontal in enumerate(channel.varnames):
                        # look for elements to sum
                        if i0 > 0:
                            if i0 in sum_sv.keys():
                                sum_sv[i0].append([len(sum_sv[0])-1, len(prod_sv[0])-1 + i0])
                            else:
                                sum_sv[i0] = [[len(sum_sv[0])-1, len(prod_sv[0])-1 + i0]]
                        # index in names_sv of first state variable product
                        sv_ind = len(names_sv) + i0*len(sv_horizontal)
                        # run over al the state variables in the product
                        for i1, sv in enumerate(sv_horizontal[1:]):
                            # prod_sv[ind > 0] contains [[index of state variable that is to be multiplied, factor of multiplication]]
                            i_aux = i1 + 1
                            if i_aux in prod_sv.keys():
                                prod_sv[i_aux].append([len(prod_sv[0]), sv_ind + i_aux])
                            else:
                                prod_sv[i_aux] = [[len(prod_sv[0]), sv_ind + i_aux]]
                        # append the correct index for the first terms of the product
                        prod_sv[0].append(sv_ind)                    
                    # add state variables to relevant arrays
                    for ind, sv in np.ndenumerate(channel.varnames):
                        v2x.append(i)
                        names_sv.append(str(inloc['ID']) + '_' + key + '_' + sv)
                        x_sv.append(channel.statevar[ind])
                        pow_sv.append(channel.powers[ind])
                        # create function tables
                        fun_inf = sp.lambdify(channel.spV, channel.varinf[ind], 'numpy')
                        fun_tau = sp.lambdify(channel.spV, channel.tau[ind], 'numpy')
                        x_inf = fun_inf(V_range)
                        x_tau = fun_tau(V_range)
                        if type(x_inf) == float: x_inf = x_inf * np.ones(V_range.shape)
                        if type(x_tau) == float: x_tau = x_tau * np.ones(V_range.shape)
                        g_sv.append([x_inf, x_tau])
        # cast to numpy arrays
        gs_arr = np.array(gs_arr); es_arr = np.array(es_arr)
        v2x = np.array(v2x); v2c = np.array(v2c)
        x_sv = np.array(x_sv); pow_sv = np.array(pow_sv); g_sv = np.array(g_sv)
        g_sv_arr = np.zeros((g_sv.shape[0], g_sv.shape[1], len(V_range)), dtype=float)
        for i in range(g_sv.shape[0]):
            for j in range(g_sv.shape[1]):
                g_sv_arr[i,j,:] = g_sv[i,j]
        for key in prod_sv: prod_sv[key] = np.array(prod_sv[key])
        for key in sum_sv: sum_sv[key] = np.array(sum_sv[key])
        for key in c2i: c2i[key] = np.array(c2i[key])
        gL = np.array(gL); eL = np.array(eL)
        sv_dict = {'gs': gs_arr, 'es': es_arr, 'v2x': v2x, 'v2c': v2c, 'c2i': c2i,
                    'x_sv': x_sv, 'pow_sv': pow_sv, 'g_sv': g_sv_arr,
                    'prod_sv': prod_sv, 'sum_sv': sum_sv,
                    'names_sv': names_sv,
                    'Vlim': Vlim, 'V_range': V_range, 'dV': dV,
                    'gL': gL, 'eL': eL}
        return sv_dict


    def construct_lin_nonlin_matrices(self, inlocs, E_eq, gs, es, Vlim=[-100., 100.], dV=.98, temp=6.3):
        ''' Unfinished! '''
        gs_arr = [] # array containing all ion channels conductances
        es_arr = [] # array containing all ion channels reversals
        sv0_arr = [] # array containing equilibirum opening probabilities
        ions = {'ca': {}, 'k': {}, 'na': {}} # dictionary containing ion channel ions list
        v2x = [] # index array to convert V into channel state variables
        v2c = [] # index array to convert V into channel currents
        c2i = {0: []} # index array to convert channel currents in inloc currents
        names_sv = [] # names of the state variables
        x_sv = [] # state variables
        pow_sv = [] # powers for state variables
        prod_sv = {0: []} # index arrays to know which state variables must be multiplied
        sum_sv = {0: []} # index arrays for which state variables to sum
        # V values for function table
        V_range = np.arange(Vlim[0], Vlim[1], dV)
        # loop over input locations
        for i, inloc in enumerate(inlocs):
            # assing local conductances
            gs_point = gs[inloc['ID']]
            es_point = es[inloc['ID']]
            # set first vector of c2i
            c2i[0].append(len(gs_arr))
            # loop over inlcluded ion channels
            for j0, key in enumerate(gs_point.keys()):
                # make ion channel
                channel = eval('ionc.' + key + '(g=gs_point[\'' + key + '\'], e=es_point[\'' + key + '\'], \
                                    V0=E_eq[' + str(i) + '], calc=False, nonlinear=True, temp=' + str(temp) + ')')
                if key != 'L' and gs_point[key] > 1e-12:
                    # channel does not read concentrations
                    if len(channel.concentrations) == 0:
                        # ion channel is added at input location i
                        v2c.append(i)
                        # append local conductances
                        gs_arr.append(gs_point[key])
                        es_arr.append(es_point[key])
                        if channel.ion != '':
                            try:
                                ions[channel.ion][i].append([len(gs_arr)-1, len(x_sv), channel.statevar.size])
                            except KeyError:
                                ion[channel.ion][i] = [[len(gs_arr)-1, len(x_sv), channel.statevar.size]]
                        # append channel opening probability
                        sv0_arr.append(np.sum(channel.factors * np.prod(channel.statevar**channel.powers, 1)[:,None]))
                        # set other vectors of c2i
                        if j0 > 0:
                            if j0 in c2i.keys():
                                c2i[j0].append([len(c2i[0])-1, len(gs_arr)-1])
                            else:
                                c2i[j0] = [[len(c2i[0])-1, len(gs_arr)-1]]
                        # append the correct index for the first terms of the sum
                        sum_sv[0].append(len(prod_sv[0]))
                        # create index arrays for multiplication
                        for i0, sv_horizontal in enumerate(channel.varnames):
                            # look for elements to sum
                            if i0 > 0:
                                if i0 in sum_sv.keys():
                                    sum_sv[i0].append([len(sum_sv[0])-1, len(prod_sv[0])-1 + i0])
                                else:
                                    sum_sv[i0] = [[len(sum_sv[0])-1, len(prod_sv[0])-1 + i0]]
                            # index in names_sv of first state variable product
                            sv_ind = len(names_sv) + i0*len(sv_horizontal)
                            # run over al the state variables in the product
                            for i1, sv in enumerate(sv_horizontal[1:]):
                                # prod_sv[ind > 0] contains [[index of state variable that is to be multiplied, factor of multiplication]]
                                i_aux = i1 + 1
                                if i_aux in prod_sv.keys():
                                    prod_sv[i_aux].append([len(prod_sv[0]), sv_ind + i_aux])
                                else:
                                    prod_sv[i_aux] = [[len(prod_sv[0]), sv_ind + i_aux]]
                            # append the correct index for the first terms of the product
                            prod_sv[0].append(sv_ind)                    
                        # add state variables to relevant arrays
                        for ind, sv in np.ndenumerate(channel.varnames):
                            # state variable is added at input location i
                            v2x.append(i)
                            names_sv.append(str(inloc['ID']) + '_' + key + '_' + sv)
                            x_sv.append(channel.statevar[ind])
                            pow_sv.append(channel.powers[ind])
                            ## create function tables
                            # expressions for the two ionchannel functions
                            expr_1 = channel.varinf[ind] / channel.tau[ind]
                            expr_2 = - 1. / channel.tau[ind]
                            f_1 = expr_1 - expr_2.subs(channel.spV, self.E_eq[i]) - \
                                    (sp.diff(expr_1, channel.spV).subs(channel.spV, self.E_eq[i]) +
                                     sp.diff(expr_2, channel.spV).subs(channel.spV, self.E_eq[i]) * channel.varinf[ind].subs(self.spV, self.E_eq[i])) * \
                                     (self.spV - self.E_eq[i])
                            f_2 = expr_2.subs(self.spV, self.E_eq[i]) - expr_2
                            # evaluate the functions
                            fun_1 = sp.lambdify(channel.spV, f_1, 'numpy')
                            fun_2 = sp.lambdify(channel.spV, f_2, 'numpy')
                            f_tab_1 = fun_1(V_range)
                            f_tab_2 = fun_2(V_range)
                            # store the functions
                            g_sv_nl.append([f_tab_1, f_tab_2])
            # put None index in c2i[0] if there are no ion channels at integration point
            if len(gs_arr) == c2i[0][-1]: c2i[0][-1] = None
        # cast to numpy arrays
        gs_arr = np.array(gs_arr); es_arr = np.array(es_arr)
        v2x = np.array(v2x); v2c = np.array(v2c)
        x_sv = np.array(x_sv); pow_sv = np.array(pow_sv); g_sv = np.array(g_sv)
        g_sv_arr = np.zeros((g_sv.shape[0], g_sv.shape[1], len(V_range)), dtype=float)
        for i in range(g_sv.shape[0]):
            for j in range(g_sv.shape[1]):
                g_sv_arr[i,j,:] = g_sv[i,j]
        for key in prod_sv: prod_sv[key] = np.array(prod_sv[key])
        for key in sum_sv: sum_sv[key] = np.array(sum_sv[key])
        for key in c2i: c2i[key] = np.array(c2i[key])
        for key in ions:
            for key2 in ions[key]:
                ions[key][key2] = np.array(ions[key][key2])
        sv_dict = {'gs': gs_arr, 'es': es_arr, 'v2x': v2x, 'v2c': v2c, 'c2i': c2i,
                    'x_sv': x_sv, 'pow_sv': pow_sv, 'g_sv': g_sv_arr,
                    'prod_sv': prod_sv, 'sum_sv': sum_sv,
                    'names_sv': names_sv,
                    'Vlim': Vlim, 'V_range': V_range, 'dV': dV,
                    'ions': ions}
        return sv_dict


    def construct_current_input_matrix(self, dt, tmax, inlocs, Iclamps):
        Nstep = int(tmax/dt)
        IDs_arr = np.array([inloc['ID'] for inloc in inlocs])

        I_in = np.zeros((len(inlocs), Nstep))

        for Iclamp in Iclamps:
            ind = np.where(IDs_arr == Iclamp['ID'])[0]
            i0 = int(Iclamp['delay'] / dt)
            i1 = int((Iclamp['delay'] + Iclamp['dur']) / dt)
            I_in[ind, i0:i1] = Iclamp['amp'] * np.ones(i1-i0)

        return I_in

    def construct_spiketimes_matrix(self, dt, tmax, spiketimes):
        spikes = [[] for i in range(int(tmax/dt)+1)]
        multiplicity = [[] for i in range(int(tmax/dt)+1)]
        for ind, spktms in enumerate(spiketimes):
            for i, spk in enumerate(spktms['spks']):
                spikes[int(np.ceil(spk/dt))].append(ind)
        for l in range(len(spikes)):
            if  len(spikes[l]) == 0: 
                spikes[l] = None
            else:
                spikes[l] = np.array(spikes[l])
                multiplicity[l] = np.ones(np.unique(spikes[l]).shape)
                for i, spkind in enumerate(np.unique(spikes[l])):
                    inds = np.where(spikes[l]==spkind)[0]
                    multiplicity[l] = len(inds)
        return spikes, multiplicity

    # def do_teststuff(self, *args):
    #         AA = args[0]
    #         print '\nmatrix shape: ', AA.shape
    #         print 'number of elements: ', AA.size
    #         AAmax = np.max(np.abs(AA))
    #         inds = np.where(np.abs(AA) > AAmax*1e-15)[0]
    #         AAapprox = AA + np.dot(AA, AA)
    #         AAapproxmax = np.max(np.abs(AA))
    #         indsapprox = np.where(np.abs(AAapprox) > AAapproxmax*1e-15)[0]
    #         AAexp = la.expm(AA)
    #         AAexpmax = np.max(np.abs(AAexp))
    #         indsexp = np.where(np.abs(AAexp) > AAexpmax*1e-15)[0]
    #         print 'nonzero elements: ' + str(len(inds))
    #         print 'nonzero elements exponential approximation: ' + str(len(indsapprox))
    #         print 'nonzero elements exponential: ' + str(len(indsexp))
    #         ys = np.ones((AA.shape[0], 1))
    #         import scipy.sparse
    #         AA_sp = scipy.sparse.csr_matrix(AA)
    #         AA_diag = np.diag(AA)[:, np.newaxis]
    #         start = posix.times()[0]
    #         for i in range(100): AA_sp.dot(ys)
    #         stop = posix.times()[0]
    #         print 'time sparse multiplication: ' + str((stop-start)/100. * 1e3) + ' us'
    #         start = posix.times()[0]
    #         for i in range(100): AA.dot(ys)
    #         stop = posix.times()[0]
    #         print 'time dense multiplication: ' + str((stop-start)/100. * 1e3) + ' us \n'
    #         start = posix.times()[0]
    #         for i in range(100): AA_diag * ys
    #         stop = posix.times()[0]
    #         print 'time array multiplication: ' + str((stop-start)/100. * 1e3) + ' us \n'

    # def construct_matrices_from_kernels(self, v2y, y2v, alphas, pairs, cutoff, dt, tau_min=None, full=False):
    #     '''
    #     Constructs the necessary matrices to simulate the SGF formalism
    #     '''
    #     if full:
    #         prop1, prop2 = self._construct_simple_propagators(alphas, dt)
    #         magnitudes = None
    #         return alphas, prop1, prop2, v2y, y2v, magnitudes
    #     else:
    #         # set up arrays for simulation
    #         pairs_int = np.concatenate((pairs[0], pairs[1]))
    #         alphas_int = np.concatenate((alphas[0], alphas[1]))
    #         y2v_int = np.concatenate((y2v[0], y2v[1]), axis=1)
    #         v2y_int = np.concatenate((v2y[0], v2y[1]))
    #         AA = np.diag(alphas_int)
    #         AA[0:len(alphas[0]), :] += y2v_int[v2y[0], :]
    #         # do some teststuff
    #         # self.do_teststuff(AA)
    #         #~ # cast AA to real via a similarity transform and transform other matrices accordingly
    #         #~ AA, y2v_int, Q, Q_inv = self._compute_similarity_transform(AA, y2v_int, pairs_int)
    #         # calculate the new eigenvalues
    #         alphas, v_inv_partial, y2v_v_prod, magnitudes = \
    #                 self._compute_eigenvalue_transform(AA, y2v_int, len(v2y[0]), Q_inv=None, cutoff=cutoff, filter_alphas=tau_min)
    #         # detect pairs of complex conjugate eigenvalues
    #         auxarr = np.abs((np.abs(alphas[:-1]) - np.abs(alphas[1:])) / np.abs(alphas[:-1]))
    #         auxarr2 = np.abs(alphas.imag) > 1e-8 # np.abs(alphas.imag) > 1e-15
    #         pairs = np.logical_and(np.concatenate((auxarr < 1e-8, np.zeros(1, dtype=bool))), auxarr2)
    #         # construct additional matrices
    #         v_inv_partial = self._compress_v_inv(v_inv_partial, v2y[1])
    #         prop1, prop2 = self._construct_simple_propagators(alphas, dt)
    #         return alphas, prop1, prop2, v_inv_partial, y2v_v_prod, magnitudes, pairs

    # def construct_matrices_nest_model(self, v2y, y2v, alphas, pairs, cutoff, dt, output_full_matrices=False):
    #     alphas, prop1, prop2, v_inv_partial, y2v_v_prod, magnitudes, pairs = \
    #         self.construct_matrices_from_kernels(v2y, y2v, alphas, pairs, cutoff, dt, full=False)
    #     returndict =  {'alphas': self._cmat_2_rvec(alphas),
    #                     'input_rescale': self._cmat_2_rvec(v_inv_partial),
    #                     'mode_rescale': self._cmat_2_rvec(y2v_v_prod),
    #                     'number_of_modes': len(alphas)}
    #     if output_full_matrices:
    #         returndict['fullmat'] = (alphas, prop1, prop2, v_inv_partial, y2v_v_prod, magnitudes, pairs)
    #     return returndict

    # def _cmat_2_rvec(self, cmat):
    #     rvec = []
    #     if len(cmat.shape) == 1:
    #         for cel in cmat:
    #             rvec.append(cel.real); rvec.append(cel.imag)
    #     elif len(cmat.shape) == 2:
    #         for crow in cmat:
    #             for cel in crow:
    #                 rvec.append(cel.real); rvec.append(cel.imag)
    #     else:
    #         raise Exception('invalid array type')
    #     return np.array(rvec)

    # def _construct_simple_propagators(self, alphas, dt):
    #     prop1 = np.exp(alphas*dt)
    #     prop2 = - (1. - np.exp(alphas*dt)) / alphas
    #     return prop1, prop2

    # def _compute_eigenvalue_transform(self, A, y2v, N, Q_inv=None, cutoff=None, pprint=False, filter_alphas=None):
    #     if Q_inv == None:
    #         Q_inv = np.diag(np.ones(A.shape[0], dtype=complex))
        
    #     # shift-invert method to compute eigenvalues
    #     alphas, v = la.eig(la.inv(A), right=True, left=False)
    #     alphas = 1. / alphas
    #     v_inv = la.inv(v)
    #     v_inv = np.dot(v_inv, Q_inv)
        
    #     # construct matrices to implement transform
    #     v_inv_partial = v_inv[:,N:]
    #     y2v_v_prod = np.dot(y2v, v)

    #     # import matplotlib.pyplot as pl
    #     # pl.semilogy(range(len(alphas)), 1./np.abs(alphas), 'b.')
    #     # pl.axvline(filter_alphas, color='r')
    #     # pl.show()

    #     if filter_alphas != None:
    #         inds = np.where(1. / np.abs(alphas) > filter_alphas)[0]
    #         # i = np.where(inds < filter_alphas)[0]
    #         # inds = inds[i]
    #         alphas = alphas[inds]
    #         v_inv_partial = v_inv_partial[inds, :]
    #         y2v_v_prod = y2v_v_prod[:, inds]
        
    #     if cutoff != None:
    #         # if there is a cutoff we compute the magnitudes
    #         vmag = np.sum(np.abs(v_inv_partial), 1)
    #         magnitudes = np.abs(y2v_v_prod / alphas[None,:])
    #         magnitudes_sum = np.sum(magnitudes, axis=0)
    #         magmag = vmag * magnitudes_sum
    #         maxmag = np.max(magmag)
    #         if type(cutoff) == float:
    #             # if the cutoff is float we use it as the fraction of the maximum
    #             # magnitude below which we truncate the exponentials
    #             inds = np.where(magmag > cutoff*maxmag)[0]
    #             if pprint: print "sparse reduction: ", len(alphas), len(inds)
    #             sortind = np.argsort(magmag[inds])[::-1]
    #             return alphas[inds[sortind]], v_inv_partial[inds[sortind],:], y2v_v_prod[:,inds[sortind]], magmag[inds[sortind]]
    #         elif type(cutoff) == int:
    #             # if cutoff is int we use [cutoff] exponentials with the biggest magnitudes
    #             sortind = np.argsort(magmag)[::-1]
    #             inds = sortind[0:cutoff]
    #             if pprint: print "sparse reduction: ", len(alphas), len(inds)
    #             return alphas[inds], v_inv_partial[inds,:], y2v_v_prod[:,inds], magmag[inds]
    #     else:
    #         vmag = np.sum(np.abs(v_inv_partial), 1)
    #         magnitudes = np.abs(y2v_v_prod / alphas[None,:])
    #         magnitudes_sum = np.sum(magnitudes, axis=0)
    #         magmag = vmag * magnitudes_sum
    #         return alphas, v_inv_partial, y2v_v_prod, magmag
        
    # def _compress_v_inv(self, v_inv, v2y):
    #     maxloc = np.max(v2y)
    #     v_inv_new = np.zeros((v_inv.shape[0], maxloc+1), dtype=complex)
    #     for i in range(maxloc+1):
    #         inds = np.where(v2y == i)[0]
    #         v_inv_new[:,i] = np.sum(v_inv[:,inds], 1)
    #     return v_inv_new

    # def make_mode_model(self, alpha, gammas, pair, dt=0.1, cutoff=None, pprint=False):
    #     v_inp_rescale = gammas.T
    #     factor_arr = 1. / gammas[0]
    #     v_mode_rescale = gammas * np.tile(factor_arr[None,:], (len(gammas), 1))
    #     # look whether delete low-influence nodes
    #     alphas, v_inp_rescale, v_mode_rescale, magnitudes = self._get_magnitudes(alpha,
    #                                 v_inp_rescale, v_mode_rescale, cutoff=cutoff, pprint=pprint)
    #     prop1, prop2 = self._construct_simple_propagators(alpha, dt)
    #     return alpha, prop1, prop2, v_inp_rescale, v_mode_rescale, magnitudes

    # def _get_magnitudes(self, alpha, v_inp_rescale, v_mode_rescale, cutoff=None, pprint=False):
    #     if cutoff != None:
    #         # if there is a cutoff we compute the magnitudes
    #         vmag = np.sum(np.abs(v_inp_rescale), 1)
    #         magnitudes = np.abs(v_mode_rescale / alpha[None,:])
    #         magnitudes_sum = np.sum(v_mode_rescale, axis=0)
    #         magmag = vmag * magnitudes_sum
    #         maxmag = np.max(magmag)
    #         if type(cutoff) == float:
    #             # if the cutoff is float we use it as the fraction of the maximum
    #             # magnitude below which we truncate the exponentials
    #             inds = np.where(magmag > cutoff*maxmag)[0]
    #             if pprint: print "sparse reduction: ", len(alpha), len(inds)
    #             sortind = np.argsort(magmag[inds])[::-1]
    #             return alpha[inds[sortind]], v_inp_rescale[inds[sortind],:], v_mode_rescale[:,inds[sortind]], magmag[inds[sortind]]
    #         elif type(cutoff) == int:
    #             # if cutoff is int we use [cutoff] exponentials with the biggest magnitudes
    #             sortind = np.argsort(magmag)[::-1]
    #             inds = sortind[0:cutoff]
    #             if pprint: print "sparse reduction: ", len(alpha), len(inds)
    #             return alpha[inds], v_inp_rescale[inds,:], v_mode_rescale[:,inds], magmag[inds]
    #     else:
    #         vmag = np.sum(np.abs(v_inp_rescale), 1)
    #         magnitudes = np.abs(v_mode_rescale / alpha[None,:])
    #         magnitudes_sum = np.sum(magnitudes, axis=0)
    #         magmag = vmag * magnitudes_sum
    #         inds = np.argsort(magmag)[::-1]
    #         return alpha, v_inp_rescale, v_mode_rescale, magmag
########################################################################
