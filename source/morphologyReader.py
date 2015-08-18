"""
Author: Willem Wybo
Date: 18/08/2015
Place: BBP, Geneva
"""

import numpy as np
import math
import scipy.optimize
import sympy as sp

from sys import argv, stdout
import os
import time
import copy
import unittest

import btstructs

import ionchannels as ionc
import functionFitter as funF
import neuronModels as neurM

## auxiliary functions #################################################
def tanh(x):
    return (1.-np.exp(-2.*x))/(1.+np.exp(-2.*x))

def tanhtanh(x1,x2):
    return (1. + np.exp(-2.*(x1+x2)) - np.exp(-2.*x1) - np.exp(-2.*x2))/(1. + np.exp(-2.*(x1+x2)) + np.exp(-2.*x1) + np.exp(-2.*x2))

def one_minus_tanhtanh(x1,x2):
    return 2.*(np.exp(-2*x1) + np.exp(-2*x2))/(1. + np.exp(-2.*(x1+x2)) + np.exp(-2.*x1) + np.exp(-2.*x2))
    
# index array
inds = np.array([0])
def get_inds(freqs):
    if inds.shape != (len(freqs), len(freqs)):
        global inds
        inds = np.zeros((len(freqs), len(freqs)), dtype=int)
        for k in range(len(freqs)):
            for l in range(len(freqs)):
                inds[k,l] = k+l
        return inds
    else:
        return inds
########################################################################


## Tools to evaluate impedances given ion channel constitutions ########
def calc_equilibrium_currents(E, gs={'L': 1, 'Na': 0., 'K': 0., 'Klva': 0., 'h': 0.},
            es={'L': -54.3, 'Na': 50., 'K': -77., 'Klva': -106., 'h': -43.}):
    for ind, key in enumerate(gs.keys()): pass


def calc_membraneParams(E, gs={'L': 1, 'Na': 0., 'K': 0., 'Klva': 0., 'h': 0.},
            es={'L': -54.3, 'Na': 50., 'K': -77., 'Klva': -106., 'h': -43.}, 
            gcalctype={'L': 'pas', 'Na': 'pas', 'K': 'pas', 'Klva': 'pas', 'h': 'pas'}):
    g = gs['L']
    Tc = []; Nc = []
    for ind, chan in enumerate(gs.keys()):
        if chan != 'L' and gs[chan] > 1e-12:
            channel = eval('ionc.' + chan + '(g=gs[\'' + chan + '\'], e=es[\'' + chan + '\'], V0=' + str(self.V0) + ', calc=True)')
            if gcalctype[ind] == 'pas':
                g += channel.g0
            elif gcalctype[ind] == 'lin':
                # !!! needs work !!!
                cV, cX = channel.compute_lincoeff()
                cX[0] *= 1e3; cX[1] *= 1e3
                g_soma += channel.g*cV[0]
                for ind, var in np.ndenumerate(channel.varnames):
                    Nvar += 1
                    Tc.append(- channel.g * cV[1][ind] * (self.V0 - channel.e) *  cX[0][ind] / cX[1][ind] )
                    Nc.append(- 1. / (cX[1][ind] * tau_dend)) # !!! Needs to be modified !!!
    return {'g_seg': g}


def calc_membraneImpedance(s, E, gs={'L': 1, 'Na': 0., 'K': 0., 'Klva': 0., 'h': 0.},
            es={'L': -54.3, 'Na': 50., 'K': -77., 'Klva': -106., 'h': -43.}, 
            gcalctype={'L': 'pas', 'Na': 'pas', 'K': 'pas', 'Klva': 'pas', 'h': 'pas'},
            C_m=1.0, **kwargs):
    lincoeff = np.zeros((len(gs), len(s)), dtype=complex)
    for ind, key in enumerate(gs.keys()):
        if gcalctype[key] == 'lin':
            channel = eval('ionc.' + key + '(g=gs[\'' + key + '\'], e=es[\'' + key + '\'], V0=' + str(E) + ', calc=True)')
            # channel.set_expansion_point(E)
            lincoeff[ind,:] = channel.calc_linear(s)['V']
        elif gcalctype[key] == 'pas':
            channel = eval('ionc.' + key + '(g=gs[\'' + key + '\'], e=es[\'' + key + '\'], V0=' + str(E) + ', calc=False)')
            # channel.set_expansion_point(E)
            lincoeff[ind,:] = channel.calc_passive(s)

    if 'radius' in kwargs.keys():
        # then membrane-impedance along cylinder
        z_m = 1./(2.*math.pi*kwargs['radius']*(s*C_m  +  np.sum(lincoeff, 0)))
    elif 'somaA' in kwargs.keys():
        # then somaimpedance
        z_m = 1./(kwargs['somaA']*(s*C_m  + np.sum(lincoeff, 0)))
    
    return z_m
    
def calc_second_order(s, E, gs={'L': 1, 'Na': 0., 'K': 0., 'Klva': 0., 'h': 0.},
                    es={'L': -54.3, 'Na': 50., 'K': -77., 'Klva': -106., 'h': -43.}, **kwargs): 
    quadcoeff = np.zeros((len(gs), len(s), len(s)), dtype=complex)
    for ind, key in enumerate(gs.keys()):
        channel = eval('ionc.' + key + '(g=gs[\'' + key + '\'], e=es[\'' + key + '\'], V0=' + str(E) + ')')
        channel.set_expansion_point(E)
        quadcoeff[ind,:,:] = channel.calc_quadratic(s)['V']
    
    if 'radius' in kwargs.keys():
        # then membrane-impedance along cylinder
        quad = 2.*math.pi*kwargs['radius'] * np.sum(quadcoeff, 0)
    elif 'somaA' in kwargs.keys():
        # then somaimpedance
        quad = kwargs['somaA'] * np.sum(quadcoeff, 0)
        
    return quad
########################################################################


## auxiliary functions #################################################
def distribute_conc_mech(conc_mech, distance2soma, ion, distrtype, params, maxLen=1.):
    if distrtype == 'flat':
        conc_mech[ion] = {'gamma': params[0], 'tau': params[1], 'inf': params[2]}
    return conc_mech

def distribute_channel(gs, es, cm, distance2soma, ionctype, distrtype, params, E, pprint=False, maxLen=1.):
    if distrtype=='flat':
        gs[ionctype] = params[0]
        es[ionctype] = E
    elif distrtype=='exp':
        gs[ionctype] = params[0] * (params[1] + params[2] * np.exp(params[3]*(distance2soma/maxLen - params[4])))
        es[ionctype] = E
    elif distrtype=='lin':
        gs[ionctype] = params[0] * (params[1] + params[2] * distance2soma/maxLen)
        es[ionctype] = E
    elif distrtype=='piecewise':
        if distance2soma > params[2] and distance2soma < params[3]:
            gs[ionctype] = params[1]
        else:
            gs[ionctype] = params[0]
        es[ionctype] = E
    elif distrtype=='fit':
        # preferentially only use for passive current and use only if all
        # other currents have been set
        gsum = 0.
        Ieq = 0.
        for key in set(gs.keys()) - set([ionctype]):
            channel = eval('ionc.' + key + '(g=gs[\'' + key + '\'], e=es[\'' + key + '\'], V0=params[0])')
            # Ichan = channel.getCurrent(params[0])
            # geff = Ichan / (params[0] - es[key])
            Ichan = channel.g0 * (params[0] - channel.e)
            gsum -= channel.g0 # geff
            Ieq -= Ichan
        if cm / (params[1]*1e-3) < gsum:
            # timescale is chosen bigger than physically possible
            tau = cm / (gsum+300.) 
        else:
            tau = (params[1]*1e-3)
        gs[ionctype] = cm / tau - gsum
        es[ionctype] = params[0] + Ieq / gs[ionctype]
        if pprint:
            print 'membrane conductance without leak:', str(gsum), 'uS/cm^2'
            print 'Membrane time constant:', str(cm / (gsum + gs[ionctype])), 's'
    else:
        print 'Error: invalid distribution type'
        exit(1)
    return gs, es
    
def evaluate_membrane_conductance(gs, es, Eq=-65.):
    '''
    Evaluates the conductance density and effective reversal of a set of currents
    
    -input:
        - [gs]: dictionnary with ion channel conductances (uS/cm^2)
        - [es]: dictionnary with ion channel reversals (mV)
        - [Eq]: float, resting potential of neuron (mV)
    
    -output: 
        - [gsum]: float, effective conductance (uS/cm^2)
        - [esum]: float, effective reversal potential of the resulting current (mV)
    '''
    gsum = 0.; esum = 0.;
    for key in gs.keys():
        channel = eval('ionc.' + key + '(g=gs[\'' + key + '\'], e=es[\'' + key + '\'], calc=True)')
        channel.set_expansion_point(Eq)
        channel.set_expansion_coeffs(order=0)
        offset = channel.calc_offset(np.array([0]))[0].real
        # catch 0/0 
        if np.abs(Eq - es[key]) < 1e-9:
            geff = gs[key]
        else:
            geff = offset / (Eq - es[key])
        gsum += geff
        esum += geff * es[key]
    esum /= gsum
    return gsum, esum
    
def evaluate_membrane_constants(gs, es, radius, Eq=-65., r_a=100./1e6, cm=1.0):
    '''
    Evaluate the fundamental membrane time and distance constants.
    
    -input:
        - [gs]: dictionnary with ion channel conductances (uS/cm^2)
        - [es]: dictionnary with ion channel reversals (mV)
        - [radius]: float, radius of segment  (cm)
        - [Eq]: float, resting potential of neuron (mV)
        - [r_a]: float, axial resistance (MOhm*cm)
        - [cm]: membrane capacitance (uF/cm^2)
        
    -output:
        - [lambda_m]: float, electrical length constant (cm)
        - [tau_m]: float, membrane time constant (ms)
    '''
    gsum, _ = evaluate_membrane_conductance(gs, es, Eq=Eq)
    lambda_m = np.sqrt(radius / (2*gsum*r_a)) * 1e4 # convert to um
    tau_m = cm / gsum
    return lambda_m, tau_m*1000.
    
def distance_to_soma(tree, node):
    path = tree.path_to_root(node)
    r, length = get_cylinder_radius_length(path)
    return length
    
def get_cylinder_radius_length(nodes):
    '''
    Returns the total length and average radius of the closest 
    cylindrical approximation of nodes, where nodes is a path of 
    nodes (consecutive!!!)
    
    intype: list of consecutive noders
    rtype: radius, length in micron
    
    '''
    nodes = nodes[::-1] # from lower to higher node
    radii = np.zeros(len(nodes))
    coord = np.zeros((len(nodes), 3))
    for i in range(len(nodes)):
        radii[i] = nodes[i].get_content()['p3d'].radius
        coord[i] = [nodes[i].get_content()['p3d'].x, nodes[i].get_content()['p3d'].y, nodes[i].get_content()['p3d'].z]
    coord = np.array(coord)
    length = 0.
    radius = 0.
    for i in range(1,len(nodes)):
        length += np.linalg.norm(coord[i,:]-coord[i-1,:])
    for i in range(1,len(nodes)):
        if length!=0.:
            if nodes[i-1]._index != 1:
                radius += (radii[i-1]+radii[i])/2. * (np.linalg.norm(coord[i,:]-coord[i-1,:]))/length
            else:
                radius += radii[i] * (np.linalg.norm(coord[i,:]-coord[i-1,:]))/length
        else:
            radius = (radii[i-1]+radii[i])/2.
    #~ print radius, length
    return radius, length   # um
    
def get_true_loc(loc, nodes):
    ''' takes reduced loc and returns full loc 
    nodes: path between loc and next changenode
    '''
    nodes = nodes[::-1]
    if loc['node'] != nodes[-1]._index:
        raise Exception("loc[\'node\'] has to be index of last node in path")
    coord = np.zeros((len(nodes), 3))
    for i in range(len(nodes)):
        coord[i] = [nodes[i].get_content()['p3d'].x, nodes[i].get_content()['p3d'].y, nodes[i].get_content()['p3d'].z]
    coord = np.array(coord)
    length = 0.
    lengths = np.zeros(len(nodes))
    for i in range(1,len(nodes)):
        lengths[i] = np.linalg.norm(coord[i,:]-coord[i-1,:])
        length += lengths[i]
    j=0
    found = False
    while not found:
        if loc['x'] > lengths[j] and loc['x'] <= lengths[j+1]:
            found = True
        j += 1
    loc['node'] = nodes[j]._index
    loc['x'] = (loc['x'] - np.sum(lengths[0:j-1]))/(loc['x'] - np.sum(lengths[0:j]))
    return loc
    
def get_reduced_loc(loc, nodes):
    ''' takes full loc and returns reduced loc
    nodes: path between loc and next changenode
    '''
    loc = copy.deepcopy(loc)
    if loc['node'] != 1:
        nodes = nodes[::-1]
        if loc['node'] != nodes[0]._index:
            raise Exception("loc[\'node\'] has to be index of first node in path")
        coord = np.zeros((len(nodes),3))
        for i in range(len(nodes)):
            coord[i] = [nodes[i].get_content()['p3d'].x, nodes[i].get_content()['p3d'].y, nodes[i].get_content()['p3d'].z]
        coord = np.array(coord)
        nnp = nodes[0].get_parent_node()
        p3d = nnp.get_content()['p3d']
        conp = np.array([p3d.x, p3d.y, p3d.z])
        deltax = (1. - loc['x']) * np.linalg.norm(coord[0,:] - conp)*1e-4 / nodes[-1].get_content()['impedance'].length
        length = 0.
        for i in range(1,len(nodes)):
            length += np.linalg.norm(coord[i,:]-coord[i-1,:])
        length *= 1e-4 # cm
        loc['node'] = nodes[-1]._index
        loc['x'] = 1. - deltax - (length / nodes[-1].get_content()['impedance'].length)
    
    return loc
########################################################################


## Tools for finding changenodes #######################################
def is_changenode(node):
    ''' 
    check if node is changenode
    
    :intype: SNode (btstructs)
    :rtype: boolean
    '''
    if 'changenode' in node.get_content().keys(): return True
    else: return False

def find_next_changenode(node):
    '''
    find parent changenode of node
    
    :intype: SNode (btstructs) 
    :rtype: SNode (btstructs) 
    '''
    if node._parent_node == None:
        return None # node is soma, so no children
    else:
        found = False
        while not found:
            node = node.get_parent_node()
            if node._parent_node != None:
                if is_changenode(node):
                    found = True
            else:
                found = True # this node is soma
        return node

def find_previous_changenode(node):
    '''
    find children changenodes of node
    
    :intype: SNode (btstructs) 
    :rtype: list of SNode's (btstructs) 
    '''
    childnodes = node.get_child_nodes()
    if len(childnodes) == 0:
        return []
    elif len(childnodes) == 1:
        node = childnodes
        while not is_changenode(node[0]):
            node = node[0].get_child_nodes()
        return node
    elif len(childnodes) == 2:
        node1 = childnodes[0]
        while not is_changenode(node1):
            node1 = node1.get_child_nodes()[0]
        node2 = childnodes[1]
        while not is_changenode(node2):
            node2 = node2.get_child_nodes()[0]
        return [node1, node2]
    elif len(childnodes) > 2 and node._index !=1 :
        raise Exception('Error: node can have at max 2 children!')
    else:
        childnodes = childnodes[2:]
        cnodes = []
        for cnode in childnodes:
            xnode = cnode
            while not is_changenode(xnode):
                xnode = xnode.get_child_nodes()[0]
            cnodes.append(xnode)
        return cnodes
                
def find_other_changenode(node, child):
    '''
    find childnode of node that is not child
    
    :intype: parent SNode (btstructs), child SNode (btstructs) 
    :rtype: SNode (btstructs) 
    '''
    cnodes = find_previous_changenode(node)
    if len(cnodes) != 2:
        return None
    else:
        cnodes = list(set(cnodes) - set([child])) # element in cnodes that is not node
        return cnodes[0]
    
def path_between_changenodes(from_node, to_node):
    '''
    return shortest path between two nodes
    
    intype: from_node, to_node SNode (btstructs)
    rtype: list of SNode's (btstructs)
    '''
    path1 = path_to_soma_cn(from_node)
    path2 = path_to_soma_cn(to_node)
    meeting_point = max([node._index for node in path1 + path2 if (node in path1) and (node in path2)])
    path1 = [x for x in path1 if x._index >= meeting_point]
    path2 = [x for x in path2 if x._index > meeting_point]
    return path1 + path2[::-1]
    
def path_to_soma_cn(node):
    path = [node]
    while node._index != 1:
        node = find_next_changenode(node)
        path.append(node)
    return path
    
def path_between_nodes(from_node, to_node):
    '''
    return shortest path between two nodes
    
    intype: from_node, to_node SNode (btstructs)
    rtype: list of SNode's (btstructs)
    '''
    path1 = path_to_soma(from_node)
    path2 = path_to_soma(to_node)
    meeting_point = max([node._index for node in path1 + path2 if (node in path1) and (node in path2)])
    path1 = [x for x in path1 if x._index >= meeting_point]
    path2 = [x for x in path2 if x._index > meeting_point]
    return path1 + path2[::-1]
    
def path_to_soma(node):
    path = [node]
    while node._index != 1:
        node = node.get_parent_node()
        path.append(node)
    return path
########################################################################


## tools for generating different trees ################################
def get_equivalent_greenstrees(morphfile, distr_sim=None, distr_calc=None, 
        s_distr_sim=None, s_distr_calc=None, pprint=True):
    ''' 
    return 3 types of greenstrees: 1 for neuron simulation, 1 for transfer
    function calculation and 1 for reduced order simulations. For now, Eq is 
    still hard-coded at -65. mV
    '''
    
    if distr_sim == None:
        distr_sim = {'Na': {'type': 'flat', 'param': [0.00*1e6], 'E': 50.},
                'K': {'type': 'flat', 'param': [0.000*1e6], 'E':-77.},
                'Klva': {'type': 'flat', 'param': [0.00*1e6], 'E': -106.},
                'h': {'type': 'exp', 'param': [0.000*1e6, 0.000*1e6, 1000.], 'E': -43.},
                'L': {'type': 'fit', 'param': [-65., 15.], 'E': -65.}}
    if s_distr_sim == None:
        s_distr_sim = {'Na': {'type': 'flat', 'param': [0.0*1e6], 'E': 50.},
            'K': {'type': 'flat', 'param': [0.00*1e6], 'E':-77.},
            'Klva': {'type': 'flat', 'param': [0.00*1e6], 'E': -106.},
            'h': {'type': 'flat', 'param': [0.00*1e6], 'E': -43.},
            'L': {'type': 'fit', 'param': [-65., 15.], 'E': -65.}}
    if distr_calc == None:
        distr_calc = {'Na': {'type': 'flat', 'param': [0.00*1e6], 'E': 50.},
            'K': {'type': 'flat', 'param': [0.000*1e6], 'E':-77.},
            'Klva': {'type': 'flat', 'param': [0.00*1e6], 'E': -106.},
            'h': {'type': 'exp', 'param': [0.000*1e6, 0.000*1e6, 1000.], 'E': -43.},
            'L': {'type': 'fit', 'param': [-65., 15.], 'E': -65.}}
    if s_distr_calc == None:
        s_distr_calc = {'Na': {'type': 'flat', 'param': [0.0*1e6], 'E': 50.},
            'K': {'type': 'flat', 'param': [0.0*1e6], 'E':-77.},
            'Klva': {'type': 'flat', 'param': [0.00*1e6], 'E': -106.},
            'h': {'type': 'flat', 'param': [0.00*1e6], 'E': -43.},
            'L': {'type': 'fit', 'param': [0.0,15], 'E': -65.}}

    # membranecurrents (!! only leak can have 'type':'fit' !!)
    distr_point = copy.deepcopy(distr_sim)
    s_distr_point = copy.deepcopy(s_distr_sim)
    for key in distr_sim.keys():
        if distr_sim[key]['type'] != 'fit':
            if distr_sim[key]['param'][0] > 0. and distr_calc[key]['param'][0] == 0.:
                pass
            elif distr_sim[key]['param'][0] == distr_calc[key]['param'][0]:
                distr_point[key]['param'][0] = 0.
        else:
            distr_point[key]['type'] = 'flat'
            distr_point[key]['param'] = [0.]
        if s_distr_sim[key]['type'] != 'fit':
            if s_distr_sim[key]['param'][0] > 0. and s_distr_calc[key]['param'][0] == 0.:
                pass
            elif s_distr_sim[key]['param'][0] == s_distr_calc[key]['param'][0]:
                s_distr_point[key]['param'][0] = 0.
        else:
            s_distr_point[key]['type'] = 'flat'
            s_distr_point[key]['param'] = [0.]
    s_distr_point['C'] = {'type': 'flat', 'param': [0.], 'E': [0.]} # parameters need to be set
    distr_point['C'] = {'type': 'flat', 'param': [0.], 'E': [0.]} # parameters need to be set
    #~ print s_distr_point
    #~ print s_distr_sim
    
    # NEURON greens tree
    greenstree_NEURON = greensTree(morphfile, soma_distr=s_distr_sim, 
                                    ionc_distr=distr_sim, pprint=False)
    greenstree_NEURON.set_changenodes()
    greenstree_NEURON.set_impedance(np.array([0j]))
    #~ greenstree_NEURON.set_length_and_radii()

    # GF calculation greens tree
    greenstree_calc = greensTree(morphfile, soma_distr=s_distr_calc, 
                                    ionc_distr=distr_calc, pprint=False)
    greenstree_calc.set_changenodes()

    # point simulation greenstree
    greenstree_point = greensTree(morphfile, soma_distr=s_distr_point, 
                                    ionc_distr=distr_point, pprint=False)
    greenstree_point.set_changenodes()

    # bit of a hack to set correct conductances, but there seems to be no way around it
    nodes_sim = greenstree_NEURON.tree.get_nodes()
    changenodes_sim = [node for node in nodes_sim if is_changenode(node)]
    for node_sim in changenodes_sim:
        phys_sim = node_sim.get_content()['physiology']
        phys_point = greenstree_point.tree.get_node_with_index(node_sim._index).get_content()['physiology']
        phys_calc = greenstree_calc.tree.get_node_with_index(node_sim._index).get_content()['physiology']
        if pprint: print 'gs_sim: ', phys_sim.gs; print 'es_sim: ', phys_sim.es
        if pprint: print 'gs_calc: ', phys_calc.gs; print 'es_calc: ', phys_calc.es
        phys_calc.gs['L'] = phys_sim.gs['L']
        phys_calc.es['L'] = phys_sim.es['L']
        # set constant current that compensates for equilibrium point not being
        # where the linearized currents cancel
        g, E = evaluate_membrane_conductance(phys_calc.gs, phys_calc.es, Eq=-65.)
        phys_point.gs['C'] = g
        phys_point.es['C'] = E
        if pprint: print 'gs_point: ', phys_point.gs; print 'es_point: ', phys_point.es
        
    greenstree_point.set_impedance(np.array([0j]))
    #~ greenstree_point.set_length_and_radii()
    
    return greenstree_NEURON, greenstree_point, greenstree_calc


def make_axon_trees(morphfile, soma_chan=None, ais_chan=None, nor_chan=None, nodes_of_ranvier=[], ais_node_index=None, V0=-65, pprint=False):
    # standard parameters (Moore et al., 1978)
    cm = 1.0 # uF/cm^2
    cm_myelin = 0.005 # uF/cm^2
    r_a = 100.*1e-6 # MOhm*cm
    gL_myelin = 1.5 # uS/cm^2
    if soma_chan == None:
        soma_chan = {'gs': {'L': 0.0003*1e6, 'Na': .120*1e6, 'K': 0.036*1e6},
                     'es': {'L': -54.3, 'Na': 50., 'K': -77.}}
    if ais_chan == None:
        ais_chan = {'gs': {'L': 0.003*1e6, 'Na': 1.20*1e6, 'K': 0.36*1e6},
                     'es': {'L': -54.3, 'Na': 50., 'K': -77.}}
    if nor_chan == None:
        nor_chan = {'gs': {'L': 0.003*1e6, 'Na': 1.20*1e6, 'K': 0.36*1e6},
                     'es': {'L': -54.3, 'Na': 50., 'K': -77.}}
    if ais_node_index == None:
        ais_node_index = 4
    # dummy channel distribution 
    dummy_distr = {'Na': {'type': 'flat', 'param': [0.00*1e6], 'E': 50., 'calctype': 'pas'},
                'K': {'type': 'flat', 'param': [0.000*1e6], 'E':-77., 'calctype': 'pas'},
                'L': {'type': 'fit', 'param': [-65., 50.], 'E': -65., 'calctype': 'pas'}}
    # dummy soma channel distribution 
    dummy_soma_distr = {'Na': {'type': 'flat', 'param': [0.0*1e6], 'E': 50., 'calctype': 'pas'},
                'K': {'type': 'flat', 'param': [0.0*1e6], 'E':-77., 'calctype': 'pas'},
                'L': {'type': 'fit', 'param': [-65., 50.], 'E': -65., 'calctype': 'pas'}}
    # initialize greenstree's
    greenstree = greensTree(morphfile, ionc_distr=dummy_distr, soma_distr=dummy_soma_distr, cnodesdistr='all', axon=True, pprint=pprint)
    # set axon parameters
    nodes = greenstree.tree.get_nodes(somanodes=False)
    ais_node = greenstree.tree.get_node_with_index(ais_node_index)
    nodes_of_ranvier = [n for n in nodes if n._index in nodes_of_ranvier]
    myelin_nodes = list(set(nodes) - set(nodes_of_ranvier) - set(nodes[0:1]) - set([ais_node]))
    # make second greenstree to only contains the passive properties
    greenstree_pas = copy.deepcopy(greenstree)
    # set somanode parameters
    phys = nodes[0].get_content()['physiology']
    phys.cm = cm
    phys.r_a = r_a
    phys.gs = copy.deepcopy(soma_chan['gs'])
    phys.es = copy.deepcopy(soma_chan['es'])
    phys.gs, phys.es = distribute_channel(phys.gs, phys.es, phys.cm, 0., 'L', 'fit', [V0, 100.], -65., pprint=False)
    # set in other greenstree
    node2 = greenstree_pas.tree.get_node_with_index(nodes[0]._index)
    phys = node2.get_content()['physiology']
    phys.gs = {}; phys.es = {}
    phys.cm = cm
    phys.r_a = r_a
    for key in copy.deepcopy(soma_chan['gs'].keys()): phys.gs[key] = 0.
    phys.es = copy.deepcopy(soma_chan['es'])
    # set ais parameters
    phys = ais_node.get_content()['physiology']
    phys.cm = cm
    phys.r_a = r_a
    phys.gs = copy.deepcopy(ais_chan['gs'])
    phys.es = copy.deepcopy(ais_chan['es'])
    phys.gs, phys.es = distribute_channel(phys.gs, phys.es, phys.cm, 0., 'L', 'fit', [V0, 100.], -65., pprint=False)
    # set in other greenstree
    node2 = greenstree_pas.tree.get_node_with_index(ais_node_index)
    phys = node2.get_content()['physiology']
    phys.gs = {}; phys.es = {}
    phys.cm = cm
    phys.r_a = r_a
    for key in copy.deepcopy(ais_chan['gs'].keys()): phys.gs[key] = 0.
    phys.es = ais_chan['es']
    # nodes of Ranvier
    for node in nodes_of_ranvier:
        phys = node.get_content()['physiology']
        phys.cm = cm
        phys.r_a = r_a
        phys.gs = copy.deepcopy(nor_chan['gs'])
        phys.es = copy.deepcopy(nor_chan['es'])
        phys.gs, phys.es = distribute_channel(phys.gs, phys.es, phys.cm, 0., 'L', 'fit', [V0, 100.], -65., pprint=False)
        # set other greenstree
        node2 = greenstree_pas.tree.get_node_with_index(node._index)
        phys = node2.get_content()['physiology']
        phys.gs = {}; phys.es = {}
        phys.cm = cm
        phys.r_a = r_a
        for key in copy.deepcopy(nor_chan['gs'].keys()): phys.gs[key] = 0.
        phys.es = copy.deepcopy(nor_chan['es'])
    # myelinated nodes
    for node in myelin_nodes:
        phys = node.get_content()['physiology']
        phys.gs = {}; phys.es = {}
        phys.cm = cm_myelin
        phys.r_a = r_a
        for key in nor_chan['gs'].keys(): phys.gs[key] = 0.
        phys.gs['L'] = gL_myelin
        phys.es = copy.deepcopy(nor_chan['es'])
        phys.es['L'] = V0
        # set other greenstree
        node2 = greenstree_pas.tree.get_node_with_index(node._index)
        phys = node2.get_content()['physiology']
        phys.gs = {}; phys.es = {}
        phys.cm = cm_myelin
        phys.r_a = r_a
        for key in copy.deepcopy(nor_chan['gs'].keys()): phys.gs[key] = 0.
        phys.gs['L'] = gL_myelin
        phys.es = copy.deepcopy(nor_chan['es'])
        phys.es['L'] = V0

    return greenstree, greenstree_pas


def get_axon_node_conductances(greenstree, nodes_of_ranvier, ais_node):
    inlocs = []; gs = {}; es = {}
    # soma
    somanode = greenstree.tree.get_node_with_index(1)
    inlocs.append({'node': somanode._index, 'x': 0.5, 'ID': 0})
    phys = somanode.get_content()['physiology']
    imp = somanode.get_content()['impedance']
    gs[inlocs[-1]['ID']] = {}; es[inlocs[-1]['ID']] = {}
    for key in phys.gs.keys():
        gs[inlocs[-1]['ID']][key] = phys.gs[key] * imp.somaA
        es[inlocs[-1]['ID']][key] = phys.es[key]
    # other nodes
    if ais_node == None:    node_inds = nodes_of_ranvier
    else:                   node_inds = [ais_node] + nodes_of_ranvier
    active_nodes = [greenstree.tree.get_node_with_index(ind) for ind in node_inds]
    for ind, node in enumerate(active_nodes):
            inlocs.append({'node': node._index, 'x': 0.5, 'ID': ind+1})
            phys = node.get_content()['physiology']
            imp = node.get_content()['impedance']
            gs[inlocs[-1]['ID']] = {}; es[inlocs[-1]['ID']] = {}
            for key in phys.gs.keys():
                gs[inlocs[-1]['ID']][key] = phys.gs[key] * 2.*np.pi*imp.radius * imp.length
                es[inlocs[-1]['ID']][key] = phys.es[key]

    return gs, es
########################################################################
        

## auxiliary classes ###################################################
class sparseArrayDict(dict):
    def __init__(self, dtype=complex, shape=None, el_shape=None, *args, **kwargs):
        # initialize dict
        dict.__init__(self, *args, **kwargs)
        self.dtype = dtype
        # check if no key is bigger than shape, or set shape if it is not
        # given
        keys = super(sparseArrayDict, self).keys()
        if shape == None:
            if keys:
                item = super(sparseArrayDict, self).__getitem__(keys[0])
                if type(item) is not np.ndarray:
                    raise TypeError('Elements need to be numpy arrays')
                self.el_shape = item.shape
                if type(keys[0]) is int:
                    self.dim = 1
                    self.shape = max(keys) + 1
                elif type(keys[0]) is tuple:
                    self.dim = len(keys[0])
                    self.shape = [0 for _ in range(self.dim)]
                    for key in keys:
                        if len(key) != self.dim:
                            raise IndexError('Index dimension error')
                        for ind, x in enumerate(key):
                            if x >= self.shape[ind]:
                                self.shape[ind] = x + 1
                else:
                    raise ValueError('Invalid index type')
            else:
                raise Exception('No shape specified')
        else:
            self.shape = shape
            self.dim = len(shape)
            if el_shape == None:
                raise Exception('No element shape specified')
            if type(el_shape) is not tuple:
                if type(el_shape) is int:
                    el_shape = (el_shape,)
                else:
                    raise TypeError('Element shape not tuple or int')
            self.el_shape = el_shape
            
    def __setitem__(self, key, value):
        # checks if key and array shape are correct
        if type(key) is int:
            if key >= self.shape or key < 0:
                raise IndexError('Index out of bounds')
        elif type(key) is tuple:
            if len(key) != self.dim:
                raise IndexError('Invalid index dimension')
            for ind, x in enumerate(key):
                if x >= self.shape[ind] or x < 0:
                    raise IndexError('Index out of bounds')
        else:
            raise ValueError('Invalid index type')
        if type(value) is not np.ndarray:
            raise TypeError('Value needs to be numpy.ndarray')
        if value.shape != self.el_shape:
            raise TypeError('Array does not have same dimensions as other elements')
        # performs function
        super(sparseArrayDict, self).__setitem__(key, value)
        
    def setelement2zero(self, key):
        # checks if key is valid
        if type(key) is int:
            if key >= self.shape or key < 0:
                raise IndexError('Index out of bounds')
        elif type(key) is tuple:
            if len(key) != self.dim:
                raise IndexError('Invalid index dimension')
            for ind, x in enumerate(key):
                if x >= self.shape[ind] or x < 0:
                    raise IndexError('Index out of bounds')
        else:
            raise ValueError('Invalid index type')
        value = np.zeros(self.el_shape, dtype=self.dtype)
        super(sparseArrayDict, self).__setitem__(key, value)
        
    def __getitem__(self, key):
        # checks if key and array shape are correct
        if type(key) is int:
            if key >= self.shape or key < 0:
                raise IndexError('Index out of bounds')
        elif type(key) is tuple:
            if len(key) != self.dim:
                raise IndexError('Invalid index dimension')
            for ind, x in enumerate(key):
                if x >= self.shape[ind] or x < 0:
                    raise IndexError('Index out of bounds')
        else:
            raise ValueError('Invalid index type')
        # perform function
        keys = super(sparseArrayDict, self).keys()
        if key in keys:
            return super(sparseArrayDict, self).__getitem__(key)
        else:
            if type(key) is int:
                if key >= self.shape[0] or key < 0:
                    raise IndexError('Index out of bounds')
            else:
                for ind, x in enumerate(key):
                    if x >= self.shape[ind] or x < 0:
                        raise IndexError('Index out of bounds')
            super(sparseArrayDict, self).__setitem__(key, np.zeros(self.el_shape, dtype=self.dtype))
            return super(sparseArrayDict, self).__getitem__(key)
            
            
class objectArrayDict(dict):
    def __init__(self, shape=None, *args, **kwargs):
        # initialize dict
        dict.__init__(self, *args, **kwargs)
        # check if no key is bigger than shape, or set shape if it is not
        # given
        keys = super(objectArrayDict, self).keys()
        if shape == None:
            if keys:
                item = super(sparseArrayDict, self).__getitem__(keys[0])
                if type(item) is not np.ndarray:
                    raise TypeError('Elements need to be numpy arrays')
                #~ self.el_shape = item.shape
                if type(keys[0]) is int:
                    self.dim = 1
                    self.shape = max(keys) + 1
                elif type(keys[0]) is tuple:
                    self.dim = len(keys[0])
                    self.shape = [0 for _ in range(self.dim)]
                    for key in keys:
                        if len(key) != self.dim:
                            raise IndexError('Index dimension error')
                        for ind, x in enumerate(key):
                            if x >= self.shape[ind]:
                                self.shape[ind] = x + 1
                else:
                    raise ValueError('Invalid index type')
            else:
                raise Exception('No shape specified')
        else:
            self.shape = shape
            self.dim = len(shape)
            #~ if el_shape == None:
                #~ raise Exception('No element shape specified')
            #~ if type(el_shape) is not tuple:
                #~ if type(el_shape) is int:
                    #~ el_shape = (el_shape,)
                #~ else:
                    #~ raise TypeError('Element shape not tuple or int')
            #~ self.el_shape = el_shape
            
    def __setitem__(self, key, value):
        # checks if key and array shape are correct
        if type(key) is int:
            if key >= self.shape or key < 0:
                raise IndexError('Index out of bounds')
        elif type(key) is tuple:
            if len(key) != self.dim:
                raise IndexError('Invalid index dimension')
            for ind, x in enumerate(key):
                if x >= self.shape[ind] or x < 0:
                    raise IndexError('Index out of bounds')
        else:
            raise ValueError('Invalid index type')
        if type(value) is not np.ndarray:
            raise TypeError('Value needs to be numpy.ndarray')
        #~ if value.shape != self.el_shape:
            #~ raise TypeError('Array does not have same dimensions as other elements')
        # performs function
        super(objectArrayDict, self).__setitem__(key, value)
        
    def __getitem__(self, key):
        # checks if key and array shape are correct
        if type(key) is int:
            if key >= self.shape or key < 0:
                raise IndexError('Index out of bounds')
        elif type(key) is tuple:
            if len(key) != self.dim:
                raise IndexError('Invalid index dimension')
            for ind, x in enumerate(key):
                if x >= self.shape[ind] or x < 0:
                    raise IndexError('Index out of bounds')
        else:
            raise ValueError('Invalid index type')
        # perform function
        keys = super(objectArrayDict, self).keys()
        if key in keys:
            return super(objectArrayDict, self).__getitem__(key)
        else:
            if type(key) is int:
                if key >= self.shape[0] or key < 0:
                    raise IndexError('Index out of bounds')
            else:
                for ind, x in enumerate(key):
                    if x >= self.shape[ind] or x < 0:
                        raise IndexError('Index out of bounds')
            return None
            
        
class dummyImpedances:
    def __init__(self, radius, length):
        self.radius = radius/1e4    # cm
        self.length = length/1e4    # cm
        
class dummySomaImpedances:
    def __init__(self, node1, node2, node3):
        r = abs(node2.get_content()['p3d'].radius) /1e4 # convert radius to cm
        self.radius = r
        self.somaA = 4.0*np.pi*r*r # in cm^2

class segmentGeometries:
    def __init__(self, radius, length, distance2soma):
        self.radius = radius/1e4    # cm
        self.length = length/1e4    # cm
        self.distance2soma = distance2soma/1e4    # cm

class segmentImpedances:
    def __init__(self, freqs, radius, length, pprint=False):
        self.radius = radius/1e4    # cm
        self.length = length/1e4    # cm
        self.freqs = freqs # Hz
        self.order = 1  #flag variable
        self.quadint = False #flag variable
        if pprint:
            print '    length: %.4f cm' % self.length
            print '    radius: %.4f cm' % self.radius
    
    def set_impedances(self, E0, gs, es, gcalctype, C_m, r_a):
        self.z_m = calc_membraneImpedance(self.freqs, E0, gs, es, gcalctype, C_m, radius=self.radius)
        self.z_a = r_a/(math.pi*self.radius*self.radius)
        self.gamma = np.sqrt(self.z_a / self.z_m)
        self.z_c = self.z_a/self.gamma
    
    def set_extended_impedances(self, E0, gs, es, gcalctype, C_m, r_a):
        self.order = 2
        # necessary frequency array
        freq_ext = 1j * np.arange(2.*self.freqs[0].imag, 2.*self.freqs[-1].imag + 2*(self.freqs[1].imag-self.freqs[0].imag), self.freqs[1].imag-self.freqs[0].imag)
        # extended impedances
        self.z_m_ext = calc_membraneImpedance(freq_ext, E0, gs, es, gcalctype, C_m, radius=self.radius)
        self.gamma_ext = np.sqrt(self.z_a / self.z_m_ext)
        self.z_c_ext = self.z_a / self.gamma_ext
    
    def set_quadratic_coeff(self, E0, gs, es):
        self.quadratic_coeff = calc_second_order(self.freqs, E0, gs, es, radius=self.radius)
        
    # def set_quadratic_integrals(self):
    #     self.quadint = True
    #     # compute index array
    #     inds = get_inds(self.freqs)
    #     if np.max(np.abs(self.quadratic_coeff)) > 0.:
    #         # extended gammas
    #         gammas = [self.gamma[:, np.newaxis] * np.ones(inds.shape, dtype=complex),
    #                     self.gamma[np.newaxis, :] * np.ones(inds.shape, dtype=complex),
    #                     self.gamma_ext[inds]]
    #         z_0s = [self.z_0[:, np.newaxis] * np.ones(inds.shape, dtype=complex),
    #                     self.z_0[np.newaxis, :] * np.ones(inds.shape, dtype=complex),
    #                     self.z_0_ext[inds]]
    #         z_1s = [self.z_1[:, np.newaxis] * np.ones(inds.shape, dtype=complex),
    #                     self.z_1[np.newaxis, :] * np.ones(inds.shape, dtype=complex),
    #                     self.z_1_ext[inds]]
    #         z_cs = [self.z_c[:, np.newaxis] * np.ones(inds.shape, dtype=complex),
    #                     self.z_c[np.newaxis, :] * np.ones(inds.shape, dtype=complex),
    #                     self.z_c_ext[inds]]
    #         # integrals
    #         print 'evaluating integrals'
    #         if self.z_1[0] == np.infty:
    #             self.integrals = {}
    #             self.integrals['i1i2o--'] = fundict_inf['xi <= xj <= xo'](gammas[0], z_0s[0], z_1s[0], z_cs[0], gammas[1], z_0s[1], z_1s[1], z_cs[1], gammas[2], z_0s[2], z_1s[2], z_cs[2], 0., 0., 0., self.length)
    #             print 1
    #             self.integrals['i1i2--o'] = fundict_inf['xi <= xj <= xo'](gammas[0], z_0s[0], z_1s[0], z_cs[0], gammas[1], z_0s[1], z_1s[1], z_cs[1], gammas[2], z_0s[2], z_1s[2], z_cs[2], 0., 0., self.length, self.length)
    #             print 1
    #             self.integrals['i1--i2o'] = fundict_inf['xi <= xj <= xo'](gammas[0], z_0s[0], z_1s[0], z_cs[0], gammas[1], z_0s[1], z_1s[1], z_cs[1], gammas[2], z_0s[2], z_1s[2], z_cs[2], 0., self.length, self.length, self.length)
    #             print 1
    #             self.integrals['i2o--i1'] = fundict_inf['xj <= xo <= xi'](gammas[0], z_0s[0], z_1s[0], z_cs[0], gammas[1], z_0s[1], z_1s[1], z_cs[1], gammas[2], z_0s[2], z_1s[2], z_cs[2], self.length, 0., 0., self.length)
    #             print 1
    #             self.integrals['i2--i1o'] = fundict_inf['xj <= xo <= xi'](gammas[0], z_0s[0], z_1s[0], z_cs[0], gammas[1], z_0s[1], z_1s[1], z_cs[1], gammas[2], z_0s[2], z_1s[2], z_cs[2], self.length, 0., self.length, self.length)
    #             print 1
    #             self.integrals['o--i1i2'] = fundict_inf['xo <= xi <= xj'](gammas[0], z_0s[0], z_1s[0], z_cs[0], gammas[1], z_0s[1], z_1s[1], z_cs[1], gammas[2], z_0s[2], z_1s[2], z_cs[2], self.length, self.length, 0., self.length)
    #             print 1
    #             self.integrals['--i1i2o'] = fundict_inf['xo <= xi <= xj'](gammas[0], z_0s[0], z_1s[0], z_cs[0], gammas[1], z_0s[1], z_1s[1], z_cs[1], gammas[2], z_0s[2], z_1s[2], z_cs[2], self.length, self.length, self.length, self.length)
    #             print 1
    #             self.integrals['i1o--i2'] = fundict_inf['xo <= xi <= xj'](gammas[0], z_0s[0], z_1s[0], z_cs[0], gammas[1], z_0s[1], z_1s[1], z_cs[1], gammas[2], z_0s[2], z_1s[2], z_cs[2], 0., self.length, 0., self.length)
    #         else:
    #             self.integrals = {}
    #             self.integrals['i1i2o--'] = fundict['xi <= xj <= xo'](gammas[0], z_0s[0], z_1s[0], z_cs[0], gammas[1], z_0s[1], z_1s[1], z_cs[1], gammas[2], z_0s[2], z_1s[2], z_cs[2], 0., 0., 0., self.length)
    #             print 1
    #             self.integrals['i1i2--o'] = fundict['xi <= xj <= xo'](gammas[0], z_0s[0], z_1s[0], z_cs[0], gammas[1], z_0s[1], z_1s[1], z_cs[1], gammas[2], z_0s[2], z_1s[2], z_cs[2], 0., 0., self.length, self.length)
    #             print 1
    #             self.integrals['i1--i2o'] = fundict['xi <= xj <= xo'](gammas[0], z_0s[0], z_1s[0], z_cs[0], gammas[1], z_0s[1], z_1s[1], z_cs[1], gammas[2], z_0s[2], z_1s[2], z_cs[2], 0., self.length, self.length, self.length)
    #             print 1
    #             self.integrals['i2o--i1'] = fundict['xj <= xo <= xi'](gammas[0], z_0s[0], z_1s[0], z_cs[0], gammas[1], z_0s[1], z_1s[1], z_cs[1], gammas[2], z_0s[2], z_1s[2], z_cs[2], self.length, 0., 0., self.length)
    #             print 1
    #             self.integrals['i2--i1o'] = fundict['xj <= xo <= xi'](gammas[0], z_0s[0], z_1s[0], z_cs[0], gammas[1], z_0s[1], z_1s[1], z_cs[1], gammas[2], z_0s[2], z_1s[2], z_cs[2], self.length, 0., self.length, self.length)
    #             print 1
    #             self.integrals['o--i1i2'] = fundict['xo <= xi <= xj'](gammas[0], z_0s[0], z_1s[0], z_cs[0], gammas[1], z_0s[1], z_1s[1], z_cs[1], gammas[2], z_0s[2], z_1s[2], z_cs[2], self.length, self.length, 0., self.length)
    #             print 1
    #             self.integrals['--i1i2o'] = fundict['xo <= xi <= xj'](gammas[0], z_0s[0], z_1s[0], z_cs[0], gammas[1], z_0s[1], z_1s[1], z_cs[1], gammas[2], z_0s[2], z_1s[2], z_cs[2], self.length, self.length, self.length, self.length)
    #             print 1
    #             self.integrals['i1o--i2'] = fundict['xo <= xi <= xj'](gammas[0], z_0s[0], z_1s[0], z_cs[0], gammas[1], z_0s[1], z_1s[1], z_cs[1], gammas[2], z_0s[2], z_1s[2], z_cs[2], 0., self.length, 0., self.length)
    #     else:
    #         self.integrals['i1i2o--'] = np.zeros(inds.shape, dtype=complex)
    #         self.integrals['i1i2--o'] = np.zeros(inds.shape, dtype=complex)
    #         self.integrals['i1--i2o'] = np.zeros(inds.shape, dtype=complex)
    #         self.integrals['i2o--i1'] = np.zeros(inds.shape, dtype=complex)
    #         self.integrals['i2--i1o'] = np.zeros(inds.shape, dtype=complex)
    #         self.integrals['o--i1i2'] = np.zeros(inds.shape, dtype=complex)
    #         self.integrals['--i1i2o'] = np.zeros(inds.shape, dtype=complex)
    #         self.integrals['i1o--i2'] = np.zeros(inds.shape, dtype=complex)
    #     print 'Done evaluating integrals'
        
    # def calc_integral(self, inloc1, inloc2, outloc): 
    #     inds = get_inds(self.freqs)
    #     if np.max(np.abs(self.quadratic_coeff)) > 0.:
    #         # extended gammas
    #         gammas = [self.gamma[:, np.newaxis] * np.ones(inds.shape, dtype=complex),
    #                     self.gamma[np.newaxis, :] * np.ones(inds.shape, dtype=complex),
    #                     self.gamma_ext[inds]]
    #         z_0s = [self.z_0[:, np.newaxis] * np.ones(inds.shape, dtype=complex),
    #                     self.z_0[np.newaxis, :] * np.ones(inds.shape, dtype=complex),
    #                     self.z_0_ext[inds]]
    #         if self.z_1[0] == np.infty:
    #             z_1s = [self.z_1[:, np.newaxis] * np.ones(inds.shape, dtype=complex),
    #                     self.z_1[np.newaxis, :] * np.ones(inds.shape, dtype=complex),
    #                     self.z_1_ext[inds]]
    #         else:
    #             z_1s = [self.z_1[:, np.newaxis],
    #                         self.z_1[np.newaxis, :],
    #                         self.z_1_ext[inds]]
    #         z_cs = [self.z_c[:, np.newaxis] * np.ones(inds.shape, dtype=complex),
    #                     self.z_c[np.newaxis, :] * np.ones(inds.shape, dtype=complex),
    #                     self.z_c_ext[inds]]
    #         i1 = inloc1['x']; i2 = inloc2['x']; o = outloc['x']
    #         x1 = self.length * inloc1['x']; x2 = self.length * inloc2['x']; xo = self.length * outloc['x']
    #         if self.quadint and i1==0. and i2==0. and o==0.:
    #             return self.integrals['i1i2o--']
    #         elif self.quadint and i1==0. and i2==0. and o==1.:
    #             return self.integrals['i1i2--o']
    #         elif self.quadint and i1==0. and i2==1. and o==1.:
    #             return self.integrals['i1--i2o']
    #         elif self.quadint and i1==1. and i2==0. and o==0.:
    #             return self.integrals['i2o--i1']
    #         elif self.quadint and i1==1. and i2==0. and o==1.:
    #             return self.integrals['i2--i1o']
    #         elif self.quadint and i1==1. and i2==1. and o==0.:
    #             return self.integrals['o--i1i2']
    #         elif self.quadint and i1==1. and i2==1. and o==1.:
    #             return self.integrals['--i1i2o']
    #         elif self.quadint and i1==0. and i2==1. and o==0.:
    #             return self.integrals['i1o--i2']
    #         elif i1<=i2 and i2<=o:
    #             if self.z_1[0] == np.infty:
    #                 return fundict_inf['xi <= xj <= xo'](gammas[0], z_0s[0], z_1s[0], z_cs[0], gammas[1], z_0s[1], z_1s[1], z_cs[1], gammas[2], z_0s[2], z_1s[2], z_cs[2], x1, x2, xo, self.length)
    #             else:
    #                 return fundict['xi <= xj <= xo'](gammas[0], z_0s[0], z_1s[0], z_cs[0], gammas[1], z_0s[1], z_1s[1], z_cs[1], gammas[2], z_0s[2], z_1s[2], z_cs[2], x1, x2, xo, self.length)
    #         elif i1<=o and o<=i2:
    #             if self.z_1[0] == np.infty:
    #                 return fundict_inf['xi <= xo <= xj'](gammas[0], z_0s[0], z_1s[0], z_cs[0], gammas[1], z_0s[1], z_1s[1], z_cs[1], gammas[2], z_0s[2], z_1s[2], z_cs[2], x1, x2, xo, self.length)
    #             else:
    #                 return fundict['xi <= xo <= xj'](gammas[0], z_0s[0], z_1s[0], z_cs[0], gammas[1], z_0s[1], z_1s[1], z_cs[1], gammas[2], z_0s[2], z_1s[2], z_cs[2], x1, x2, xo, self.length)
    #         elif i2<=i1 and i1<=o:
    #             if self.z_1[0] == np.infty:
    #                 return fundict_inf['xj <= xi <= xo'](gammas[0], z_0s[0], z_1s[0], z_cs[0], gammas[1], z_0s[1], z_1s[1], z_cs[1], gammas[2], z_0s[2], z_1s[2], z_cs[2], x1, x2, xo, self.length)
    #             else:
    #                 return fundict['xj <= xi <= xo'](gammas[0], z_0s[0], z_1s[0], z_cs[0], gammas[1], z_0s[1], z_1s[1], z_cs[1], gammas[2], z_0s[2], z_1s[2], z_cs[2], x1, x2, xo, self.length)
    #         elif i2<=o and o<=i1:
    #             if self.z_1[0] == np.infty:
    #                 return fundict_inf['xj <= xo <= xi'](gammas[0], z_0s[0], z_1s[0], z_cs[0], gammas[1], z_0s[1], z_1s[1], z_cs[1], gammas[2], z_0s[2], z_1s[2], z_cs[2], x1, x2, xo, self.length)
    #             else:
    #                 return fundict['xj <= xo <= xi'](gammas[0], z_0s[0], z_1s[0], z_cs[0], gammas[1], z_0s[1], z_1s[1], z_cs[1], gammas[2], z_0s[2], z_1s[2], z_cs[2], x1, x2, xo, self.length)
    #         elif o<=i1 and i1<=i2:
    #             if self.z_1[0] == np.infty:
    #                 return fundict_inf['xo <= xi <= xj'](gammas[0], z_0s[0], z_1s[0], z_cs[0], gammas[1], z_0s[1], z_1s[1], z_cs[1], gammas[2], z_0s[2], z_1s[2], z_cs[2], x1, x2, xo, self.length)
    #             else:
    #                 return fundict['xo <= xi <= xj'](gammas[0], z_0s[0], z_1s[0], z_cs[0], gammas[1], z_0s[1], z_1s[1], z_cs[1], gammas[2], z_0s[2], z_1s[2], z_cs[2], x1, x2, xo, self.length)
    #         elif o<=i2 and i2<=i1:
    #             if self.z_1[0] == np.infty:
    #                 return fundict_inf['xo <= xj <= xi'](gammas[0], z_0s[0], z_1s[0], z_cs[0], gammas[1], z_0s[1], z_1s[1], z_cs[1], gammas[2], z_0s[2], z_1s[2], z_cs[2], x1, x2, xo, self.length)
    #             else:
    #                 return fundict['xo <= xj <= xi'](gammas[0], z_0s[0], z_1s[0], z_cs[0], gammas[1], z_0s[1], z_1s[1], z_cs[1], gammas[2], z_0s[2], z_1s[2], z_cs[2], x1, x2, xo, self.length)
    #     else:
    #         return np.zeros(inds.shape, dtype=complex)
        
    def set_greensfunctions(self):
        # transfer impedance between ends of segment
        if self.z_1[0] == np.infty:
            self.z_trans = 1. / ((1./self.z_0) * np.cosh(self.gamma*self.length) + \
                        1./self.z_c * np.sinh(self.gamma*self.length))
            self.z_in = self.collapse_branch0()
        else: 
            self.z_trans = 1. / ((1./self.z_0 + 1./self.z_1) * np.cosh(self.gamma*self.length) + \
                        (self.z_c/(self.z_0*self.z_1) + 1./self.z_c) * np.sinh(self.gamma*self.length))
            self.z_in = 1./(1./self.z_1 + 1./self.collapse_branch0())
            
        if self.order==2:
            if self.z_1_ext[0] == np.infty:
                self.z_trans_ext = 1. / ((1./self.z_0_ext) * np.cosh(self.gamma_ext*self.length) + \
                            1./self.z_c_ext * np.sinh(self.gamma_ext*self.length))
                self.z_in_ext = self.collapse_branch0(size='ext')
            else: 
                self.z_trans_ext = 1. / ((1./self.z_0_ext + 1./self.z_1_ext) * np.cosh(self.gamma_ext*self.length) + \
                            (self.z_c_ext/(self.z_0_ext*self.z_1_ext) + 1./self.z_c_ext) * np.sinh(self.gamma_ext*self.length))
                self.z_in_ext = 1./(1./self.z_1_ext + 1./self.collapse_branch0(size='ext'))
                
    def set_voltagetransfers(self):
        # voltage transfers between ends of the segment:
        if self.z_1[0] == np.infty:
            self.GV_01 = 1. / np.cosh(self.gamma*self.length)
        else:
            self.GV_01 = 1. / (np.cosh(self.gamma*self.length) * (1. + (self.z_c / self.z_1) * tanh(self.gamma * self.length)))
        self.GV_10 = 1. / (np.cosh(self.gamma*self.length) * (1. + (self.z_c / self.z_0) * tanh(self.gamma * self.length)))
        
        if self.order==2:
            if self.z_1[0] == np.infty:
                self.GV_01_ext = 1. / np.cosh(self.gamma_ext*self.length)
            else:
                self.GV_01_ext = 1. / (np.cosh(self.gamma_ext*self.length) * (1. + (self.z_c_ext / self.z_1_ext) * tanh(self.gamma_ext * self.length)))
            self.GV_10_ext = 1. / (np.cosh(self.gamma_ext*self.length) * (1. + (self.z_c_ext / self.z_0_ext) * tanh(self.gamma_ext * self.length)))
        
    def set_impedance_1(self, cnodes):
        if not cnodes:
            self.z_1 = np.infty*np.ones(len(self.freqs))
        elif len(cnodes) == 1:
            z_child = cnodes[0].get_content()['impedance'].collapse_branch1()
            self.z_1 = z_child
        elif len(cnodes) == 2:
            z_child1 = cnodes[0].get_content()['impedance'].collapse_branch1()
            z_child2 = cnodes[1].get_content()['impedance'].collapse_branch1()
            self.z_1 = 1./(1./z_child1 + 1./z_child2)
        
        if self.order==2:
            freq_ext = 1j * np.arange(2.*self.freqs[0].imag, 2.*self.freqs[-1].imag + 2.*(self.freqs[1].imag-self.freqs[0].imag), self.freqs[1].imag-self.freqs[0].imag)
            if not cnodes:
                self.z_1_ext = np.infty*np.ones(len(freq_ext))
            elif len(cnodes) == 1:
                z_child = cnodes[0].get_content()['impedance'].collapse_branch1(size='ext')
                self.z_1_ext = z_child
            elif len(cnodes) == 2:
                z_child1 = cnodes[0].get_content()['impedance'].collapse_branch1(size='ext')
                z_child2 = cnodes[1].get_content()['impedance'].collapse_branch1(size='ext')
                self.z_1_ext = 1./(1./z_child1 + 1./z_child2)
    
    def set_impedance_0(self, pnode, cnode=None, order=2):
        if cnode == None:
            z_c = np.infty
        else:
            z_c = cnode.get_content()['impedance'].collapse_branch1()
        z_p = pnode.get_content()['impedance'].collapse_branch0()
        self.z_0 = 1./(1./z_c + 1./z_p)
        
        if self.order == 2:
            if cnode == None:
                z_c = np.infty
            else:
                z_c = cnode.get_content()['impedance'].collapse_branch1(size='ext')
            z_p = pnode.get_content()['impedance'].collapse_branch0(size='ext')
            self.z_0_ext = 1./(1./z_c + 1./z_p)
        
    def set_impedance_0_soma(self, somaimp, somaimp_ext=0.):
        self.z_0 = somaimp
        if self.order == 2:
            self.z_0_ext = somaimp_ext
        
    def collapse_branch0(self, size='normal'): 
        if size=='normal':
            return self.z_c * (self.z_0*np.cosh(self.gamma*self.length) + self.z_c*np.sinh(self.gamma*self.length)) / \
                        (self.z_0*np.sinh(self.gamma*self.length) + self.z_c*np.cosh(self.gamma*self.length))
        elif size=='ext':
            return self.z_c_ext * (self.z_0_ext*np.cosh(self.gamma_ext*self.length) + self.z_c_ext*np.sinh(self.gamma_ext*self.length)) / \
                        (self.z_0_ext*np.sinh(self.gamma_ext*self.length) + self.z_c_ext*np.cosh(self.gamma_ext*self.length))
        else:
            print 'Wrong size flag'
            exit(1)
                    
    def collapse_branch1(self, size='normal'):
        if size=='normal':
            if self.z_1[0] == np.infty:
                zr = self.z_c / tanh(self.gamma*self.length)
            else:
                zr = self.z_c * (self.z_1*np.cosh(self.gamma*self.length) + self.z_c*np.sinh(self.gamma*self.length)) / \
                        (self.z_1*np.sinh(self.gamma*self.length) + self.z_c*np.cosh(self.gamma*self.length))
            return zr
        elif size=='ext':
            if self.z_1_ext[0] == np.infty:
                zr = self.z_c_ext / tanh(self.gamma_ext*self.length)
            else:
                zr = self.z_c_ext * (self.z_1_ext*np.cosh(self.gamma_ext*self.length) + self.z_c_ext*np.sinh(self.gamma_ext*self.length)) / \
                        (self.z_1_ext*np.sinh(self.gamma_ext*self.length) + self.z_c_ext*np.cosh(self.gamma_ext*self.length))
            return zr
        else:
            print 'Wrong size flag'
            exit(1)
        

class somaImpedance:
    def __init__(self, freqs, node1, node2, node3, E0, gs, es, gcalctype, C_m, pr=False): 
        self.freqs = freqs
        self.order = 1  #flag variable
        r = abs(node2.get_content()['p3d'].radius) /1e4 # convert radius to cm
        self.radius = r
        self.somaA = 4.0*np.pi*r*r # in cm^2
        if pr:
            print 'In GF calculation:'
            print 'soma Radius = %.8f cm' % r
            print 'soma Surface = %.8f cm^2' % self.somaA
        self.z_soma = calc_membraneImpedance(self.freqs, E0, gs, es, gcalctype, C_m, somaA=self.somaA)
        # input impedance
        somachildren = node1.get_child_nodes()[2:]
        scnodes = somachildren
        for ind, somachild in enumerate(somachildren):
            if not is_changenode(somachild):
                scnodes[ind] = find_previous_changenode(somachild)[0]
        self.z_in = self.soma_impedance(scnodes)
        #~ if len(freqs)>1:
            #~ import matplotlib.pyplot as pl
            #~ pl.plot(freqs.imag, self.z_soma, 'r')
            #~ pl.plot(freqs.imag, self.z_in, 'b')
            #~ pl.show()
        
    def set_extended_impedance(self, somanode, E0, gs, es, gcalctype, C_m):
        self.order = 2
        # necessary frequency array
        freq_ext = 1j * np.arange(2.*self.freqs[0].imag, 2.*self.freqs[-1].imag + 2.*(self.freqs[1].imag-self.freqs[0].imag), self.freqs[1].imag-self.freqs[0].imag)
        # extended impedance
        self.z_soma_ext = calc_membraneImpedance(freq_ext, E0, gs, es, gcalctype, C_m, somaA=self.somaA)
        # input impedance
        somachildren = somanode.get_child_nodes()[2:]
        scnodes = somachildren
        for ind, somachild in enumerate(somachildren):
            if not is_changenode(somachild):
                scnodes[ind] = find_previous_changenode(somachild)[0]
        self.z_in_ext = self.soma_impedance(scnodes, size='ext')
        
    def soma_impedance(self, nodes, size='normal'):
        z_soma = self.z_soma 
        if size=='ext':
            z_soma = self.z_soma_ext
        if nodes:
            for node in nodes:
                z_soma = 1./((1./z_soma) + (1./node.get_content()['impedance'].collapse_branch1(size=size)))
        return z_soma
        
    def set_quadratic_coeff(self, E0, gs, es, gcalctype, C_m):
        self.order = 2
        # necessary frequency array
        freq_ext = 1j * np.arange(2.*self.freqs[0].imag, 2.*self.freqs[-1].imag + 2.*(self.freqs[1].imag-self.freqs[0].imag), self.freqs[1].imag-self.freqs[0].imag)
        # extended impedances
        self.z_soma_ext = calc_membraneImpedance(freq_ext, E0, gs, es, gcalctype, C_m, somaA=self.somaA)
        # quadrqtic coefficient
        self.quadratic_coeff = calc_second_order(self.freqs, E0, gs, es, gcalctype, C_m, somaA=self.somaA)
        
    def calc_integral(self, inloc1, inloc2, outloc): 
        inds = get_inds(self.freqs)
        return np.ones(inds.shape, dtype=complex)


class segmentPhysiology:
    def __init__(self, E0=-65., r_a=100./1e6, gs={'L': 1e6/50000.}, es={'L': -65.}, conc_mechs={}):
        self.cm = 1.0#1.0    # uF/cm^2
        self.gs = copy.deepcopy(gs)    # uS/cm^2
        self.es = copy.deepcopy(es)    # mV
        self.gcalctype = {key: 'pas' for key in gs.keys()}
        self.conc_mechs = conc_mechs
        self.Veq = E0   # mV, exact spatial equilibrium potential
        self.E0 = E0    # mV, equilibrium potential used for computation
        self.C0 = {}    # mM, equilibrium concentrations
        self.r_a = r_a  # MOhm*cm (attention, to calculate the impedances,
                            # MOhm is used, for NEURON, convert to Ohm!!!)
                            
    def set_membrane_constants(self, lambda_m=None, tau_m=None):
        self.lambda_m = lambda_m
        self.tau_m = tau_m

    def set_E0(self, E0):
        self.E0 = E0

    def __str__(self):
        phys_string = 'c_m = ' + str(self.cm) + ' uF/cm^2, r_a = ' + str(self.r_a) + ' MOhm*cm\n'
        for chan in self.gs:
            if self.gs[chan] > 1e-10:
                phys_string += chan + ': gbar = ' + str(self.gs[chan]) + ' uS/cm^2, E = ' + str(self.es[chan]) + ' mV \n'
        phys_string += 'E_eq = ' + str(self.Veq) + ' mV, E_comp = ' + str(self.E0) + ' mV \n'
        return phys_string 
        
########################################################################


# mechanism_name_translation NEURON
mechname = {'L': 'pas', 'Na': 'INa', 'K': 'IK', 'Klva': 'IKlva', 'h': 'Ih'}
# default channel distribution 
default_distr = {'Na': {'type': 'flat', 'param': [0.00*1e6], 'E': 50.},
            'K': {'type': 'flat', 'param': [0.000*1e6], 'E':-77.},
            'Klva': {'type': 'flat', 'param': [0.000*1e6], 'E': -106.},
            'h': {'type': 'flat', 'param': [0.000*1e6], 'E': -43.},
            'L': {'type': 'fit', 'param': [-65., 50.], 'E': -65.}}
# soma channel distribution 
default_soma_distr = {'Na': {'type': 'flat', 'param': [0.0*1e6], 'E': 50.},
            'K': {'type': 'flat', 'param': [0.0*1e6], 'E':-77.},
            'Klva': {'type': 'flat', 'param': [0.00*1e6], 'E': -106.},
            'h': {'type': 'flat', 'param': [0.00*1e6], 'E': -43.},
            'L': {'type': 'fit', 'param': [-65., 50.], 'E': -65.}}


## main class ##########################################################
class greensTree:
    def __init__(self, morphFile, ionc_distr=default_distr, ionc_distr_apical=None, 
                    soma_distr=default_soma_distr, conc_mech_distr=None, conc_mech_distr_apical=None, 
                    conc_mech_distr_soma=None, cnodesdistr='exact', axon=False, 
                    record_concentrations=False, pprint=True):
        '''
        Initializes a greensTree object that contains the membrane information
        and from which the Greens functions are computed.

        input:
            [morphFile]: morphology .swc file
            [ionc_distr]: the desired dendritic ion channel distribution. When [ionc_distr_apical] 
                is None, this is considered to be the distribution throughout the dendritic tree.
                When [ionc_distr_apical] is not None, it is the basel distribution. It is a dictionary
                of the form:
                    {'ionc_type_Key': {'type': '', param: [], 'E': float}}
            [ionc_distr_apical]: when not None, the apical ion channel distribution
            [soma_distr]: the distribution of ion channels at the soma
            [cnodesdistr]: where to put the changenodes (nodes where impedances are stored). 
                'exact' means that the changenodes are distributed at location where the membrane
                parameters change and at bifurcations. 'all' means that all nodes are changenodes.
            [axon]: whether to include an axon.
            [pprint]: print membrane parameters to screen
        '''
        self.tree = btstructs.STree()
        self.tree.read_SWC_tree_from_file(morphFile, axon=axon)
        self.tree.remove_trifuractions()
        nodes = self.tree.get_nodes()


        maxLength = max([distance_to_soma(self.tree, n) for n in nodes if len(n.get_child_nodes()) == 0])
        # set ionchannel conductances
        if ionc_distr_apical == None:
            # one global channel distribution throughout the tree
            keys = ionc_distr.keys()
            for node in nodes[3:]:
                cont = node.get_content()
                cont['physiology'] = segmentPhysiology()
                distance2soma = distance_to_soma(self.tree, node)
                for key in set(keys)-set('L'):
                    gs, es = distribute_channel(copy.copy(cont['physiology'].gs), copy.copy(cont['physiology'].es), cont['physiology'].cm, distance2soma, key, 
                            ionc_distr[key]['type'], ionc_distr[key]['param'], ionc_distr[key]['E'], maxLen=maxLength)
                    cont['physiology'].gs = gs
                    cont['physiology'].es = es
                    cont['physiology'].gcalctype[key] = ionc_distr[key]['calctype']
                key = 'L' # leak has to go last
                gs, es = distribute_channel(copy.copy(cont['physiology'].gs), copy.copy(cont['physiology'].es), cont['physiology'].cm, distance2soma, key, 
                        ionc_distr[key]['type'], ionc_distr[key]['param'], ionc_distr[key]['E'], maxLen=maxLength)
                cont['physiology'].gs = gs
                cont['physiology'].es = es
                cont['physiology'].gcalctype[key] = 'pas'
                node.set_content(cont)
        else:
            # apical and basal channel distribution
            for node in nodes[3:]:
                if node.get_content()['p3d'].type == 3:
                    keys = ionc_distr.keys()
                    # basal distribution
                    cont = node.get_content()
                    cont['physiology'] = segmentPhysiology()
                    distance2soma = distance_to_soma(self.tree, node)
                    for key in set(keys)-set('L'):
                        gs, es = distribute_channel(copy.copy(cont['physiology'].gs), copy.copy(cont['physiology'].es), cont['physiology'].cm, distance2soma, key, 
                                ionc_distr[key]['type'], ionc_distr[key]['param'], ionc_distr[key]['E'], maxLen=maxLength)
                        cont['physiology'].gs = gs
                        cont['physiology'].es = es
                        cont['physiology'].gcalctype[key] = ionc_distr[key]['calctype']
                    key = 'L' # leak has to go last
                    gs, es = distribute_channel(copy.copy(cont['physiology'].gs), copy.copy(cont['physiology'].es), cont['physiology'].cm, distance2soma, key, 
                            ionc_distr[key]['type'], ionc_distr[key]['param'], ionc_distr[key]['E'], maxLen=maxLength)
                    cont['physiology'].gs = gs
                    cont['physiology'].es = es
                    cont['physiology'].gcalctype[key] = 'pas'
                    node.set_content(cont)
                else:
                    # apical distribution
                    keys = ionc_distr_apical.keys()
                    cont = node.get_content()
                    cont['physiology'] = segmentPhysiology()
                    distance2soma = distance_to_soma(self.tree, node)
                    for key in set(keys)-set('L'):
                        gs, es = distribute_channel(copy.copy(cont['physiology'].gs), copy.copy(cont['physiology'].es), cont['physiology'].cm, distance2soma, key, 
                                ionc_distr_apical[key]['type'], ionc_distr_apical[key]['param'], ionc_distr_apical[key]['E'], maxLen=maxLength)
                        cont['physiology'].gs = gs
                        cont['physiology'].es = es
                        cont['physiology'].gcalctype[key] = ionc_distr_apical[key]['calctype']
                    key = 'L' # leak has to go last
                    gs, es = distribute_channel(copy.copy(cont['physiology'].gs), copy.copy(cont['physiology'].es), cont['physiology'].cm, distance2soma, key, 
                            ionc_distr_apical[key]['type'], ionc_distr_apical[key]['param'], ionc_distr_apical[key]['E'], maxLen=maxLength)
                    cont['physiology'].gs = gs
                    cont['physiology'].es = es
                    cont['physiology'].gcalctype[key] = 'pas'
                    node.set_content(cont)
        # soma channels
        keys = soma_distr.keys()
        somanode = nodes[0]
        cont = somanode.get_content()
        cont['physiology'] = segmentPhysiology()
        for key in set(keys)-set('L'):
            gs, es = distribute_channel(copy.copy(cont['physiology'].gs), copy.copy(cont['physiology']).es, cont['physiology'].cm, 0., key, 
                        soma_distr[key]['type'], soma_distr[key]['param'], soma_distr[key]['E'])
            cont['physiology'].gs = gs
            cont['physiology'].es = es
            cont['physiology'].gcalctype[key] = soma_distr[key]['calctype']
        key = 'L' # leak has to go last
        gs, es = distribute_channel(copy.copy(cont['physiology'].gs), copy.copy(cont['physiology']).es, cont['physiology'].cm, 0., key, 
                    soma_distr[key]['type'], soma_distr[key]['param'], soma_distr[key]['E'])
        cont['physiology'].gs = gs
        cont['physiology'].es = es
        cont['physiology'].gcalctype[key] = 'pas'
        somanode.set_content(cont)

        if (conc_mech_distr != None) and (conc_mech_distr_apical == None):
            # concentration mechanisms
            for node in nodes[3:]:
                cont = node.get_content()
                for key in conc_mech_distr.keys():
                    cont['physiology'].conc_mechs = distribute_conc_mech(copy.copy(cont['physiology'].conc_mechs), distance2soma, key, 
                                                conc_mech_distr[key]['type'], conc_mech_distr[key]['param'], maxLen=maxLength)
                node.set_content(cont)
        elif (conc_mech_distr != None) and (conc_mech_distr_apical != None):
            for node in nodes[3:]:
                cont = node.get_content()
                if node.get_content()['p3d'].type == 3:
                    for key in conc_mech_distr.keys():
                        cont['physiology'].conc_mechs = distribute_conc_mech(copy.copy(cont['physiology'].conc_mechs), distance2soma, key, 
                                                conc_mech_distr[key]['type'], conc_mech_distr[key]['param'], maxLen=maxLength)
                else:
                    for key in conc_mech_distr.keys():
                        cont['physiology'].conc_mechs = distribute_conc_mech(copy.copy(cont['physiology'].conc_mechs), distance2soma, key, 
                                                conc_mech_distr_apical[key]['type'], conc_mech_distr_apical[key]['param'], maxLen=maxLength)
                node.set_content(cont)
        elif (conc_mech_distr == None) and (conc_mech_distr_apical != None):
            for node in nodes[3:]:
                cont = node.get_content()
                if node.get_content()['p3d'].type == 4:
                    for key in conc_mech_distr_apical.keys():
                        cont['physiology'].conc_mechs = distribute_conc_mech(copy.copy(cont['physiology'].conc_mechs), distance2soma, key, 
                                                conc_mech_distr_apical[key]['type'], conc_mech_distr_apical[key]['param'], maxLen=maxLength)
                node.set_content(cont)
        if conc_mech_distr_soma != None:
            # soma mechanism
            somanode = nodes[0]
            cont = somanode.get_content()
            for key in conc_mech_distr_soma.keys():
                cont['physiology'].conc_mechs = distribute_conc_mech(copy.copy(cont['physiology'].conc_mechs), distance2soma, key, 
                                            conc_mech_distr_soma[key]['type'], conc_mech_distr_soma[key]['param'], maxLen=maxLength)
            somanode.set_content(cont)

        # set changenodes
        if cnodesdistr == 'exact':
            cnodes = []
            somachildren = nodes[0].get_child_nodes()[2:]
            nnodes = list(set(nodes[3:]) - set(somachildren))
            for ind, node in enumerate(nnodes):
                pnode = node.get_parent_node()
                gs1 = pnode.get_content()['physiology'].gs
                gs2 = node.get_content()['physiology'].gs
                for key in gs1.keys():
                    if gs1[key] != gs2[key]:
                        cnodes.append(pnode)
                        break
            self.set_changenodes(cnodes)
        elif cnodesdistr == 'all':
            # nodes = self.tree.get_nodes()
            cnodes = nodes[0:1] + nodes[3:]
            self.set_changenodes(cnodes)

        # set equilibrium membrane potential if leak conductance does not fix it
        if (ionc_distr['L']['type'] != 'fit') or (soma_distr['L']['type'] != 'fit'):
            Veqs, Ceqs = self.compute_resting_potential(record_concentrations=record_concentrations)
            for ind, Veq in enumerate(Veqs):
                node = self.tree.get_node_with_index(Veq['node'])
                phys = node.get_content()['physiology']
                phys.E0 = Veq['E0']; phys.Veq = Veq['E0']
                phys.C0 = Ceqs[ind]['C0']
            self.set_functional_equilibrium_potential()

        if pprint:
            for node in nodes[0:1]+nodes[3:]:
                print node
                print node.get_content()['physiology']
                
    def set_changenodes(self, cnodes = []):
        '''
        Sets nodes that are bifurcations, leafs or soma as changenodes.
        Additional nodes can be added in the cnodes list.

        input:
            [cnodes]: list of nodes that should be set as changenodes.
        '''
        nodes = self.tree.get_nodes()
        for node in nodes:
            if len(node.get_child_nodes()) > 1:
                cont = node.get_content()
                cont['changenode'] = 1
                node.set_content(cont)
            if len(node.get_child_nodes()) == 0 and node._index != 2 and node._index != 3:
                cont = node.get_content()
                cont['changenode'] = 1
                node.set_content(cont)
            if node in cnodes:
                cont = node.get_content()
                cont['changenode'] = 1
                node.set_content(cont)
    
    def get_changenodes(self):
        '''
        - output
            - list of changenodes
        '''
        nodes = self.tree.get_nodes()
        return [node for node in nodes if is_changenode(node)]

    def compute_resting_potential(self, record_concentrations=True):
        '''
        compute the resting membrane potential of the defined neuron. Uses the NEURON
        simulator.

        output:
            [Veqs]: list of dictionary, containing the equilibrium potential for each
                node, keys are: 'E0', the resting membrane potential, 'node': the node
                index and 'x': the location on the node.
            [Ceqs]: list of dictionary, containing equilibirum concentration for each ion
                at each node. Keys are: 'C0', the equilibirum concentration, 'node': the node
                index and 'x': the location on the node.
        '''
        NEURONneuron = neurM.NeuronNeuron(self, dt=0.1, truemorph=True, factorlambda=1)
        nodes = self.tree.get_nodes()
        nodes = nodes[0:1] + nodes[3:]
        reclocs = [{'ID': ind, 'x': 1.0, 'node': node._index} for ind, node in enumerate(nodes)]
        NEURONneuron.add_recorder(reclocs)
        Vm = NEURONneuron.run(tdur=2000., record_concentrations=record_concentrations)
        Veqs = [{'E0': Vm[recloc['ID']][-1], 'node': recloc['node'], 'x': recloc['x']} for recloc in reclocs]
        Ceqs = []            
        ions = ['ca']
        for recloc in reclocs:
            Ceqs.append({'C0': {}, 'node': recloc['node'], 'x': recloc['x']})
            if record_concentrations:
                for ion in ions:
                    Ceqs[-1]['C0'][ion] = Vm['conc'][recloc['ID']][ion][-1]
        #print Veqs
        return Veqs, Ceqs

    def set_functional_equilibrium_potential(self):
        somanode = self.tree.get_node_with_index(1)
        E0_start = somanode.get_content()['physiology'].E0
        self._set_E0_up(somanode, E0_start)

    def _set_E0_up(self, node, E0_start):
        if node._index == 1:
            cnodes = node.get_child_nodes()[2:]
        else:
            cnodes = node.get_child_nodes()
        for cnode in cnodes:
            cont = cnode.get_content()
            E0 = cont['physiology'].E0
            if (np.abs(E0_start - E0) > .5) or is_changenode(cnode):
                # make cnode changenode if it is not changenode yet
                if not is_changenode(cnode):
                    cont['changenode'] = 1
                # set equilibrium potential
                cont['physiology'].set_E0((E0_start + E0)/2.)
                self._set_E0_up(cnode, E0)
            else:
                self._set_E0_up(cnode, E0_start)

    def get_equilibrium_potentials(self, inlocs):
        '''
        Returns array of equilibrium potentials for each inloc in inlocs.

        input:
            [inlocs]: list of input locations.

        output:
            [Veqs]: numpy array of floats, containing the equilibrium potentials
                corresponding to each inloc.
        '''
        Veqs = []
        for inloc in inlocs:
            node = self.tree.get_node_with_index(inloc['node'])
            phys = node.get_content()['physiology']
            Veqs.append(phys.Veq)
        return np.array(Veqs)

    def create_simplified_tree(self, deltaL=50.):
        swc_file = open(self.morphFile.replace('.swc', '') + '_simplified.swc', 'w')
        swc_file.write('# This swc file is automatically generated by the function \n' + \
                        '# greensTree.create_simplified_tree() in mophologyReader.py \n' + \
                        '# by pooling modes with dx = ' + str(deltaL) + ' um \n#\n')
        somanode = self.tree.get_node_with_index(1)
        # write soma stuff
        snode2 = self.tree.get_node_with_index(2)
        snode3 = self.tree.get_node_with_index(3)
        p3ds = [somanode.get_content()['p3d'], 
                snode2.get_content()['p3d'],
                snode3.get_content()['p3d']]
        for i, p3d in enumerate(p3ds):
            string = str(i+1) + ' ' + str(1) + ' ' + str(p3d.x) + ' ' + str(p3d.y) + ' ' + str(p3d.z) + ' ' + str(p3d.radius) + ' '
            if i == 0: string += str(-1) + '\n'
            else: string += str(1) + '\n'
            swc_file.write(string)
        # start recursion
        self._add_to_simplified_tree(somanode, 0., somanode, 1, np.array([0.,0.,0.]), [somanode, snode2, snode3], swc_file, deltaL)
        swc_file.close()

    def _add_to_simplified_tree(self, node, L0, pnode_object, pnode_index, x_pnode, pnodes, swc_file, deltaL=50.):
        p3d_parent = node.get_content()['p3d']
        if node._index == 1:
            cnodes = node.get_child_nodes()[2:]
            x_parent = np.array([0., 0., 0.])
        else:
            cnodes = node.get_child_nodes()
            x_parent = np.array([p3d_parent.x, p3d_parent.y, p3d_parent.z])
        for cnode in cnodes:
            p3d = cnode.get_content()['p3d']
            x = np.array([p3d.x, p3d.y, p3d.z])
            Lnew = L0 + np.linalg.norm(x - x_parent)
            if is_changenode(cnode):
                pnodes.append(cnode)
                path = self.tree.path_between_nodes(cnode, pnode_object)
                radius, length = get_cylinder_radius_length(path)
                if pnode_object._index == 1:
                    radius = p3d.radius
                    x0 = np.array([0.,0.,0.])
                else:
                    p3d0 = pnode_object.get_content()['p3d']
                    x0 = np.array([p3d0.x, p3d0.y, p3d0.z])
                xnew = x_pnode + Lnew * (x-x0) / np.linalg.norm(x-x0)
                # radius = p3d.radius
                swc_file.write(str(len(pnodes)) + ' ' + str(p3d.type) + ' ' + \
                                str(xnew[0]) + ' ' + str(xnew[1]) + ' ' + str(xnew[2]) + ' ' + \
                                str(radius) + ' ' + str(pnode_index) + '\n')
                self._add_to_simplified_tree(cnode, 0., cnode, len(pnodes), xnew, pnodes, swc_file, deltaL)
            else:
                if Lnew > deltaL:
                    pnodes.append(cnode)
                    path = self.tree.path_between_nodes(cnode, pnode_object)
                    radius, length = get_cylinder_radius_length(path)
                    if pnode_object._index == 1:
                        radius = p3d.radius
                        x0 = np.array([0.,0.,0.])
                    else:
                        p3d0 = pnode_object.get_content()['p3d']
                        x0 = np.array([p3d0.x, p3d0.y, p3d0.z])
                    xnew = x_pnode + Lnew * (x-x0) / np.linalg.norm(x-x0)
                    radius = p3d.radius
                    swc_file.write(str(len(pnodes)) + ' ' + str(p3d.type) + ' ' + \
                                    str(xnew[0]) + ' ' + str(xnew[1]) + ' ' + str(xnew[2]) + ' ' + \
                                    str(radius) + ' ' + str(pnode_index) + '\n')
                    self._add_to_simplified_tree(cnode, 0., cnode, len(pnodes), xnew, pnodes, swc_file, deltaL)
                else:
                    self._add_to_simplified_tree(cnode, Lnew, pnode_object, pnode_index, x_pnode, pnodes, swc_file, deltaL)
        
    def set_electrical_constants(self):
        '''
        computes the electrical lenght and time constants of the membrane.
        Sets these lengths in the content['physiology'] object of the nodes.
        
        - output:
            - [memconstants]: dictionary of tuples. Indices are changenode 
                indices and elements are tuples (lambda (um), tau (ms)) 
        '''
        cnodes = self.get_changenodes()
        memconstants = {}
        for node in cnodes:
            cont = node.get_content()
            imp = cont['impedance']
            phys = cont['physiology']
            lambda_m, tau_m = evaluate_membrane_constants(phys.gs, phys.es,
                                imp.radius, r_a=phys.r_a, Eq=phys.E0, cm=phys.cm)
            phys.set_membrane_constants(lambda_m=lambda_m, tau_m=tau_m)
            memconstants[node._index] = (lambda_m, tau_m)
        return memconstants
        
    def get_lengths_radii(self):
        '''
        Returns the lengths and radii of compartments
        
        - output:
            - [compartment_lengths_radii]: dictonary of tuples. Indices are 
                changenode indices and elements tuples (length (um), radius (um))
        '''
        cnodes = self.get_changenodes()
        compartment_lengths_radii = {}
        for node in cnodes:
            imp = node.get_content()['impedance']
            if node._index != 1:
                compartment_lengths_radii[node._index] = (imp.length*1e4, imp.radius*1e4)
            else:
                compartment_lengths_radii[node._index] = (imp.somaA*1e8, imp.radius*1e4)
        return compartment_lengths_radii
        
    def calc_greensfunction(self, inloc={'node': 10, 'x': 0.6}, 
                outloc={'node': 400, 'x': 0.3}, size='normal', voltage=False):
        '''
        Compute the Greens Function between inloc and outloc.
        
        input:
            [inloc]: dictionary containing \'node\'- and \'x\'-entry
            [outloc]: dictionary containing \'node\'- and \'x\'-entry
            [size]: string, default \'normal\', for normal gf, \'ext\' for extended
                    for 2d-kernel calculation. Only use \'ext\' when impedances
                    have been set with \'x\' flag.
            [voltage]: Boolean, default False. If False, computes the transfer from
                    current to voltage. If True, computes the transfer from voltage
                    to voltage.
                    
        output:
            [G]: numpy array of complex numbers. Contains the Green's function 
                    evaluated at the frequencies in self.freqs.
        '''
        in_node = self.tree.get_node_with_index(inloc['node'])
        out_node = self.tree.get_node_with_index(outloc['node'])
        
        if inloc['node'] != 1:
            if not is_changenode(in_node):
                cnin = find_previous_changenode(in_node)[0]
            else:
                cnin = in_node
            path = self.tree.path_between_nodes(cnin, in_node)
            inloc = get_reduced_loc(copy.copy(inloc), path)
        else:
            cnin = in_node
            
        if outloc['node'] != 1:
            if not is_changenode(out_node):
                cnout = find_previous_changenode(out_node)[0]
            else:
                cnout = out_node
            path = self.tree.path_between_nodes(cnout, out_node)
            outloc = get_reduced_loc(copy.copy(outloc), path)
        else:
            cnout = out_node
        
        if cnin == cnout:
            if cnin._index == 1:
                # node is soma node, so G is input impedance at soma
                if size=='normal':
                    if voltage:
                        G = np.ones(cnin.get_content()['impedance'].z_in.shape, dtype=complex)
                    else:
                        G = copy.copy(cnin.get_content()['impedance'].z_in)
                elif size=='ext':
                    if voltage:
                        G = np.ones(cnin.get_content()['impedance'].z_in_ext.shape, dtype=complex)
                    else:
                        G = copy.copy(cnin.get_content()['impedance'].z_in_ext)
                else:
                    print 'Error: wrong flag'
                    exit(1)
            else:
                if size=='normal':
                    gamma = cnout.get_content()['impedance'].gamma
                    z_c = cnout.get_content()['impedance'].z_c
                    z_0 = cnout.get_content()['impedance'].z_0
                    z_1 = cnout.get_content()['impedance'].z_1
                elif size=='ext':
                    gamma = cnout.get_content()['impedance'].gamma_ext
                    z_c = cnout.get_content()['impedance'].z_c_ext
                    z_0 = cnout.get_content()['impedance'].z_0_ext
                    z_1 = cnout.get_content()['impedance'].z_1_ext
                else:
                    print 'Error: wrong flag'
                    exit(1)
                L = cnout.get_content()['impedance'].length
                D = inloc['x']*L
                x = outloc['x']*L
                if x==D:
                    if x!=0: 
                        if z_1[0] == np.infty:
                            if voltage:
                                G = np.ones(z_c.shape, dtype=complex)
                            else:
                                G = np.exp(2.*np.log(z_c) + 2.*gamma*D) * np.power((1.+np.exp(-2.*gamma*D))/2.,2) \
                                        * (tanh(gamma*D) + (z_0/z_c)) \
                                        * one_minus_tanhtanh(gamma*D, gamma*L) / (z_c + z_0*tanh(gamma*L))
                        else:
                            if voltage:
                                G = np.ones(z_c.shape, dtype=complex)
                            else:
                                G = np.cosh(gamma*D) * one_minus_tanhtanh(gamma*L, gamma*D) * \
                                    (z_0/z_c + tanh(gamma*D)) * (z_1 + z_c*tanh(gamma*(L-D))) / \
                                    ((z_0+z_1)/z_c + (1. + (z_0/z_c)*(z_1/z_c))*tanh(gamma*L)) * np.cosh(gamma*D)
                    elif x==0:
                        if voltage:
                            G = np.ones(z_c.shape, dtype=complex)
                        else:
                            nin = find_next_changenode(cnin)
                            #~ print "gfcalculation nin ", nin
                            # G = copy.copy(cnin.get_content()['impedance'].z_in)
                            if z_1[0] == np.infty:
                                G = (z_0 / z_c) / (1./z_c + (z_0 / z_c**2) * tanh(gamma*L))
                            else:
                                G = z_0 * (z_1/z_c + tanh(gamma*L)) / \
                                    ((z_0+z_1)/z_c + (1. + (z_0/z_c)*(z_1/z_c)) * tanh(gamma*L))
                            #~ G = nin.G_in
                    elif x==L:
                        if voltage:
                            G = np.ones(z_c.shape, dtype=complex)
                        else:
                            #~ print "gfcalculation nin ", cnin
                            #G = copy.copy(cnin.get_content()['impedance'].z_in)
                            if z_1[0] == np.infty:
                                G = (z_0/z_c + tanh(gamma*L)) / (1./z_c + (z_0 / z_c**2) * tanh(gamma*L)) 
                            else:
                                G = z_1 * (z_0/z_c + tanh(gamma*L)) / \
                                    ((z_0+z_1)/z_c + (1. + (z_0/z_c)*(z_1/z_c)) * tanh(gamma*L))
                            #~ G = cnin.G_in
                elif x<D:
                    if z_1[0] == np.infty:
                        if voltage:
                            G = (np.cosh(gamma*x) / np.cosh(gamma*D)) * \
                                (1. + (z_c/z_0)*tanh(gamma*x)) / (1. + (z_c/z_0)*tanh(gamma*D))
                        else:
                            G = np.power(z_c,2) * np.cosh(gamma*D) * (np.sinh(gamma*x) + (z_0/z_c)*np.cosh(gamma*x)) * \
                                one_minus_tanhtanh(gamma*D, gamma*L) / (z_c + z_0*tanh(gamma*L))
                    else:
                        if voltage:
                            G = (np.cosh(gamma*x) / np.cosh(gamma*D)) * \
                                (1. + (z_c/z_0)*tanh(gamma*x)) / (1. + (z_c/z_0)*tanh(gamma*D))
                        else:
                            G = np.cosh(gamma*x) * one_minus_tanhtanh(gamma*L, gamma*D) * \
                                (tanh(gamma*x) + z_0/z_c) * (z_c*tanh(gamma*(L-D)) + z_1) / \
                                ((z_0+z_1)/z_c + (1. + (z_0/z_c)*(z_1/z_c))*tanh(gamma*L)) * np.cosh(gamma*D)
                elif x>D:
                    if z_1[0] == np.infty:
                        if voltage:
                            G = np.cosh(gamma*(L-x)) / np.cosh(gamma*(L-D))
                        else:
                            G = np.power(z_c,2) * np.cosh(gamma*x) * (np.sinh(gamma*D) + (z_0/z_c)*np.cosh(gamma*D)) * \
                                one_minus_tanhtanh(gamma*x, gamma*L) / (z_c + z_0*tanh(gamma*L))
                    else:
                        if voltage:
                            G = np.cosh(gamma*(L-x)) / np.cosh(gamma*(L-D)) * \
                                (1. + (z_c/z_1)*tanh(gamma*(L-x))) / (1. + (z_c/z_1)*tanh(gamma*(L-D)))
                        else:
                            G = np.cosh(gamma*D) * one_minus_tanhtanh(gamma*L, gamma*x) * \
                                (tanh(gamma*D) + z_0/z_c) * (z_c*tanh(gamma*(L-x)) + z_1) / \
                                ((z_0+z_1)/z_c + (1. + (z_0/z_c)*(z_1/z_c))*tanh(gamma*L)) * np.cosh(gamma*x)
            
        else:
            changenodes = path_between_changenodes(cnin, cnout)
            
            # set G_start
            if cnin._index == 1:
                if size=='normal':
                    if voltage:
                        G_start = np.ones(cnin.get_content()['impedance'].z_in.shape, dtype=complex)
                    else:
                        G_start = copy.copy(cnin.get_content()['impedance'].z_in)
                elif size=='ext':
                    if voltage:
                        G_start = np.ones(cnin.get_content()['impedance'].z_in_ext.shape, dtype=complex)
                    else:
                        G_start = copy.copy(cnin.get_content()['impedance'].z_in_ext)
            else:
                if size=='normal':
                    gamma = cnin.get_content()['impedance'].gamma
                    z_c = cnin.get_content()['impedance'].z_c
                    z_0 = cnin.get_content()['impedance'].z_0
                    z_1 = cnin.get_content()['impedance'].z_1
                elif size=='ext':
                    gamma = cnin.get_content()['impedance'].gamma_ext
                    z_c = cnin.get_content()['impedance'].z_c_ext
                    z_0 = cnin.get_content()['impedance'].z_0_ext
                    z_1 = cnin.get_content()['impedance'].z_1_ext
                else:
                    print 'Error: wrong flag'
                    exit(1)
                L = cnin.get_content()['impedance'].length
                D = inloc['x']*L
                
                if changenodes[1] in find_previous_changenode(cnin): # if true, path goes further from soma
                    if z_1[0] == np.infty:
                        if voltage:
                            G_start = 1. / np.cosh(gamma*(L-D))
                        else:
                            G_start = (np.cosh(gamma*D)/np.cosh(gamma*L)) * (z_c*tanh(gamma*D) + z_0) / \
                                (1. + z_0/z_c *tanh(gamma*L))
                    else:
                        if voltage:
                            G_start = 1. / (np.cosh(gamma*(L-D)) * (1. + (z_c/z_1)*tanh(gamma*(L-D))))
                        else:
                            G_start = (np.cosh(gamma*D)/np.cosh(gamma*L)) * z_1 * (tanh(gamma*D) + z_0/z_c) / \
                                ((z_0+z_1)/z_c + (1. + (z_0/z_c)*(z_1/z_c))*tanh(gamma*L))
                elif changenodes[1] == find_next_changenode(cnin): # else, path goes to soma
                    if z_1[0] == np.infty:
                        if voltage:
                            G_start = 1. / (np.cosh(gamma*D) * (1. + (z_c/z_0)*tanh(gamma*D)))
                        else:
                            G_start = (np.cosh(gamma*(L-D))/np.cosh(gamma*L)) * z_0 / \
                                (1. + z_0/z_c *tanh(gamma*L))
                    else:
                        if voltage:
                            G_start = 1. / (np.cosh(gamma*D) * (1. + (z_c/z_0)*tanh(gamma*D)))
                        else:
                            G_start = (np.cosh(gamma*(L-D))/np.cosh(gamma*L)) * z_0/z_c * (z_c*tanh(gamma*(L-D)) + z_1) / \
                                ((z_0+z_1)/z_c + (1. + (z_0/z_c)*(z_1/z_c))*tanh(gamma*L))
                else:
                    print 'Error: wrong input node assignment'
                    exit(1)
                            
            # set G_stop
            if self.volt:
                if cnout._index == 1:
                    G_stop = np.ones(G_start.shape, dtype=complex)
                else:
                    if size=='normal':
                        gamma = cnout.get_content()['impedance'].gamma
                        z_c = cnout.get_content()['impedance'].z_c
                        z_0 = cnout.get_content()['impedance'].z_0
                        z_1 = cnout.get_content()['impedance'].z_1
                    elif size=='ext':
                        gamma = cnout.get_content()['impedance'].gamma_ext
                        z_c = cnout.get_content()['impedance'].z_c_ext
                        z_0 = cnout.get_content()['impedance'].z_0_ext
                        z_1 = cnout.get_content()['impedance'].z_1_ext
                    else:
                        print 'Error: wrong flag'
                        exit(1)
                    L = cnout.get_content()['impedance'].length
                    x = outloc['x']*L
                    
                    if changenodes[-2] in find_previous_changenode(cnout): # if true, path comes from further from soma 
                        G_stop = (np.cosh(gamma*x) / np.cosh(gamma*L)) * \
                            ((1. + z_c/z_0 * np.tanh(gamma*x)) / (1. + z_c/z_0 * np.tanh(gamma*L)))
                    elif changenodes[-2] == find_next_changenode(cnout): # else, path comes from soma
                        if z_1[0] == np.infty:
                            G_stop = np.cosh(gamma*(L-x)) / np.cosh(gamma*L)
                        else:
                            G_stop = (np.cosh(gamma*(L-x)) / np.cosh(gamma*L)) * \
                                ((1. + z_c/z_1 * np.tanh(gamma*(L-x))) / (1. + z_c/z_1 * np.tanh(gamma*L)))
                    else:
                        print 'Error: wrong output node assignment'
                        exit(1)    
                if find_next_changenode(cnin) == changenodes[1]:
                    path = changenodes[1:]
                    G = self._multiply_to_path_voltage(path, G_start, G_stop, size=size)
                else:
                    path = changenodes
                    G = self._multiply_to_path_voltage(path, G_start, G_stop, size=size)

            else:
                if cnout._index == 1:
                    if size=='normal':
                        G_stop = copy.copy(cnout.get_content()['impedance'].z_in)
                    elif size=='ext':
                        G_stop = copy.copy(cnout.get_content()['impedance'].z_in_ext)
                else:
                    if size=='normal':
                        gamma = cnout.get_content()['impedance'].gamma
                        z_c = cnout.get_content()['impedance'].z_c
                        z_0 = cnout.get_content()['impedance'].z_0
                        z_1 = cnout.get_content()['impedance'].z_1
                    elif size=='ext':
                        gamma = cnout.get_content()['impedance'].gamma_ext
                        z_c = cnout.get_content()['impedance'].z_c_ext
                        z_0 = cnout.get_content()['impedance'].z_0_ext
                        z_1 = cnout.get_content()['impedance'].z_1_ext
                    else:
                        print 'Error: wrong flag'
                        exit(1)
                    L = cnout.get_content()['impedance'].length
                    x = outloc['x']*L
                    
                    if changenodes[-2] in find_previous_changenode(cnout): # if true, path comes from further from soma 
                        if z_1[0] == np.infty:
                            G_stop = (np.cosh(gamma*x)/np.cosh(gamma*L)) * (z_c*tanh(gamma*x) + z_0) / \
                                    (1. + z_0/z_c *tanh(gamma*L))
                        else:
                            G_stop = (np.cosh(gamma*x)/np.cosh(gamma*L)) * z_1 * (tanh(gamma*x) + z_0/z_c) / \
                                        ((z_0+z_1)/z_c + (1. + (z_0/z_c)*(z_1/z_c))*tanh(gamma*L))
                    elif changenodes[-2] == find_next_changenode(cnout): # else, path comes from soma
                        if z_1[0] == np.infty:
                            G_stop = (np.cosh(gamma*(L-x))/np.cosh(gamma*L)) * z_0 / \
                                    (1. + z_0/z_c *tanh(gamma*L))
                        else:
                            G_stop = (np.cosh(gamma*(L-x))/np.cosh(gamma*L)) * z_0/z_c * (z_c*tanh(gamma*(L-x)) + z_1) / \
                                    ((z_0+z_1)/z_c + (1. + (z_0/z_c)*(z_1/z_c))*tanh(gamma*L))
                    else:
                        print 'Error: wrong output node assignment'
                        exit(1)
            
                G = G_start
                if find_next_changenode(cnin) == changenodes[1]:
                    path = changenodes[1:]
                    G = self._multiply_to_path(path, G_start, G_stop, size=size)
                else:
                    path = changenodes
                    G = self._multiply_to_path(path, G_start, G_stop, size=size)

        return G
        
        
    def _multiply_to_path(self, path, G, G_stop, size='normal'):
        node = path[0]
        if len(path) == 1:
            if size=='normal':
                G *= G_stop / path[0].get_content()['impedance'].z_in
            elif size=='ext':
                G *= G_stop / path[0].get_content()['impedance'].z_in_ext
            else:
                print 'Error: wrong output node assignment'
                exit(1)
            return G
        elif find_next_changenode(node) == path[1]:
            if size=='normal':
                G *= node.get_content()['impedance'].z_trans / node.get_content()['impedance'].z_in
            elif size=='ext':
                G *= node.get_content()['impedance'].z_trans_ext / node.get_content()['impedance'].z_in_ext
            else:
                print 'Error: wrong output node assignment'
                exit(1)
            path = path[1:]
            return self._multiply_to_path(path, G, G_stop, size=size)
        elif len(path) == 2:
            if size=='normal':
                G *= G_stop / path[0].get_content()['impedance'].z_in
            elif size=='ext':
                G *= G_stop / path[0].get_content()['impedance'].z_in_ext
            else:
                print 'Error: wrong output node assignment'
                exit(1)
            return G
        else:
            if size=='normal':
                G *= path[1].get_content()['impedance'].z_trans / node.get_content()['impedance'].z_in
            elif size=='ext':
                G *= path[1].get_content()['impedance'].z_trans_ext / node.get_content()['impedance'].z_in_ext
            else:
                print 'Error: wrong output node assignment'
                exit(1)
            path = path[1:]
            return self._multiply_to_path(path, G, G_stop, size=size)
    
            
    def _multiply_to_path_voltage(self, path, GV, GV_stop, size='normal'):
        node = path[0]
        if len(path) == 1:
            if size=='normal':
                GV *= GV_stop
            elif size=='ext':
                GV *= GV_stop
            else:
                print 'Error: wrong output node assignment'
                exit(1)
            return GV
        elif find_next_changenode(node) == path[1]:
            if size=='normal':
                GV *= node.get_content()['impedance'].GV_10
            elif size=='ext':
                GV *= node.get_content()['impedance'].GV_10_ext
            else:
                print 'Error: wrong output node assignment'
                exit(1)
            path = path[1:]
            return self._multiply_to_path_voltage(path, GV, GV_stop, size=size)
        elif len(path) == 2:
            if size=='normal':
                GV *= GV_stop
            elif size=='ext':
                GV *= GV_stop
            else:
                print 'Error: wrong output node assignment'
                exit(1)
            return GV
        else:
            if size=='normal':
                GV *= path[1].get_content()['impedance'].GV_01
            elif size=='ext':
                GV *= path[1].get_content()['impedance'].GV_01_ext
            else:
                print 'Error: wrong output node assignment'
                exit(1)
            path = path[1:]
            return self._multiply_to_path_voltage(path, GV, GV_stop, size=size)
        
        
    def set_length_and_radii(self):
        '''
        Sets the lengths and radii under ['impedance'] in the cont-dictionaries of the
        changenodes.
        '''
        nodes = self.tree.get_nodes()
        for node in nodes:
            if node._index == 1:
                cont = node.get_content()
                cont['impedance'] = dummySomaImpedances(node, nodes[1], nodes[2])
            if node._index not in [1,2,3]:
                if is_changenode(node):
                    # find next changenode and path between this and present node
                    cnode = find_next_changenode(node)
                    path = self.tree.path_between_nodes(node, cnode)
                    # calculate length and radii and save them to content
                    radius, length = get_cylinder_radius_length(path)
                    cont = node.get_content()
                    cont['impedance'] = dummyImpedances(radius, length)
                    node.set_content(cont)
        
        
    def set_impedance(self, freqs, size='normal', pprint=False, volt=True):
        '''
        Calculates the impedances throughout the tree. Initiates a \'segmentImpedance\'-
        object that is stored under the [\'impedance\'] entry in the \'self.content\'
        dictionary of the nodes in the three. If the node is soma, a \'somaImpedance\'-
        object is used. 
        
        input:
            [freqs]: Numpy array of purely imaginary numbers representing the frequences
                    at which function and impedances will be evaluated
            [size]: String, default \'normal\'. Choose \'normal\' or \'ext\'. Normal is used 
                    for normal gf-calculations, \'ext\' is used for 2d-kernels.
            [pprint]: Boolean, default False. If True prints info on screen while running
            [volt]; Boolean, default True. If True, uses voltage-based transfers to compute gf.
                    Recommended to be set on True.
        '''
        self.volt = volt
        self.freqs = freqs
        nodes = self.tree.get_nodes()
        bnodes = [node for node in nodes if len(node.get_child_nodes()) == 2]
        leafs = [node for node in nodes if self.tree.is_leaf(node) and node._index != 2 and node._index != 3]
        if pprint:
            for leaf in leafs: print 'leaf: ' + str(leaf)
        
        # recursive call for z_1
        leaf0 = leafs[0]
        leafs.remove(leaf0)
        self._impedance_from_leaf(leaf0, None, leafs, count=0, size=size, pprint=pprint)
                
        # find soma
        somanode = self.tree.get_node_with_index(1)
        somachildren = somanode.get_child_nodes()[2:]
        scnodes = somachildren
        for ind, somachild in enumerate(somachildren):
            if not is_changenode(somachild):
                scnodes[ind] = find_previous_changenode(somachild)[0]
        
        # impedance of children from soma and other nodes
        for scnode in scnodes:
            nodes = list(set(scnodes) - set([scnode]))
            
            cont = scnode.get_content()

            z_soma = somanode.get_content()['impedance'].soma_impedance(nodes)
            if size=='ext':
                z_soma_ext = somanode.get_content()['impedance'].soma_impedance(nodes, size=size)
                cont['impedance'].set_impedance_0_soma(z_soma, z_soma_ext)
            else:
                cont['impedance'].set_impedance_0_soma(z_soma)
            if self.volt:
                cont['impedance'].set_voltagetransfers() 
            else:
                cont['impedance'].set_greensfunctions() 
            scnode.set_content(cont)
            
            # recursive call
            nodes = find_previous_changenode(scnode)
            self._impedance_from_soma(nodes, pprint=pprint)
                
                
    def _impedance_from_leaf(self, leafnode, prevleaf, leafs, count=0, size='normal', pprint=False):
        # print count
        if pprint:
            print 'Forward sweep: ' + str(leafnode)
        if leafnode._parent_node != None:
            changenode = find_next_changenode(leafnode)
            if pprint and leafnode._index == 1:
                print 'soma'
            if len(leafnode.get_child_nodes()) > 1:
                if  'impedance' in leafnode.get_content().keys():
                    cont = leafnode.get_content()
                    cont['impedance'].set_impedance_1(find_previous_changenode(leafnode))
                    leafnode.set_content(cont)
                    self._impedance_from_leaf(changenode, leafnode, leafs, count=count+1, size=size)
                else:
                    path = self.tree.path_between_nodes(leafnode, changenode)
                    radius, length = get_cylinder_radius_length(path)
                    cont = leafnode.get_content()
                    cont['impedance'] = segmentImpedances(self.freqs, radius, length) # impedance
                    cont['impedance'].set_impedances(cont['physiology'].E0, cont['physiology'].gs, cont['physiology'].es, 
                                                    cont['physiology'].gcalctype, cont['physiology'].cm, cont['physiology'].r_a)
                    if size=='ext':
                        cont['impedance'].set_extended_impedances(cont['physiology'].E0, cont['physiology'].gs, cont['physiology'].es, 
                                                                    cont['physiology'].gcalctype, cont['physiology'].r_a)
                    leafnode.set_content(cont)
                    
                    cnode = find_other_changenode(leafnode, prevleaf)
                    if cnode.get_child_nodes():
                        sub_tree = self.tree.get_sub_tree(cnode)
                        st_nodes = sub_tree.get_nodes()
                        st_leafs = [node for node in st_nodes if sub_tree.is_leaf(node)]
                        leaf0 = st_leafs[0]
                    else:
                        leaf0 = cnode
                    leafs.remove(leaf0)
                    self._impedance_from_leaf(leaf0, leafnode, leafs, count=count+1, size=size)
            else:
                path = self.tree.path_between_nodes(leafnode, changenode)
                radius, length = get_cylinder_radius_length(path)
                cont = leafnode.get_content()
                cont['impedance'] = segmentImpedances(self.freqs, radius, length) # impedance (convert radius, length to cm)
                cont['impedance'].set_impedances(cont['physiology'].E0, cont['physiology'].gs, cont['physiology'].es, 
                                                cont['physiology'].gcalctype, cont['physiology'].cm, cont['physiology'].r_a)
                if size=='ext':
                    cont['impedance'].set_extended_impedances(cont['physiology'].E0, cont['physiology'].gs, cont['physiology'].es, 
                                                            cont['physiology'].gcalctype, cont['physiology'].cm, cont['physiology'].r_a)
                cont['impedance'].set_impedance_1(find_previous_changenode(leafnode))
                leafnode.set_content(cont)
                self._impedance_from_leaf(changenode, leafnode, leafs, count=count+1, size=size)
        else:
            # if leafnode is soma, start in other three
            if leafs:
                leaf0 = leafs[0]
                leafs.remove(leaf0)
                self._impedance_from_leaf(leaf0, prevleaf, leafs, count=count+1, size=size)
            else:
                cont = leafnode.get_content()
                cont['impedance'] = somaImpedance(self.freqs, leafnode, self.tree.get_node_with_index(2), self.tree.get_node_with_index(3), 
                                                    cont['physiology'].E0, cont['physiology'].gs, cont['physiology'].es, 
                                                    cont['physiology'].gcalctype, cont['physiology'].cm)
                if size=='ext':
                    cont['impedance'].set_extended_impedance(leafnode, cont['physiology'].E0, cont['physiology'].gs, cont['physiology'].es, 
                                                            cont['physiology'].gcalctype, cont['physiology'].cm)
                leafnode.set_content(cont)

    def _impedance_from_soma(self, nodes, pprint=False):
        if nodes:
            for node in nodes:
                if pprint and not node.get_child_nodes(): print 'Backward sweep: ' + str(node)
                pnode = find_next_changenode(node)
                cnode = find_other_changenode(pnode, node)
                cont = node.get_content()
                cont['impedance'].set_impedance_0(pnode, cnode)
                if self.volt:
                    cont['impedance'].set_voltagetransfers() 
                else:
                    cont['impedance'].set_greensfunctions() 
                node.set_content(cont)
                # recursive call
                cnodes = find_previous_changenode(node)
                self._impedance_from_soma(cnodes)
    
    def delete_impedances(self):
        nodes = self.tree.get_nodes()
        cnodes = [node for node in nodes if is_changenode(node)]
        for node in cnodes:
            cont = node.get_content()
            if 'impedance' in cont.keys():
                del cont['impedance']
            node.set_content(cont)
    
    
    def calc_quadratic_coeff(self, node='all'):
        cnodes = self.get_changenodes()
        for node in cnodes:
            print 'Quadratic coefficient for: ' + str(node)
            cont = node.get_content()
            cont['impedance'].set_quadratic_coeff(cont['physiology'].E0, cont['physiology'].gs, cont['physiology'].es)
            
    def calc_quadratic_integrals(self, node='all'):
        cnodes = self.get_changenodes()
        for node in cnodes[1:]:
            print 'Quadratic integral for: ' + str(node)
            cont = node.get_content()
            cont['impedance'].set_quadratic_integrals()
            
    def get_nearest_neighbours(self, inlocs, reduced=True, test=False, add_leaves=True, separate=True):
        '''
        For each inloc in inlocs, returns a list of nearest neighbours of that
        inloc that are farther away from the soma.
        
        - input:
            - [inlocs]: list of dictionnaries specifying the specific inlocs
            - [reduced]: boolean, if True, returns the neighbours with reference
                to the changenodes. If False, returns the neighbours with the original 
                node specification
            - [test]: boolean, if True, print the lists of neighbours
            - [add_leaves]: boolean, if True, adds the leafs as neighbours of 
                inlocs that are the furthest away from the soma. If False,
                these inlocs have no neighbours [empty lists].
            - [seperate]: boolean, if True, separates sets of neighbours even
                if parent inloc is not as the end of the branch. If False, those
                are returned as one set.
        - output:
            - [neighbours]: list with at each entry the neighbours of the corresponding
                entry in [inlocs], in original or reduced format
            - [inlocs]: the inlocs, in original or reduced format
        '''
        nodes = [self.tree.get_node_with_index(inloc['node']) for inloc in inlocs]
        aux_x = np.array([inloc['x'] for inloc in inlocs])
        # find nearest neighbour nodes
        nearest_neighbours = []
        for ind, inloc in enumerate(inlocs):
            if nodes.count(nodes[ind]) > 1:
                # if more than one inloc in node, find the one with next
                # highest 'x', or if 'x' is highest, the nearest neigbours
                # of that node
                indnodes = [i for i in range(len(nodes)) if nodes[i]==nodes[ind]]
                inl = aux_x[indnodes]
                rind = np.where(inl==aux_x[ind])[0][0]
                indsort = np.argsort(inl)
                if indsort[-1] == rind: 
                    nearest_neighbours.append(
                            self._find_nearest_inlocs(ind, inlocs, nodes, add_leaves=add_leaves, separate=separate))
                else:
                    rrind = np.where(indsort==rind)[0][0]
                    nearest_neighbours.append([[inlocs[indnodes[indsort[rrind+1]]]]])
            else: 
                nearest_neighbours.append(
                    self._find_nearest_inlocs(ind, inlocs, nodes, add_leaves=add_leaves, separate=separate))
        if not reduced:
            if test:
                for ind, inloc in enumerate(inlocs):
                    print 'inloc: ', inloc
                    print 'nearest neighbours: ', nearest_neighbours[ind]
            return nearest_neighbours, inlocs
        else:
            # get reduced locations
            reduced_nodes = []
            # reduced_neighbours has to have the right nested structure
            reduced_neighbours = [[[]] for _ in nodes] 
            for ind, neighbours in enumerate(nearest_neighbours):
                for ind1, neighbour in enumerate(neighbours):
                    if ind1 > 0:
                        # reduced_neighbours has to have the right nested structure
                        reduced_neighbours[ind].append([])
                    for neighbourloc in neighbour:
                        n_node = self.tree.get_node_with_index(neighbourloc['node'])
                        if not is_changenode(n_node):
                            cnode = find_previous_changenode(n_node)[0]
                        else:
                            cnode = n_node
                        path = self.tree.path_between_nodes(cnode, n_node)
                        reduced_neighbours[ind][ind1].append(get_reduced_loc(copy.copy(neighbourloc), path))
                if nodes[ind]._index != 1:
                    if not is_changenode(nodes[ind]):
                        cnode = find_previous_changenode(nodes[ind])[0]
                    else:
                        cnode = nodes[ind]
                    path = self.tree.path_between_nodes(cnode, nodes[ind])
                    reduced_nodes.append(get_reduced_loc(copy.copy(inlocs[ind]), path))
                else:
                    reduced_nodes.append(inlocs[ind])
            if test:
                for ind, rnode in enumerate(reduced_nodes):
                    print 'inloc: ', rnode
                    print 'nearest neighbours: ', reduced_neighbours[ind]
            return reduced_neighbours, reduced_nodes


    def _find_nearest_inlocs(self, index, inlocs, nodes, add_leaves=True, separate=True):
        nearest_nodes = self.tree.find_nearest_neighbours(nodes[index], nodes)
        if separate or np.abs(inlocs[index]['x']-1.) < 1e-2:
            locs = [[] for _ in nearest_nodes]
        else:
            locs = [[]]
        if len(nearest_nodes) > 0:
            for ind0 in range(len(nearest_nodes)):
                for ind, node in enumerate(nearest_nodes[ind0]):
                    inlocs_node = [inloc for inloc in inlocs if inloc['node'] == node._index]
                    #~ print 'ind: ', ind, ', inlocs: ', inlocs_node
                    if not inlocs_node: 
                    # leaf node has to be inserted 
                        if add_leaves:
                            locs[ind0].append({'node': node._index, 'x': 1.})
                        else: pass
                    elif len(inlocs_node) > 1: 
                        # we have to take the inloc closes to start of segment
                        inlx = np.array([inloc_node['x'] for inloc_node in inlocs_node])
                        indsort = np.argsort(inlx)
                        if separate or np.abs(inlocs[index]['x']-1.) < 1e-2:
                            locs[ind0].append(inlocs_node[indsort[0]])
                        else:
                            locs[0].append(inlocs_node[indsort[0]])
                    else:
                        # we can just use this inloc
                        if separate or np.abs(inlocs[index]['x']-1.) < 1e-2:
                            locs[ind0].append(inlocs_node[0])
                        else:
                            locs[0].append(inlocs_node[0])
        else: # leaf node has to be inserted
            if add_leaves:
                locs.append([{'node': nodes[index]._index, 'x': 1.}])
            else: pass
        return locs
    
    def calc_IP_conductances(self, inlocs, external_tree=None, test=False, pprint=False):
        '''
        Calculates the rescaled conductances of continuously distributed
        currents in the dendritic tree.

        !!! Bad function, needs to be rewritten !!!
        
        - input: 
            - inlocs: list of dictionnaries specifying inlocs
            - external_tree: greensTree object with same underlying tree but different
                conductances
            - test: boolean, returns extra arrays for testing purposes
            - pprint: boolean, print test results
        
        - output:
            - [gs_inloc]: dictionnary, indexed by inloc['ID'], of dictionnaries 
                of conductances at the different inlocs (uS)
            - [es_inloc]: dictionnary  indexed by inloc['ID'], of dictionnaries 
                of reversal potentials at different inlocs (mV). Set in a generic
                fashion
        '''
        # auxiliary variable for testing purposes
        self.passlentest = None

        rneighbours, rinlocs = self.get_nearest_neighbours(inlocs, reduced=True)
        # make dictionnary for conductances of integration points
        snode = self.tree.get_node_with_index(1)
        somaA = snode.get_content()['impedance'].somaA
        gsoma = copy.copy(snode.get_content()['physiology'].gs)
        gs_inloc = {inloc['ID']: {} for inloc in inlocs}
        es_inloc = {inloc['ID']: snode.get_content()['physiology'].es for inloc in inlocs}
        gcalctype_inloc = {inloc['ID']: snode.get_content()['physiology'].gcalctype for inloc in inlocs}        
        # es_inloc = {inloc['ID']: {} for inloc in inlocs}
        # gcalctype_inloc = {inloc['ID']: {} for inloc in inlocs}
        for inloc in inlocs:
            for key in gsoma.keys():
                gs_inloc[inloc['ID']][key] = 0.
                # directly update soma conductance
                if inloc['node'] == 1:
                    gs_inloc[inloc['ID']][key] += gsoma[key] * somaA
        # set the distances
        for ind, inloc in enumerate(rinlocs):
            node = self.tree.get_node_with_index(inloc['node'])
            cont = node.get_content()
            imp = cont['impedance']
            if node._index == 1: 
                # if node is soma, L and x1 are 0 by definition
                L = 0.; x1 = 0.
                cont.update({'L': L, 'x1': x1, 'L2': {}})
                node.set_content(cont)
                cnodes = node.get_child_nodes()
                for cnode in cnodes[2:]:
                    if not is_changenode(cnode):
                        cnode = find_previous_changenode(cnode)[0]
                    self._set_lengths_up(x1, cnode, inloc, rneighbours[ind][0], node._index, external_tree=external_tree)
                    self._calc_integrals_in_subtree(cnode, inloc, rneighbours[ind][0], gs_inloc, external_tree=external_tree, test=test, pprint=pprint)
            elif len(rneighbours[ind]) == 1:
                # if only one subtree starts at node, we just set the lengths in that one
                if rneighbours[ind][0][0]['node'] == inloc['node']:
                    # get the relevant gamma constant
                    if external_tree == None:
                        gamma = imp.gamma[len(imp.gamma)/2].real
                    else:
                        node_ext = external_tree.tree.get_node_with_index(inloc['node'])
                        gamma = node_ext.get_content()['impedance'].gamma
                        gamma = gamma[len(gamma)/2].real
                    # if nearest neighbour is on the same node,
                    # only one nearest neighbour exists. We can just take the distance 
                    # to that one. The distance is saved in the node contents in a 
                    # dictionnary with the 'ID' as key (as opposed to node index for L1 and L2)
                    if not find_previous_changenode(node) and rneighbours[ind][0][0]['x'] == 1:
                        # but not if nearest node is a leaf and not a real inloc
                        # however, we need to set the simple integral
                        cont['L'] = imp.length * gamma
                        cont['x2'] = inloc['x'] * imp.length * gamma
                        self._calc_simple_integral(node, inloc, None, gs_inloc, external_tree=external_tree, test=test, pprint=pprint)
                    else:
                        D = (rneighbours[ind][0][0]['x'] - inloc['x']) * imp.length * gamma
                        if 'deltax' not in cont.keys():
                            cont['deltax'] = {}
                        cont['deltax'].update({inloc['ID']: D})
                        # we need to set the simple integral
                        self._calc_simple_integral(node, inloc, rneighbours[ind][0][0], gs_inloc, external_tree=external_tree, test=test, pprint=pprint)
                else:
                    # get the relevant gamma constant
                    if external_tree == None:
                        gamma = imp.gamma[len(imp.gamma)/2].real
                    else:
                        node_ext = external_tree.tree.get_node_with_index(inloc['node'])
                        gamma = node_ext.get_content()['impedance'].gamma
                        gamma = gamma[len(gamma)/2].real    
                    # nearest neighbour(s) is (are) on another node, by consequence
                    # we have to look for the distances trough the tree structure
                    node = self.tree.get_node_with_index(inloc['node'])
                    cont = node.get_content()
                    L = imp.length * gamma
                    x1 = imp.length * (1. - inloc['x']) * gamma
                    cont.update({'L': L, 'x1': x1, 'L2': {}})
                    node.set_content(cont)
                    childnodes = find_previous_changenode(node)
                    for cnode in childnodes:
                        self._set_lengths_up(x1, cnode, inloc, rneighbours[ind][0], node._index, external_tree=external_tree)
                    self._set_length_up_invert(node, rneighbours[ind][0])
                    self._calc_integrals_in_subtree(node, inloc, rneighbours[ind][0], gs_inloc, external_tree=external_tree, test=test, pprint=pprint)
            elif len(rneighbours[ind]) > 1:
                # get the relevant gamma constant
                if external_tree == None:
                    gamma = imp.gamma[len(imp.gamma)/2].real
                else:
                    node_ext = external_tree.tree.get_node_with_index(inloc['node'])
                    gamma = node_ext.get_content()['impedance'].gamma
                    gamma = gamma[len(gamma)/2].real    
                # if node has more than one subtree, we need to treat both
                # separately if 'x'==1, but we need to merge them if 'x'!=1
                if inloc['x'] == 1.:
                    L = imp.length * gamma  
                    x1 = 0.
                    cont.update({'L': L, 'x1': x1})
                    node.set_content(cont)
                    childnodes = find_previous_changenode(node)
                    for ind2, cnode in enumerate(childnodes):
                        self._set_lengths_up(x1, cnode, inloc, rneighbours[ind][ind2], node._index, external_tree=external_tree)
                        self._calc_integrals_in_subtree(node, inloc, rneighbours[ind][ind2], gs_inloc, external_tree=external_tree, test=test, pprint=pprint)
                else:
                    rn = []
                    for neighbour in rneighbours[ind]:
                        for n in neighbour:
                            rn.append(n)
                    L = imp.length*gamma
                    x1 = (1. - inloc['x']) * imp.length * gamma
                    cont.update({'L': L, 'x1': x1, 'L2': {}})
                    node.set_content(cont)
                    childnodes = find_previous_changenode(node)
                    for cnode in childnodes:
                        self._set_lengths_up(x1, cnode, inloc, rn, node._index, external_tree=external_tree)
                    self._set_length_up_invert(node, rn)
                    self._calc_integrals_in_subtree(node, inloc, rn, gs_inloc, external_tree=external_tree, test=test, pprint=pprint)
            else:
                raise Exception("Wrong neighour assignment")
        # set es and gcalctype
        # for inloc in inlocs:
        #     node = self.tree.get_node_with_index(inloc['node'])
        #     phys = node.get_content()['physiology']
        #     for key in 
        #     es_inloc[inloc['ID']] = copy.deepcopy(phys.gs)
        #     gcalctype_inloc[inloc['ID']] = copy.deepcopy(phys.gcalctype)
        if test:
            nodes = [node for node in self.tree.get_nodes() if is_changenode(node)]
            gtotal = {key: 0. for key in gsoma.keys()}
            for node in nodes:
                gnode = node.get_content()['physiology'].gs
                for key in gtotal.keys():
                    if node._index != 1:
                        length = node.get_content()['impedance'].length
                        radius = node.get_content()['impedance'].radius
                        gtotal[key] += gnode[key] * length * 2. * math.pi * radius
                    else:
                        gtotal[key] += gnode[key] * somaA
            gtotalpoint = {key: 0. for key in gsoma.keys()}
            for inloc in inlocs:
                for key in gtotalpoint.keys():
                    gtotalpoint[key] += gs_inloc[inloc['ID']][key]
            if pprint:
                print 'Total conductances of original neuron model:'
                print gtotal
                print 'Total conductances of the integration points:'
                print gtotalpoint
                for key in gs_inloc.keys():
                    print 'conductances of integration point ', key, ':'
                    print gs_inloc[key]
            return gs_inloc, es_inloc, gcalctype_inloc, gtotal, gtotalpoint, self.passlentest
        else:
            return gs_inloc, es_inloc, gcalctype_inloc
    
    
    def _calc_integrals_in_subtree(self, node, inloc, rneighbours, gs, external_tree=None, test=False, pprint=False):
        cont = node.get_content()
        if test:
            print node
        # some variable assigments for notational clarity
        nn = [rn['node'] for rn in rneighbours]
        nx = [rn['x'] for rn in rneighbours]
        L2 = cont['L2'] # dictionary with node indices
        L1 = cont['L1'] # dictionary with node indices
        L = cont['L'] # float
        if 'x1' in cont.keys():
            x1 = cont['x1'] # float
        if 'x2' in cont.keys():
            x2 = cont['x2'] # float
        imp = cont['impedance']
        nodegs = cont['physiology'].gs
        factor = 2.* math.pi * cont['impedance'].radius
        # get the relevant gamma constant
        if external_tree == None:
            gamma = imp.gamma[len(imp.gamma)/2].real
        else:
            node_ext = external_tree.tree.get_node_with_index(node._index)
            gamma = node_ext.get_content()['impedance'].gamma
            gamma = gamma[len(gamma)/2].real
            
        # main part of function
        if node._index == inloc['node']:
            # if inloc is in segment, first form of integral is used and recursion
            # is started
            # calculate integral of integration point on current node
            a1 = 1
            b1 = np.sum(np.array([np.exp(cont['x1'] - cont['L2'][key] - 2*cont['L']) for key in cont['L2'].keys()]))
            I1 = np.log((a1 * np.exp(2.*(cont['x1']-cont['L'])) + b1) / (a1 * np.exp(-2*cont['L']) + b1)) / (2.*a1)
            # update conductances of first integration point
            for keyg in nodegs.keys():
                gs[inloc['ID']][keyg] += I1 * nodegs[keyg] * factor / gamma
            # calculate integrals of other integration points
            I = {}
            for key in cont['L2'].keys():
                a = np.sum(np.array([np.exp(cont['L2'][key] - cont['L2'][key2]) for key2 in cont['L2'].keys()]))
                b = np.exp(2.*cont['L'] + cont['L2'][key] - cont['x1'])
                I[key] = np.log((a * np.exp(2*cont['L']) + b) / (a * np.exp(2.*(cont['L'] - cont['x1'])) + b)) / (2.*a)
                # update conductances of other integration points
                rnID = [rn['ID'] for rn in rneighbours if key==rn['node']][0]
                for keyg in nodegs.keys():
                    gs[rnID][keyg] += I[key] * nodegs[keyg] * factor / gamma
            if test:
                Isum = np.sum(np.array([I[key] for key in I.keys()])) + I1
                if np.abs(Isum - cont['x1']) > 0.01: self.passlentest = node._index
                if pprint:
                    print 'Sum of integrals: %.8f, length fraction: %.8f' % (Isum, cont['x1'])
            childnodes = find_previous_changenode(node)
            for cnode in childnodes:
                self._calc_integrals_in_subtree(cnode, inloc, rneighbours, gs, external_tree=external_tree, test=test, pprint=pprint)
        
        elif node._index in nn:
            # if we have arrived at a neighbour node, we have to check whether
            # it's a dummy leaf or not
            indx = [i for i in range(len(nn)) if node._index == nn[i]][0]
            xcur = nx[indx]
            if (not self.tree.is_leaf(node)) or (xcur != 1):
                # node is not a dummy leaf, so we use final form of integral.
                # recursion is not continued
                a = 1
                b = np.sum(np.array([np.exp(x2 - L1[key]) for key in L1.keys()]))
                I2 = np.log((a * np.exp(2.*x2) + b) / (a + b)) / (2.*a)
                # update conductances of integration point on node
                rnID = [rn['ID'] for rn in rneighbours if node._index==rn['node']][0]
                for keyg in nodegs.keys():
                    gs[rnID][keyg] += I2 * nodegs[keyg] * factor / gamma 
                I = {}
                for key in L1.keys():
                    a = np.sum(np.array([np.exp(L1[key] - L1[key2]) for key2 in L1.keys()]))
                    b = np.exp(L1[key] - x2)
                    I[key] = np.log((a + b) / (a * np.exp(-2*x2) + b)) / (2.*a)
                    # update conductances of other integration points
                    if key==inloc['node']:
                        ID = inloc['ID']
                    else:
                        ID = [rn['ID'] for rn in rneighbours if key==rn['node']][0]
                    for keyg in nodegs.keys():
                        gs[ID][keyg] += I[key] * nodegs[keyg] * factor / gamma
                if test:
                    Isum = np.sum(np.array([I[key] for key in I.keys()])) + I2
                    if np.abs(Isum - x2) > 0.01: self.passlentest = node._index
                    if pprint:
                        print 'Sum of integrals: %.8f, length final segment: %.8f' % (Isum, x2)
            
            else:
                # node is a dummy leaf, so we use other form of integral.
                # recursion is not continued
                I={}
                for key in L1.keys():
                    a = np.sum(np.array([np.exp(L1[key] - L1[key2]) for key2 in L1.keys()]))
                    b = np.sum(np.array([np.exp(L1[key] - L - L2[key2]) for key2 in L2.keys()]))
                    I[key] = np.log((a + b) / (a * np.exp(-2.*L) + b)) / (2.*a)
                    # update conductances of other integration points
                    if key==inloc['node']:
                        ID = inloc['ID']
                    else:
                        ID = [rn['ID'] for rn in rneighbours if key==rn['node']][0]
                    for keyg in nodegs.keys():
                        gs[ID][keyg] += I[key] * nodegs[keyg] * factor / gamma
                if test:
                    Isum = np.sum(np.array([I[key] for key in I.keys()]))
                    if np.abs(Isum - L) > 0.01: self.passlentest = node._index
                    if pprint:
                        print 'Sum of integrals: %.8f, length segment: %.8f' % (Isum, L)
        else:
            # node is a central node, without any inlocs, so central form of 
            # integral is used. Recursion is continued
            I={}
            for key in L1.keys():
                a = np.sum(np.array([np.exp(L1[key] - L1[key2]) for key2 in L1.keys()]))
                b = np.sum(np.array([np.exp(L1[key] - L - L2[key2]) for key2 in L2.keys()]))
                I[key] = np.log((a + b) / (a * np.exp(-2.*L) + b)) / (2.*a)
                # update conductances of other integration points
                if key==inloc['node']:
                    ID = inloc['ID']
                else:
                    ID = [rn['ID'] for rn in rneighbours if key==rn['node']][0]
                for keyg in nodegs.keys():
                    gs[ID][keyg] += I[key] * nodegs[keyg] * factor / gamma
            for key in L2.keys():
                a = np.sum(np.array([np.exp(L2[key] - L2[key2]) for key2 in L2.keys()]))
                b = np.sum(np.array([np.exp(L + L2[key] - L1[key2]) for key2 in L1.keys()]))
                I[key] = np.log((a * np.exp(2.*L) + b) / (a + b)) / (2.*a)
                # update conductances of other integration points
                ID = [rn['ID'] for rn in rneighbours if key==rn['node']][0]
                for keyg in nodegs.keys():
                    gs[ID][keyg] += I[key] * nodegs[keyg] * factor / gamma
            if test:
                Isum = np.sum(np.array([I[key] for key in I.keys()]))
                if np.abs(Isum - L) > 0.01: self.passlentest = node._index
                if pprint:
                    print 'Sum of integrals: %.8f, length segment: %.8f' % (Isum, L)
            childnodes = find_previous_changenode(node)
            for cnode in childnodes:
                self._calc_integrals_in_subtree(cnode, inloc, rneighbours, gs, external_tree=external_tree, test=test, pprint=pprint)
                
    def _calc_simple_integral(self, node, inloc, neighbour, gs, external_tree=None, test=True, pprint=False):
        cont = node.get_content()
        imp = cont['impedance']
        #~ gamma = imp.gamma[len(imp.gamma)/2].real
        factor = 2. * math.pi * imp.radius 
        nodegs = cont['physiology'].gs
        # get the relevant gamma constant
        if external_tree == None:
            gamma = imp.gamma[len(imp.gamma)/2].real
        else:
            node_ext = external_tree.tree.get_node_with_index(node._index)
            gamma = node_ext.get_content()['impedance'].gamma
            gamma = gamma[len(gamma)/2].real
        if pprint:
            print node
        if neighbour == None:
            # calculate integral
            I = cont['L'] - cont['x2']
            # set conductance of integration point
            for keyg in nodegs.keys():
                gs[inloc['ID']][keyg] += I * nodegs[keyg] * factor / gamma
            if test:
                if np.abs(I - I) > 0.01: self.passlentest = node._index
                if pprint:
                    print 'Sum of integrals: %.8f, length segment: %.8f' % (I, I)
        else:
            # calculate integrals
            I = {inloc['ID']: cont['deltax'][inloc['ID']]/2.}
            I[neighbour['ID']] = cont['deltax'][inloc['ID']]/2.
            # set conductances of integration points
            for keyg in nodegs.keys():
                gs[inloc['ID']][keyg] += I[inloc['ID']] * nodegs[keyg] * factor / gamma
                gs[neighbour['ID']][keyg] += I[neighbour['ID']] * nodegs[keyg] * factor / gamma
            if test:
                Isum = np.sum(np.array([I[k] for k in I.keys()]))
                if np.abs(Isum - cont['deltax'][inloc['ID']]) > 0.01: self.passlentest = node._index
                if pprint:
                    print 'Sum of integrals: %.8f, length segment: %.8f' % (Isum, cont['deltax'][inloc['ID']])
        
        
    def _set_lengths_up(self, lower_length, node, rinloc, rneighbours, root_index, external_tree=None):
        # set L1, the distance to root from bottom of segment
        #~ print node
        pnode = find_next_changenode(node)
        cont = node.get_content()
        imp = cont['impedance']
        #~ gamma = imp.gamma[len(imp.gamma)/2].real
        # get the relevant gamma constant
        if external_tree == None:
            gamma = imp.gamma[len(imp.gamma)/2].real
        else:
            node_ext = external_tree.tree.get_node_with_index(node._index)
            gamma = node_ext.get_content()['impedance'].gamma
            gamma = gamma[len(gamma)/2].real
        L = imp.length*gamma
        L1 = lower_length
        cont.update({'L': L, 'L1': {root_index: L1}, 'L2': {}})
        node.set_content(cont)
        # set distance from bottom of segment to inloc if inloc in segment
        # and start to go down if node is inloc or leaf, otherwise go up
        if node._index in [rn['node'] for rn in rneighbours]:
            for ind in range(len(rneighbours)):
                if rneighbours[ind]['node']==node._index:
                    break
            if not find_previous_changenode(node) and rneighbours[ind]['x']==1.:
                # if node is leaf changenode and 'x'==1, node is not a real
                # inloc
                pass
            else:
                x2 = rneighbours[ind]['x'] * imp.length * gamma
                cont.update({'x2': x2})
                node.set_content(cont)
                pnode = find_next_changenode(node)
                self._set_length_down(x2, pnode, rinloc, rneighbours[ind]['node'])
        else:
            childnodes = find_previous_changenode(node)
            for cnode in childnodes:
                self._set_lengths_up(L1+L, cnode, rinloc, rneighbours, root_index, external_tree=external_tree)
            self._set_length_up_invert(node, rneighbours)
    
    
    def _set_length_down(self, upper_length, node, rinloc, leaf_index):
        # we have to go down to set the length from the higher nearest neighbours
        # in the subtree, untill we reach the root
        cont = node.get_content()
        cont['L2'][leaf_index] = upper_length
        node.set_content(cont)
        if node._index != 1 and node._index != rinloc['node']:
            pnode = find_next_changenode(node)
            self._set_length_down(upper_length + cont['L'], pnode, rinloc, leaf_index)
                    
                    
    def _set_length_up_final(self, node, rneighbours):
        childnodes = find_previous_changenode(node)
        if childnodes:
            # if node is not a leaf, we need to set the lengths 'L1' (lower lengths)
            # of it's children if they not already have been set
            for cnode in childnodes:
                cont = node.get_content()
                ccont = cnode.get_content()
                for key in (set(cont['L1'].keys()) - set(ccont['L1'].keys())):
                    ccont['L1'][key] = cont['L1'][key] + cont['L']
                cnode.set_content(ccont)
                if cnode._index not in [rn['node'] for rn in rneighbours]:
                    # if cnode is a nearest neighbour of the root node, the 
                    # recursion can be stopped
                    self._set_length_up_final(cnode, rneighbours)
            
            
    def _set_length_up_invert(self, node, rneighbours):
        childnodes = find_previous_changenode(node)
        if len(childnodes) > 1:
            # if more than one childnode, we have to go up again after recursion 
            # to set the other childnodes
            for cnode in childnodes:
                ocnode = find_other_changenode(node, cnode)
                ccont = cnode.get_content()
                occont = ocnode.get_content()
                ccont['L1'].update(occont['L2'].copy())
                if occont.has_key('x2'):
                    ccont['L1'].update({ocnode._index: occont['x2']})
                cnode.set_content(ccont)
                if cnode._index not in [rn['node'] for rn in rneighbours]:
                    self._set_length_up_final(cnode, rneighbours)


    def get_conc_mechs(self, inlocs):
        '''
        Returns the concentration dynamics mechanisms at each inloc

        intput:
            [inlocs]: dictionnary with inlocs

        output:
            [conc_mechs]: dictionnary, keys are inloc IDs and elements are dictionnaries
                    of concentration mechanisms for each ion
        '''
        conc_mechs = {}
        for inloc in inlocs:
            node = self.tree.get_node_with_index(inloc['node'])
            if not is_changenode(node):
                node = find_previous_changenode(node)[0]
            conc_mechs[inloc['ID']] = node.get_content()['physiology'].conc_mechs
            for ion in conc_mechs[inloc['ID']].keys():
                conc_mechs[inloc['ID']][ion]['conc0'] = node.get_content()['physiology'].C0[ion]
        return conc_mechs


    def distribute_inlocs(self, num=10, distrtype='random', radius=0.001, pprint=False, split_radius=0.0020, frac=0.6, type='all'):
        '''
        Returns a list of input locations for a tree according to a specified distribution type
        
        input:
            - [num]: int, number of inputs
            - [distrtype]: string, type of distribution
                'random': randomly distributed, but with a minimal spacing
                'uniform': uniformly distributed with a given spacing (here
                        [num] is irrelevant)
                'single': one inloc on each branch at a given distance
                'nosplit': one inloc on each main branch (a branch that splits 
                        closer to soma than split_radius)
                'hines': distributed so that the maximal number of nearest neighbours
                        of an inloc is 2
            - [radius]: float, minimal or given distance between input locations (cm)
            - [type]: if 'all', both apical and basal inlocs, 3 for basal, 4 for apical
        
        output:
            - [inlocs]: list of dictionnaries representing inlocs.
        '''
        nodes = self.tree.get_nodes()[3:]
        if type == 3 or type == 4:
            nodes = [n for n in nodes if n.get_content()['p3d'].type == type]
        inlocs = [{'ID': 0, 'node': 1, 'x': 0.}]
        if distrtype == 'random':
            for i in range(num):
                nodes_left = [n._index for n in nodes if 'tag' not in n.get_content().keys()]
                if len(nodes_left) < 1:
                    break
                ind = np.random.randint(len(nodes_left))
                x = np.random.random()
                inlocs.append({'ID': i+1, 'node': nodes_left[ind], 'x': x})
                node = self.tree.get_node_with_index(nodes_left[ind])
                self._tag_nodes_up(node, node, radius=radius)
                self._tag_nodes_down(node, node, radius=radius)
                if pprint: print 'added node: ', node
            self.remove_tags()
        elif distrtype == 'uniform':
            node = self.tree.get_node_with_index(1)
            self._add_inlocs_up(node, 0., inlocs, radius=radius, type=type)
        elif distrtype == 'single':
            node = self.tree.get_node_with_index(1)
            self._add_inlocs_up(node, 0., inlocs, radius=radius, type=type, stop_if_found=True)
        elif distrtype == 'nosplit':
            node = self.tree.get_node_with_index(1)
            self._add_inlocs_up(node, 0., inlocs, radius=radius, type=type, stop_if_found=True, split_radius=split_radius, loc_at_branch=True)
        elif distrtype == 'fromleaf':
            inlocs_2 = copy.deepcopy(inlocs)
            L_to_leaf = []
            node = self.tree.get_node_with_index(1)
            self._count_length_up(node, 0., L_to_leaf)
            L_to_leaf_arr = np.array(L_to_leaf)
            sortind = np.argsort(L_to_leaf_arr[:,1])
            L_to_leaf_arr = L_to_leaf_arr[sortind][::-1]
            for tup in L_to_leaf_arr[:num]:
                if tup[1] > 0.0070:
                    Ls = (0.0070 + (tup[1]-0.0070)*np.random.rand(5))
                    Ls = tup[1] - np.sort(Ls)[::-1]
                    L0 = np.mean(Ls)
                    self._add_loc_from_leaf(self.tree.get_node_with_index(tup[0]), 0., Ls, inlocs)
                    IDs = [inloc['ID'] for inloc in inlocs[-5:]]
                    self._add_loc_from_leaf(self.tree.get_node_with_index(tup[0]), 0., np.array([L0]), inlocs_2)
                    inlocs_2[-1]['IDs'] = IDs
            inlocs = (inlocs, inlocs_2)
        elif distrtype == 'hines':
            inlocs = [{'ID': 0, 'node': 1, 'x': 1.}]
            ID = 0
            for i in range(num):
                ID +=1
                nodes_left = [n._index for n in nodes if 'tag' not in n.get_content().keys()]
                if len(nodes_left) < 1:
                    break
                ind = np.random.randint(len(nodes_left))
                x = np.random.random()
                inlocs.append({'ID': ID, 'node': nodes_left[ind], 'x': x})
                node = self.tree.get_node_with_index(nodes_left[ind])
                self._tag_nodes_up(node, node, radius=radius)
                self._tag_nodes_down(node, node, radius=radius)
                NNs, _ = self.get_nearest_neighbours(inlocs, add_leaves=False, separate=False, test=False)
                N_nh_inds = self._check_hines_criterion(NNs)
                if N_nh_inds[0] != -1:
                    ID += 1
                    snode = self.tree.get_node_with_index(1)
                    Nnode1 = self.tree.get_node_with_index(NNs[N_nh_inds[0]][N_nh_inds[1]][0]['node'])
                    Nnode2 = self.tree.get_node_with_index(NNs[N_nh_inds[0]][N_nh_inds[1]][1]['node'])
                    path1 = path_between_nodes(Nnode1, snode)[::-1]
                    path2 = path_between_nodes(Nnode2, snode)[::-1]
                    L1 = len(path1); L2 = len(path2)
                    L = L1
                    if L < L2: L = L2
                    j = 0
                    while (path1[j] == path2[j]) and (j < L):
                        j += 1
                    inlocs.insert(-1, {'node': path1[j-1]._index, 'x': 1., 'ID': ID})
                # print ">>> iter = ", i
                # print inlocs
        else:
            raise Exception('Invalid distribution type')
        return inlocs

    def _check_hines_criterion(self, NNs):
        for i, NN in enumerate(NNs):
            for j, N in enumerate(NN):
                if len(N) > 1:
                    return (i,j)
        return (-1,)
    
    def _tag_nodes_up(self, start_node, node, radius=0.001):
        if 'tag' not in node.get_content().keys():
            if node._index == start_node._index:
                length = 0.
            else:
                path = path_between_nodes(node, start_node)
                _, length = get_cylinder_radius_length(path)
                length *= 1e-4
            if length < radius:
                node.get_content()['tag'] = 1
                cnodes = node.get_child_nodes()
                for cn in cnodes:
                    self._tag_nodes_up(start_node, cn, radius=radius)
                    
    def _tag_nodes_down(self, start_node, node, cnode=None, radius=0.001):
        if node._index == start_node._index:
            length = 0.
        else:
            path = path_between_nodes(node, start_node)
            _, length = get_cylinder_radius_length(path)
            length *= 1e-4
        if length < radius:
            node.get_content()['tag'] = 1
            cnodes = node.get_child_nodes()
            if len(cnodes) > 1:
                if cnode != None:
                    cnodes = list(set(cnodes) - set([cnode]))
                for cn in cnodes:
                    self._tag_nodes_up(start_node, cn, radius=radius)
            pnode = node.get_parent_node()
            if pnode != None:
                self._tag_nodes_down(start_node, pnode, node, radius=radius)
                
    def remove_tags(self):
        nodes = self.tree.get_nodes()
        for n in nodes:
            cont = n.get_content()
            if 'tag' in cont.keys():
                del cont['tag']
                
    def _add_inlocs_up(self, node, L, inlocs, radius=0.001, type='all', stop_if_found=False, loc_at_branch=True, split_radius=None):
        cnodes = node.get_child_nodes()
        if node._index == 1:
            cnodes = cnodes[2:]
        if type == 3 or type == 4:
            cnodes = [cn for cn in cnodes if cn.get_content()['p3d'].type == type]
        pnode = node.get_parent_node()
        #~ path = path_between_nodes(pnode, node)
        #~ _, pL = get_cylinder_radius_length(path)
        if loc_at_branch:
            if len(cnodes) > 1 and node._index != 1:
                inlocs.append({'node': node._index, 'x': 1, 'ID': len(inlocs)})
        if split_radius != None and L > split_radius and len(cnodes) > 1:
            ind = np.random.randint(len(cnodes))
            cnodes = [cnodes[ind]]
        for cnode in cnodes:
            path = path_between_nodes(node, cnode)
            _, cL = get_cylinder_radius_length(path)
            cL *= 1e-4
            if L+cL > radius:
                if radius-L < 0.:
                    print 'problem!!'
                x = (radius-L) / cL
                ind = cnode._index
                inlocs.append({'node': ind, 'x': x, 'ID': len(inlocs)})
                if not stop_if_found:
                    while (1.-x)*cL > radius:
                        x = x + radius / cL 
                        inlocs.append({'node': ind, 'x': x, 'ID': len(inlocs)})
                    self._add_inlocs_up(cnode, (1.-x)*cL, inlocs, radius=radius, type=type, 
                                    stop_if_found=stop_if_found, loc_at_branch=loc_at_branch, split_radius=split_radius)
            else:
                self._add_inlocs_up(cnode, L+cL, inlocs, radius=radius, type=type, 
                                    stop_if_found=stop_if_found, loc_at_branch=loc_at_branch, split_radius=split_radius)

    def _count_length_up(self, node, L, L_to_leaf, split_radius=None):
        cnodes = node.get_child_nodes()
        if node._index == 1:
            cnodes = cnodes[2:]
        if type == 3 or type == 4:
            cnodes = [cn for cn in cnodes if cn.get_content()['p3d'].type == type]
        if len(cnodes) == 0:
            L_to_leaf.append((node._index, L))
        if split_radius != None and L > split_radius and len(cnodes) > 1:
            ind = np.random.randint(len(cnodes))
            cnodes = [cnodes[ind]]
        else:
            for cnode in cnodes:
                path = path_between_nodes(node, cnode)
                _, cL = get_cylinder_radius_length(path)
                cL *= 1e-4
                L += cL
                self._count_length_up(cnode, L, L_to_leaf, split_radius=split_radius)

    def _add_loc_from_leaf(self, node, L, Ls, inlocs):
        if node._index == 1:
            inlocs.append({'node': 1, 'x': 0.5, 'ID': len(inlocs)})
        else:
            pnode = node.get_parent_node()

            path = path_between_nodes(pnode, node)
            _, cL = get_cylinder_radius_length(path)
            cL *= 1e-4

            L += cL
            if L > Ls[0]:
                inlocs.append({'node': pnode._index, 'x': 0.5, 'ID': len(inlocs)})
                Ls = Ls[1:]
                if len(Ls) > 0:
                    self._add_loc_from_leaf(pnode, L, Ls, inlocs)
            else:
                self._add_loc_from_leaf(pnode, L, Ls, inlocs)


########################################################################


## to calculate Greens functions in the time domain ####################
class greensFunctionCalculator:
    '''
    Implements input-output layer on top of the greenstree class. Provides
    the combination of different types of kernels that require the greens-
    functions and returns them in the time domain by means of the FFT-algorithm.
    '''
    def __init__(self, greenstree):
        '''
        Instantiates a greensFunctionCalculator
        
        input:
            [greenstree]: instance of the greenstree class
        '''
        self.greenstree = greenstree
        self.E_eq = -65.
    
        
    def set_impedances(self, N=np.power(2,10), dt=0.5*1e-3, size='normal', volt=True, pprint=False):
        '''
        Set the impedances in the self.greenstree, frequencies are chosen
        to fit for FFT transform.
        
        input:
            [N]: Integer, number of points at which GFs are sampled
            [dt]: floating point number. Timestep at which the GFs are sampled
            [size]: flag, \'normal\' for GFs for simulation and for ONLY first-
                order Volterra kernels, \'ext\' for second-order Volterra kernels

        Recommended N=2^10 and dt=0.5*1e-3 for the Volterra approach and 
        N=2^16 and dt=0.1*1e-3 for convolution approach
        '''
        # setting parameters
        self.FFT = True
        self.N = N
        self.dt = dt    # s
        self.smax = math.pi/dt # Hz
        self.ds = math.pi/(N*dt) # Hz
        self.s = np.arange(-self.smax,self.smax,self.ds)*1j  # Hz
        self.size = size
        self.tallbot = False
        # set the impedances
        self.greenstree.set_impedance(self.s, size=size, volt=volt, pprint=pprint)
        
    def set_impedances_logscale(self, fmax=6, base=10, num=500, pprint=False):
        '''
        Set the impedances in the self.greenstree, frequencies are on a logscale
        
        input:
            [fmax]: float, maximal frequency is [base]**[fmax]
            [base]: float, see above
            [num]: int, has to be even, eventual number of points on 
                logscale is 3*[num]-2

        Recommended N=2^10 and dt=0.5*1e-3 for the Volterra approach and 
        N=2^16 and dt=0.1*1e-3 for convolution approach
        '''
        # setting parameters
        self.FFT = False
        self.N = (3*num-2)/2
        self.dt = None
        self.smax = base**fmax
        self.ds = None
        a = np.logspace(1, fmax, num=num, base=base)
        b = np.linspace(-base, base, num=num)
        self.s = 1j * np.concatenate((-a[::-1], b[1:-1], a))
        self.size = 'normal'
        self.tallbot = False
        # set the impedances
        self.greenstree.set_impedance(self.s, size='normal', volt=True, pprint=pprint)

    def set_impedances_Tallbot(self, N, mu, nu, pprint = False):
        # setting parameters
        self.FFT = False
        self.N = N
        self.dt = None
        self.ds = None
        self.smax = None
        self.theta = (2.*np.arange(N) + 1.) * np.pi / (2.*N)
        self.s = sigma + mu * (self.theta * np.cos(self.theta) / np.sin(self.theta) + 1j * nu * self.theta)
        self.sprime = mu * (np.cos(self.theta) / np.sin(self.theta) + self.theta / np.cos(self.theta)**2 + 1j * nu)
        self.tallbot = True
        # set the impedances
        self.greenstree.set_impedance(self.s, size='normal', volt=True, pprint=pprint)
    
    def set_quadratic(self):
        '''only call after having called \'set_impedances\' with size=\'ext\' '''
        if self.size == 'normal':
            raise Exception("Cannot call this funtion when size in \'normal\'")
        else:
            self.greenstree.calc_quadratic_coeff()

    def greensFunctionTime(self, inloc={'node': 10, 'x': 0.6}, outloc={'node': 400, 'x': 0.3}, 
                                tmax=0.3, voltage=False):
        '''
        Compute a GF in the timedomain between [inloc] and [outloc]
        
        input:
            [inloc]: dictionnary that has to contain \'node\'-entry and \'x\' entry,
                represents input location of GF
            [outloc]: same as for [inloc], but represents output location of GF
            [tmax]: cutoff time for GF, in s
        
        output:
            [Gt]: GF, in MOhm/ms
            [t]: corresponding time array, in ms
        '''
        if (not set(['node', 'x']).issubset(set(inloc.keys()))) or \
            (not set(['node', 'x']).issubset(set(outloc.keys()))):
            print 'Error: inloc and outloc has to contain entries for \'node\' and \'x\''
            exit(1)
        # calc greens function
        G = self.greenstree.calc_greensfunction(inloc, outloc, voltage=voltage)
        #~ print 'GF(f=0) = ', G[len(G)/2]
        Gt, t = self.calcFFT(G)
        # keep relevant parts
        Gt = Gt[0:tmax/self.dt]
        t = t[0:tmax/self.dt]
        
        return Gt/1000., t*1000.   
        
    def kernelSet_standard(self, inputloclist=[{'node': 10, 'x': 0.6}], tmax=0.3, freqdomain=False, FFT=True):
        '''
        Compute the the standard kernelset relating input with output
                
        input:
            [inputloclist]: list of dictionnaries that each have to contain a 
                \'node\'-entry and \'x\' entry, representing input and output locations
            [tmax]: float, cutoff time for GF, in s
            [freqdomain]: boolean, if True, return GF's in frequency domain
        
        output:
            if freqdomain is True:
                [kernelset]: GFs in time domain, in MOhm/ms
                [t]: corresponding time array, in ms
            else:
                [inpmat]: GFs in frequency domain, in MOhm
                [self.s]: corresponding frequency array, in Hz
        '''
        for inp in inputloclist:
            if not set(['node', 'x']).issubset(set(inp.keys())):
                print 'Error: inputloclist has to contain entries for \'node\' and \'x\''
                exit(1)
        # calculate matrix
        M = len(inputloclist)
        inpmat = np.zeros((M,M,2*self.N), dtype=complex)
        for i in range(M):
            for j in range(M):
                inpmat[i,j,:] = self.greenstree.calc_greensfunction(inputloclist[i], inputloclist[j])
        # return stuff
        if freqdomain:
            return inpmat, self.s
        elif FFT:
            if self.FFT:
                if tmax < self.N*self.dt:
                    kernelset = np.zeros((M,M,tmax/self.dt), dtype=float)
                else:
                    kernelset = np.zeros((M,M,self.N), dtype=float)
                for i in range(M):
                    for j in range(M):
                        Gt, t = self.calcFFT(inpmat[i,j,:])
                        if tmax < self.N*self.dt:
                            kernelset[i,j,:] = Gt[0:tmax/self.dt]
                        else:
                            kernelset[i,j,:] = Gt
                if tmax < self.N*self.dt:
                    t = t[0:tmax/self.dt]
                return kernelset/1000., t*1000.
            else:
                raise Exception('No FFT possible on logscale')
        else:
            alphai, v2y, y2v, pairs, indices = self.fit_kernels(inpmat)
            return alphai, v2y, y2v, pairs, indices 
        
    def kernelSet_sparse_inv(self, inputloclist=[{'node': 10, 'x': 0.6}], tmax=0.3, freqdomain=False, FFT=True):
        '''
        Compute the the sparse kernelset relating input with output in a brute-force way,
        by using numerical matrix inversion (not recommended for large amounts of inputlocs)
                
        input:
            [inputloclist]: list of dictionnaries that each have to contain a 
                \'node\'-entry and \'x\' entry, representing input and output locations
            [tmax]: float, cutoff time for GF, in s
            [freqdomain]: boolean, if True, return GF's in frequency domain
        
        output:
            if freqdomain is False:
                [kernelset_inp]: input kernels in time domain, in MOhm/ms
                [kernelset_trans]: voltage transfer kernels in time domain, in MOhm/ms
                [t]: corresponding time array, in ms
            else:
                [inpmat]: input kernels in frequency domain, in MOhm
                [transfmat]: voltage transfer kernels in frequency domain, in MOhm
                [self.s]: corresponding frequency array, in Hz
        '''
        for inp in inputloclist:
            if not set(['node', 'x']).issubset(set(inp.keys())):
                print 'Error: inputloclist has to contain entries for \'node\' and \'x\''
                exit(1)
        N = self.N
        dt = self.dt
        # calculate matrix
        M = len(inputloclist)
        inpmat = np.zeros((M,M,2*N), dtype=complex)
        inpvect = np.zeros((M,1,2*N), dtype=complex)
        transfmat = np.zeros((M,M,2*N), dtype=complex)
        # calculate GF matrix
        for i in range(M):
            for j in range(M):
                inpmat[i,j,:] = self.greenstree.calc_greensfunction(inputloclist[i], inputloclist[j])
        # do the inversion
        for k in range(2*N):
            invmat = np.linalg.inv(inpmat[:,:,k])
            for i in range(M):
                inpvect[i,0,k] = 1./invmat[i,i]
                for j in range(M):
                    if i==j:
                        transfmat[i,j,k] = 0
                    else:
                        transfmat[i,j,k] = -invmat[i,j]/invmat[i,i]
        # return stuff
        if freqdomain:
            return inpvect, transfmat, self.s
        elif FFT:
            if self.FFT:
                if tmax < N*dt:
                    kernelset_inp = np.zeros((M,1,tmax/dt), dtype=float)
                    kernelset_trans = np.zeros((M,M,tmax/dt), dtype=float)
                else:
                    kernelset_inp = np.zeros((M,1,N), dtype=float)
                    kernelset_trans = np.zeros((M,M,N), dtype=float)
                for i in range(M):
                    inp, t = self.calcFFT(inpvect[i,0,:])
                    if tmax < N*dt:
                        kernelset_inp[i,0,:] = inp[0:tmax/dt]
                    else:
                        kernelset_inp[i,0,:] = inp
                    for j in range(M):
                        trans, t = self.calcFFT(transfmat[i,j,:])
                        if tmax < N*dt:
                            kernelset_trans[i,j,:] = trans[0:tmax/dt]
                        else:
                            kernelset_trans[i,j,:] = trans
                if tmax < N*dt:
                    t = t[0:tmax/dt]
                return kernelset_inp/1000., kernelset_trans/1000., t*1000.
            else:
                raise Exception('No FFT possible on logscale')
        else:
            transfmat_dict = sparseArrayDict(shape=transfmat[:,:,0].shape, \
                                    el_shape=self.s.shape, dtype=complex)
            for u in range(transfmat.shape[0]):
                for v in range(transfmat.shape[1]):
                    if np.max(np.abs(transfmat[u,v])) > 1e-7:
                        transfmat_dict[u,v] = transfmat[u,v]
            alphai, v2y, y2v, pairs, indices = self.fit_kernels([inpvect, transfmat_dict])
            return alphai, v2y, y2v, pairs, indices
            
    def kernelSet_sparse(self, inputloclist=[{'node': 1, 'x': 0.6}], tmax=0.3, freqdomain=False, FFT=True, kernelconstants=False, pprint=False):
        '''
        Compute the the sparse kernelset relating input with output by using reduced
        formulas
                
        input:
            [inputloclist]: list of dictionnaries that each have to contain a 
                \'node\'-entry, \'x\' entry, representing input and output locations
            [tmax]: float, cutoff time for GF, in s
            [freqdomain]: boolean, if True, return GF's in frequency domain
        
        output:
            if freqdomain is False:
                [kernelset_inp]: input kernels in time domain, in MOhm/ms
                [kernelset_trans]: voltage transfer kernels in time domain, in MOhm/ms
                [t]: corresponding time array, in ms
            else:
                [inpmat]: input kernels in frequency domain, in MOhm
                [transfmat]: voltage transfer kernels in frequency domain, in MOhm
                [self.s]: corresponding frequency array, in Hz
        '''
        inlocs = inputloclist
        nosoma = True
        for inloc in inlocs: 
            if inloc['node'] == 1: 
                nosoma = False; break
        if nosoma: raise Exception('Soma needs to be in list of inlocs')
        # transfer function matrix
        transfmat = sparseArrayDict(shape=(len(inlocs), len(inlocs)), 
                                    el_shape=self.s.shape, dtype=complex)
        inpvect = np.zeros((len(inlocs), 1, len(self.s)), dtype=complex)
        # make list of ID's
        inlocIDs = [inloc['ID'] for inloc in inlocs]
        # find sets of nearest neighbours
        NNs, _ = self.greenstree.get_nearest_neighbours(inlocs, add_leaves=False, reduced=False)
        # for ind, NN in enumerate(NNs):
            # print 'inloc: ', inlocs[ind]['ID'], ' neighbours: ', NN 
        # list of ID's of neighbours of soma
        somaind = 0
        for ind, inloc in enumerate(inlocs): 
            if inloc['node'] == 1: somaind = ind; break
        somaNNIDs = []
        for setNN in NNs[somaind]:
            for elNN in setNN:
                somaNNIDs = [elNN['ID']]
        # create connection matrices for inputs in sets of nearest neighbours
        connections = []
        connectionIDs_row = []
        connectionIDs_col = []
        kf = []
        for ind, inloc in enumerate(inlocs):
            i = 0
            # find all sets of nearest neighbours to which inloc belongs
            NNsets = []
            found = False
            for inind, inNN in enumerate(NNs):
                if inind != ind:
                    for setind, setNN in enumerate(inNN):
                        for elind, elNN in enumerate(setNN):
                            if inloc['ID'] == elNN['ID']:
                                if inloc['node'] != 1 and inlocs[inind]['x'] < 1. \
                                        and inlocs[inind]['node'] != 1:
                                    # here we have to add all child neighbours of the 
                                    # parent neighbour
                                    setNNcopy = []
                                    for setNN2 in inNN:
                                        for elNN2 in setNN2:
                                            setNNcopy.append(copy.deepcopy(elNN2))
                                else:
                                    # here we only need to add one set, since the parent
                                    # neighbours splits up the set of his child
                                    # neighbours
                                    setNNcopy = copy.deepcopy(setNN)
                                # append the element closest to soma to this set
                                # of nearest neighbours
                                setNNcopy.append(copy.deepcopy(inlocs[inind]))
                                # append the rest of the set
                                NNsets.append(setNNcopy)
                                found = True; break
                else: # inloc is by definition part of all these sets
                    for setind, setNN in enumerate(inNN):
                        if setNN:
                            NNsets.append(copy.deepcopy(setNN))
            # inloc is also part of this set of nearest neighbours
            if not found: NNsets.append([copy.deepcopy(inloc)])#; print NNs
            # flatten inloc set
            NNinloc = [inlocNN for NN in NNsets for inlocNN in NN]
            # set up symbolic and real connection matrices
            connections.append(sp.zeros(len(NNinloc), len(NNinloc)))
            kf.append(np.zeros((len(NNinloc), len(NNinloc), 2*self.N), dtype=complex))
            # for bookkeeping, keeps a list of IDs of each NN of inloc
            connectionIDs_row.append(np.zeros((len(NNinloc)), dtype=int))
            connectionIDs_col.append(np.zeros((len(NNinloc)), dtype=int))
            for ind1, inloc1 in enumerate(NNinloc):
                connectionIDs_row[ind][ind1] = inloc1['ID']
                for ind2, inloc2 in enumerate(NNinloc):
                    connectionIDs_col[ind][ind2] = inloc2['ID']
                    # fill the sympy matrix
                    if ind1 == ind2:
                        connections[ind][ind1, ind2] = 1
                    else:
                        connections[ind][ind1, ind2] = sp.symbols('kf[' + str(ind) + '][' + \
                                                                str(ind1) + '][' + str(ind2) + ']')
                    # if transfer does not yet exist, we have to compute it
                    try:
                        eval('g_' + str(inloc1['ID']) + '_' + str(inloc2['ID']))
                    except NameError:
                        try:
                            exec('g_' + str(inloc1['ID']) + '_' + str(inloc2['ID']) + \
                                ' = self.greenstree.calc_greensfunction(inloc1, inloc2, voltage=True)')
                        except IndexError:
                            print inloc1, inloc2
                    # different entries that comprise the same transfer function will be pointers
                    # to the same object, thus saving memory
                    if ind1 == ind2:
                        kf[ind][ind1][ind2][:] = np.ones(self.s.shape, dtype=complex)
                    else:
                        kf[ind][ind1][ind2][:] = copy.deepcopy(eval('g_' + str(inloc1['ID']) + '_' + str(inloc2['ID'])))

        # calculate minors
        for ind, inloc in enumerate(inlocs):
            # find index of inloc in local connection matrix
            ID = inlocIDs[ind]
            indlocal = np.where(connectionIDs_row[ind] == ID)[0][0]
            # if submatrix is smal enough we can do analytic computation
            if connections[ind].shape[0] < 5:
                M = connections[ind]#.transpose()
                #~ print M
                M_det = M.det()
                M_i = copy.deepcopy(M)
                M_i.row_del(indlocal)
                M_ii = copy.deepcopy(M_i)
                M_ii.col_del(indlocal)
                M_ii_det = M_ii.det()
                inputk = M_det / M_ii_det
                inpvect[ind,0,:] = eval(str(inputk))
                # iterate over local connections
                for indlocal2, ID2 in enumerate(connectionIDs_col[ind]):
                    if indlocal2 != indlocal:
                        M_ij = copy.deepcopy(M_i)
                        M_ij.col_del(indlocal2)
                        ind2 = inlocIDs.index(ID2)
                        transfer = M_ij.det() / M_ii_det * (-1)**(indlocal + indlocal2 + 1) 
                        #~ if ind == 0 and ind2==1:
                            #~ print transfer
                        transfmat[ind, ind2] = eval(str(transfer))
            # else we have to do numerical inversion
            else:
                # k=0, bit of hack because of sparseArrayDict
                invmat = np.zeros(connections[ind].shape, dtype=complex)
                invmat[:,:] = np.linalg.inv(np.transpose(kf[ind][:,:,0]))
                inpvect[ind,0,0] = 1. / invmat[indlocal,indlocal]
                for indlocal2, ID2 in enumerate(connectionIDs_col[ind]):
                    if indlocal2 != indlocal:
                        ind2 = inlocIDs.index(ID2)
                        transfmat.setelement2zero((ind, ind2))
                        transfmat[ind, ind2][0] = -invmat[indlocal, indlocal2] / invmat[indlocal, indlocal]
                # rest of k's
                for k in range(1,2*self.N):
                    invmat = np.zeros(connections[ind].shape, dtype=complex)
                    invmat[:,:] = np.linalg.inv(np.transpose(kf[ind][:,:,k]))
                    inpvect[ind,0,k] = 1. / invmat[indlocal,indlocal]
                    for indlocal2, ID2 in enumerate(connectionIDs_col[ind]):
                        if indlocal2 != indlocal:
                            ind2 = inlocIDs.index(ID2)
                            transfmat[ind, ind2][k] = -invmat[indlocal, indlocal2] / invmat[indlocal, indlocal]
            inpvect[ind,0,:] *= self.greenstree.calc_greensfunction(inloc, inloc)

        # return stuff
        if freqdomain:
            return inpvect, transfmat, self.s
        if not self.tallbot:
            if FFT:
                if self.FFT:
                    if tmax < self.N * self.dt:
                        kernelset_inp = np.zeros((len(inlocs),1,int(tmax/self.dt)), dtype=float)
                        kernelset_trans = sparseArrayDict(shape=(len(inlocs), len(inlocs)), 
                                            el_shape=int(tmax/self.dt), dtype=complex)
                    else:
                        kernelset_inp = np.zeros((len(inlocs),1,self.N), dtype=float)
                        kernelset_trans = sparseArrayDict(shape=(len(inlocs), len(inlocs)), 
                                            el_shape=self.N, dtype=complex)
                    for i in range(len(inlocs)):
                        kt, tk = self.calcFFT(inpvect[i,0,:])
                        if tmax < self.N * self.dt:
                            kernelset_inp[i,0,:] = kt[0:int(tmax/self.dt)] / 1000.
                        else:
                            kernelset_inp[i,0,:] = kt / 1000.
                        for j in range(len(inlocs)):
                            if (i,j) in transfmat.keys():
                                kt, tk = self.calcFFT(transfmat[i,j])
                                if tmax < self.N * self.dt:
                                    kernelset_trans[i,j] = kt[0:int(tmax/self.dt)] / 1000.
                                else:
                                    kernelset_trans[i,j] = kt / 1000.
                    if tmax < self.N * self.dt:
                        tk = tk[0:tmax/self.dt]
                    return kernelset_inp, kernelset_trans, tk*1000.
                else:
                    raise Exception('No FFT possible on logscale')
            elif kernelconstants:
                alphas, gammas, pairs, Ms, K1, K2 = self._fit_sp(inpvect, transfmat, pprint=pprint)
                return alphas, gammas, pairs, Ms
            else:
                alphai, v2y, y2v, pairs, indices = self.fit_kernels([inpvect, transfmat], pprint=pprint)
                return alphai, v2y, y2v, pairs, indices
        else:
            alphas, gammas, pairs, Ms, K1, K2 = self._compute_Tallbot_quadrature(inpvect, transfmat, pprint=pprint)
            return alphas, gammas, pairs, Ms


    def calcFFT(self, arr, test=0, method='hannig', improve_tails=False, rtol=1e-3, smoothen=False):
        '''
        Calc 1d fourier transform of arr by the FFT algorithm
        
        input: 
            [arr]: 1d complex numpy array
            [test]: flag, if 1, print test result
            [method]: use 'hannig'
            [improve_tails]: boolean, applies tail corrections if True, only
                necessary when function in freqdomain are non-zero at cutoff
        output: 
            [fftarr.real]: real part of FFT array
            [t]: corresponding times, in s
        '''
        N = self.N
        dt = self.dt
        ds = self.ds
        smax = self.smax
        
        if improve_tails:
            imp = False
            if np.abs(arr[-1]) / np.max(np.abs(arr)) > rtol:
                imp = True
                alpha, gamma = self._fit_exp(arr)
                fexp = gamma / (alpha + self.s)
                arr = arr - fexp
        
        if method == 'standard':
            #fft
            t = np.arange(0.,N*dt,dt)
            arr = np.array(arr)
            fftarr = np.fft.ifft(arr)
            fftarr = fftarr[0:len(fftarr)/2]
            # scale factor
            scale = 2*N*ds*np.exp(-1j*t*smax)/(2*math.pi)
            fftarr = scale*fftarr
            if smoothen:
                from scipy.ndimage.filters import gaussian_filter
                fftarr = gaussian_filter(fftarr.real, sigma=1., mode='reflect')
                #~ temparr = 0.25*fftarr[0:-2] + 0.5*fftarr[1:-1] + 0.25*fftarr[2:]
                #~ fftarr = np.concatenate((fftarr[0:1], temparr, fftarr[-1:]))
        elif method == 'trapezoid':
            #fft
            t = np.arange(0.,N*dt,dt)
            arr = np.array(arr)
            fftarr = np.fft.ifft(arr)
            fftarr = fftarr[0:len(fftarr)/2]
            # scale factor
            prefactor = (1.-np.cos(ds*t))/(ds*np.power(t,2))
            prefactor[0] = 0.5*ds
            scale = 4.*N*prefactor*np.exp(-1j*t*smax)/(2*math.pi)
            fftarr = scale*fftarr
        elif method == 'hannig':
            #fft
            t = np.arange(0.,N*dt,dt)
            window = 0.5*(1.+np.cos(math.pi*self.s.imag/smax))
            arr = np.array(arr*window)
            fftarr = np.fft.ifft(arr)
            fftarr = fftarr[0:len(fftarr)/2]
            # scale factor
            scale = 2*N*ds*np.exp(-1j*t*smax)/(2*math.pi)
            fftarr = scale*fftarr
        elif method == 'simpson':
            #fft
            t = np.arange(0.,N*dt,dt)
            arreven = arr[::2]
            arrodd = arr[1::2]
            arreventrans = np.fft.ifft(arreven)
            arroddtrans = np.fft.ifft(arrodd)
            fftarr = ds/(6.*math.pi)*(-arr[0]*np.exp(-1j*smax*t) + 2.*N*np.exp(-1j*smax*t)*arreventrans \
                    +4.*N*np.exp(-1j*(smax-ds)*t)*arroddtrans + arr[-1]*np.exp(1j*smax*t))
        elif method == 'quadrature':
            import scipy.integrate as integ
            t = np.arange(0.,N*dt,dt)
            fftarr = np.zeros(N)
            smax = 1000.
            for i in range(len(t)):
                fftarr[i] = integ.romberg(self._aux_func,-smax,smax,args = [t[i]])
                
        #~ if improve_tails:
            #~ if imp:
                #~ import matplotlib.pyplot as pl
                #~ pl.figure('improved_tails')
                #~ pl.plot(t, fftarr, 'g')
                #~ pl.plot(t, gamma * np.exp(-alpha*t), 'r')
                #~ pl.plot(t, fftarr + gamma * np.exp(-alpha*t), 'b')
                #~ pl.show()
                #~ fftarr += gamma * np.exp(-alpha*t) #/ (2.*math.pi)
            #~ tailcorr = self._calc_tailcorrection(temparr, ds)
            #~ fftarr[0] += tailcorr

        if test==1:
            print 'Tests:'
            print 'Int: ', sum((arr[0:-1]+arr[1:])/2.)*(ds)/(2*math.pi), ', Trans zero freq: ', fftarr[0]
            print 'G zero_freq: ', arr[len(samplefreqs)/2].real, ', G integral: ', (fftarr[0].real/2. + sum(fftarr[1:-1].real) + fftarr[-1].real/2.)*dt

        return fftarr.real, t
        
    def _fit_exp(self, arr):
        N = 100
        sm = self.smax
        f0 = np.mean(arr[-N:].real)
        f1 = np.mean((arr[-(N-1):].real - arr[-N:-1].real) / self.ds)
        alpha = np.sqrt(-2.*sm*(f0/f1) - sm**2)
        gamma = f0 * (alpha**2 + sm**2) / alpha
        import matplotlib.pyplot as pl
        #~ pl.figure('_fit_exp')
        #~ pl.plot(self.s.imag, gamma / (alpha + self.s), 'r')
        #~ pl.plot(self.s.imag, arr, 'b')
        #~ pl.plot(self.s.imag, arr - gamma / (alpha + self.s), 'g')
        #~ pl.show()
        return alpha, gamma
        
    def _calc_tailcorrection(self, arr, ds):
        ind = int(len(self.s)/3.)
        fx1 = arr[-ind].real
        dfdx1 = ((arr[-ind].real - arr[-ind-10].real) / (10*ds))
        #~ print -0.5 * fx1**2 / dfdx1
        return -0.5 * fx1**2 / dfdx1
    
    def _aux_func(self, s, t):
        return (self.greensFunctionFrequency(0, 0.95, np.array([s]))[0] * np.exp(1j*s*t))[0]
        
    def calcFFT2d(self, arr, test=0):
        '''
        Calc 1d fourier transform of arr by the FFT algorithm
        
        input: 
            [arr]: 1d complex numpy array
            [test]: flag, if 1, print test result
        output: 
            [fftarr.real]: real part of FFT array
            [t]: corresponding times, in s
        '''
        N = self.N
        dt = self.dt
        ds = self.ds
        smax = self.smax
        # fft
        t = np.arange(0.,N*dt,dt)
        arr = np.array(arr)
        fftarr = np.fft.ifft2(arr)
        fftarr = fftarr[0:len(fftarr)/2, 0:len(fftarr)/2]
        scale1 = np.outer(np.exp(-1j*smax*t), np.exp(-1j*smax*t))
        #~ scale1 = 1
        scale2 = np.power(2.*N*ds/(2.*math.pi), 2)
        fftarr = scale2*scale1*fftarr
        
        return fftarr.real, t
        
    def fit_kernels(self, kernels, pprint=False, return_original_matrices=False):
        if type(kernels) is list:
            kin = kernels[0]; ktrans = kernels[1]
            alphai, v2y, y2v, pairi, indices = self._sp_2_matrix(kin, ktrans, pprint=pprint)
        else:
            alphai, v2y, y2v, pairi, indices = self._fit_full(kernels, pprint=pprint)
        return alphai, v2y, y2v, pairi, indices
    
    def _fit_full(self, kernels, pplot=False, pprint=False):
        kall = kernels
        # compute exponential expansion
        FEF = funF.fExpFitter()
        alphas = objectArrayDict(shape=kall.shape[0:2])
        gammas = objectArrayDict(shape=kall.shape[0:2])
        pairs = objectArrayDict(shape=kall.shape[0:2])
        Ms = np.zeros(kall.shape[0:2], dtype=int)
        K = 0
        for u in range(kall.shape[0]):
            for v in range(kall.shape[1]):
                if u == v:
                    #~ print 'full in'
                    alpha, gamma, pair, rms = FEF.fitFExp(self.s, kall[u,v], rtol=1e-5, deg=5, maxiter=10, 
                            initpoles='log10', realpoles=True, zerostart=False, constrained=True, reduce_numexp=False)
                    alpha = alpha / 1000.; gamma = gamma / 1000.
                else:
                    #~ print 'full trans'
                    alpha, gamma, pair, rms = FEF.fitFExp(self.s, kall[u,v], rtol=1e-5, deg=5, maxiter=10, 
                            initpoles='log10', realpoles=True, zerostart=True, constrained=True, reduce_numexp=False)
                    alpha = alpha / 1000.; gamma = gamma / 1000.
                if pprint and rms > 1e-3:
                    print 'rmse, full: ', rms
                if pplot and rms > 1e-3:
                    print 'loc: ', (u,v)
                    print 'numexp: ', len(alpha)
                    import matplotlib.pyplot as pl
                    kernel = FEF.sumFExp(self.s, alpha*1000., gamma*1000.)
                    pl.figure('expfit')
                    pl.plot(self.s.imag, kall[u,v].real, 'b')
                    pl.plot(self.s.imag, kernel.real, 'b--', lw=2)
                    pl.plot(self.s.imag, kall[u,v].imag, 'r')
                    pl.plot(self.s.imag, kernel.imag, 'r--', lw=2)
                    pl.figure('difference')
                    pl.plot(self.s.imag, kall[u,v].real - kernel.real, 'b--', lw=2)
                    pl.plot(self.s.imag, kall[u,v].imag - kernel.imag, 'r--', lw=2)
                    EF = funF.ExpFitter()
                    pl.figure('transform')
                    #~ fftarr, t = self.calcFFT(kall[u,v])
                    #~ t *= 1000.; fftarr = fftarr / 1000.
                    #~ pl.plot(t, fftarr, 'g')
                    t = np.arange(0.,50.,0.01)
                    pl.plot(t, EF.sumExp(t, -alpha, gamma), 'g--', lw=2)
                    pl.show()
                    pl.show()
                alphas[u,v] = alpha
                gammas[u,v] = gamma
                pairs[u,v] = pair
                Ms[u,v] = len(gamma)
                K += len(gamma)
                
        #~ np.set_printoptions(precision=2)
        #~ print 'alphas fit:', alphas
        #~ print 'pairs fit:', pairs
                
        # store the expansion in convenient format for further simulation
        y2v = np.zeros((len(kall), K), dtype=complex)
        v2y = np.zeros(K, dtype=int)
        alphai = np.zeros(K, dtype=complex)
        pairi = np.zeros(K, dtype=bool)
        indices = np.zeros(K, dtype=object)
        
        l = 0
        for u in range(kall.shape[0]):
            for v in range(kall.shape[1]):
                y2v[u, l:l+Ms[u,v]] = gammas[u,v]
                v2y[l:l+Ms[u,v]] = v*np.ones(Ms[u,v], dtype=int)
                alphai[l:l+Ms[u,v]] = alphas[u,v]
                pairi[l:l+Ms[u,v]] = pairs[u,v]
                auxarr = np.empty(Ms[u,v], dtype=object)
                for i in range(Ms[u,v]): auxarr[i] = (u,v)
                indices[l:l+Ms[u,v]] = auxarr
                l += Ms[u,v];
                
        return -alphai, v2y, y2v, pairi, indices
    
    def _sp_2_matrix(self, kin, ktrans, pplot=False, pprint=False, return_original_matrices=False):
        alphas, gammas, pairs, Ms, K1, K2 = self._fit_sp(kin, ktrans, pprint=pprint)

        # store the expansion in convenient format for further simulation
        y2v = [np.zeros((ktrans.shape[0], K1), dtype=complex), \
                np.zeros((ktrans.shape[0], K2), dtype=complex)]
        v2y = [np.zeros(K1, dtype=int), np.zeros(K2, dtype=int)]
        alphai = [np.zeros(K1, dtype=complex), np.zeros(K2, dtype=complex)]
        pairi = [np.zeros(K1, dtype=bool), np.zeros(K2, dtype=bool)]
        indices = [np.zeros(K1, dtype=object), np.zeros(K2, dtype=object)]
        
        l1 = 0; l2 = 0
        for key in ktrans.keys():
            y2v[0][key[0], l1:l1+Ms[key]] = gammas[key]
            v2y[0][l1:l1+Ms[key]] = key[1] * np.ones(Ms[key], dtype=int)
            alphai[0][l1:l1+Ms[key]] = alphas[key]
            pairi[0][l1:l1+Ms[key]] = pairs[key]
            auxarr = np.empty(Ms[key], dtype=object)
            for i in range(Ms[key]): auxarr[i] = key
            indices[0][l1:l1+Ms[key]] = auxarr
            l1 += Ms[key]
        for u in range(kin.shape[0]):
            y2v[1][u, l2:l2+Ms[u,u]] = gammas[u,u]
            v2y[1][l2:l2+Ms[u,u]] = u * np.ones(Ms[u,u], dtype=int)
            alphai[1][l2:l2+Ms[u,u]] = alphas[u,u]
            pairi[1][l2:l2+Ms[u,u]] = pairs[u,u]
            auxarr = np.empty(Ms[u,u], dtype=object)
            for i in range(Ms[u,u]): auxarr[i] = (u,u)
            indices[1][l2:l2+Ms[u,u]] = auxarr
            l2 += Ms[u,u]
        
        return alphai, v2y, y2v, pairi, indices

    def _fit_sp(self, kin, ktrans, pplot=False, pprint=True):
        # compute exponential expansion
        rtol = 1e-8
        FEF = funF.fExpFitter()
        alphas = objectArrayDict(shape=ktrans.shape[0:2])
        gammas = objectArrayDict(shape=ktrans.shape[0:2])
        pairs = objectArrayDict(shape=ktrans.shape[0:2])
        Ms = np.zeros(ktrans.shape[0:2], dtype=int)
        MIs = np.zeros(ktrans.shape[0:2], dtype=int)
        K1 = 0; K2 = 0
        for key in ktrans.keys():
            # alpha, gamma, pair, rms = FEF.fitFExp(self.s, ktrans[key], rtol=1e-5, deg=35, maxiter=20, 
            #         initpoles='log10', realpoles=True, zerostart=False, constrained=True, reduce_numexp=False)
            # alpha, gamma, pair, rms = FEF.fitFExp(self.s, ktrans[key], rtol=1e-5, deg=5, maxiter=20, 
            #         initpoles='log10', realpoles=False, zerostart=False, constrained=True, reduce_numexp=False)
            alpha, gamma, pair, rms = FEF.fitFExp_increment(self.s, ktrans[key], \
                            rtol=rtol, maxiter=50, realpoles=False, constrained=True, zerostart=False)
            alpha = alpha / 1000.; gamma = gamma / 1000.
            if (pprint or pplot) and rms > rtol:
                print 'rmse, sparse, trans:', rms
            if pplot and rms > rtol:
                print 'loc: trans, ', key
                print 'numexp: ', len(alpha) 
                print 'alphas: ', 1./np.abs(alpha)
                print 'influences: ', np.abs(gamma/alpha), '\n'
                print 'gammas: ', np.abs(gamma), '\n'
                import matplotlib.pyplot as pl
                pl.figure('expfit')
                kernel = FEF.sumFExp(self.s, alpha*1000., gamma*1000.)
                pl.plot(self.s.imag, (ktrans[key]).real, 'b')
                pl.plot(self.s.imag, kernel.real, 'b--', lw=2)
                pl.plot(self.s.imag, (ktrans[key]).imag, 'r')
                pl.plot(self.s.imag, kernel.imag, 'r--', lw=2)
                pl.figure('difference')
                pl.plot(self.s.imag, (ktrans[key]).real - kernel.real, 'b--', lw=2)
                pl.plot(self.s.imag, (ktrans[key]).imag - kernel.imag, 'r--', lw=2)
                EF = funF.ExpFitter()
                pl.figure('transform')
                t = np.arange(0.,50.,0.001)
                pl.plot(t, EF.sumExp(t, -alpha, gamma), 'g--', lw=2)
                # ind = np.where(1/np.abs(alpha) > 1.)[0]
                # ind_ = np.where(1/np.abs(alpha) < 1.)[0]
                # pl.plot(t, EF.sumExp(t, -alpha[ind], gamma[ind]), 'r--', lw=2)
                # pl.plot(t, EF.sumExp(t, -alpha[ind_], gamma[ind_]), 'b--', lw=2)
                pl.show()
            alphas[key] = - alpha
            gammas[key] = gamma
            pairs[key] = pair
            Ms[key] = len(gamma)
            K1 += len(gamma)

        if pprint:
            print 'number of exponentials / trans kernel = ', float(K1) / float(len(ktrans.keys())) 
        
        for u in range(kin.shape[0]):
            # alpha, gamma, pair, rms = FEF.fitFExp(self.s, kin[u,0], rtol=1e-3, deg=35, maxiter=20, 
            #         initpoles='log10', realpoles=True, zerostart=False, constrained=True, reduce_numexp=False)
            # alpha, gamma, pair, rms = FEF.fitFExp(self.s, kin[u,0], rtol=1e-5, deg=5, maxiter=20, 
            #         initpoles='log10', realpoles=False, zerostart=False, constrained=True, reduce_numexp=False)
            alpha, gamma, pair, rms = FEF.fitFExp_increment(self.s, kin[u,0], \
                            rtol=rtol, maxiter=50, realpoles=False, constrained=True, zerostart=False)
            alpha = alpha / 1000.; gamma = gamma / 1000. # adjust units ([alpha] = 1/ms)
            if (pprint or pplot) and rms > rtol:
                print 'rmse, sparse, inp:', rms
            if pplot and rms > rtol:
                print 'loc: in, ', u
                print 'numexp: ', len(alpha)
                print 'alphas: ', 1./np.abs(alpha)
                print 'influences: ', np.abs(gamma/alpha), '\n'
                print 'gammas: ', np.abs(gamma), '\n'
                import matplotlib.pyplot as pl
                kernel = FEF.sumFExp(self.s, alpha*1000., gamma*1000.)
                pl.figure('expfit')
                pl.plot(self.s.imag, kin[u,0].real, 'b')
                pl.plot(self.s.imag, kernel.real, 'b--', lw=2)
                pl.plot(self.s.imag, kin[u,0].imag, 'r')
                pl.plot(self.s.imag, kernel.imag, 'r--', lw=2)
                pl.figure('difference')
                pl.plot(self.s.imag, kin[u,0].real - kernel.real, 'b--', lw=2)
                pl.plot(self.s.imag, kin[u,0].imag - kernel.imag, 'r--', lw=2)
                EF = funF.ExpFitter()
                pl.figure('transform')
                t = np.arange(0.,50.,0.001)
                pl.plot(t, EF.sumExp(t, -alpha, gamma), 'g--', lw=2)
                # pl.loglog(t, EF.sumExp(t, -alpha, gamma), 'g--', lw=2)
                pl.show()
                pl.show()
            alphas[(u,u)] = - alpha
            gammas[(u,u)] = gamma
            pairs[(u,u)] = pair
            Ms[u,u] = len(gamma)
            K2 += len(gamma)

        if pprint:
            print 'number of exponentials / input kernel = ', float(K2) / float(kin.shape[0])

        return alphas, gammas, pairs, Ms, K1, K2
########################################################################

