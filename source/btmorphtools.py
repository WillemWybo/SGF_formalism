"""
Author: Benjamin Torben-Nielsen
Date: 18/08/2015
"""

import h5py
import numpy as np

import neuron
from neuron import h
import btstructs    

"""
- Convert BBP's H5 morphology format to the more standard SWC format.
- Create a passive model from an SWC file
"""

class h5_point(object) :
    def __init__(self,x,y,z,radius) :
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius

class h5_structure(object) :
    def __init__(self,start_index,n_type,parent_section_index) :
        self.start_index = start_index
        self.n_type = n_type
        self.parent_section_index = parent_section_index

def _get_H5_points(h5_file) :
    points = {}
    h5_points = h5_file['points']
    for index, item in zip(range(len(h5_points)),h5_points) :
        points[index] = h5_point(item[0],item[1],item[2],item[3])
    return points

def _get_h5_structure(h5_file) :
    structures = {}
    if 'structure' in h5_file :
        h5_structures = h5_file['structure']
        for index, item in zip(range(len(h5_structures)),h5_structures) :
            structures[index] = h5_structure(item[0], \
                                             item[1],item[2])
    return structures    

def _create_three_point_soma(structure,points) :
    """
    Create a three-point soma assuming that the first entry in the H5 \
    structure field is the soma

     1 1 xs ys zs rs -1
     2 1 xs (ys-rs) zs rs 1
     3 1 xs (ys+rs) zs rs 1
     
    with xs,ys,zs being point in the middle of the contour and rs being
    the average distance between the center and the contour

    http://neuromorpho.org/neuroMorpho/SomaFormat.html
    """
    xs,ys,zs = [],[],[]
    end_of_soma_index = structure[1].start_index-1
    for index in range(end_of_soma_index) :
        p = points[index]
        xs.append(p.x)
        ys.append(p.y)
        zs.append(p.z)
    center_x = np.mean(xs)
    center_y = np.mean(ys)
    center_z = np.mean(zs)

    rs = 0
    for x,y,z in zip(xs,ys,zs) :
        rs = rs + np.sqrt( (x-center_x)**2 + (y-center_y)**2 + (z-center_z)**2  )
    rs = rs / len(xs)
    print 'rs=',rs
    
    line = '1 1 '+str(center_x)+' '+str(center_y)+' '+str(center_z) + ' ' + str(rs) + ' -1\n'
    line += '2 1 '+str(center_x)+' '+str(center_y-rs)+' '+str(center_z) + ' ' + str(rs) + ' 1\n'
    line += '3 1 '+str(center_x)+' '+str(center_y+rs)+' '+str(center_z) + ' ' + str(rs) + ' 1\n'
    return line

def _fetch_structure_information(index,structure) :
    for seg_id in structure :
        struct = structure[seg_id]
        if struct.start_index == index :
            # print 'index=',index
            parent_section_index = struct.parent_section_index
            if parent_section_index == 0 :
                real_parent_index = 1
            else :
                real_parent_index = structure[parent_section_index+1].start_index-1
            section_type = struct.n_type #structure[parent_section_index].n_type
            return real_parent_index,section_type

    # print 'returning -10 for index=',index
    # raw_input('Press ENTER')
    return index - 1, None

def convert_h5_to_SWC(h5_file_name, types=[3,4], swc_given_name=None) :
    """
    Convert h5 file to SWC file
    Arguments:
    - h5_file_name: string with the file name
    - types: list with the neurite types to include, 2: axon, 3: basal , 4: apical
    - swc_given_name: filename of the SWC output file. If not set, default \
    behaviour simpy replaces with *.h5 with *.swc
    """

    # load points and structure from H5 file
    h5_file = h5py.File(h5_file_name,'r')
    points = _get_H5_points(h5_file)
    structure = _get_h5_structure(h5_file)        
    structure2 = h5_file['structure']
    h5_file.close()

    # directly convert into an SWC file
    if swc_given_name == None :
        swc_file_name = h5_file_name[:-2]+'swc'
    else :
        swc_file_name = swc_given_name
    swc_file = open(swc_file_name,'w')

    # main loop
    end_of_soma_index = 1000 
    for index in points :
        p = points[index]
        if index == 0 :
            end_of_soma_index = structure[1].start_index-1
            swc_line = _create_three_point_soma(structure,points)
            # raw_input('soma, end=%i, press ENTER' % end_of_soma_index)
            swc_file.write(swc_line)
        elif index <= end_of_soma_index:
            #skip the soma
            pass
        else :
            parent_index,point_type = _fetch_structure_information(index,structure)
            point_type = point_type if point_type != None else int(swc_line.split(' ')[1])
            swc_line = str(index)+' '+str(point_type)+' ' \
              +str(p.x)+' '+str(p.y)+' '+str(p.z)+' '+str(p.radius) \
              + ' '+str(parent_index) + '\n'
            if point_type in types :
                swc_file.write(swc_line)
        swc_file.flush()
    swc_file.close()
    return 0
    

rs = 0
def create_NRN_from_SWC(file_name,**kwargs) :
    global rs
    """
    Create a passive multi-compartmental model in pyNRN from an SWC file
    """
    swc_tree = btstructs.STree()
    swc_tree.read_SWC_tree_from_file(file_name)
    nodes = swc_tree.get_nodes()
    rs = nodes[1].get_content()['p3d'].y
    sections = {}
    h.load_file("stdlib.hoc") # contains the lambda rule
    for node in nodes :
        sections.update({node.get_index(): \
                               _make_section(node,node.get_index,sections,**kwargs)})
    return sections

def _make_section(node,index,sections,**kwargs) :
    compartment = neuron.h.Section(name=str(index)) # NEW NRN SECTION
    # assume three point soma
    if node.get_index() not in [1,2,3] :
        pPos = node.get_parent_node().get_content()['p3d']
        cPos = node.get_content()['p3d']
        compartment.push()
        h.pt3dadd(float(pPos.x),float(pPos.y),float(pPos.z),float(pPos.radius))
        h.pt3dadd(float(cPos.x),float(cPos.y),float(cPos.z),float(cPos.radius))
        # nseg according to NEURON book
        compartment.nseg =int(((compartment.L/(0.1*h.lambda_f(100))+0.9)/2)*2+1)

        # passive properties
        compartment.cm = kwargs['cm'] if 'cm' in kwargs else 0.9
        compartment.Ra = kwargs['ra'] if 'ra' in kwargs else 200
        compartment.insert('pas')
        compartment.e_pas =  kwargs['e_pas'] if 'e_pas' in kwargs else -65
        compartment.g_pas =  kwargs['g_pas'] if 'g_pas' in kwargs else 1.0/25000
        
        h.pop_section()
        compartment.connect(sections.get(node.get_parent_node().get_index()),\
                            1,0)
        return compartment
    else :
        if node.get_index() == 1 :
            # root of SWC tree = soma
            cPos = node.get_content()['p3d']
            compartment.push()
            compartment.diam=rs#cPos.radius
            compartment.L=rs#cPos.radius

            # passive properties
            compartment.cm = kwargs['cm'] if 'cm' in kwargs else 0.9
            compartment.Ra = kwargs['ra'] if 'ra' in kwargs else 200
            compartment.insert('pas')
            compartment.e_pas =  kwargs['e_pas'] if 'e_pas' in kwargs else -65
            compartment.g_pas =  kwargs['g_pas'] if 'g_pas' in kwargs else 1.0/25000
                
            h.pop_section()
            #self._soma = compartment
            return compartment
    #return compartment


