"""
Basic visualization of neurite morphologies. Color coding for individual \
sections is supported. Also synapse can be drawn.

B. Torben-Nielsen @ BBP (updated BTN legacy code)
"""
import sys,time
#sys.path.append('/home/torben/work/epfl/BBP_packages/btmorph')
sys.setrecursionlimit(10000)

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

import btstructs

""" internal constants required for the dendrogram generation """
H_SPACE = 20
V_SPACE = 0
C = 'k'

max_width = 0
max_height = 0

def plot_2D_SWC(tree=None, file_name=None, cs=None, synapses=None, locs=None, syn_cs=None, outN=None, draw_cbar=True,
                draw_scale=True, color_scale=None, num_ticks=5, no_axon=True, my_color='k', special_syn=None,
                syn_labels=None, cbar_orientation='horizontal', cbar_label='Vm', lwidth_factor=1, show_axes=False,
                radial_projection=False, alpha=1.):
    '''
    Colors can be
    None: uniform/default matplotlib color
    Any color code: uniform but specified color
    array of values:
    colormap?
    '''
    import matplotlib.patheffects as PathEffects
    
    xlim = [0,0]
    ylim = [0,0]
    frame1 = plt.gca()
    if file_name != None:
        # read the SWC into a dictionary: key=index, value=(x,y,z,d,parent)
        x = open(file_name,'r')
        SWC = {}
        for line in x :
            if(not line.startswith('#')) :
                splits = line.split()
                index = int(splits[0])
                n_type = int(splits[1])
                x = float(splits[2])
                y = float(splits[3])
                z = float(splits[4])
                r = float(splits[5])
                parent = int(splits[-1])
                SWC[index] = (x,y,z,r,parent,n_type)
                if x > xlim[1]: xlim[1] = x
                if x < xlim[0]: xlim[0] = x
                if y > ylim[1]: ylim[1] = y
                if y < ylim[0]: xlim[0] = y
    elif tree != None :
        SWC = {}
        nodes = tree.get_nodes()
        for node in nodes:
            p3d = node.get_content()['p3d']
            index = p3d.index
            n_type = p3d.type
            x = p3d.x
            y = p3d.y
            z = p3d.z
            r = p3d.radius
            parent = p3d.parent_index
            SWC[index] = (x,y,z,r,parent,n_type)
            if x > xlim[1]: xlim[1] = x
            if x < xlim[0]: xlim[0] = x
            if y > ylim[1]: ylim[1] = y
            if y < ylim[0]: xlim[0] = y
    else:
        print 'Error: input is either \'tree\' or \'filename\''
        exit(1)
    if locs != None:
        # reshape location list
        loc_dict = {}
        for loc in locs:
            loc_dict[loc['node']] = loc['x']
    
    #if use_colors:
        #my_color_list = ['r','g','b','c','m','y','r--','b--','g--', 'y--']
    #else:
        #my_color_list = ['k','k','k','k','k','k','k--','k--','k--', 'k--']
    # for color scale plotting
    if cs == None: 
        pass
    elif color_scale != None:
        max_cs = color_scale[1]
        min_cs = color_scale[0]
        norm_cs = (max_cs - min_cs) * (1. + 1./100.)
    elif isinstance(cs, np.ndarray):
        max_cs = np.max(cs)
        min_cs = np.min(cs)
        norm_cs = (max_cs - min_cs) * (1. + 1./100.)
    elif isinstance(cs, list):
        max_cs = max(cs)
        min_cs = min(cs)
        norm_cs = (max_cs - min_cs) * (1. + 1./100.)
    elif isinstance(cs, dict):
        arr = np.array([cs[key] for key in cs.keys()])
        max_cs = np.max(arr)
        min_cs = np.min(arr)
        norm_cs = (max_cs - min_cs) * (1. + 1./100.)
    else:
        raise Exception('cs type is invalid')
    if cs != None:
        cm = plt.get_cmap('jet')
        Z = [[0,0],[0,0]]
        levels = np.linspace(min_cs, max_cs, 100)
        CS3 = plt.contourf(Z, levels, cmap=cm)
        
    min_y = 100000.0
        
    for index in SWC.keys() : # not ordered but that has little importance here
        # draw a line segment from parent to current point
        current_SWC = SWC[index]
        #print 'index: ', index, ' -> ', current_SWC
        c_x = current_SWC[0]
        c_y = current_SWC[1]
        c_z = current_SWC[2]
        c_r = current_SWC[3]*2.
        parent_index = current_SWC[4]

        if(c_y < min_y) :
            min_y = c_y
                
        if(index <= 3) :
            print 'do not draw the soma and its CNG, 2 point descriptions'
        else :
            if (not no_axon) or (current_SWC[5] !=2):
                parent_SWC = SWC[parent_index]
                p_x = parent_SWC[0]
                p_y = parent_SWC[1]
                p_z = parent_SWC[2]
                p_r = parent_SWC[3]
                if(p_y < min_y) :
                    min_y= p_y
                # print 'index:', index, ', len(cs)=', len(cs)
                if(cs == None) :
                    if radial_projection:
                        pl = plt.plot([np.sqrt(p_x**2+p_z**2), np.sqrt(c_x**2+c_z**2)], [p_y,c_y], c=my_color, linewidth=c_r*lwidth_factor, alpha=alpha)
                    else:
                        pl = plt.plot([p_x,c_x], [p_y,c_y], c=my_color, linewidth=c_r*lwidth_factor, alpha=alpha)
                else :
                    if radial_projection:
                        pl = plt.plot([np.sqrt(p_x**2+p_z**2), np.sqrt(c_x**2+c_z**2)], [p_y,c_y], c=cm((cs[index]-min_cs)/norm_cs), linewidth=c_r*lwidth_factor, alpha=alpha)
                    else:
                        pl = plt.plot([p_x,c_x], [p_y,c_y], c=cm((cs[index]-min_cs)/norm_cs), linewidth=c_r*lwidth_factor, alpha=alpha)
        # add the synapses
        if synapses != None:
            if index in synapses:
                # plot synapse marker
                if syn_cs == None:
                    if radial_projection:
                        plt.plot(np.sqrt(c_x**2+c_z**2),c_y,'ro', markersize=5*lwidth_factor)
                    else:
                        plt.plot(c_x,c_y,'ro', markersize=5*lwidth_factor)
                else:
                    if radial_projection:
                        plt.plot(np.sqrt(c_x**2+c_z**2),c_y, 'o', mfc=syn_cs[index], markersize=5*lwidth_factor)
                    else:
                        plt.plot(c_x,c_y, 'o', mfc=syn_cs[index], markersize=5*lwidth_factor)
                # plot synapse label
                if syn_labels != None and index in syn_labels.keys():
                    txt = frame1.annotate(syn_labels[index], xy=(c_x, c_y), xycoords='data', xytext=(5,5), textcoords='offset points', fontsize='large')
                    txt.set_path_effects([PathEffects.withStroke(foreground="w", linewidth=2)])
        if locs != None:
            if index in loc_dict.keys():
                # plot synapse marker
                p_x = SWC[parent_index][0]
                p_y = SWC[parent_index][1]
                p_z = SWC[parent_index][2]
                if radial_projection:
                    x_plot = p_x + (c_x - p_x) * loc_dict[index]
                    z_plot = p_z + (c_z - p_z) * loc_dict[index]
                    point_plot = np.sqrt(x_plot**2 + z_plot**2)
                else:
                    point_plot = p_x + (c_x - p_x) * loc_dict[index]
                y_plot = p_y + (c_y - p_y) * loc_dict[index]
                if syn_cs == None:
                    plt.plot(point_plot, y_plot, 'ro', markersize=5*lwidth_factor)
                else:
                    plt.plot(point_plot, y_plot, 'o', mfc=syn_cs[index], markersize=5*lwidth_factor)
                # plot synapse label
                if syn_labels != None and index in syn_labels.keys():
                    txt = frame1.annotate(syn_labels[index], xy=(x_plot, y_plot), xycoords='data', xytext=(5,5), textcoords='offset points', fontsize='large')
                    txt.set_path_effects([PathEffects.withStroke(foreground="w", linewidth=2)])
    
    if not show_axes:
        frame1.axes.get_xaxis().set_ticks([])
        frame1.axes.get_yaxis().set_ticks([])
        frame1.set_xlabel('X')
        frame1.set_ylabel('Y')

    frame1.axes.get_xaxis().set_visible(show_axes)
    frame1.axes.get_yaxis().set_visible(show_axes)
    frame1.axison = show_axes

    # draw a scale bar
    if draw_scale:
        scale = 100
        plt.plot([0,scale],[min_y*1.1,min_y*1.1],'k',linewidth=5) # 250 for MN, 100 for granule
        txt = frame1.annotate(r'' + str(scale) + ' $\mu$m', xy=(scale/2., min_y*1.1), xycoords='data', xytext=(-28,8), textcoords='offset points', fontsize='medium')
        txt.set_path_effects([PathEffects.withStroke(foreground="w", linewidth=2)])
        #~ frame1.text(, min_y+(ylim[1]-ylim[0])/30, str(scale) + ' um')
    
    if(cs != None and draw_cbar) :
        # so that colorbar works with tight_layout
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(frame1)
        if cbar_orientation=='horizontal':
            cax = divider.append_axes("bottom", "5%", pad="3%")
        else:
            cax = divider.append_axes("right", "5%", pad="3%")
        cb = plt.colorbar(None, cax=cax, orientation=cbar_orientation)
        ticks_f = np.round(np.linspace(min_cs, max_cs, num_ticks+2), decimals=1)
        #~ print ticks_f
        ticks_i = ticks_f
        cb.set_ticks(ticks_i)
        if cbar_orientation=='horizontal':
            cb.ax.xaxis.set_ticks_position('bottom')
        cb.set_label(r'$V_m$ (mV)')

    if(outN != None) :
        plt.savefig(outN)
    
    #~ if ax != None:
        #~ ax = frame1
    
    return frame1
    

def plot_collapsed_to_1D(tree=None, values=None, special_nodes=None, special_node_labels=None,
    lwidth_factor=1.):
    """
    plot figure
    """
    ax = plt.gca()
    somanode = tree.get_node_with_index(1)
    # soma is at zero
    ax.plot(0., values[1], 'kD', ms=4.*lwidth_factor)
    
    somachildren = somanode.get_child_nodes()
    for cnode in somachildren[2:]:
        p3d0 = somanode.get_content()['p3d']
        p3d1 = cnode.get_content()['p3d']
        D = np.linalg.norm(np.array([p3d0.x-p3d1.x, p3d0.y-p3d1.y, p3d0.z-p3d1.z]))
        ax.plot([0,D], [values[1], values[cnode._index]], 'k', lw=lwidth_factor)
        if (special_nodes != None) and (cnode._index in special_nodes):
            ax.plot(D, values[cnode._index], 'ro', ms=4.*lwidth_factor)
        _plot_values_up(cnode, D, values, ax, special_nodes, special_node_labels)
        

def _plot_values_up(node, D, values, ax, special_nodes=None, special_node_labels=None, lwidth_factor=1.):
    cnodes = node.get_child_nodes()
    if cnodes:
        for cnode in cnodes:
            p3d0 = node.get_content()['p3d']
            p3d1 = cnode.get_content()['p3d']
            Dnew = np.linalg.norm(np.array([p3d0.x-p3d1.x, p3d0.y-p3d1.y, p3d0.z-p3d1.z]))
            Dnew += D
            ax.plot([D,Dnew], [values[node._index], values[cnode._index]], 'k', lw=lwidth_factor)
            if (special_nodes != None) and (cnode._index in special_nodes):
                ax.plot(Dnew, values[cnode._index], 'ro', ms=4.*lwidth_factor)
            _plot_values_up(cnode, Dnew, values, ax, special_nodes, special_node_labels, lwidth_factor)


def plot_3D_SWC(file_name='P20-DEV139.CNG.swc',cs=None,synapses=None,outN=None) :
    """
    Matplotlib rendering of a SWC described morphology in 3D
    Colors can be
    None: uniform/default matplotlib color
    Any color code: uniform but specified color
    array of values:
    colormap?
    """
    my_color_list = ['r','g','b','c','m','y','r--','b--','g--']
    
    if(cs == None) :
        pass
    else :
        norm = colors.normalize(np.min(cs),np.max(cs))
        Z = [[0,0],[0,0]]
        levels=range(int(np.min(cs)),int(np.max(cs)),1)
        levels = np.linspace(np.min(cs),np.max(cs),1)
        CS3 = plt.contourf(Z,levels,cmap=cm.jet)
        plt.clf()
    
    # read the SWC into a dictionary: key=index, value=(x,y,z,d,parent)
    x = open(file_name,'r')
    SWC = {}
    for line in x :
        if(not line.startswith('#')) :
            splits = line.split()
            index = int(splits[0])
            n_type = int(splits[1])
            x = float(splits[2])
            y = float(splits[3])
            z = float(splits[4])
            r = float(splits[5])
            parent = int(splits[-1])
            SWC[index] = (x,y,z,r,parent,n_type)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for index in SWC.keys() : # not ordered but that has little importance here
        # draw a line segment from parent to current point
        current_SWC = SWC[index]
        #print 'index: ', index, ' -> ', current_SWC
        c_x = current_SWC[0]
        c_y = current_SWC[1]
        c_z = current_SWC[2]
        c_r = current_SWC[3]
        parent_index = current_SWC[4]
                
        if(index <= 3) :
            print 'do not draw the soma and its CNG, !!! 2 !!! point descriptions'
        else :
            parent_SWC = SWC[parent_index]
            p_x = parent_SWC[0]
            p_y = parent_SWC[1]
            p_z = parent_SWC[2]
            p_r = parent_SWC[3]
            # print 'index:', index, ', len(cs)=', len(cs)
            if cs == None :
                pl = plt.plot([p_x,c_x],[p_y,c_y],[p_z,c_z],my_color_list[current_SWC[5]-1],linewidth=c_r)
            else :
                try :
                    pl = plt.plot([p_x,c_x],[p_y,c_y], \
                                  c=cm.jet(norm(cs[index])),linewidth=c_r)
                except Exception :
                    print 'something going wrong here'
                    # pass# it's ok: it's the list size...

        # add the synapses
        if(synapses != None) :
            if(index in synapses) :
                plt.plot(c_x,c_y,'ro')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    if(cs != None) :
        cb = plt.colorbar(CS3) # bit of a workaround, but it seems to work
        ticks_f = np.linspace(np.min(cs)-1,np.max(cs)+1,5)
        ticks_i = map(int,ticks_f)
        cb.set_ticks(ticks_i)

    if(outN != None) :
        plt.savefig(outN)

def plot_dendrogram(file_name,transform='plain',shift=0,c='k',radius=True,rm=20000.0,ra=200,outN=None) :
    global C, RM, RA, max_width, max_height # n.a.s.t.y.
    '''
    Generate a dendrogram from an SWC file. The SWC has to be formatted with a "three point soma"
    '''
    swc_tree = btstructs.STree()
    swc_tree = swc_tree.read_SWC_tree_from_file(file_name)
    RM = rm
    RA = ra
    C = c
    max_height = 0
    max_width = 0
    plt.clf()
    print 'Going to build the dendrogram. This might take a while...'
    ttt = time.time()
    _expand_dendrogram(swc_tree.get_root(),swc_tree,shift,0,radius=radius,transform=transform)
    if(transform == 'plain') :
        plt.ylabel('L (micron)')
    elif(transform == 'lambda') :
        plt.ylabel('L (lambda)')
    print (time.time() - ttt), ' later the dendrogram was finished. '

    print 'max_widht=%f, max_height=%f' % (max_width,max_height)
    x_bound = (max_width / 2.0) + (0.1*max_width)
    max_y_bound = max_height + 0.1*max_height
    plt.axis([-1.0*x_bound,x_bound,-0.1*max_height,max_y_bound])

    plt.plot([x_bound,x_bound],[0,100],'k', linewidth=5) # 250 for MN, 100 for granule

    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    
    if(outN != None) :
        plt.savefig(outN)

def _expand_dendrogram(cNode,swc_tree,off_x,off_y,radius,transform='plain') :
    global max_width,max_height # middle name d.i.r.t.y.
    '''
    Gold old fashioned recursion... sys.setrecursionlimit()!
    '''
    place_holder_h = H_SPACE
    max_degree = swc_tree.degree_of_node(cNode)
    required_h_space = max_degree * place_holder_h
    start_x = off_x-(required_h_space/2.0)
    if(required_h_space > max_width) :
        max_width = required_h_space

    
    children = cNode.get_child_nodes()

    if swc_tree.is_root(cNode) :
        print 'i am expanding the root'
        children.remove(swc_tree.get_node_with_index(2))
        children.remove(swc_tree.get_node_with_index(3))
    
    for cChild in  children :
        l = _path_between(swc_tree,cChild,cNode,transform=transform)
        r = cChild.get_content()['p3d'].radius

        cChild_degree = swc_tree.degree_of_node(cChild)
        new_off_x = start_x + ( (cChild_degree/2.0)*place_holder_h )
        new_off_y = off_y+(V_SPACE*2)+l
        r = r if radius  else 1
        plt.vlines(new_off_x,off_y+V_SPACE,new_off_y,linewidth=r,colors=C)
        if((off_y+(V_SPACE*2)+l) > max_height) :
            max_height = off_y+(V_SPACE*2)+l

        _expand_dendrogram(cChild,swc_tree,new_off_x,new_off_y,radius=radius,transform=transform)

        start_x = start_x + (cChild_degree*place_holder_h)
        plt.hlines(off_y+V_SPACE,off_x,new_off_x,colors=C)

def _path_between(swc_tree,deep,high,transform='plain') :
    path = swc_tree.path_to_root(deep)
    pl = 0
    pNode = deep
    for node in path[1:] :
        pPos = pNode.get_content()['p3d']
        cPos = node.get_content()['p3d']
        pl += np.sqrt( (pPos.x - cPos.x)**2 + (pPos.y - cPos.y)**2 + (pPos.z - cPos.z)**2 )
        pNode = node
        if(node == high) : break
        
    if(transform == 'plain'):
        return pl
    elif(transform == 'lambda') :
        DIAM = (deep.get_content()['p3d'].radius*2.0 + high.get_content()['p3d'].radius*2.0) /2.0 # naive...
        c_lambda = np.sqrt(1e+4*(DIAM/4.0)*(RM/RA))
        return pl / c_lambda
