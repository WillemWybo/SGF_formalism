"""
Module contains:
    P3D
    SNode   
    STree

B. Torben-Nielsen, 2013-01 @ BBP (from BTN legacy code)
"""

        
import copy


class P3D:
    """
    Wrapper for 3D positions
    A P3D contains all information required for the SWC format,
    plus, order / degree / path length in case the user sets it.
    """
        
    def __init__(self,x=0,y=0,z=0,radius=1,type=7) :
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        self.type = type
        
    def set_index(self,index) :
        self.index = index
        
    def set_parent_index(self,index) :
        self.parent_index = index
        
    def set_order(self,order) :
        self.order=order
        
    def distance(self,pos) :
        dx = self.x - pos.x
        dy = self.y - pos.y
        dz = self.z - pos.z
        dist = np.sqrt(dx*dx + dy*dy + dz*dz)
        del dx; del dy; del dz
        return dist

    def __str__(self) :
        return 'P3D: '+str(self.x)+','+str(self.y)+','+str(self.z)

class SNode :
    """
    Simple Node for use with a simple Tree (STree)
    By design, the "_content" should be a dictionary. (2013-03-08)
    """
    
    def __init__(self,index) :
        self._parent_node = None
        self._index = index
        self._child_nodes = []
        
    def get_index(self) :
        return self._index
        
    def get_parent_node(self) :
        return self._parent_node
        
    def get_child_nodes(self) :
        return self._child_nodes
        
    def get_content(self) :
        return self._content
    
    def set_index(index) :
        self._index = index
    
    def set_parent_node(self,parent_node) :
        self._parent_node = parent_node
        
    def set_content(self,content) :
        if isinstance(content,dict) :
            self._content = content 
        else :
            raise Exception("SNode.set_content must receive a dict")    
        
    def add_child(self,child_node) :
        self._child_nodes.append(child_node)
            
    def make_empty(self):
        self._parent_node = None
        self._content = None
        self._child_nodes = []
            
    def remove_child(self, child_node) :
        self._child_nodes.remove(child_node)
        
    def __str__(self) :
        return 'SNode (ID: '+str(self._index)+')'

    # def swc_str(self) :
    #     return 'SNode (ID: '+str(self._index)+', PID: '+ \
    #       str(self._parent_node.get_index())+'): '+ str(self._content)    

    def __lt__(self,other):
        if self._index < other._index :
            return True
    def __le__(self,other):
        if self._index <= other._index :
            return True
    def __gt__(self,other):
        if self._index > other._index :
            return True
    def __ge__(self,other):
        if self._index >= other._index :
            return True
    
    def __copy__(self) : # customization of copy.copy
        ret = SNode(self._index)
        for child in self._child_nodes :
            ret.add_child(child)
        ret.content = self._content
        ret.set_parent_node(self._parent_node)
        return ret
        
class STree :
    """
    Simple tree for use with a simple Node (SNode)
    """
    
    def __init__(self) :
        """ Initialize an empty tree by default
        """
        _root = None
        
    def set_root(self,node) :
        self._root = node
        self._root.set_parent_node(None)
        
    def get_root(self) :
        return self._root
        
    def is_root(self,node) :
        if node.get_parent_node() != None :
            return False
        else :
            return True
            
    def is_leaf(self,node) :
        if len(node.get_child_nodes()) == 0  :
            return True
        else :
            return False
            
    def add_node_with_parent(self,node,parent) :
        node.set_parent_node(parent)
        parent.add_child(node)
        
    def remove_node(self,node) :
        node.get_parent_node().remove_child(node)
        self._deep_remove(node)
                    
    def _deep_remove(self,node) :
        children = node.get_child_nodes()
        node.make_empty()
        for child in children :
            self._deep_remove(child)        

    def get_nodes(self, somanodes=True) :
        n = []
        self._gather_nodes(self._root,n)
        if somanodes: 
            return n
        else:
            return n[0:1]+n[3:] 

    def get_nodes(self,somanodes=True) :
        n = []
        self._gather_nodes(self._root,n)
        if somanodes: 
            return n 
        else:
            return n[0:1] + n[3:]
        
    def _gather_nodes(self,node,node_list) :
        node_list.append(node)
        for child in node.get_child_nodes() :
            self._gather_nodes(child,node_list)
            
    def get_sub_tree(self,fake_root) :
        ret = STree()
        cp = fake_root.__copy__()
        cp.set_parent_node(None)
        ret.set_root(cp)
        return ret
            
    def get_node_with_index(self, index) :
        return self._find_node(self._root,index)
        
    def get_node_in_subtree(self,index,fake_root) :
        return self._find_node(fake_root,index)
        
    def _find_node(self,node,index) :
        """
        Sweet breadth-first/stack iteration to replace the recursive call. 
        Traverses the tree until it finds the node you are looking for.     
        Returns SNode when found and None when not found
        """
        stack = []; 
        stack.append(node)
        while(len(stack) != 0) :
            for child in stack :
                if child.get_index() == index  :
                    return child
                else :
                    stack.remove(child)
                    for cchild in child.get_child_nodes() :
                        stack.append(cchild)
        return None # Not found!
        
    def degree_of_node(self,node) :
        sub_tree = self.get_sub_tree(node)
        st_nodes = sub_tree.get_nodes()
        leafs = 0
        for n in st_nodes :
            if sub_tree.is_leaf(n) :
                leafs = leafs +1
        return leafs
        
    def order_of_node(self,node) :
        ptr =self.path_to_root(node)
        order = 0
        for n in ptr :
            if len(n.get_child_nodes()) > 1  :
                order = order +1
        """ order is on [0,max_order] thus subtract 1 from this calculation """
        return order -1 
                
    def path_to_root(self,node) :
        n = []
        self._go_up_from(node,n)            
        return n
        
    def _go_up_from(self,node,n):
        n.append(node)
        p_node = node.get_parent_node()
        if p_node != None :
            self._go_up_from(p_node,n)

    def path_between_nodes(self,from_node,to_node) :
        """
        Find the path between two nodes. The from_node needs to be of higher \
        order than the to_node. In case there is no path between the nodes, \
        the path from the from_node to the soma is given.
        """
        n = []
        self._go_up_from_until(from_node,to_node,n)
        return n

    def _go_up_from_until(self,from_node,to_node,n) :
        n.append(from_node)
        if from_node == to_node :
            return
        p_node = from_node.get_parent_node()
        if p_node != None :
            self._go_up_from_until(p_node,to_node,n)
            
    def find_nearest_neighbours(self, node, nodes):
        '''
        searches for the nearest neighbours of node higher up in the tree \
        (the ones on a direct path from node) in the list nodes. \
        If no such node is found, the nearest neighbour is either a leaf, \
        and that leaf is returned. If node is a bifurcation, the nearest \
        neighbours in seperate subtrees are returned in seperate lists. \
        If node is a leaf, an empty list is returned
        '''
        childnodes = node.get_child_nodes()
        if node._index == 1:
            childnodes = childnodes[2:]
        nearest = [[] for node in childnodes]
        for ind, child in enumerate(childnodes):
            if child not in nodes:
                self._search_neighbour_up(child, nodes, nearest[ind])
            else:
                nearest[ind].append(child)
        return nearest
    
    def _search_neighbour_up(self, node, nodes, nearest):
        childnodes = node.get_child_nodes()
        if childnodes:
            for child in childnodes:
                if child not in nodes:
                    self._search_neighbour_up(child, nodes, nearest)
                else:
                    nearest.append(child)
        else:
            nearest.append(node) # node is leaf
            
    def remove_trifuractions(self):
        pnode = self.get_root()
        cnodes = pnode.get_child_nodes()
        for node in cnodes[2:]:
            self._check_for_trifurcations(node)
            
    def _check_for_trifurcations(self, node):
        cnodes = copy.copy(node.get_child_nodes())
        if len(cnodes)==3:
            # adds a newparent for node 1 and 2 in cnodes
            cn1 = cnodes[1]; cn2 = cnodes[2]
            #~ cnodes = copy.deepcopy(cnodes) # reference crap
            newparent = SNode(2*self.max_index)
            self.max_index += 2
            newcontent = copy.deepcopy(node.get_content())
            p3d = newcontent['p3d']
            p3d.x += 1;
            newparent.set_content(newcontent)
            newparent.get_content()['p3d'].set_index(newparent._index)
            newparent.get_content()['p3d'].set_parent_index(node._index)
            self.add_node_with_parent(newparent, node)
            cn1.set_parent_node(newparent)
            cn2.set_parent_node(newparent)
            newparent.add_child(cn1)
            newparent.add_child(cn2)
            node.remove_child(cn1)
            node.remove_child(cn2)
            cn1.get_content()['p3d'].set_parent_index(newparent._index)
            cn2.get_content()['p3d'].set_parent_index(newparent._index)
            #~ subtree = self.get_sub_tree(newparent)
            #~ self._increase_indices_in_subtree(newparent)
        for cnode in cnodes:
            self._check_for_trifurcations(cnode)
    
    #~ def _increase_indices_in_subtree(self, node):
        #~ for cnode in node.get_child_nodes():
            #~ cnode._index += 1
            #~ cnode.get_content()['p3d'].set_index(cnode._index)
            #~ cnode.get_content()['p3d'].set_parent_index(cnode.get_parent_node()._index)
            #~ self._increase_indices_in_subtree(cnode)
        
    def write_SWC_tree_to_file(self,file_n) :
        """
        Save a tree to an SWC file
        TODO
        """
        raise Exception("Not yet implemented")
        writer = open(file_n,'w')
        nodes = self.get_nodes()
        nodes.sort()
        for node in nodes :
            p3d = node.get_content()['p3d'] # update 2013-03-08
            p3d_string = p3d.swc_str()
            print 'p3d_string: ', p3d_string
            writer.write( p3d_string + '\n' )
            writer.flush()
        writer.close()          
        #print 'STree::writeSWCTreeToFile -> finished. Tree in >',fileN,'<'
        
    def read_SWC_tree_from_file(self,file_n,axon=False) :
        """
        Read and load a morphology from an SWC file. 
        On the NeuroMorpho.org website, 5 types of somadescriptions are 
        considered. (http://neuromorpho.org/neuroMorpho/SomaFormat.html)
        Arguments:
        - file_n : file name
        - soma_type : [1-5], see specs
        """
        file = open(file_n,'r')
        all_nodes = dict()
        for line in file :
            if not line.startswith('#') :
                split= line.split()
                index = int(split[0].rstrip())
                type = int(split[1].rstrip())
                x = float(split[2].rstrip())
                y = float(split[3].rstrip())
                z = float(split[4].rstrip())
                radius = float(split[5].rstrip())
                parent = int(split[6].rstrip())
                
                if type!=2:
                    tP3D = P3D(x=x,y=y,z=z,radius=radius,type=type)
                    tP3D.set_index(index)
                    tP3D.set_parent_index(parent)
                    t_node = SNode(index)
                    t_node.set_content({'p3d':tP3D})
                    all_nodes[index] = t_node
                elif axon and type==2:
                    tP3D = P3D(x=x,y=y,z=z,radius=radius,type=type)
                    tP3D.set_index(index)
                    tP3D.set_parent_index(parent)
                    t_node = SNode(index)
                    t_node.set_content({'p3d':tP3D})
                    all_nodes[index] = t_node
        self.max_index = 0
        for index, node in all_nodes.items() :
            parent_index = node.get_content()['p3d'].parent_index          
            parent_node = all_nodes.get(parent_index)
            if index == 1 :
                self.set_root(node)
            else :
                self.add_node_with_parent(node,parent_node)
            if index > self.max_index:
                self.max_index = index
                
        return self

                        
        
    
