import numpy as np
import torch
import copy
# @torch.jit.script
class Individuals():
  """Individual class: genes, network, and fitness
  """ 
  def __init__(self, conn, node,rng, weights=[]):
    """Intialize individual with given genes
    Args:
      node - (np_array) - node genes
            [4 X nUniqueGenes]
            [0,:] => Node Id
            [1,:] => Type (1=input, 2=output 3=hidden 4=bias)
            [2,:] => Activation function (as int)
            [3,:] => architecture component tracker

      conn - (np_array) - connection genes
            [6 X nUniqueGenes] 
            [0,:] => Innovation Number (unique Id)
            [1,:] => Source Node Id
            [2,:] => Destination Node Id
            [3,:] => Weight Value
            [4,:] => Enabled?
            [5,:] => architecture component tracker
  
    Attributes:
      node    - (np_array) - node genes (see args)
      conn    - (np_array) - conn genes (see args)
      nInput  - (int)      - number of inputs
      nOutput - (int)      - number of outputs
      wMat    - (np_array) - weight matrix, one row and column for each node
                [N X N]    - rows: connection from; cols: connection to
      wVec    - (np_array) - wMat as a flattened vector
                [N**2 X 1]    
      aVec    - (np_array) - activation function of each node (as int)
                [N X 1]    
      nConn   - (int)      - number of connections
      fitness - (double)   - fitness averaged over all trials (higher better)
      X fitMax  - (double)   - best fitness over all trials (higher better)
      rank    - (int)      - rank in population (lower better)
      birth   - (int)      - generation born

    """
    self.rng     = rng
    self.node    = np.copy(node)
    self.conn    = np.copy(conn)
    self.nInput  = sum(node[1,:]==1)
    self.nOutput = sum(node[1,:]==2)
    self.wMat    = []
    self.wVec    = []
    self.aVec    = []
    self.vKey    = []
    self.nConn   = 0
    self.nNodes  = 0
    self.fitness = [] # Mean fitness over trials
    self.fitMax  = [] # Best fitness over trials
    self.rank    = []
    # self.gen     = gen
    self.w_adjusted = weights
    
    self.fun = None
    

  def setFun(self,fn):
    if fn !=None:
      self.fun = fn
    
  def express(self):
    """Converts genes to weight matrix and activation vector
    """
    order, wMat = getNodeOrder(self.node, self.conn)
    self.order = order
    if order is False:
      return False
    else:
      wMat[np.isnan(wMat)] = 0
      
      self.aVec = self.node[2,order]
      nNodes = self.aVec.shape[0] # saving NN number of nodes
      self.node_track = np.zeros([4,nNodes],dtype=int)
      self.node_track[0,:] = np.arange(0,nNodes)       # new index 
      self.node_track[1,:] = order                          # storing original order sorted
      self.node_track[2,:] = self.node[1,order].astype(int) # tracking node type
      # self.node_track[3,:] = self.node[3,order].astype(int) # tracking architecture stage

      wVec = wMat.flatten()
      wVec[np.isnan(wVec)] = 0
      vKey = np.where(wVec!=0)[0]
      self.nConn = np.sum(wVec!=0)
        
      if len(self.w_adjusted) > 0:
        wVec[vKey] = self.w_adjusted
        wMat = np.reshape(wVec,(nNodes,nNodes))
        
      self.wVec  = wVec
      self.vKey  = vKey
      self.wMat  = wMat
      self.nNodes= nNodes
        
      return True

      
  
    
  def createChild(self, p, rng, weights=None):
    """Create new individual with this individual as a parent

      Args:
        p      - (dict)     - algorithm hyperparameters (see p/hypkey.txt)


    Returns:
        child  - (Ind)      - newly created individual

    """     
    child = Individuals(self.conn, self.node, rng)
    child.setFun(self.fun)
    child.mutateArch(p)
    return child

# -- 'Single Weight Network' topological mutation ------------------------ -- #

  def mutateArch(self, p):
    """Randomly alter topology of individual
    Note: This operator forces precisely ONE topological change 

    Args:
      p      - (dict)     - algorithm hyperparameters (see p/hypkey.txt)

    Returns:
        child   - (Ind)      - newly created individual

    """

    # Choose topological mutation
    topoRoulette = np.array((p['prob_addConn'], p['prob_addNode'], \
                             p['prob_enable'] , p['prob_mutAct']))

    spin = self.rng.random()*np.sum(topoRoulette)
    slot = topoRoulette[0]
    choice = topoRoulette.size
    for i in range(1,topoRoulette.size):
      if spin < slot:
        choice = i
        break
      else:
        slot += topoRoulette[i]

    # Add Connection
    if choice == 1:
      if p['mut_new_conns'] == 0:
        self.mutAddConn(p)             
      else:  
        self.mutAddConn2(p,nNewConns=p['mut_new_conns'])
 

    # Add Node
    elif choice == 2:
      self.mutAddNode(p)

    # Enable Connection
    elif choice == 3:
      self.mutEnableConn(p)

    # Mutate Activation
    elif choice == 4:
      self.mutActivation2(p)
#       self.fun(self,p) 


#   def stagnationFree(self, p, increase_number=20):
#     for _ in range(increase_number):
#         self.mutAddNode(p)
    
#     for _ in range(increase_number):copy
#       self.mutAddConn(p)

#     self.express()

  def mutAddNode(self,p):
    """Add new node to the agent NN

    Args:
      p        - (dict)     - algorithm hyperparameters (see p/hypkey.txt)

    Returns:
      connG    - (np_array) - updated connection genes
      nodeG    - (np_array) - updated node genes

    """
    connG = self.conn
    nodeG = self.node

    new_nodeId = int(max(nodeG[0,:]+1))
    new_connId = connG[0,-1]+1    
       
    # Choose connection to split
    conn_active = connG[0,(connG[4,:] == 1)].astype(int)

    
    if len(conn_active) < 1:
      return connG, nodeG # No active connections, nothing to split
    conn_split  = conn_active[self.rng.integers(len(conn_active))]
    
    # Create new node
    new_activation = p['nn_ActFunc'][self.rng.integers(len(p['nn_ActFunc']))]
    new_node = np.array([[new_nodeId, 3, new_activation]]).T
    
    # Add connections to and from new node
    # -- Effort is taken to minimize disruption from node addition:
    # The 'weight to' the node is set to 1, the 'weight from' is set to the
    # original  weight. With a near linear activation function the change in performance should be minimal.

    conn_to    = connG[:,conn_split].copy()
    conn_to[0] = new_connId
    conn_to[2] = new_nodeId
    conn_to[3] = 1 # weight set to 1
      
    conn_from    = connG[:,conn_split].copy()
    conn_from[0] = new_connId + 1
    conn_from[1] = new_nodeId
    conn_from[3] = connG[3,conn_split] # weight set to previous weight value
        
    new_conns = np.vstack((conn_to,conn_from)).T
        
   
    # Disable original connection
    connG[4,conn_split] = 0
    
          
    # Add new structures to genome
    nodeG = np.hstack((nodeG,new_node))
    connG = np.hstack((connG,new_conns))

    self.conn = connG
    self.node = nodeG

    if p['mut_new_conns'] > 0:
      self.mutAddConn2(p,nNewConns=p['mut_new_conns'],destSelected=new_nodeId)
    
  
  def mutAddConn(self, p):
    """Add new connection to genome.
    To avoid creating recurrent connections all nodes are first sorted into
    layers, connections are then only created from nodes to nodes of the same or
    later layers.

    Args:
      p        - (dict)     - algorithm hyperparameters (see p/hypkey.txt)


    Returns:
      connG    - (np_array) - updated connection genes

    """
    connG = self.conn
    nodeG = self.node
    newConnId = connG[0,-1]+1


    nIns = len(nodeG[0,nodeG[1,:] == 1]) + len(nodeG[0,nodeG[1,:] == 4])
    nOuts = len(nodeG[0,nodeG[1,:] == 2])
    order, wMat = getNodeOrder(nodeG, connG)   # Topological Sort of Network
    hMat = wMat[nIns:-nOuts,nIns:-nOuts]
    hLay = getLayer(hMat)+1

    # To avoid recurrent connections nodes are sorted into layers, and connections are only allowed from lower to higher layers
    if len(hLay) > 0:
      lastLayer = max(hLay)+1
    else:
      lastLayer = 1
    L = np.r_[np.zeros(nIns), hLay, np.full((nOuts),lastLayer) ]
    nodeKey = np.c_[nodeG[0,order], L] # Assign Layers

    sources = self.rng.permutation(len(nodeKey))
    for src in sources:
      srcLayer = nodeKey[src,1]
      dest = np.where(nodeKey[:,1] > srcLayer)[0]
      
      # Finding already existing connections:
      #   ) take all connection genes with this source (connG[1,:])
      #   ) take the destination of those genes (connG[2,:])
      #   ) convert to nodeKey index (Gotta be a better numpy way...)   
      srcIndx = np.where(connG[1,:]==nodeKey[src,0])[0]
      exist = connG[2,srcIndx]
      existKey = []
      for iExist in exist:
        existKey.append(np.where(nodeKey[:,0]==iExist)[0])
      dest = np.setdiff1d(dest,existKey) # Remove existing connections
      
      # Add a random valid connection
      self.rng.shuffle(dest)
      if len(dest)>0:  # (there is a valid connection)
        connNew = np.empty((5,1))
        connNew[0] = newConnId
        connNew[1] = nodeKey[src,0]
        connNew[2] = nodeKey[dest[0],0]
        connNew[3] = (self.rng.random()-0.5)*2*p['cae_Sw_lim']
        connNew[4] = 1
        connG = np.c_[connG,connNew]
        break;

    self.conn = connG
    self.node = nodeG


  def mutEnableConn(self,p):
    """Enable a connection to the agent NN

    Args:

    Returns:
      connG    - (np_array) - updated connection genes

    """
    connG = self.conn
    
    disabled = np.where(connG[4,:] == 0)[0]
    if len(disabled) > 0:
      enable = self.rng.integers(len(disabled))
      connG[4,disabled[enable]] = 1
    self.conn = connG

  def mutActivation2(self,p):
    """Changes the activation function of node in the agent NN

    Args:
      p        - (dict)     - algorithm hyperparameters (see p/hypkey.txt)
    Returns:
      connG    - (np_array) - updated connection genes

    """
    nodeG = self.node
    connG = self.conn

    # selecting active connections
    conn_active = connG[:,((connG[4,:] == 1))].astype(int)

    # addressing nodes with active connections inputs
    node_active_idx = np.unique(conn_active[2,:])
    # node_selected_idx = node_active[0,node_active].astype(int)

    # selecting which node to mutate to a new activation function
    # if there are active nodes
    if len(node_active_idx) > 0:
        mut_act_node = self.rng.choice(node_active_idx)
        act_fun = [int(nodeG[2,mut_act_node])]
        available_act = list(p['nn_ActFunc'])
        # new_act_pool = [r for r in act_fun+available_act if (r not in act_fun) or (r not in available_act)]
        new_act_pool = []
        for r in act_fun+available_act:
            if (r not in act_fun) or (r not in available_act):
                new_act_pool = np.append(new_act_pool,r)

        # changing to a different activation function
        nodeG[2,mut_act_node] = self.rng.choice(new_act_pool)
    
    self.node = nodeG

 
  def mutAddConn2(self, p, nNewConns = 3, destSelected = None):
    """Add new connection to genome.
    To avoid creating recurrent connections all nodes are first sorted into
    layers, connections are then only created from nodes to nodes of the same or
    later layers.

    Args:
      p        - (dict)     - algorithm hyperparameters (see p/hypkey.txt)


    Returns:
      connG    - (np_array) - updated connection genes

    """
    connG = self.conn
    nodeG = self.node
    newConnId = connG[0,-1]+1


    nIns = len(nodeG[0,nodeG[1,:] == 1]) + len(nodeG[0,nodeG[1,:] == 4])
    nOuts= len(nodeG[0,nodeG[1,:] == 2])
    order, wMat = getNodeOrder(nodeG, connG)   # Topological Sort of Network
    hMat = wMat[nIns:-nOuts,nIns:-nOuts]
    hLay = getLayer(hMat)+1

    # To avoid recurrent connections nodes are sorted into layers, and connections are only allowed from higher to lower layers
    lastLayer = 1
    if len(hLay) > 0:
      lastLayer = max(hLay)+1

    L = np.r_[np.zeros(nIns), hLay, np.full((nOuts),lastLayer) ]
    nodeKey = np.c_[nodeG[0,order], L] # list of nodes and layer assignation

    destList = nodeKey[nodeKey[:,1] > 0,0] # Selecting all nodes that are not the input or bias layer
    destNode = self.rng.choice(destList,size=1,replace=False)
    if destSelected != None:
      if destSelected in destList:
        destNode = np.array(destSelected)
      else:
        raise NameError('destination node selected does not exist in destList')
        
    # getting all the connections already exist to the destination node and removing from the source list
    connExistIdx = np.where(connG[2,:] == destNode)
    connExists = connG[1,connExistIdx]
    existLay   = nodeKey[nodeKey[:,0]==destNode,1] 
    sourceList = nodeKey[nodeKey[:,1] < existLay][:,0]
    sourceList = np.setdiff1d(sourceList,connExists) # Remove existing connections

    #Checking there is enough source nodes for nNewconns
    sourceSize = sourceList.shape[0]
    if sourceSize < nNewConns:
      nNewConns = 0
      if sourceSize > 0:
        nNewConns = sourceSize

    # creating a list of source nodes times nNweConns
    newSources = self.rng.choice(sourceList,size=nNewConns,replace=False)
    # creating new connections and adding to connGroup
    for i in range(nNewConns):
      connNew = np.empty((6,1))
      connNew[0] = newConnId+i
      connNew[1] = newSources[i]
      connNew[2] = destNode
      connNew[3] = 1
      connNew[4] = 1
      connG = np.c_[connG,connNew]       

    self.conn = connG
    self.node = nodeG

def getNodeOrder(nodeG,connG):
  """Builds connection matrix from genome through topological sorting.

  Args:
    nodeG - (np_array) - node genes
            [4 X nUniqueGenes]
            [0,:] => Node Id
            [1,:] => Type (1=input, 2=output 3=hidden 4=bias)
            [2,:] => Activation function (as int)
            [3,:] => architecture component tracker

    connG - (np_array) - connection genes
            [6 X nUniqueGenes] 
            [0,:] => Innovation Number (unique Id)
            [1,:] => Source Node Id
            [2,:] => Destination Node Id
            [3,:] => Weight Value
            [4,:] => Enabled?
            [5,:] => architecture component tracker

  Returns:
    Q    - [int]      - sorted node order as indices
    wMat - (np_array) - ordered weight matrix
           [N X N]

    OR

    False, False      - if cycle is found
  """
#   conn = np.copy(connG)
#   node = np.copy(nodeG)
#   nIns = len(node[0,node[1,:] == 1]) + len(node[0,node[1,:] == 4])
#   nOuts = len(node[0,node[1,:] == 2])
  
#   # Create connection and initial weight matrices
#   conn[3,conn[4,:]==0] = np.nan # disabled but still connected
#   src  = conn[1,:].astype(int)
#   dest = conn[2,:].astype(int)
  
#   lookup = node[0,:].astype(int)
#   for i in range(len(lookup)): 
#     src[np.where(src==lookup[i])] = i
#     dest[np.where(dest==lookup[i])] = i
  
#   wMat = np.zeros((np.shape(node)[1],np.shape(node)[1]))
#   wMat[src,dest] = conn[3,:]
#   connMat = wMat[nIns+nOuts:,nIns+nOuts:]
#   connMat[connMat!=0] = 1

#   # Topological Sort of Hidden Nodes
#   edge_in = np.sum(connMat,axis=0)
#   Q = np.where(edge_in==0)[0]  # Start with nodes with no incoming connections
#   for i in range(len(connMat)):
#     if (len(Q) == 0) or (i >= len(Q)):
#       Q = []
#       return False, False # Cycle found, can't sort
#     edge_out = connMat[Q[i],:]
#     edge_in  = edge_in - edge_out # Remove nodes' conns from total
#     nextNodes = np.setdiff1d(np.where(edge_in==0)[0], Q)
#     Q = np.hstack((Q,nextNodes))

#     if sum(edge_in) == 0:
#       break
  
#   # Add In and outs back and reorder wMat according to sort
#   Q += nIns+nOuts
#   Q = np.r_[lookup[:nIns], Q, lookup[nIns:nIns+nOuts]]
#   wMat = wMat[np.ix_(Q,Q)]
  
  conn = np.copy(connG)
  node = np.copy(nodeG)

  in_idx  = node[0,((node[1,:] == 1) | (node[1,:] == 4))].astype(int)
  out_idx = node[0,node[1,:] == 2].astype(int)
  hid_idx = node[0,node[1,:] == 3].astype(int)



  # Create connection and initial weight matrices
  conn[3,conn[4,:]==0] = np.nan # disabled but still connected
  src  = conn[1,:].astype(int)
  dest = conn[2,:].astype(int)

  lookup = node[0,:].astype(int)
  for i in range(len(lookup)): 
    src[np.where(src==lookup[i])] = i
    dest[np.where(dest==lookup[i])] = i
  
  wMat = np.zeros((np.shape(node)[1],np.shape(node)[1]))
  wMat[src,dest] = conn[3,:]
  #   connMat = wMat[nIns+nOuts:,nIns+nOuts:]
  conn_mat = wMat[np.ix_(hid_idx,hid_idx)]
  conn_mat[conn_mat!=0] = 1

  # Topological Sort of Hidden Nodes
  edge_in = np.sum(conn_mat,axis=0)
  Q = np.where(edge_in==0)[0]  # Start with nodes with no incoming connections
  for i in range(len(conn_mat)):
    if (len(Q) == 0) or (i >= len(Q)):
      Q = []
      return False, False # Cycle found, can't sort
    edge_out = conn_mat[Q[i],:]
    edge_in  = edge_in - edge_out # Remove nodes' conns from total
    next_nodes = np.setdiff1d(np.where(edge_in==0)[0], Q)
    Q = np.hstack((Q,next_nodes))

    if sum(edge_in) == 0:
      break
  
  # Add In and outs back and reorder wMat according to sort
  Q = np.r_[lookup[in_idx], hid_idx[Q], lookup[out_idx]]
  wMat = wMat[np.ix_(Q,Q)]
  
  return Q, wMat

def getLayer(wMat):
  """Get layer of each node in weight matrix
  Traverse wMat by row, collecting layer of all nodes that connect to you (X).
  Your layer is max(X)+1. Input and output nodes are ignored and assigned layer
  0 and max(X)+1 at the end.

  Args:
    wMat  - (np_array) - ordered weight matrix
           [N X N]

  Returns:
    layer - [int]      - layer # of each node

  Todo:
    * With very large networks this might be a performance sink -- especially, 
    given that this happen in the serial part of the algorithm. There is
    probably a more clever way to do this given the adjacency matrix.
  """
  wMat[np.isnan(wMat)] = 0  
  wMat[wMat!=0]=1
  nNode = np.shape(wMat)[0]
  layer = np.zeros((nNode))
  while (True): # Loop until sorting is stable
    prevOrder = np.copy(layer)
    for curr in range(nNode):
      srcLayer=np.zeros((nNode))
      for src in range(nNode):
        srcLayer[src] = layer[src]*wMat[src,curr]   
      layer[curr] = np.max(srcLayer)+1    
    if all(prevOrder==layer):
      break
  return layer-1

''' functions deprecated'''

#   def mutActivation(self,p):
#     """Changes the activation function of node in the agent NN

#     Args:
#       p        - (dict)     - algorithm hyperparameters (see p/hypkey.txt)
#     Returns:
#       connG    - (np_array) - updated connection genes

#     """
#     nodeG = self.node
    
#     start = 1+self.nInput + self.nOutput
#     end = nodeG.shape[1]
#     if start != end:
#       mutNode = self.rng.integers(start,end)
#       #returns elements in list a and they don't share
#       a = [int(nodeG[2,mutNode])]
#       b = list(p['nn_ActFunc'])
#       newActPool = [r for r in a+b if (r not in a) or (r not in b)]
#       nodeG[2,mutNode] = int(newActPool[self.rng.integers(len(newActPool))])
    
#     self.node = nodeG