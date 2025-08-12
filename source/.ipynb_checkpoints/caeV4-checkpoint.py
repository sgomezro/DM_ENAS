import numpy as np
import copy
import json

from ._nsgaV4 import nsga_sort
from ._individualsV4 import Individuals
from ._agentV4 import nn_agent
from ._variationV4 import get_parents


class cae_class():
  """DNN-CAE main class. Evolves population given fitness values of individuals.
  """
  def __init__(self, p,rng=None,dtype=None):
      
    """Intialize DNN-CAE algorithm with hyperparameters
    Args:
    hyp - (dict) - algorithm hyperparameters
    
    Attributes:
    p       - (dict)     - algorithm hyperparameters (see p/hypkey.txt)
    pop     - (Ind)      - Current population
    innov   - (np_array) - innovation record
              [5 X nUniqueGenes]
              [0,:] == Innovation Number
              [1,:] == Source
              [2,:] == Destination
              [3,:] == New Node?
              [4,:] == Generation evolved
    gen     - (int)      - Current generation
    """

    self.pop_size = p['popSize']
    self.max_gen  = p['maxGen'] 
    self.init_act = p['nn_initAct']
    self.sw_lim   = p['cae_Sw_lim']
    self.direction= p['opt_direction']
    self.tourn_size  = p['select_tournSize']
    self.input_size  = p['nn_input_size']
    self.output_size = p['nn_output_size']
    self.dtype = dtype

    
    
    self.pop     = []
    
    # Settting share weight values to test the fitness of the architecture
    sw_lim   = p['cae_Sw_lim']
    n_vals   = p['cae_Sw_nVals']
    sw_list = np.linspace(-sw_lim, sw_lim ,n_vals+1)
    self.sw_list = sw_list[sw_list !=0]
    
    self.iteration= 0
    self.rng = rng
    self.indType = Individuals
    
    self.fn = self.probMoo2()
    self.set_optimize_funs()
    
  def set_population(self,pop):
      self.pop = pop

  def set_optimize_funs(self):
    ''' defining optimizing methods, if direction is minimize
        the optomization is min and argmin, if is maximize then
        the optimization is max and argmax.
    '''
    if self.direction == 'minimize':
      print('Minimizing')
      def optimize(arg1,arg2):
        return min(arg1,arg2)
      def argoptimize(arg1):
        return np.argmin(arg1)
    elif self.direction == 'maximize':
      print('Maximizing')
      def optimize(arg1,arg2):
        return max(arg1,arg2)
      def argoptimize(arg1):
        return np.argmax(arg1)
    
    self.opt = optimize
    self.argopt = argoptimize



  def ask_cae(self):
    """Returns parents and elite children to bread new population
       Args:
       Returns:
         eliteChildren = list with the first children for the new offspring
         parents       = list of parents to bread new children, after breading
                         the new children must be added to eliteChildren to
                         complete the population size.
    """
    parents = None
    elite_children = []
    if self.iteration == 0:
      self.iteration += 1  
      if len(self.pop) == 0:
        self.init_population()      # Initialize population
    else:
      self.iteration +=1
      self.probMoo2()      # Rank population according to objectivess
      elite_children,parents = get_parents(self.pop,self.pop_size,self.rng,self.tourn_size)
    return elite_children,parents
    
  def get_nn_population(self):
    ''' Generates a list with NN information for each individual 
        in the population.
    Args:
    Returns:
      list of elements:
        [0,:] == Number of nodes each NN agent has.
        [1,:] == Activation function (as int)
        [2,:] == Vector keys (where weight matrix is flaten and keys
                 indicate active weigths)
     '''
    
    return [nn_agent(self.input_size,None,ind=ind,dtype=self.dtype) for ind in self.pop]

  def tell(self,reward):
    """Assigns fitness to current population

    Args:
      reward - (np_array) - fitness value of each individual
               [nInd X nTrails]

    """
    self.best = np.mean(reward[0,:])
    for i in range(np.shape(reward)[0]):
      self.pop[i].fitness = np.mean(reward[i,:])
      fitMax_idx = self.argopt(reward[i,:])
      self.pop[i].fitMax  = reward[i,fitMax_idx]
      self.pop[i].best_sw = self.sw_list[fitMax_idx]
      self.pop[i].nConn   = self.pop[i].nConn
      # print(f' best {self.best} dtpe {self.best.dtype}, pop i {self.pop[i].fitness} dtype {np.dtype(self.pop[i].fitness)}')
      self.best = self.opt(self.best,self.pop[i].fitness)
    idx = np.where(reward.mean(axis=1) == self.best)[0][0]
    self.best_nWeights = self.pop[idx].vKey.size
  
  def stop(self):
    stop_flag = False
    if self.iteration < 1:
      return stop_flag
    else:
      if (self.iteration >= self.max_gen):
        stop_flag = True
        
      return stop_flag

  def init_population(self):
    """Initialize population with a list of random individuals
    """

    #setting probability of enabling a connection
    prob_init_enable = 0.25
    
    # - Create Nodes -
    node_id = np.arange(0,self.input_size+self.output_size+1,1)
    node = np.empty((3,len(node_id)))
    node[0,:] = node_id
    
    # Node types: Type (1=input, 2=output 3=hidden 4=bias)
    node[1,0] = 4 # Bias
    node[1,1:self.input_size+1] = 1 # Input Nodes
    node[1,(self.input_size+1):\
           (self.input_size+self.output_size+1)]  = 2 # Output Nodes
    
    # Node Activations
    node[2,:] = int(self.init_act)
    
    # - Create Conns -
    n_conn = (self.input_size+1) * self.output_size
    ins   = np.arange(0,self.input_size+1,1)            # Input and Bias Ids
    outs  = (self.input_size+1) + np.arange(0,self.output_size) # Output Ids
    
    conn      = np.empty((5,n_conn,))
    conn[0,:] = np.arange(0,n_conn,1)      # Connection Id
    conn[1,:] = np.tile(ins, len(outs))   # Source Nodes
    conn[2,:] = np.repeat(outs,len(ins) ) # Destination Nodes
    conn[3,:] = 0                         # Weight Values
    conn[4,:] = 1                         # Enabled?
        
    # Create population of individuals with varied weights
    pop = []
    for i in range(self.pop_size):
        newInd = self.indType(conn, node,None)
        newInd.setFun(self.fn)
        newInd.conn[3,:] = (2*(self.rng.random((1,n_conn))-0.5))*self.sw_lim 
        newInd.conn[4,:] = self.rng.random((1,n_conn)) < prob_init_enable
        newInd.express()
        pop.append(copy.deepcopy(newInd))  
    
    self.pop = pop
    

    
  def probMoo2(self):
    """Rank population according to Pareto dominance.
    """
    # Compile objectives
    sw_fitness = np.asarray([ind.fitness for ind in self.pop])
    n_weights = np.asarray([len(ind.vKey) for ind in self.pop])
    max_fit  = np.asarray([ind.fitMax  for ind in self.pop])
#     n_conns  = np.asarray([ind.n_conn   for ind in self.pop])
#     nNodes  = np.asarray([ind.nNodes  for ind in self.pop])
#     n_conns[n_conns==0] = 1 # No conns is always pareto optimal (but boring)
    if self.direction == 'maximize':
      objVals = np.c_[sw_fitness,n_weights] # Maximize
      # objVals = np.c_[sw_fitness,max_fit] # Maximize
    
    elif self.direction == 'minimize':
      objVals = np.c_[-sw_fitness,n_weights] # Minimize by maximizing
      # objVals = np.c_[-sw_fitness,max_fit] # Minimize by maximizing
    
    #nsga always maximize
    rank = nsga_sort(objVals)


    # Assign ranks
    for i in range(len(self.pop)):
      self.pop[i].rank = rank[i]
