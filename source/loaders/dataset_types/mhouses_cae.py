import numpy as np
from gym.utils import seeding
from .loadMHdata import dataSet

class CaeMHEnv():
  """
  Multiple houses set architecture search environment
  """

  def __init__(self,filename):
    """
    Data set setList is a tuple containing 
    [0] trainXL: [nSamples all houses x nInputs]
    [1] trainY:  [nSamples all houses x nOutputs]
    [3] setXL:   [nHouses x nSamples x nInputs]
    [4] setYL:   [nHouses x nSamples x nOutputs]
    [3] cmaXL:   [nHouses x nSamples x nInputs]
    [4] cmaYL:   [nHouses x nSamples x nOutputs]    
    """

    self.t = 0          # Current starting point in the training set

    self.batch   = 1344 # Number hours per batch, set to 1 month
    self.seed()

    self.house = 0
    #filename = '/data/MH_all_norm_multTargets.csv'
    self.dataSet = dataSet(filename)
    setList = self.dataSet.getMHSet()
    self.trainXL = setList[0]
    self.trainYL = setList[1]
    self.testXL  = setList[2]
    self.testYL  = setList[3]
    self.cmaXL   = setList[4]
    self.cmaYL   = setList[5]
    self.trainX =  []
    self.trainY =  []
    self.testX  =  []
    self.testY  =  []
    self.cmaX   =  []
    self.cmaY   =  []
 

    self.setType = None
    self.state = None
    self.trainOrder = None
    self.currIndx = None

  def SetUp(self, h, setType,stepAhead):
    if stepAhead == 0:
        self.setType = setType
        self.house = h
        self.trainX = self.trainXL[h]
        self.trainY = self.trainYL[h]
        self.testX  = self.testXL[h]
        self.testY  = self.testYL[h]
        self.cmaX   = self.cmaXL[h]
        self.cmaY   = self.cmaYL[h]
        print('\t*** Working with house: '+str(h)+' first row in testY and cmaY: '+str([self.testY[0],self.cmaY[0]]))
    else:
        raise Exception('StepAhead is not implemented yet on mhouses_cae.py')
    
  def seed(self, seed=None):
    ''' Randomly select from training set'''
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
  
  def reset(self):
    ''' 
    Return the set type and set index for evolution of the ANN
    Args:
      setType   - (String) - define which type of set will be retrieved
                              - 'Training', retrieves a batch from the
                                  training set.
                              - 'CMA', retrieves a batch with the same
                                  lengt of the test set for CMA evolution
                              - 'Test', retrieves the test set

    '''
    #checking setup has ran first
    if (len(self.testX) ==0) and (not self.setType =='Training'):
      raise Exception('Must run the SetUp first, to select house and setype')
  
    
    if self.setType is 'Training': # training set
      self.t = np.random.randint(len(self.trainY)-self.batch)
      self.currIndx = [i for i in range (self.t,self.t+self.batch)]
      self.state = self.trainX[self.currIndx,:]
    elif self.setType is 'CMA':   # CMA batch
      self.trainOrder = np.arange(0,len(self.cmaY))
      self.t = 0 
      self.currIndx = self.trainOrder
      self.state = self.cmaX
    elif self.setType is 'Test':  # Test set
      self.trainOrder = np.arange(0,len(self.testY))
      self.t = 0 
      self.currIndx = self.trainOrder
      self.state = self.testX
    else:
      raise Exception('Must select a setType, between Training, CMA or Test')
              
    return self.state
  
  def step(self, action):
    ''' 
    Return the reward for each type of setType
    Args:
      setType   - (String) - define which type of set will be based the
                              the reward calculation.
                              - 'Training', target from the training set.
                              - 'CMA', target formt he training set the last
                                  batch with same size as the test set.
                              - 'Test', target from the test set
      action    - (numpy.array) with the output results from the ANN
   
    '''
    if self.setType is 'Training':
      y = self.trainY[self.currIndx]
    elif self.setType is 'CMA':
      y = self.cmaY[self.currIndx]
    elif self.setType is 'Test':
      y = self.testY[self.currIndx]
    else:
      raise Exception('Must select a setType, between Training, CMA or Test')
    
    err = np.abs(action.flatten() - y.flatten())
    loss = np.sum(err) 
    reward = -loss
    
    done = True
    obs = self.state

    return obs, reward, done, {}


    
 
