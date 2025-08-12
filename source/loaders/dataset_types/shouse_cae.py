import numpy as np
from .loadSH_MA import dataSet

class CaeSHEnv():
  """
  Multiple houses set architecture search environment
  """

  def __init__(self,filename):
    """
    Data set setList is a tuple containing 
    [0] trainXL: [nSamples single house x nInputs]
    [1] trainY:  [nSamples single house x nOutputs]
    [3] setXL:   [nSamples x nInputs]
    [4] setYL:   [nSamples x nOutputs]
    [3] cmaXL:   [nSamples x nInputs]
    [4] cmaYL:   [nSamples x nOutputs]    
    """

    self.t = 0          # Current starting point in the training set

    self.batch   = 1344 # Number hours per batch, set to 1 month

    self.house = 0
    self.dataSet = dataSet(filename)
    setList = self.dataSet.getMHMSSet()
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

  def SetUp(self, h, setType, stepAhead):
    self.setType = setType
    self.house  = h
    self.trainX = self.trainXL[0]
    self.trainY = self.trainYL[0][:,stepAhead]
    self.testX  = self.testXL[0]
    self.testY  = self.testYL[0][:,stepAhead]
    self.cmaX   = self.cmaXL[0]
    self.cmaY   = self.cmaYL[0][:,stepAhead]
    print('\t*** Working with house: defined by dataset first row in testY and cmaY: '+str([self.testY[0],self.cmaY[0]]))

  def setRng(self,rng):
    self.rng = rng
    
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
  
    
    if self.setType == 'Training': # training set
      self.t = self.rng.integers(len(self.trainY)-self.batch)
      self.currIndx = [i for i in range (self.t,self.t+self.batch)]
      self.state = self.trainX[self.currIndx,:]
    elif self.setType == 'CMA':   # CMA batch
      self.trainOrder = np.arange(0,len(self.cmaY))
      self.t = 0 
      self.currIndx = self.trainOrder
      self.state = self.cmaX
    elif self.setType == 'Test':  # Test set
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


    
 
