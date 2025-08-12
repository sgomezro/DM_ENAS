import numpy as np
from gym.utils import seeding
from .loadSeries import dataSet



class SeriesEnv():
  """Classification as an unsupervised OpenAI Gym RL problem.
  Includes scikit-learn digits dataset, MNIST dataset
  """

  def __init__(self,filename='/data/h3_norm_multTargetsv1.csv'):
    """
    Data set is a tuple of 
    [0] input data: [nSamples x nInputs]
    [1] labels:     [nSamples x 1]

    Example data sets are given at the end of this file
    """

    self.t = 0          # Current batch number

    self.batch   = 1344 # Number hours per batch, set to 1 month
    self.seed()
    
    self.dataSet = dataSet(filename)
    self.trainSet, self.trainTarget, self.testSet, self.testTarget = self.dataSet.getSet()
    self.cmaSet = self.trainSet[-len(self.testTarget):,:]
    self.cmaTarget = self.trainTarget[-len(self.testTarget):]
 

    self.setType = None
    self.state = None
    self.trainOrder = None
    self.currIndx = None

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
    if self.setType is 'Training': # training set
      self.t = np.random.randint(len(self.trainTarget)-self.batch)
      self.currIndx = [i for i in range (self.t,self.t+self.batch)]
      self.state = self.trainSet[self.currIndx,:]
    elif self.setType is 'CMA':   # CMA batch
      sizeTest = len(self.testTarget)
      sizeTrain = len(self.trainTarget)
      self.trainOrder = np.arange(sizeTrain-sizeTest, sizeTrain)
      self.t = 0
      self.currIndx = self.trainOrder
      self.state = self.trainSet[self.currIndx,:]
    elif self.setType is 'Test':  # Test set
      self.trainOrder = np.arange(0,len(self.testTarget))
      self.t = 0 
      self.currIndx = self.trainOrder
      self.state = self.testSet[self.currIndx,:]
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
      y = self.trainTarget[self.currIndx]
    elif self.setType is 'CMA':
      y = self.trainTarget[self.currIndx]
    elif self.setType is 'Test':
      y = self.testTarget[self.currIndx]
    else:
      raise Exception('Must select a setType, between Training, CMA or Test')
    
    err = np.abs(action - y)
    loss = np.sum(err) 
    reward = -loss

    done = True
    obs = self.state

    return obs, reward, done, {}


    
 
