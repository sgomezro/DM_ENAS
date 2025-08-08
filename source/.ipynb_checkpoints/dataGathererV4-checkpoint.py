from ._helpersV4 import timeDisp,export_agent,export_weights

import os
import numpy as np
import copy
import time
from datetime import datetime



class gatherer_class():
    ''' Class to record best NN architectures and keep track
    '''
    def __init__(self, p, fn_pre_adjw): 
        """
        Args:
          p        - (dict)   - algorithm parameters
        """
        self.filename = p['filename'] # File name path + prefix
        self.path     = p['experiment_path']
        self.save_path = self.path + self.filename
        self.direction = p['opt_direction']

        # Initialize empty fields
        self.elite     = []
        self.best_sw   = []
        self.best_pre_adjw = []
        self.w_pre_adjw = []

        self.t0    = time.time()
        self.time_elap = np.array([])

        self.fitness = np.array([])
        self.n_nodes = np.array([])
        self.n_conns = np.array([])

        self.new_best = False
        self.save_p_used(p)
        self.get_adjw_fitness = fn_pre_adjw
        self.set_optimize_funs()
        
    def set_optimize_funs(self):
        ''' defining optimizing methods, if direction is minimize
            the optomization is min and argmin, if is maximize then
            the optimization is max and argmax.
        '''
        if self.direction == 'minimize':
            def optimize(arg1,arg2):
                return min(arg1,arg2)
            def argoptimize(arg1):
                return np.argmin(arg1)
            def is_true(arg1,arg2):
                return arg1 < arg2
        elif self.direction == 'maximize':
            def optimize(arg1,arg2):
                return max(arg1,arg2)
            def argoptimize(arg1):
                return np.argmax(arg1)
            def is_true(arg1,arg2):
                return arg1 > arg2
    
        self.opt = optimize
        self.argopt = argoptimize
        self.is_true = is_true

    def gather_data(self, pop, gen):

        fitness = [ind.fitness for ind in pop]
        peakfit = [ind.fitMax for ind in pop]
        n_nodes = np.asarray([np.shape(ind.node)[1] for ind in pop])
        n_conns = np.asarray([ind.nConn for ind in pop])


        # --- Save complete fitness ----------------------------------------------
        if len(self.fitness) == 0:
            self.fitness = np.copy(np.array(fitness)[None,:])
            self.n_nodes = np.copy(np.array(n_nodes)[None,:])
            self.n_conns = np.copy(np.array(n_conns)[None,:])
        else:
            self.fitness = np.append(self.fitness, np.array(fitness)[None,:],axis=0)
            self.n_nodes = np.append(self.n_nodes, np.array(n_nodes)[None,:],axis=0)
            self.n_conns = np.append(self.n_conns, np.array(n_conns)[None,:],axis=0)

        # --- Best Individual ----------------------------------------------------
        self.elite.append(pop[self.argopt(fitness)])
        elite = self.elite[-1]

        if len(self.best_sw) == 0:
            self.best_sw = copy.deepcopy(self.elite)
        elif self.is_true(self.elite[-1].fitness, self.best_sw[-1].fitness):
            # print(f'For gen {gen} elite fitness {elite.fitness}, fit max {elite.fitMax} sw value {elite.best_sw}')
            if self.is_true(self.elite[-1].fitMax, self.best_sw[-1].fitMax):
                self.best_sw= np.append(self.best_sw,copy.deepcopy(self.elite[-1]))
                self.new_best = True
        else:
            self.best_sw= np.append(self.best_sw,copy.deepcopy(self.best_sw[-1]))   
            self.new_best = False
        # ------------------------------------------------------------------------ 

        #---- Checking if new best is actually best with adjusted weights
        self.check_new_best(gen)

        self.time_elap = np.append(self.time_elap, int(time.time()-self.t0))
    
    def display(self,gen):
        print ('{}-{} elapsed| Elite fit {:.4e}, peak {:.2e}/sw{} \t| Best Shared {:.4e}, peak {:.2e}/sw{} \t| Best pre adjw {:.4e}'.format(
            str(gen).zfill(3),
            timeDisp(self.time_elap[-1]),
            self.elite[-1].fitness,
            self.elite[-1].fitMax,
            self.elite[-1].best_sw,
            self.best_sw[-1].fitness,
            self.best_sw[-1].fitMax,
            self.best_sw[-1].best_sw,
            self.best_pre_adjw[-1].fitness))

    def save_p_used(self,p):
        ''' Save at file all parameters used to evolve NN architecture
        '''
        loc = self.save_path + '/best/'
        if not os.path.exists(loc):
            os.makedirs(loc)
        loc = self.save_path+'/elite/'
        if not os.path.exists(loc):
            os.makedirs(loc)
        loc = self.save_path+'/pre_adjw/cma/'
        if not os.path.exists(loc):
            os.makedirs(loc)
        
        f = open(self.save_path + '/parameters_used.out','w')
        f.write("Parameters used to ran experiment "+self.filename+"\n")
        f.write("experiment started at "+datetime.now().strftime("%m/%d/%Y, %H:%M"))
        f.write("\n{\n")
        for k in p.keys():
            f.write("'{}':'{}'\n".format(k, p[k]))
        f.write("}")
        f.close()

    def save(self,gen):
        ''' Save data to disk '''
        ##saving best with shared weights
        export_agent(f'{self.save_path}/best/ind_{str(gen).zfill(4)}.npy', self.best_sw[-1])

        ## Saving elite
        export_agent(f'{self.save_path}/elite/ind_{str(gen).zfill(4)}.npy', self.elite[-1])
        
        # #saving best NN agent with pre adjusted weights
        export_agent(f'{self.save_path}/pre_adjw/ind_{str(gen).zfill(4)}.npy', self.best_pre_adjw[-1])
        
        ## saving pre adjusted weights
        best= self.best_pre_adjw[-1]
        best_weights = best.wVec[best.vKey]
        adj_w_fname = f'{self.save_path}/pre_adjw/cma/cma_{str(gen).zfill(4)}'
        export_weights(adj_w_fname,best_weights)

 
    def check_new_best(self,gen):
        if (self.new_best) | (len(self.best_pre_adjw) == 0):
            #Saving best shared weights NN agent as tmp for weight adjustment check best
            export_agent(f'{self.save_path}/tmp.npy',self.best_sw[-1])
            adjw_fitness,adjw_weigths = self.get_adjw_fitness()
            if len(self.best_pre_adjw) == 0:
                best = copy.deepcopy(self.elite)
                best[-1].fitness = adjw_fitness
                best[-1].wVec[best[-1].vKey] = adjw_weigths
                self.best_pre_adjw = best

            elif self.is_true(adjw_fitness,self.best_pre_adjw[-1].fitness):
                best = copy.deepcopy(self.elite[-1])
                best.fitness = adjw_fitness
                best.wVec[best.vKey] = adjw_weigths              
                self.best_pre_adjw = np.append(self.best_pre_adjw,best)
                #saving best NN agent with pre adjusted weights
                self.save(gen)

            else:
                self.best_pre_adjw = np.append(self.best_pre_adjw,copy.deepcopy(self.best_pre_adjw[-1]))
        else:
            self.best_pre_adjw = np.append(self.best_pre_adjw,copy.deepcopy(self.best_pre_adjw[-1]))

        self.new_best = False
    
    
    
