import numpy as np


def get_parents(pop,n_offspring,rng,tournament_size):
    """ Creates the parent list to bread new children solutions.

    Procedure:
    ) Sort all individuals by rank
    ) Eliminate lower percentage of individuals from breeding pool
    ) Pass upper percentage of individuals to child population unchanged
    ) Select parents by tournament selection
    ) Produce new population through crossover and mutation

    Args:
      pop list of individuals

    Returns:
      children - [Ind]      - newly created population
      innov   - (np_array)  - updated innovation record

    """
    elite_children = []
    
    #setting some ratios
    cull_ratio = 0.2
    elite_ratio = 0.2
    # tournament_size = 32
    
    # Sort by rank
    pop.sort(key=lambda x: x.rank)

    # Cull  - eliminate worst individuals from breeding pool
    to_cull = int(np.floor(cull_ratio * len(pop)))
    if to_cull > 0:
        pop[-to_cull:] = []     

    # Elitism - keep best individuals unchanged
    n_elites = int(np.floor(len(pop)*elite_ratio))
    for i in range(n_elites):
        elite_children.append(pop[i])
        n_offspring -= 1

    # Get parent pairs via tournament selection
    # -- As individuals are sorted by fitness, index comparison is 
    # enough. In the case of ties the first individual wins
    parents = rng.integers(len(pop),size=(n_offspring,tournament_size))
    parents_id = parents.min(axis=1)

    # Breed child population
    parents = []

    for i in range(n_offspring):  
        # Mutation only: take only highest fit parent
        ind = pop[parents_id[i]]
        parents.append(ind)
  
    return elite_children,parents

# def evolve_pop(pop,mpiActive=False):
#     """ Evolves new population.
#     Wrapper which calls 'recombine' and combines all
#     offspring into a new population.
#     """
#     children,parents = get_parents(pop)

#     if mpiActive:
#         return children,parents
#     else:
#         for parent in parents:
#             child = parent.createChild(self.p,self.rng,self.iteration)
#             child.express()
#             children.append(child)
#         self.pop = children



# def getParentsOriginal(self, pop):
#   """ Creates the parent list to bread new children solutions.

#   Procedure:
#     ) Sort all individuals by rank
#     ) Eliminate lower percentage of individuals from breeding pool
#     ) Pass upper percentage of individuals to child population unchanged
#     ) Select parents by tournament selection
#     ) Produce new population through crossover and mutation

#   Args:
#       pop list of individuals

#   Returns:
#       children - [Ind]      - newly created population
#       innov   - (np_array)  - updated innovation record

#   """
#   p = self.p
#   nOffspring = p['popSize']
#   eliteChildren = []
 
#   # Sort by rank
#   pop.sort(key=lambda x: x.rank)

#   # Cull  - eliminate worst individuals from breeding pool
#   numberToCull = int(np.floor(p['select_cullRatio'] * len(pop)))
#   if numberToCull > 0:
#     pop[-numberToCull:] = []     

#   # Elitism - keep best individuals unchanged
#   nElites = int(np.floor(len(pop)*p['select_eliteRatio']))
#   for i in range(nElites):
#     eliteChildren.append(pop[i])
#     nOffspring -= 1

#   # Get parent pairs via tournament selection
#   # -- As individuals are sorted by fitness, index comparison is 
#   # enough. In the case of ties the first individual wins
#   parentA = self.rng.integers(len(pop),size=(nOffspring,p['select_tournSize']))
#   parentB = self.rng.integers(len(pop),size=(nOffspring,p['select_tournSize']))
#   parentsId = np.vstack( (np.min(parentA,1), np.min(parentB,1) ) )
#   parentsId = np.sort(parentsId,axis=0)[0,:] # Higher fitness parent first
  
  
#   # Breed child population
#   parents = []

#   for i in range(nOffspring):  
#     # Mutation only: take only highest fit parent
#     ind = pop[parentsId[i]]
#     parents.append(ind)
  
#   return eliteChildren,parents


# def breedChildren(p,subParents,gen):
#   subChildren = []
#   for ind in subParents:
#     child = ind.createChild(p, gen)
#     child.express()
#     subChildren.append(child)
#   return subChildren


