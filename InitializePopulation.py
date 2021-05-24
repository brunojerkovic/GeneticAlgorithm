import numpy as np
from Chromosome import Chromosome
from Problem import Problem
import random

class InitializePopulation:
    @staticmethod
    def init_population(pop_size, path_len, problem: Problem, kp_hc_solver):
        pop = []

        # Initialize population as multiple random permutations of ints between 0 and pop_size
        for i in range(pop_size):
            # Initialize path
            path = [i for i in range(path_len)]
            random.shuffle(path)

            # Initialize knapsack
            knapsack = [[random.randint(0, 1) for _ in range(int(dim))] for dim in problem.knapsack_dimension]

            # Initialize entire chromosome (and add it to new population)
            chromosome = Chromosome(path=path, knapsack=knapsack)
            chromosome = kp_hc_solver(chromosome, problem)
            pop.append(chromosome)

        return pop