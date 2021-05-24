import numpy as np
from MyGA import MultiComponentGeneticAlgorithm, AlgorithmType
import LoadDataset
from Fitness import Fitness
from InitializePopulation import InitializePopulation
from NextGen import NextGen
from Selection import SelectionTSP, SelectionKP
from Crossover import CrossoverTSP, CrossoverKP
from Mutation import MutationTSP, MutationKP
from TestingGUI import open_GUI
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    open_GUI(graph_name='Berlin')









