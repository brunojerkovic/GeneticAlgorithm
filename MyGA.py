import numpy as np
import random
import matplotlib.pyplot as plt
from enum import Enum, auto
from Chromosome import Chromosome
from Problem import Problem
import os
from KPHardConstraintSolver import KPHardConstraintSolver
from Fitness import Fitness
from statistics import mean

class AlgorithmType(Enum):
    GENERATIONAL = 'generational'
    STEADY_STATE = 'steady_state'

class MultiComponentGeneticAlgorithm:
    def __init__(self, problem: Problem, algorithm_type, init_population_fn, insert_offspring_into_population, kp_hc_solver_fn, local_search_fn, fitness_fn,
                 selection_tsp_fn, crossover_tsp_fn, mutation_tsp_fn, gen_neighborhood_tsp_fn,
                 selection_kp_fn, crossover_kp_fn, mutation_kp_fn, gen_neighborhood_kp_fn,
                 pop_size=5, mortality_rate=None, survival_rate=None, elitism=True, n_neighbors=20, max_loc_search_iters=1, gui=None,
                 p_c_tsp=1., p_m_tsp=1., p_c_kp=1., p_m_kp=1.,
                 perform_separately=False, ret_avg_fit=True):
        # Initialize type of the algorithm
        self.algorithm_type = algorithm_type
        self.perform_separately = perform_separately
        self.best: Chromosome = None

        # Initialize probabilities
        self.p_c_tsp = p_c_tsp
        self.p_m_tsp = p_m_tsp
        self.p_c_kp = p_c_kp
        self.p_m_kp = p_m_kp

        # Initialize general operators
        self.ret_avg_fit = ret_avg_fit
        self.init_population = init_population_fn
        self.insert_offspring_into_population = insert_offspring_into_population
        self.fitness = fitness_fn
        self.kp_hc_solver_fn = kp_hc_solver_fn
        self.local_search = local_search_fn
        self.gui = gui

        # Initialize TSP operators
        self.selection_tsp = selection_tsp_fn
        self.crossover_tsp = crossover_tsp_fn
        self.mutation_tsp = mutation_tsp_fn
        self.gen_neighborhood_tsp = gen_neighborhood_tsp_fn

        # Initialize KP operators
        self.selection_kp = selection_kp_fn
        self.crossover_kp = crossover_kp_fn
        self.mutation_kp = mutation_kp_fn
        self.gen_neighborhood_kp = gen_neighborhood_kp_fn

        # Initialize parameters
        self.problem = problem
        self.pop_size = pop_size
        self.n_neighbors = n_neighbors
        self.max_loc_search_iters = max_loc_search_iters
        self.average_fits = []
        self.best_fits = []
        self.iter_num = 0

        # Initialize generational/steady_state parameters
        self.mortality_rate = round(mortality_rate * pop_size if mortality_rate is not None else 0.5 * pop_size)
        self.survival_rate = round(survival_rate * pop_size if survival_rate is not None else 0.1 * pop_size)
        # Even though survival rate can be 0, if elitism is on, survival rate will be 1
        self.elitism = elitism

        # Initialize population (and evaluate it)
        self.pop = self.init_population(pop_size, len(problem.cities), self.problem, self.kp_hc_solver_fn)
        for chromosome in self.pop:
            chromosome.fit = self.fitness(chromosome, problem)
        self.pop.sort(key=lambda x: x.fit, reverse=True)


    def run(self, iter_num=10):
        """
        Run the GA.
        :param iter_num: Number of iterations of GA
        :return: Best chromosome after these iterations
        """
        if self.algorithm_type == AlgorithmType.GENERATIONAL:
            return self.generational_algorithm(iter_num)
        elif self.algorithm_type == AlgorithmType.STEADY_STATE:
            return self.steady_state_algorithm(iter_num)
        else:
            raise Exception("Requested algorithm type is not yet available.")
    
    def generational_algorithm(self, iter_num):
        if self.perform_separately:
            return self.generational_separate_algortihm(iter_num)
        else:
            return self.generational_joint_algorithm(iter_num)

    def steady_state_algorithm(self, iter_num):
        if self.perform_separately:
            return self.steady_state_separate_algorithm(iter_num)
        else:
            return self.steady_state_joint_algorithm(iter_num)

    def generational_joint_algorithm(self, iter_num):
        self.iter_num += iter_num

        for i in range(iter_num):
            #region GETTING THINGS READY
            # Reinitialize new generation
            pop_new = []

            # Make chromosome one iteration older
            for chromosome in self.pop:
                chromosome.age += 1

            # Save parents if there is elitism (but only 'survival_rate' number of them) (IF ELITISM IS ACTIVATED AND POP_SIZE<=10, THERE WILL BE AT LEAST 1)
            if self.elitism:
                self.survival_rate = 1 if self.survival_rate == 0 else self.survival_rate
                for i, chromosome in enumerate(self.pop):
                    pop_new.append(chromosome)
                    if i+1 >= self.survival_rate:
                        break

            # Select parents for reproduction
            children_needed = len(self.pop) - len(pop_new)
            if children_needed == 0:
                # If there are no children needed, algorithm stops, as it does not have anything to do if it cannot produce new solution
                break
            children_needed = children_needed if children_needed % 2 == 0 else children_needed + 1 # We need even number of parents (each pair of parents always give 2 children)
            #endregion

            #region TSP COMPONENT
            # Perform selection of parents
            parents_paths = self.selection_tsp(self.pop, K=children_needed)

            # Perform crossover of parents
            offsprings_paths = []
            for parent1, parent2 in zip(parents_paths[::2], parents_paths[1::2]):
                offsprings_paths += self.crossover_tsp(parent1, parent2, p_c=self.p_c_tsp)

            # Perform mutation on offsprings
            mutated_offsprings_paths = [self.mutation_tsp(offspring, p_m=self.p_m_tsp) for offspring in offsprings_paths]
            #endregion

            #region KP COMPONENT
            # Perform selection of parents
            parents_kp = self.selection_kp(self.pop, K=children_needed)

            # Perform crossover of parents
            offsprings_kp = []
            for parent1, parent2 in zip(parents_kp[::2], parents_kp[1::2]):
                offsprings_kp += self.crossover_kp(parent1, parent2, p_c=self.p_c_kp)

            # Perform mutation of offsprings
            mutated_offsprings_kp = [self.mutation_kp(offspring, p_m=self.p_m_kp) for offspring in offsprings_kp]
            #endregion

            #region FINISH THINGS OFF
            # Join offsprings into one solution
            offsprings = [Chromosome(path=offspring_tsp.path, knapsack=offspring_kp.knapsack) for offspring_tsp, offspring_kp in zip(mutated_offsprings_paths, mutated_offsprings_kp)]
            
            # Fix chromosomes which disobey the hard KP constraint
            offsprings = [self.kp_hc_solver_fn(offspring, self.problem) for offspring in offsprings]

            # Perform evaluation of mutated offsprings
            for offspring in offsprings:
                offspring.fit = self.fitness(offspring, self.problem)

            # Perfrom Local Search
            chromosome_offsprings = [self.local_search(chromosome, self.problem, self.fitness, self.gen_neighborhood_tsp, self.gen_neighborhood_kp, self.n_neighbors, self.max_loc_search_iters) for chromosome in offsprings]

            # Add evaluated offsprings to new generation and sort it by fit
            pop_new += chromosome_offsprings[:-1] if children_needed < len(chromosome_offsprings) else chromosome_offsprings
            pop_new.sort(key=lambda x: x.fit, reverse=True)

            # New generation becomes the one usable for later
            self.pop = pop_new

            # Save average fit and best fit
            self.average_fits.append(sum([el.fit for el in self.pop]) / len(self.pop))
            self.best_fits.append(self.pop[0].fit)
            #endregion

            # Update GUI progressbar
            self.avg = mean(c.fit for c in self.pop)
            self.gui.update_progress(self.pop[0], self.avg)

        # Return best/average fit of population
        self.best = self.pop[0]
        if self.ret_avg_fit:
            self.avg = mean(c.fit for c in self.pop)
            return self.best, self.avg
        else:
            return self.best

    def steady_state_joint_algorithm(self, iter_num):
        self.iter_num += iter_num

        for i in range(iter_num):
            # Set chromosome age to +1
            for chromosome in self.pop:
                chromosome.age += 1

            for _ in range(self.mortality_rate):
                #region TSP COMPONENT
                # Select parents for reproduction
                parent1_tsp, parent2_tsp = self.selection_tsp(self.pop)

                # Perform crossover of parents
                offspring_tsp = self.crossover_tsp(parent1_tsp, parent2_tsp, p_c=self.p_c_tsp)[0]

                # Perform mutation of offsprings
                mutated_offspring_tsp = self.mutation_tsp(offspring_tsp, p_m=self.p_m_tsp)
                #endregion

                #region KP COMPONENT
                # Select parents for reproduction
                parent1_kp, parent2_kp = self.selection_kp(self.pop)

                # Perform crossover of parents
                offspring_kp = self.crossover_kp(parent1_kp, parent2_kp, p_c=self.p_c_kp)[0]

                # Perform mutation of offsprings
                mutated_offspring_kp = self.mutation_kp(offspring_kp, p_m=self.p_m_kp)
                #endregion

                #region FINISH IT OFF
                # Join offsprings into one solution
                offspring = Chromosome(path=mutated_offspring_tsp.path, knapsack=mutated_offspring_kp.knapsack)

                # Fix chromosomes which disobey the hard KP constraint
                offspring = self.kp_hc_solver_fn(offspring, self.problem)

                # Evaluate offspring
                offspring.fit = self.fitness(offspring, self.problem)

                # Perform local search
                chromosome_offspring = self.local_search(offspring, self.problem, self.fitness, self.gen_neighborhood_tsp, self.gen_neighborhood_kp,
                                                         self.n_neighbors, self.max_loc_search_iters)

                # Return next population with the offspring or not
                self.pop = self.insert_offspring_into_population(self.pop, chromosome_offspring, self.elitism)

                # Sort population by descending order
                self.pop.sort(key=lambda x: x.fit, reverse=True)
                #endregion

            #region SORT POPULATION AND DO AVERAGE FIT
            self.best = self.pop[0]

            # Save average fit and best fit of the population
            self.average_fits.append(sum([el.fit for el in self.pop]) / len(self.pop))
            self.best_fits.append(self.pop[0].fit)

            # Update GUI progressbar
            self.avg = mean(c.fit for c in self.pop)
            self.gui.update_progress(self.best, self.avg)

        # Return best/average fit of population
        self.best = self.pop[0]
        if self.ret_avg_fit:
            self.avg = mean(c.fit for c in self.pop)
            return self.best, self.avg
        else:
            return self.best

    def generational_separate_algortihm(self, iter_num):
        self.iter_num += iter_num

        for i in range(iter_num):
            # region GETTING THINGS READY
            # Reinitialize new generation
            pop_new = []

            # Make chromosome one iteration older
            for chromosome in self.pop:
                chromosome.age += 1

            # Save parents if there is elitism (but only 'survival_rate' number of them) (IF ELITISM IS ACTIVATED AND POP_SIZE<=10, THERE WILL BE AT LEAST 1)
            if self.elitism:
                self.survival_rate = 1 if self.survival_rate == 0 else self.survival_rate
                for i, chromosome in enumerate(self.pop):
                    pop_new.append(chromosome)
                    if i + 1 >= self.survival_rate:
                        break

            # Select parents for reproduction
            children_needed = len(self.pop) - len(pop_new)
            if children_needed == 0:
                # If there are no children needed, algorithm stops, as it does not have anything to do if it cannot produce new solution
                break
            children_needed = children_needed if children_needed % 2 == 0 else children_needed + 1  # We need even number of parents (each pair of parents always give 2 children)
            # endregion

            # region TSP COMPONENT
            # Perform selection of parents
            parents_paths = self.selection_tsp(self.pop, K=children_needed)

            # Perform crossover of parents
            offsprings_paths = []
            for parent1, parent2 in zip(parents_paths[::2], parents_paths[1::2]):
                offsprings_paths += self.crossover_tsp(parent1, parent2, p_c=self.p_c_tsp)

            # Perform mutation on offsprings
            mutated_offsprings_paths = [self.mutation_tsp(offspring, p_m=self.p_m_tsp) for offspring in
                                        offsprings_paths]
            # endregion

            # region FINISH THINGS OFF
            # Join offsprings into one solution
            offsprings = [Chromosome(path=offspring_tsp.path, knapsack=parent_kp.knapsack) for offspring_tsp, parent_kp in zip(mutated_offsprings_paths, parents_paths)]

            # Perform evaluation of mutated offsprings
            for offspring in offsprings:
                offspring.fit = Fitness.fitness_ttp(offspring, self.problem)

            # Perfrom Local Search
            chromosome_offsprings = [self.local_search(chromosome, self.problem, Fitness.fitness_ttp, self.gen_neighborhood_tsp,
                                  self.gen_neighborhood_kp, self.n_neighbors, self.max_loc_search_iters) for chromosome in offsprings]

            # Add evaluated offsprings to new generation and sort it by fit
            pop_new += chromosome_offsprings[:-1] if children_needed < len(chromosome_offsprings) else chromosome_offsprings
            pop_new.sort(key=lambda x: x.fit, reverse=True)

            # New generation becomes the one usable for later
            self.pop = pop_new

            # Save average fit and best fit
            self.average_fits.append(sum([el.fit for el in self.pop]) / len(self.pop))
            self.best_fits.append(self.pop[0].fit)
            # endregion

            # Update GUI progressbar
            self.avg = mean(c.fit for c in self.pop)
            self.gui.update_progress(self.pop[0], self.avg)

        for i in range(iter_num):
            # region GETTING THINGS READY
            # Reinitialize new generation
            pop_new = []

            # Make chromosome one iteration older
            for chromosome in self.pop:
                chromosome.age += 1

            # Save parents if there is elitism (but only 'survival_rate' number of them) (IF ELITISM IS ACTIVATED AND POP_SIZE<=10, THERE WILL BE AT LEAST 1)
            if self.elitism:
                self.survival_rate = 1 if self.survival_rate == 0 else self.survival_rate
                for i, chromosome in enumerate(self.pop):
                    pop_new.append(chromosome)
                    if i + 1 >= self.survival_rate:
                        break

            # Select parents for reproduction
            children_needed = len(self.pop) - len(pop_new)
            if children_needed == 0:
                # If there are no children needed, algorithm stops, as it does not have anything to do if it cannot produce new solution
                break
            children_needed = children_needed if children_needed % 2 == 0 else children_needed + 1  # We need even number of parents (each pair of parents always give 2 children)
            # endregion

            # region KP COMPONENT
            # Perform selection of parents
            parents_kp = self.selection_kp(self.pop, K=children_needed)

            # Perform crossover of parents
            offsprings_kp = []
            for parent1, parent2 in zip(parents_kp[::2], parents_kp[1::2]):
                offsprings_kp += self.crossover_kp(parent1, parent2, p_c=self.p_c_kp)

            # Perform mutation of offsprings
            mutated_offsprings_kp = [self.mutation_kp(offspring, p_m=self.p_m_kp) for offspring in offsprings_kp]
            # endregion

            # region FINISH THINGS OFF
            # Join offsprings into one solution
            offsprings = [Chromosome(path=parent_tsp.path, knapsack=offspring_kp.knapsack) for offspring_kp, parent_tsp in
                          zip(mutated_offsprings_kp, parents_kp)]

            # Fix chromosomes which disobey the hard KP constraint
            offsprings = [self.kp_hc_solver_fn(offspring, self.problem) for offspring in offsprings]

            # Perform evaluation of mutated offsprings
            for offspring in offsprings:
                offspring.fit = self.fitness(offspring, self.problem)

            # Perfrom Local Search
            chromosome_offsprings = [
                self.local_search(chromosome, self.problem, self.fitness, self.gen_neighborhood_tsp,
                                  self.gen_neighborhood_kp, self.n_neighbors, self.max_loc_search_iters) for chromosome
                in offsprings]

            # Add evaluated offsprings to new generation and sort it by fit
            pop_new += chromosome_offsprings[:-1] if children_needed < len(
                chromosome_offsprings) else chromosome_offsprings
            pop_new.sort(key=lambda x: x.fit, reverse=True)

            # New generation becomes the one usable for later
            self.pop = pop_new

            # Save average fit and best fit
            self.average_fits.append(sum([el.fit for el in self.pop]) / len(self.pop))
            self.best_fits.append(self.pop[0].fit)
            # endregion

            # Update GUI progressbar
            self.avg = mean(c.fit for c in self.pop)
            self.gui.update_progress(self.pop[0], self.avg)


        # Return best/average fit of population
        self.best = self.pop[0]
        if self.ret_avg_fit:
            self.avg = mean(c.fit for c in self.pop)
            return self.best, self.avg
        else:
            return self.best

    def steady_state_separate_algorithm(self, iter_num):
        self.iter_num += iter_num

        for i in range(iter_num):
            # Set chromosome age to +1
            for chromosome in self.pop:
                chromosome.age += 1

            for _ in range(self.mortality_rate):
                # region TSP COMPONENT
                # Select parents for reproduction
                parent1_tsp, parent2_tsp = self.selection_tsp(self.pop)

                # Perform crossover of parents
                offspring_tsp = self.crossover_tsp(parent1_tsp, parent2_tsp, p_c=self.p_c_tsp)[0]

                # Perform mutation of offsprings
                mutated_offspring_tsp = self.mutation_tsp(offspring_tsp, p_m=self.p_m_tsp)
                # endregion

                # region FINISH IT OFF
                # Join offsprings into one solution
                offspring = Chromosome(path=mutated_offspring_tsp.path, knapsack=parent1_tsp.knapsack)

                # Evaluate offspring
                offspring.fit = Fitness.fitness_ttp(offspring, self.problem)

                # Perform local search
                chromosome_offspring = self.local_search(offspring, self.problem, Fitness.fitness_ttp,
                                                         self.gen_neighborhood_tsp, self.gen_neighborhood_kp,
                                                         self.n_neighbors, self.max_loc_search_iters)

                # Return next population with the offspring or not
                self.pop = self.insert_offspring_into_population(self.pop, chromosome_offspring, self.elitism)

                # Sort population by descending order
                self.pop.sort(key=lambda x: x.fit, reverse=True)
                # endregion

            # region SORT POPULATION AND DO AVERAGE FIT
            self.best = self.pop[0]

            # Save average fit and best fit of the population
            self.average_fits.append(sum([el.fit for el in self.pop]) / len(self.pop))
            self.best_fits.append(self.pop[0].fit)
            # endregion

            # Update GUI progressbar
            self.avg = mean(c.fit for c in self.pop)
            self.gui.update_progress(self.best, self.avg)

        for i in range(iter_num):
            for _ in range(self.mortality_rate):
                # region KP COMPONENT
                # Select parents for reproduction
                parent1_kp, parent2_kp = self.selection_kp(self.pop)

                # Perform crossover of parents
                offspring_kp = self.crossover_kp(parent1_kp, parent2_kp, p_c=self.p_c_kp)[0]

                # Perform mutation of offsprings
                mutated_offspring_kp = self.mutation_kp(offspring_kp, p_m=self.p_m_kp)
                # endregion

                # region FINISH IT OFF
                # Join offsprings into one solution
                offspring = Chromosome(path=parent1_kp.path, knapsack=mutated_offspring_kp.knapsack)

                # Fix chromosomes which disobey the hard KP constraint
                offspring = self.kp_hc_solver_fn(offspring, self.problem)

                # Evaluate offspring
                offspring.fit = self.fitness(offspring, self.problem)

                # Perform local search
                chromosome_offspring = self.local_search(offspring, self.problem, self.fitness,
                                                         self.gen_neighborhood_tsp, self.gen_neighborhood_kp,
                                                         self.n_neighbors, self.max_loc_search_iters)

                # Return next population with the offspring or not
                self.pop = self.insert_offspring_into_population(self.pop, chromosome_offspring, self.elitism)

                # Sort population by descending order
                self.pop.sort(key=lambda x: x.fit, reverse=True)
                # endregion

            # region SORT POPULATION AND DO AVERAGE FIT
            self.best = self.pop[0]

            # Save average fit and best fit of the population
            self.average_fits.append(sum([el.fit for el in self.pop]) / len(self.pop))
            self.best_fits.append(self.pop[0].fit)
            # endregion

            # Update GUI progressbar
            self.avg = mean(c.fit for c in self.pop)
            self.gui.update_progress(self.best, self.avg)

        # Return best/average fit of population
        self.best = self.pop[0]
        if self.ret_avg_fit:
            self.avg = mean(c.fit for c in self.pop)
            return self.best, self.avg
        else:
            return self.best

    def save_results(self, best_solution, avg_fits, filename, img_name=random.randint(1, 1000)):
        """
        Function to plot the list of average fits of iterations.
        :return: None
        """

        # If directories do not exist, create them
        directory = 'Results'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Find out the new name for the file
        onlyfiles = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        file_nums = [int(file.split('_')[0]) for file in onlyfiles] if len(onlyfiles) > 0 else [0]
        file_nums.sort()
        #last_num = int(onlyfiles[-1].split('_')[0]) if len(onlyfiles) > 0 else 0
        new_name = str(file_nums[-1] + 1)
        new_name = new_name if len(new_name) >= 2 else '0' + new_name
        #new_name = str(file_nums[-1] + 1) if file_nums[-1] // 10 >= 1 else '0' + str(file_nums[-1] + 1)


        # Save image files
        plt.figure()
        plt.plot(np.array(range(len(self.average_fits))), np.full(len(self.average_fits), 1) * self.average_fits)
        plt.title("Average fit over the number of iterations for TTP")
        plt.xlabel("Number of iterations")
        plt.ylabel("Total fit")
        plt.savefig(f"{directory}/{new_name}_avg.png")
        plt.show(block=True)

        plt.figure()
        plt.plot(np.array(range(len(self.best_fits))), np.full(len(self.best_fits), 1) * self.best_fits)
        plt.title("Best fit over the number of iterations for TTP")
        plt.xlabel("Number of iterations")
        plt.ylabel("Total fit")
        plt.savefig(f"{directory}/{new_name}_best.png")
        plt.show(block=True)

        # Save text file
        vals = {
            'Filename: ': filename,
            'Algorithm type: ': self.algorithm_type,
            'Perform separately': self.perform_separately,
            'Population Initialization function: ': self.init_population.__name__,
            'Next Generation function: ': self.insert_offspring_into_population.__name__,
            'Fitness function: ': self.fitness.__name__,
            'Selection TSP: ': self.selection_tsp.__name__,
            'Crossover TSP': self.crossover_tsp.__name__,
            'Mutation TSP: ': self.mutation_tsp.__name__,
            'Selection KP: ': self.selection_kp.__name__,
            'Crossover KP: ': self.crossover_kp.__name__,
            'Mutation KP: ': self.mutation_kp.__name__,
            'Population size: ': self.pop_size,
            'Iteration number: ': self.iter_num,
            'Survival/mortality rate: ': self.survival_rate if self.algorithm_type.value == 'generational' else self.mortality_rate,
            'Elitism: ': self.elitism,
            'TSP path: ': self.best.path,
            'KP path: ': self.best.knapsack,
            'Fitness value of the best chromosome: ': best_solution.fit,
            'Average fitness value of the population: ': avg_fits,
            'Age of the best chromosome (num of iterations it has lived): ': self.best.age,
            'Local Search function: ': self.local_search.__name__,
            'Neighborhood TSP: ': self.gen_neighborhood_tsp.__name__,
            'Neighborhood KP: ': self.gen_neighborhood_kp.__name__,
            'Number of neighbors: ': self.max_loc_search_iters,
            'Number of max iterations in local search: ': self.n_neighbors,
            'KP constraint solver function: ': self.kp_hc_solver_fn.__name__
        }
        with open(f'{directory}/{new_name}_txt.txt', 'w+') as f:
            for key, val in zip(list(vals.keys()), list(vals.values())):
                f.write(f'{key}={val}\n_')