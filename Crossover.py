import random
import numpy as np
import enum
from Chromosome import Chromosome

class CrossoverDecorators:
    @staticmethod
    def extract_dec(crossover_fn):
        def inner(cls, p1, p2, p_c=1):
            # The probability that the crossover will happen is p_c
            if random.uniform(0, 1) > p_c:
                return parent1, parent2

            # Check if this is KP crossover or TSP crossover
            if cls.__name__.endswith('TSP'):
                # Check if sizes are equal
                if len(p1.path) != len(p2.path):
                    raise Exception("Parents' length do not match!")

                # Perform crossover
                off1_path, off2_path = crossover_fn(cls, p1.path, p2.path)
                return Chromosome(path=off1_path), Chromosome(path=off2_path)
            elif cls.__name__.endswith('KP'):
                # Check if sizes are equal
                if len(p1.knapsack) != len(p2.knapsack):
                    raise Exception("Parents' length do not match!")

                # Perform crossover
                off1 = []
                off2 = []

                # Iterate over all knapsacks and cross them over
                for kp1, kp2 in zip(p1.knapsack, p2.knapsack):
                    # If knapsacks are empty or only have 1 element, there is nothing to crossover
                    if len(kp1) == 0 and len(kp2) == 0:
                        off1.append([])
                        off2.append([])
                        continue
                    elif len(kp1) == 1 and len(kp2) == 1:
                        off1.append(kp1)
                        off2.append(kp2)
                        continue

                    # Perform crossover
                    kp1_, kp2_ = crossover_fn(cls, kp1, kp2)

                    # Add these small knapsacks to big ones
                    off1.append(kp1_)
                    off2.append(kp2_)

                return Chromosome(knapsack=off1), Chromosome(knapsack=off2)
            else:
                raise Exception("This is not TPS nor KP crossover.")
        return inner

class Crossover:
    pass

class CrossoverTSP(Crossover):
    @classmethod
    @CrossoverDecorators.extract_dec
    def order_crossover(cls, parent1, parent2):
        # Length of the path
        parent_length = len(parent1)
        p1, p2 = parent1, parent2

        # Initialize offspring
        off1 = np.full(parent_length, -1)
        off2 = np.full(parent_length, -1)
        off1.astype(int)
        off2.astype(int)

        # Create 2 cut-points and store them in a list
        while True:
            indexes = []
            indexes.append(random.randint(0, len(p1) - 1))
            indexes.append(random.randint(0, len(p1) - 1))
            if indexes[0] != indexes[1]:
                break
        indexes.sort()

        # Copy between parents' cut-points to children
        off1[indexes[0]:indexes[1]] = p1[indexes[0]:indexes[1]]
        off2[indexes[0]:indexes[1]] = p2[indexes[0]:indexes[1]]

        # Rotate parents from 2nd cut-point
        p1 = np.concatenate([p1[-(len(p1) - indexes[1]):], p1[:indexes[1]]])
        p2 = np.concatenate([p2[-(len(p2) - indexes[1]):], p2[:indexes[1]]])

        # Fill missing cities in 1st offspring
        i = indexes[1]
        for el in p2:
            if el not in off1:
                while off1[i] != -1:
                    i += 1
                    i = 0 if i == parent_length else i
                off1[i] = el

        # Fill missing cities in 2nd offspring
        i = indexes[1]
        for el in p1:
            if el not in off2:
                while off2[i] != -1:
                    i += 1
                    i = 0 if i == parent_length else i
                off2[i] = el

        return list(off1), list(off2)

    # TODO: Nesto ode nevalja?
    @classmethod
    @CrossoverDecorators.extract_dec
    def cycle_crossover(cls, parent1, parent2):
        # Path length
        parent_length = len(parent1)
        p1, p2 = np.array(parent1), np.array(parent2)

        # Initialize offspring
        off1 = np.full(parent_length, -1)
        off2 = np.full(parent_length, -1)
        off1.astype(int)
        off2.astype(int)

        # Perform cycle crossover
        while True:
            try:
                ind = np.where(off1 == -1)[0][0] # Two 0s because of the way that numpy.where method works
            except:
                # There are no more cycles
                break
            cycle_start = p1[ind]
            while True:
                off1[ind] = p1[ind]
                off2[ind] = p2[ind]
                val = p2[ind]
                ind = np.where(p1 == val)[0][0]

                # Check if cycle is completed
                if cycle_start == p1[ind]:
                    break

            off1, off2 = off2, off1
        return list(off1), list(off2)

    @classmethod
    @CrossoverDecorators.extract_dec
    def partially_mapped_crossover(cls, parent1, parent2):
        # Path length
        parent_length = len(parent1)
        p1, p2 = parent1, parent2

        # Initialize offspring
        off1 = np.full(parent_length, -1)
        off2 = np.full(parent_length, -1)
        off1.astype(int)
        off2.astype(int)

        for j in range(2):
            p1, p2 = p2, p1

            while True:
                start_ind = random.randint(0, len(p1) - 1)
                end_ind = random.randint(start_ind, len(p1) - 1)
                if start_ind != end_ind and start_ind != 0 and end_ind - start_ind > 1:
                    break

            offspring = np.hstack((p2[:start_ind], p1[start_ind:end_ind], p2[end_ind:])).ravel()

            while True:
                p1_segment = offspring[start_ind:end_ind]
                p2_segment = np.concatenate((offspring[:start_ind], offspring[end_ind:]))

                # Do this until there are no more duplicate elements
                duplicate_elems = np.array([el for el in p1_segment if el in p2_segment])
                if len(duplicate_elems) == 0:
                    break

                for i, el in enumerate(offspring):
                    # Skip elems from p1_segment (only change the ones from p2 segment)
                    if i >= start_ind and i < end_ind:
                        continue

                    if el in duplicate_elems:
                        ind_p2 = np.where(p1 == el)[0][0]
                        offspring[i] = p2[ind_p2]

            if j == 0:
                off1 = offspring
            else:
                off2 = offspring

        return list(off1), list(off2)

class CrossoverKP(Crossover):
    @classmethod
    @CrossoverDecorators.extract_dec
    def n_point_crossover(cls, parent1, parent2, n=1):
        # Number of knapsack items
        kp_item_num = len(parent1)

        # Initialize offspring
        off1_kp = np.full(kp_item_num, -1)
        off2_kp = np.full(kp_item_num, -1)
        off1_kp.astype(int)
        off2_kp.astype(int)

        # Sample indexes (0 and parent_length are definite indexes because they are boundaries)
        indexes = [0, kp_item_num]
        for i in range(n):
            while True:
                new_index = random.randint(1, kp_item_num - 1)
                if new_index not in indexes:
                    indexes.append(new_index)
                    break
        indexes = np.sort(indexes)

        # Create offsprings from n crossovers
        for i, index in enumerate(indexes[1:]):
            parent1, parent2 = parent2, parent1

            off1_kp[indexes[i]:indexes[i + 1]] = parent1[indexes[i]:indexes[i + 1]]
            off2_kp[indexes[i]:indexes[i + 1]] = parent2[indexes[i]:indexes[i + 1]]

        return list(off1_kp), list(off2_kp)