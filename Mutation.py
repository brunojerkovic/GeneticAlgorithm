import random
import enum
from Chromosome import Chromosome

class MutationDecorators:
    @staticmethod
    def extract_dec(mutation_fn):
        def inner(cls, offspring, p_m=1):
            # The probability that the crossover will happen is p_c
            if random.uniform(0, 1) > p_m:
                return offspring

            # Check if this is KP mutation or TSP mutation
            if cls.__name__.endswith('TSP'):
                # Perform mutation
                offspring.path = mutation_fn(cls, offspring.path)
            elif cls.__name__.endswith('KP'):
                for i, kp in enumerate(offspring.knapsack):
                    # Only works for bit-representations
                    if set(kp) | {0} | {1} != {0, 1}:
                        raise Exception("Only works for bit-representations.")

                    # Perform mutation
                    if len(kp) != 0:
                        offspring.knapsack[i] = mutation_fn(cls, kp)
            else:
                raise Exception("This is not KP nor TSP mutation.")
            return offspring
        return inner

class Mutation:
    @classmethod
    @MutationDecorators.extract_dec
    def insertion(cls, offspring):
        # Take random element
        ind_take = random.randint(0, len(offspring) - 1)
        element = offspring[ind_take]
        offspring = offspring[:ind_take] + offspring[(ind_take+1):]

        # Put element at random location
        ind_put = random.randint(0, len(offspring)+1) # +1 because it can go at the end (dont forget that len of offspring is now smaller by 1)
        offspring = offspring[:ind_put] + [element] + offspring[ind_put:]

        return offspring

class MutationKP(Mutation):
    @classmethod
    @MutationDecorators.extract_dec
    def exchange(cls, offspring):
        # Generate 2 indexes for mutation
        while True:
            ind1 = random.randint(0, len(offspring) - 1)
            ind2 = random.randint(0, len(offspring) - 1)
            if ind1 != ind2:
                break

        # Swap indexes of offspring
        offspring[ind1], offspring[ind2] = offspring[ind2], offspring[ind1]

        return offspring

    @classmethod
    @MutationDecorators.extract_dec
    def bit_flip(cls, offspring):
        ind = random.randint(0, len(offspring) - 1)
        offspring[ind] = 1 - offspring[ind]

        return offspring

class MutationTSP(Mutation):
    @classmethod
    @MutationDecorators.extract_dec
    def two_opt(cls, offspring, joint_segment=False):
        first_ind = random.randint(0, len(offspring) - 1)
        if joint_segment:
            second_ind = (first_ind + 2) % len(offspring)
        else:
            while True:
                second_ind = random.randint(0, len(offspring) - 1)
                if second_ind != first_ind and second_ind != (first_ind + 1) % len(offspring) and (
                        second_ind + 1) % len(offspring) != first_ind:
                    break

        offspring[first_ind], offspring[(second_ind + 1) % len(offspring)] = offspring[(second_ind + 1) % len(offspring)], offspring[first_ind]
        offspring[second_ind], offspring[(first_ind + 1) % len(offspring)] = offspring[(first_ind + 1) % len(offspring)], offspring[second_ind]

        return offspring






