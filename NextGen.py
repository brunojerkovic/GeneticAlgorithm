import random
from Chromosome import Chromosome


class NextGen:
    @staticmethod
    def replace_by_random_choice(pop, offspring, elitism):
        ind = random.randint(0, len(pop)-1)

        # Change if there is no elitism or new child is better than the random parent
        if not elitism or pop[ind].fit < offspring.fit:
            pop[ind] = offspring
        return pop

    @staticmethod
    def replace_the_worst_parent(pop, offspring, elitism):
        # Change if there is no elitism or new child is better than the worst parent
        if not elitism or pop[-1].fit < offspring.fit:
            pop[-1] = offspring
        return pop
