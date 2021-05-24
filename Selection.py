import random
import numpy as np
import copy
import enum

class Selection:
    @staticmethod
    def r_tournament_selection_without_replacement(pop, r=2, K=2):
        pop = copy.deepcopy(pop)
        parents = []

        if K > r:
            raise Exception("Cannot sample without replacement more elements than in set: r > K")
        for k in range(K):
            candidates = random.choices(pop, k=r)
            candidates.sort(key=lambda x: x.fit, reverse=True)
            parents.append(candidates[0])

            # Remove winner from original population
            for i, el in enumerate(pop):
                if id(el) == id(candidates[0]):
                    pop.pop(i)
                    break

        return parents

    @staticmethod
    def r_tournament_selection_with_replacement(pop, r=2, K=2):
        parents = []
        for k in range(K):
            candidates = random.choices(pop, k=r)
            candidates.sort(key=lambda x: x.fit, reverse=True)
            parents.append(candidates[0])

        return parents

    # This is relative proportional roulette wheel selection (relative to deal with scale problem (f-f_min))
    @staticmethod
    def roulette_wheel_selection(pop, K=2):
        weights = np.array([c.fit for c in pop])

        # Turn goal function to fit function
        weights -= min(weights)

        # Calculate probs
        weights /= sum(weights)

        # Roulette Wheel Selection
        parents = random.choices(pop, weights=weights, k=K)

        return list(parents)

    @staticmethod
    def stochastic_universal_sampling(pop, K=2):
        pop = np.array(pop, dtype=object)
        pop_size = len(pop)
        min_fit = min(c.fit for c in pop)
        fits_ = np.array([c.fit-min_fit for c in pop])
        fits = np.array([sum(fits_[:i]) for i in range(len(fits_) + 1)])
        F = np.sum(fits_)
        r = round(random.random() * F/K, 5)

        indexes = []
        j = K
        while True:
            r += j * F / K
            r %= max(fits)
            for i in range(len(fits)):
                if fits[i - 1] < r < fits[i] and (i-1)%pop_size not in indexes:
                    indexes.append((i - 1) % pop_size)
                    j -= 1
                    break
            if j == 0:
                break


        if len(indexes) != K:
            raise Exception("Error in Stohastic Universal Sampling")

        parents = [pop[i] for i in indexes]
        return list(parents)

    @staticmethod
    def truncation_selection(pop, K=2, tau=None):
        tau = len(pop)//2+1 if tau is None else tau
        probs = [1 / tau if i < tau else 0 for i in range(len(pop))]

        parents = random.choices(pop, weights=probs, k=K)
        return parents

    @staticmethod
    def linear_ranking_selection(pop, K=2):
        probs = [i + 1 for i in range(len(pop))]
        probs.reverse()
        prob_sum = sum(probs)
        probs = [prob/prob_sum for prob in probs]

        parents = random.choices(pop, weights=probs, k=K)
        return parents

class SelectionKP(Selection):
    pass

class SelectionTSP(Selection):
    pass