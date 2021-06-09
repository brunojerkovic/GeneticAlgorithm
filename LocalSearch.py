from Chromosome import Chromosome
from Fitness import Fitness
import random
from Problem import Problem
import copy

class LocalSearch:
    @staticmethod
    def greedy_hill_climbing(solution: Chromosome, problem: Problem, fitness_fn, gen_neighborhood_tsp, gen_neighborhood_kp, n_neighbors, n_max_iters):
        best_solution = Chromosome(path=solution.path, knapsack=solution.knapsack, fit=solution.fit, age=solution.age)

        for i in range(n_max_iters):
            # Generate neighborhood
            neighbors = []
            for _ in range(n_neighbors):
                path = copy.deepcopy(gen_neighborhood_tsp(best_solution, p_m=1.).path)
                knapsack = copy.deepcopy(gen_neighborhood_kp(best_solution, p_m=1.).knapsack)
                neighbors.append(Chromosome(path = path, knapsack=knapsack))

            # Evaluate neighborhood
            for neighbor in neighbors:
                neighbor.fit = fitness_fn(neighbor, problem)

            # Find best current solution
            neighbors.sort(key=lambda x: x.fit, reverse=True)
            current_solution = neighbors[0]

            # Stop if current solution is worse than the best one
            if i != 0 and current_solution.fit == prev_solution.fit: #Don't worry that for i==0 'prev_solution' is not defined, because it will never evaluate it for i==0
                break
            if current_solution.fit > best_solution.fit:
                best_solution = current_solution

            prev_solution = current_solution

        return best_solution

    @staticmethod
    def iterated_local_search(solution: Chromosome, problem: Problem, fitness_fn, gen_neighborhood_tsp, gen_neighborhood_kp, n_neighbors, n_max_iters):
        def perturbate(x: Chromosome):
            p = random.random()
            if p > 0.5:
                return Chromosome(path=gen_neighborhood_tsp(x, p_m=1.).path, knapsack=x.knapsack)
            else:
                return Chromosome(path=x.path, knapsack=gen_neighborhood_kp(x, p_m=1.).knapsack)

        best_solution = LocalSearch.greedy_hill_climbing(solution, problem, fitness_fn, gen_neighborhood_tsp, gen_neighborhood_kp, n_neighbors, n_max_iters=1)
        for _ in range(n_max_iters):
            # Find perturbation of x
            x_pert = perturbate(best_solution)
            x_pert.fit = fitness_fn(x_pert, problem)

            # Find the best solution in the neighborhood of x_
            x_pert_loc_best = LocalSearch.greedy_hill_climbing(x_pert, problem, fitness_fn, gen_neighborhood_tsp, gen_neighborhood_kp, n_neighbors, n_max_iters=1)

            # Replace the solution if it is better than original
            if x_pert_loc_best.fit > best_solution.fit:
                best_solution = x_pert_loc_best

        return best_solution
