import random
from Chromosome import Chromosome
from Problem import Problem

class KPHardConstraintSolver:
    @staticmethod
    def kp_hard_constraint_solver(chromosome: Chromosome, problem: Problem, remove=None) -> bool:
        # Remove item if it is exceeding weight
        if remove is not None:
            chromosome.knapsack[remove[0]][remove[1]] = 0

        # Count total weight, if exceeded remove item
        weight = 0
        interest = 0
        remove = (0, 0)
        instantiate_interest = True
        for city_ind, kp_list in enumerate(chromosome.knapsack):
            for item_in_kp_ind, item_picked in enumerate(kp_list):
                # If item is picked up add weight
                if item_picked == 1:
                    item_ind = problem.cities[city_ind][2][item_in_kp_ind]
                    item_profit, item_weight = problem.items[item_ind]
                    weight += item_weight

                    # If this item has lower interest, it is the next one for the removal
                    if interest > item_profit / item_weight or instantiate_interest:
                        instantiate_interest = False # This is True only for the first picked up element
                        interest = item_profit / item_weight
                        remove = (city_ind, item_in_kp_ind)

        # If weight exceeds total weight, remove it
        if weight > problem.kp_capacity:
            return KPHardConstraintSolver.kp_hard_constraint_solver(chromosome, problem, remove=remove)
        else:
            return chromosome

    @staticmethod
    def random_kp_hard_constraint_solver(chromosome: Chromosome, problem: Problem, remove=None, num_of_picked_up_items=None) -> bool:
        # Remove item if it is exceeding weight
        if remove is not None:
            try:
                chromosome.knapsack[remove[0]][remove[1]] = 0
            except:
                num_of_picked_up_items += 1

        # Count total weight, if exceeded remove item
        weight = 0
        remove = (0, 0)
        num_of_picked_up_items = sum([1 for kp_list in chromosome.knapsack for item in kp_list if item == 1]) if num_of_picked_up_items is None else num_of_picked_up_items - 1
        random_item = random.randint(0, num_of_picked_up_items)
        for city_ind, kp_list in enumerate(chromosome.knapsack):
            for item_in_kp_ind, item_picked in enumerate(kp_list):
                # If item is picked up add weight
                if item_picked == 1:
                    item_ind = problem.cities[city_ind][2][item_in_kp_ind]
                    _, item_weight = problem.items[item_ind]
                    weight += item_weight

                    # Remove random item
                    if random_item == 0:
                        remove = (city_ind, item_in_kp_ind)
                    else:
                        random_item -= 1

        # If weight exceeds total weight, remove it
        if weight > problem.kp_capacity:
            return KPHardConstraintSolver.random_kp_hard_constraint_solver(chromosome, problem, remove=remove, num_of_picked_up_items=num_of_picked_up_items)
        else:
            return chromosome
