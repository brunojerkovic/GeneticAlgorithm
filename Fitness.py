from Chromosome import Chromosome
from Problem import Problem


class Fitness:
    @staticmethod
    def fitness_tsp(chromosome: Chromosome, problem: Problem) -> float:
        # Calculate the overall distance as the Euclidian distance between each of the cities
        distance = 0
        path = chromosome.path
        for i in range(1, len(path)):
            ind1 = int(path[i - 1])
            ind2 = int(path[i])
            distance += ((problem.cities[ind2][0] - problem.cities[ind1][0]) ** 2 + (
                        problem.cities[ind2][1] - problem.cities[ind1][1]) ** 2) ** 0.5
        return distance

    @staticmethod
    def fitness_ttp(chromosome: Chromosome, problem: Problem, alpha: float = 1.) -> float:
        objective = 0

        for city_num, kp_list in enumerate(chromosome.knapsack):
            weight = 0.
            for item_in_kp_ind, item_picked in enumerate(kp_list):
                # Only count the picked up items
                if item_picked == 1:
                    item_ind = problem.cities[city_num][2][item_in_kp_ind]
                    item_profit, item_weight = problem.items[item_ind]

                    # Add profit to objective function
                    objective += item_profit

                    # Calculate collected weight of items at current city (needed for later)
                    weight += item_weight

            # Calculate distance between current city and the next
            x_curr, y_curr, _ = current_city_coord = problem.cities[city_num]
            x_next, y_next, _ = next_city_coord = problem.cities[city_num + 1 if city_num + 1 < problem.map_dimension[0] else 0]  # This calculates the route from last to first (circullar path)
            distance = ((x_next - x_curr) ** 2 + (y_next - y_curr) ** 2) ** 0.5

            # Calculate speed
            velocity = problem.max_speed - (problem.max_speed - problem.min_speed) * weight / problem.kp_capacity

            # Calculate objective
            objective -= problem.renting_ratio * distance / velocity

        return objective

    @staticmethod
    def fitness_kp(problem: Problem, chromosome: Chromosome, alpha: float = 1.) -> float:
        weight = 0
        profit = 0

        for i, kp in enumerate(chromosome.knapsack):
            weight = 0.
            for j, item in enumerate(kp):
                # Only count the picked up items
                if problem.items[i][j][2] == 1:
                    profit += problem.items[index][0]

                    # Calculate collected weight of items at current city (needed for later)
                    weight += problem.items[index][1]

        # Check if total weight satisfies the constraint
        if weight < problem.kp_capacity:
            return profit
        else:
            # This is possible only if GA uses termination condition
            return profit - alpha * (weight - problem.kp_capacity)  # This is use to guide worse solutions toward better ones (to have degree of 'badness')


