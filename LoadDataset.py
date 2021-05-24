import numpy as np
from Problem import Problem

def load_coordinates_tsp(filename):
    cities = None

    with open(filename) as f:
        for i, line in enumerate(f.readlines()):
            if line.startswith('DIMENSION'):
                cities = np.zeros((int(line.split()[2]), 2))

            if line[0].isdigit():
                instance_num, x_coord, y_coord = line.split()
                instance_num = int(instance_num) - 1
                x_coord = float(x_coord[:-2])
                y_coord = float(y_coord[:-2])
                cities[instance_num] = np.array([x_coord, y_coord])

    return cities

def load_problem_ttp(filename):
    problem = Problem()
    insert_nodes = True

    with open(filename) as f:
        for i, line in enumerate(f.readlines()):
            # Read problem's metadata
            if line.startswith('PROBLEM NAME'):
                problem.problem_name = line.split(':')[1].strip()
            elif line.startswith('DIMENSION'):
                map_dimension = (int(line.split(':')[1]), 2)
                problem.cities = [[0., 0., []] for _ in range(map_dimension[0])] # This is to add the coords and indexes of the items of this city
                problem.map_dimension = map_dimension
                problem.knapsack_dimension = np.zeros(map_dimension[0])
            elif line.startswith('NUMBER OF ITEMS'):
                item_dimension = (int(line.split(':')[1]), 2)
                problem.items = np.zeros(item_dimension)
                problem.item_dimension = item_dimension
            elif line.startswith('CAPACITY OF KNAPSACK'):
                problem.kp_capacity = int(line.split(':')[1])
            elif line.startswith('MIN SPEED'):
                problem.min_speed = float(line.split(':')[1])
            elif line.startswith('MAX SPEED'):
                problem.max_speed = float(line.split(':')[1])
            elif line.startswith('RENTING RATIO'):
                problem.renting_ratio = float(line.split(':')[1])

            if line.startswith('ITEMS SECTION'):
                insert_nodes = False

            # Insert nodes
            if line[0].isdigit() and insert_nodes == True:
                instance_num, x_coord, y_coord = line.split()
                problem.cities[int(instance_num)-1] = np.array([float(x_coord), float(y_coord), []], dtype=object)

            # Insert items
            if line[0].isdigit() and insert_nodes == False:
                instance_num, profit, weight, node_num = line.split()
                problem.total_profit += int(profit) # Add this item's profit to total profit
                problem.cities[int(node_num) - 1][2].append(int(instance_num)-1)    # Add this item to its city
                problem.items[int(instance_num)-1] = np.array([float(profit), float(weight)])   # Add this item to the list of items
                problem.knapsack_dimension[int(node_num)-1] += 1

    return problem



