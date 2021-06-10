import tkinter as tk
import numpy as np
from Fitness import Fitness
from Selection import Selection
from Crossover import Crossover
from InitializePopulation import InitializePopulation
from Mutation import Mutation
from NextGen import NextGen
from MyGA import MultiComponentGeneticAlgorithm
import time

class MapCanvas:
    def __init__(self, cities, path, fit, avg_fit, master_frame, position=(0,1), canvas_shape=(500, 500), time=0.0):
        # region INIT ATTRIBUTES
        self.canvas_shape = canvas_shape
        self.master_frame = master_frame
        self.paths = np.array(path)
        coords = np.array(cities, dtype=object)
        self.coords = self.normalize_data_on_canvas(coords)
        self.num_of_cities = coords.shape[0]
        # endregion

        # region CANVAS
        self.canvas = tk.Canvas(self.master_frame)
        self.canvas.config(width=self.canvas_shape[0], height=self.canvas_shape[1])
        self.canvas.grid(row=position[0], column=position[1], rowspan=4)
        # endregion

        # region RESULTS FRAME
        self.results_frame = tk.LabelFrame(self.master_frame, text='RESULTS', padx=5, pady=5)
        self.results_frame.grid(row=position[0]+5, column=0, columnspan=2)

        self.city_num_var = tk.StringVar()
        self.city_num_var.set(f'Number of cities: {self.num_of_cities}')
        self.city_num_lbl = tk.Label(self.results_frame, fg='black', textvariable=self.city_num_var)
        self.city_num_lbl.pack()

        self.fitness_num_var = tk.StringVar()
        self.fitness_num_var.set(f'Fitness of the best chromosome: {round(fit, 2)}')
        self.fitness_num_lbl = tk.Label(self.results_frame, fg='black', textvariable=self.fitness_num_var)
        self.fitness_num_lbl.pack()

        self.avg_fitness_num_var = tk.StringVar()
        self.avg_fitness_num_var.set(f'Average fitness of population: {round(avg_fit, 2)}')
        self.avg_fitness_num_lbl = tk.Label(self.results_frame, fg='black', textvariable=self.avg_fitness_num_var)
        self.avg_fitness_num_lbl.pack()

        self.time_num_var = tk.StringVar()
        self.time_num_var.set(f'Time: {round(time, 2)}s')
        self.time_lbl = tk.Label(self.results_frame, fg='black', textvariable=self.time_num_var)
        self.time_lbl.pack()
        # endregion

        self.draw_city_dots()
        self.connect_cities()

    def update_paths(self, paths):
        self.paths = np.array(paths)

        # Redraw all canvas
        self.canvas.delete('all')
        self.draw_city_dots()
        self.connect_cities()

    def update_cities(self, cities):
        # Draw number of cities
        self.num_of_cities = len(cities)
        self.city_num_var.set(f'Number of cities: {self.num_of_cities}')

        # Get normalized coords
        coords = np.array(cities, dtype=object)
        self.coords = self.normalize_data_on_canvas(coords)
        
        # Draw cities
        self.canvas.delete('all')
        self.draw_city_dots()

    def update_table(self, fit, avg_fit, time):
        self.avg_fitness_num_var.set(f'Average fitness of population: {round(avg_fit, 2)}')
        self.fitness_num_var.set(f'Fitness of the best chromosome: {round(fit, 2)}')
        self.time_num_var.set(f'Time: {round(time, 2)}s')

    def normalize_data_on_canvas(self, coords, pad=10):
        canvas_shape = self.canvas_shape
        normalized_coords = np.zeros((coords.shape[0], 2))
        normalized_coords.T[0] = canvas_shape[0] - (coords.T[0] - min(coords.T[0])+pad) / \
                                 (max(coords.T[0]+pad) - min(coords.T[0])+pad) * canvas_shape[0]
        normalized_coords.T[1] = canvas_shape[1] - (coords.T[1] - min(coords.T[1])+pad) / \
                                 (max(coords.T[1]+pad) - min(coords.T[1])+pad) * canvas_shape[1]

        return normalized_coords

    def draw_city_dots(self, color='blue', radius=3):
        for point in self.coords:
            x_start, y_start = point[0]-radius, point[1]-radius
            x_end, y_end = point[0]+radius, point[1]+radius
            self.canvas.create_oval(y_start, x_start, y_end, x_end, fill=color)

    def connect_cities(self, color='red'):
        for i in range(self.num_of_cities-1):
            ind_start = int(self.paths[i])
            ind_end = int(self.paths[i+1])

            x_start, y_start = self.coords[ind_start]
            x_end, y_end = self.coords[ind_end]

            self.canvas.create_line(y_start, x_start, y_end, x_end, fill=color)

    def refresh_table(self):
        self.results_frame.update_idletasks()