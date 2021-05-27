import numpy as np
import tkinter as tk
from MyGA import AlgorithmType, MultiComponentGeneticAlgorithm
from InitializePopulation import InitializePopulation
from NextGen import NextGen
from Fitness import Fitness
from Selection import SelectionTSP, SelectionKP
from Crossover import CrossoverTSP, CrossoverKP
from Mutation import MutationTSP, MutationKP
from enum import Enum
import LoadDataset
from tkinter import ttk
from Chromosome import Chromosome
from Canvas import MapCanvas
from LocalSearch import LocalSearch
from KPHardConstraintSolver import KPHardConstraintSolver
import time

class TestingGUI:
    def __init__(self):
        self.set_gui()

    def set_gui(self):
        # Set window
        self.master_frame = tk.Tk()
        self.master_frame.title("Testing Interface")
        self.time_passed = 0.0
        self.start_time = 0.0

        #region INITIALIZATION
        self.initializations_frame = tk.LabelFrame(self.master_frame, text='INITIALIZATION', padx=5, pady=5)
        self.initializations_frame.grid(row=0, column=0, sticky='w')

        tk.Label(self.initializations_frame, fg='black', text='Dataset Path: ').grid(row=0, column=0)
        self.dataset_path_entry = tk.Entry(self.initializations_frame, fg='black', bg='white', width=50)
        self.dataset_path_entry.insert(tk.END, 'Datasets/b2.ttp') #Datasets/berlin_example.ttp  Datasets/b2.ttp
        self.dataset_path_entry.grid(row=0, column=1)

        tk.Label(self.initializations_frame, fg='black', text='Algorithm Type: ').grid(row=1, column=0)
        algorithm_type_options = [algorithm.name for algorithm in AlgorithmType]
        algorithm_type_options.reverse()
        algorithm_type_options = TestingGUI.preprocessing_enums_for_option_menu(algorithm_type_options)
        self.algorithm_type_choice = tk.StringVar(self.initializations_frame)
        self.algorithm_type_choice.set(algorithm_type_options[0])
        self.algorithm_type_menu = tk.OptionMenu(self.initializations_frame, self.algorithm_type_choice, *algorithm_type_options, command=self.set_survival_mortality_rate)
        self.algorithm_type_menu.grid(row=1, column=1)

        self.separately_choice = tk.IntVar()
        self.separately_choice.set(0)
        self.separately_cb = tk.Checkbutton(self.initializations_frame, text='Perform TSP and KP separately', variable=self.separately_choice, onvalue=1,
                                         offvalue=0)
        self.separately_cb.grid(row=3, column=0, columnspan=2)
        #endregion

        #region FUNCTIONS
        self.primary_functions_frame = tk.LabelFrame(self.master_frame, text='PRIMARY FUNCTIONS', padx=5, pady=5)
        self.primary_functions_frame.grid(row=1, column=0, sticky='w')

        tk.Label(self.primary_functions_frame, fg='black', text='Population initialization function: ').grid(row=0, column=0)
        init_pop_options = [method for method in dir(InitializePopulation) if method.startswith('_') is False]
        init_pop_options = TestingGUI.preprocessing_enums_for_option_menu(init_pop_options)
        self.init_pop_choice = tk.StringVar(self.primary_functions_frame)
        self.init_pop_choice.set(init_pop_options[0])
        self.init_pop_menu = tk.OptionMenu(self.primary_functions_frame, self.init_pop_choice, *init_pop_options)
        self.init_pop_menu.grid(row=0, column=1)

        tk.Label(self.primary_functions_frame, fg='black', text='Next Generation function: ').grid(row=1, column=0)
        next_gen_options = [method for method in dir(NextGen) if method.startswith('_') is False]
        next_gen_options = TestingGUI.preprocessing_enums_for_option_menu(next_gen_options)
        self.next_gen_choice = tk.StringVar(self.primary_functions_frame)
        self.next_gen_choice.set(next_gen_options[0])
        self.next_gen_menu = tk.OptionMenu(self.primary_functions_frame, self.next_gen_choice, *next_gen_options)
        self.next_gen_menu.grid(row=1, column=1)

        tk.Label(self.primary_functions_frame, fg='black', text='Fitness function: ').grid(row=2, column=0)
        fitness_options = [method for method in dir(Fitness) if method.startswith('_') is False]
        fitness_options = TestingGUI.preprocessing_enums_for_option_menu(fitness_options)
        fitness_options.reverse()
        self.fitness_choice = tk.StringVar(self.primary_functions_frame)
        self.fitness_choice.set(fitness_options[0])
        self.fitness_menu = tk.OptionMenu(self.primary_functions_frame, self.fitness_choice, *fitness_options)
        self.fitness_menu.grid(row=2, column=1)

        tk.Label(self.primary_functions_frame, fg='black', text='KP constraint solver function: ').grid(row=3, column=0)
        constraint_options = [method for method in dir(KPHardConstraintSolver) if method.startswith('_') is False]
        constraint_options = TestingGUI.preprocessing_enums_for_option_menu(constraint_options)
        self.constraint_choice = tk.StringVar(self.primary_functions_frame)
        self.constraint_choice.set(constraint_options[0])
        self.constraint_menu = tk.OptionMenu(self.primary_functions_frame, self.constraint_choice, *constraint_options)
        self.constraint_menu.grid(row=3, column=1)
        #endregion

        # region PARAMETERS
        self.parameters_frame = tk.LabelFrame(self.master_frame, text='PARAMETERS', padx=5, pady=5)
        self.parameters_frame.grid(row=2, column=0, sticky='w')

        tk.Label(self.parameters_frame, fg='black', text='Population size: ').grid(row=0, column=0)
        self.population_size_entry = tk.Entry(self.parameters_frame, fg='black', bg='white', width=10)
        self.population_size_entry.insert(tk.END, '5')
        self.population_size_entry.grid(row=0, column=1)

        tk.Label(self.parameters_frame, fg='black', text='Iteration number: ').grid(row=1, column=0)
        self.iteration_number_entry = tk.Entry(self.parameters_frame, fg='black', bg='white', width=10)
        self.iteration_number_entry.insert(tk.END, '10')
        self.iteration_number_entry.grid(row=1, column=1)

        self.set_survival_mortality_rate()

        self.elitism_choice = tk.IntVar()
        self.elitism_choice.set(1)
        self.elitism_cb = tk.Checkbutton(self.parameters_frame, text='Elitism', variable=self.elitism_choice, onvalue=1,
                                         offvalue=0)
        self.elitism_cb.grid(row=3, column=0, columnspan=2)
        # endregion

        # region TRAVELLING SALESMAN PROBLEM
        self.tsp_frame = tk.LabelFrame(self.master_frame, text='TRAVELLING SALESMAN PROBLEM', padx=5, pady=5)
        self.tsp_frame.grid(row=0, column=1, sticky='w')

        tk.Label(self.tsp_frame, fg='black', text='Selection: ').grid(row=0, column=0)
        selection_tsp_options = [method for method in dir(SelectionTSP) if method.startswith('_') is False]
        selection_tsp_options = TestingGUI.preprocessing_enums_for_option_menu(selection_tsp_options)
        self.selection_tsp_choice = tk.StringVar(self.tsp_frame)
        self.selection_tsp_choice.set(selection_tsp_options[0])
        self.selection_tsp_menu = tk.OptionMenu(self.tsp_frame, self.selection_tsp_choice, *selection_tsp_options)
        self.selection_tsp_menu.grid(row=0, column=1)

        tk.Label(self.tsp_frame, fg='black', text='Crossover: ').grid(row=1, column=0)
        crossover_tsp_options = [method for method in dir(CrossoverTSP) if method.startswith('_') is False and method.endswith('dec') is False]
        crossover_tsp_options = TestingGUI.preprocessing_enums_for_option_menu(crossover_tsp_options)
        self.crossover_tsp_choice = tk.StringVar(self.tsp_frame)
        self.crossover_tsp_choice.set(crossover_tsp_options[0])
        self.crossover_tsp_menu = tk.OptionMenu(self.tsp_frame, self.crossover_tsp_choice, *crossover_tsp_options)
        self.crossover_tsp_menu.grid(row=1, column=1)

        tk.Label(self.tsp_frame, fg='black', text='Crossover prob: ').grid(row=2, column=0)
        self.p_c_tsp_entry = tk.Entry(self.tsp_frame, fg='black', bg='white', width=10)
        self.p_c_tsp_entry.insert(tk.END, '1.')
        self.p_c_tsp_entry.grid(row=2, column=1)

        tk.Label(self.tsp_frame, fg='black', text='Mutation: ').grid(row=3, column=0)
        mutation_tsp_options = [method for method in dir(MutationTSP) if method.startswith('_') is False]
        mutation_tsp_options = TestingGUI.preprocessing_enums_for_option_menu(mutation_tsp_options)
        self.mutation_tsp_choice = tk.StringVar(self.tsp_frame)
        self.mutation_tsp_choice.set(mutation_tsp_options[0])
        self.mutation_tsp_menu = tk.OptionMenu(self.tsp_frame, self.mutation_tsp_choice, *mutation_tsp_options)
        self.mutation_tsp_menu.grid(row=3, column=1)

        tk.Label(self.tsp_frame, fg='black', text='Mutation prob: ').grid(row=4, column=0)
        self.p_m_tsp_entry = tk.Entry(self.tsp_frame, fg='black', bg='white', width=10)
        self.p_m_tsp_entry.insert(tk.END, '1.')
        self.p_m_tsp_entry.grid(row=4, column=1)
        # endregion

        # region KNAPSACK PROBLEM
        self.kp_frame = tk.LabelFrame(self.master_frame, text='KNAPSACK PROBLEM', padx=5, pady=5)
        self.kp_frame.grid(row=1, column=1, sticky='w')

        tk.Label(self.kp_frame, fg='black', text='Selection: ').grid(row=0, column=0)
        selection_kp_options = [method for method in dir(SelectionKP) if method.startswith('_') is False]
        selection_kp_options = TestingGUI.preprocessing_enums_for_option_menu(selection_kp_options)
        self.selection_kp_choice = tk.StringVar(self.kp_frame)
        self.selection_kp_choice.set(selection_kp_options[0])
        self.selection_kp_menu = tk.OptionMenu(self.kp_frame, self.selection_kp_choice, *selection_kp_options)
        self.selection_kp_menu.grid(row=0, column=1)

        tk.Label(self.kp_frame, fg='black', text='Crossover: ').grid(row=1, column=0)
        crossover_kp_options = [method for method in dir(CrossoverKP) if method.startswith('_') is False and method.endswith('dec') is False]
        crossover_kp_options = TestingGUI.preprocessing_enums_for_option_menu(crossover_kp_options)
        self.crossover_kp_choice = tk.StringVar(self.kp_frame)
        self.crossover_kp_choice.set(crossover_kp_options[0])
        self.crossover_kp_menu = tk.OptionMenu(self.kp_frame, self.crossover_kp_choice, *crossover_kp_options)
        self.crossover_kp_menu.grid(row=1, column=1)

        tk.Label(self.kp_frame, fg='black', text='Crossover prob: ').grid(row=2, column=0)
        self.p_c_kp_entry = tk.Entry(self.kp_frame, fg='black', bg='white', width=10)
        self.p_c_kp_entry.insert(tk.END, '1.')
        self.p_c_kp_entry.grid(row=2, column=1)

        tk.Label(self.kp_frame, fg='black', text='Mutation: ').grid(row=3, column=0)
        mutation_kp_options = [method for method in dir(MutationKP) if method.startswith('_') is False]
        mutation_kp_options = TestingGUI.preprocessing_enums_for_option_menu(mutation_kp_options)
        self.mutation_kp_choice = tk.StringVar(self.kp_frame)
        self.mutation_kp_choice.set(mutation_kp_options[0])
        self.mutation_tsp_menu = tk.OptionMenu(self.kp_frame, self.mutation_kp_choice, *mutation_kp_options)
        self.mutation_tsp_menu.grid(row=3, column=1)

        tk.Label(self.kp_frame, fg='black', text='Mutation prob: ').grid(row=4, column=0)
        self.p_m_kp_entry = tk.Entry(self.kp_frame, fg='black', bg='white', width=10)
        self.p_m_kp_entry.insert(tk.END, '1.')
        self.p_m_kp_entry.grid(row=4, column=1)
        # endregion

        # region LOCAL SEARCH
        self.local_search_frame = tk.LabelFrame(self.master_frame, text='LOCAL SEARCH', padx=5, pady=5)
        self.local_search_frame.grid(row=2, column=1, sticky='w')

        tk.Label(self.local_search_frame, fg='black', text='Local Search: ').grid(row=0, column=0)
        local_search_options = [method for method in dir(LocalSearch) if method.startswith('_') is False]
        local_search_options = TestingGUI.preprocessing_enums_for_option_menu(local_search_options)
        self.local_search_choice = tk.StringVar(self.local_search_frame)
        self.local_search_choice.set(local_search_options[0])
        self.local_search_menu = tk.OptionMenu(self.local_search_frame, self.local_search_choice, *local_search_options)
        self.local_search_menu.grid(row=0, column=1)

        tk.Label(self.local_search_frame, fg='black', text='Neighborhood TSP function: ').grid(row=1, column=0)
        neighborhood_tsp_options = [method for method in dir(MutationTSP) if method.startswith('_') is False]
        neighborhood_tsp_options = TestingGUI.preprocessing_enums_for_option_menu(neighborhood_tsp_options)
        self.neighborhood_tsp_choice = tk.StringVar(self.local_search_frame)
        self.neighborhood_tsp_choice.set(neighborhood_tsp_options[0])
        self.neighborhood_tsp_menu = tk.OptionMenu(self.local_search_frame, self.neighborhood_tsp_choice, *neighborhood_tsp_options)
        self.neighborhood_tsp_menu.grid(row=1, column=1)

        tk.Label(self.local_search_frame, fg='black', text='Neighborhood KP function: ').grid(row=2, column=0)
        neighborhood_kp_options = [method for method in dir(MutationKP) if method.startswith('_') is False]
        neighborhood_kp_options = TestingGUI.preprocessing_enums_for_option_menu(neighborhood_kp_options)
        self.neighborhood_kp_choice = tk.StringVar(self.local_search_frame)
        self.neighborhood_kp_choice.set(neighborhood_kp_options[0])
        self.neighborhood_kp_menu = tk.OptionMenu(self.local_search_frame, self.neighborhood_kp_choice, *neighborhood_kp_options)
        self.neighborhood_kp_menu.grid(row=2, column=1)

        tk.Label(self.local_search_frame, fg='black', text='Neighborhood size: ').grid(row=3, column=0)
        self.n_neighbors_entry = tk.Entry(self.local_search_frame, fg='black', bg='white', width=10)
        self.n_neighbors_entry.insert(tk.END, '20')
        self.n_neighbors_entry.grid(row=3, column=1)

        tk.Label(self.local_search_frame, fg='black', text='Number of iters in loc. search: ').grid(row=4, column=0)
        self.max_local_search_iters_entry = tk.Entry(self.local_search_frame, fg='black', bg='white', width=10)
        self.max_local_search_iters_entry.insert(tk.END, '10')
        self.max_local_search_iters_entry.grid(row=4, column=1)
        # endregion

        # region START
        self.start_frame = tk.LabelFrame(self.master_frame, text='START', padx=5, pady=5)
        self.start_frame.grid(row=3, column=0, columnspan=2)

        self.start_button = tk.Button(self.start_frame, text='START', cursor='hand2')
        self.start_button.bind('<Button-1>', self.start_button_clicked)
        self.start_button.grid(row=0, column=0, columnspan=2)

        self.save_img_choice = tk.IntVar()
        self.save_img_choice.set(1)
        self.save_img_cb = tk.Checkbutton(self.start_frame, text='Save average fit as jpg and properties as txt', variable=self.save_img_choice, onvalue=1, offvalue=0)
        self.save_img_cb.grid(row=1, column=0, columnspan=2)

        self.progressbar = ttk.Progressbar(self.start_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progressbar.grid(row=2, column=0, columnspan=2)
        # endregion

        # region CANVAS
        cities_rnd = [[565.0, 575.0, []],
                            [25.0, 185.0, []],
                            [345.0,	750.0, []],
                            [945.0,	685.0, []],
                            [845.0,	655.0, []],
                            [880.0,	660.0, []],
                            [25.0, 230.0, []],
                            [525.0,	1000.0, []],
                            [580.0,	1175.0, []],
                            [650.0,	1130.0, []]]
        path_rnd = [0,1,2,3,4,5,6,7,8,9]
        fit_rnd = 0.0
        self.canvas = MapCanvas(cities_rnd, path_rnd, fit_rnd, self.master_frame, position=(0, 2))
        # endregion

    def update_progress(self, chromosome):
        if not self.separately_choice.get():
            self.progressbar['value'] += 1 / int(self.iteration_number_entry.get()) * 100
        else:
            self.progressbar['value'] += 1 / int(self.iteration_number_entry.get()) * 100 / 2
        self.start_frame.update_idletasks()
        self.update_canvas(path=chromosome.path, fit=chromosome.fit, time=time.time()-self.start_time)

    def start_button_clicked(self, event):
        # Start timer and restart progress bar
        self.time_passed = 0.
        self.start_time = time.time()
        self.progressbar['value'] = 0.

        # Load the problem
        self.problem = LoadDataset.load_problem_ttp(self.dataset_path_entry.get().replace("\\", "/"))

        # Draw cities on canvas
        self.update_canvas(self.problem.cities)

        # Load the parameters of GA
        algorithm_type = AlgorithmType.GENERATIONAL if TestingGUI.preprocessing_option_menu_for_enum(self.algorithm_type_choice.get()) == AlgorithmType.GENERATIONAL.value else AlgorithmType.STEADY_STATE
        init_population_fn = eval('InitializePopulation.' + TestingGUI.preprocessing_option_menu_for_enum(self.init_pop_choice.get()))
        perform_separately = True if self.separately_choice.get() else False
        insert_offspring_into_population = eval('NextGen.' + TestingGUI.preprocessing_option_menu_for_enum(self.next_gen_choice.get()))
        local_search_fn = eval('LocalSearch.' + TestingGUI.preprocessing_option_menu_for_enum(self.local_search_choice.get()))
        fitness_fn = eval('Fitness.' + TestingGUI.preprocessing_option_menu_for_enum(self.fitness_choice.get()))
        selection_tsp_fn = eval('SelectionTSP.' + TestingGUI.preprocessing_option_menu_for_enum(self.selection_tsp_choice.get()))
        crossover_tsp_fn = eval('CrossoverTSP.' + TestingGUI.preprocessing_option_menu_for_enum(self.crossover_tsp_choice.get()))
        mutation_tsp_fn = eval('MutationTSP.' + TestingGUI.preprocessing_option_menu_for_enum(self.mutation_tsp_choice.get()))
        gen_neighborhood_tsp_fn = eval('MutationTSP.' + TestingGUI.preprocessing_option_menu_for_enum(self.neighborhood_tsp_choice.get()))
        selection_kp_fn = eval('SelectionKP.' + TestingGUI.preprocessing_option_menu_for_enum(self.selection_kp_choice.get()))
        crossover_kp_fn = eval('CrossoverKP.' + TestingGUI.preprocessing_option_menu_for_enum(self.crossover_kp_choice.get()))
        mutation_kp_fn = eval('MutationKP.' + TestingGUI.preprocessing_option_menu_for_enum(self.mutation_kp_choice.get()))
        gen_neighborhood_kp_fn = eval('MutationKP.' + TestingGUI.preprocessing_option_menu_for_enum(self.neighborhood_kp_choice.get()))
        pop_size = int(self.population_size_entry.get())
        iter_num = int(self.iteration_number_entry.get())
        self.progressbar.length = iter_num
        mortality_rate = float(self.survival_mortality_entry.get()) if algorithm_type == AlgorithmType.STEADY_STATE else None
        survival_rate = float(self.survival_mortality_entry.get()) if algorithm_type == AlgorithmType.GENERATIONAL else None
        elitism = True if self.elitism_choice.get() else False
        max_loc_search_iters = int(self.max_local_search_iters_entry.get())
        n_neighbors = int(self.n_neighbors_entry.get())
        kp_hc_solver = eval('KPHardConstraintSolver.' + TestingGUI.preprocessing_option_menu_for_enum(self.constraint_choice.get()))
        p_c_tsp = float(self.p_c_tsp_entry.get())
        p_m_tsp = float(self.p_m_tsp_entry.get())
        p_c_kp = float(self.p_c_kp_entry.get())
        p_m_kp = float(self.p_m_kp_entry.get())

        # Initialize GA
        myGA = MultiComponentGeneticAlgorithm(self.problem, algorithm_type, init_population_fn, insert_offspring_into_population, kp_hc_solver, local_search_fn, fitness_fn,
                 selection_tsp_fn, crossover_tsp_fn, mutation_tsp_fn, gen_neighborhood_tsp_fn,
                 selection_kp_fn, crossover_kp_fn, mutation_kp_fn, gen_neighborhood_kp_fn,
                 pop_size=pop_size, mortality_rate=mortality_rate, survival_rate=survival_rate, n_neighbors=n_neighbors, elitism=elitism, max_loc_search_iters=max_loc_search_iters, gui=self,
                 p_c_tsp=p_c_tsp, p_m_tsp=p_m_tsp, p_c_kp=p_c_kp, p_m_kp=p_m_kp,
                 perform_separately=perform_separately)

        # Find the best solution with GA
        best_solution = myGA.run(iter_num)

        # End timer
        self.time_passed = time.time() - self.start_time

        # Save the results
        if self.save_img_choice.get() == 1:
            myGA.save_results(best_solution)

    def set_survival_mortality_rate(self, *args):
        algorithm_type = TestingGUI.preprocessing_option_menu_for_enum(self.algorithm_type_choice.get())
        if algorithm_type == AlgorithmType.GENERATIONAL.value:
            tk.Label(self.parameters_frame, fg='black', text='Survival rate: ').grid(row=2, column=0)
        elif algorithm_type == AlgorithmType.STEADY_STATE.value:
            tk.Label(self.parameters_frame, fg='black', text='Mortality rate: ').grid(row=2, column=0)

        self.survival_mortality_entry = tk.Entry(self.parameters_frame, fg='black', bg='white', width=10)
        self.survival_mortality_entry.insert(tk.END, '0.5') if algorithm_type == AlgorithmType.STEADY_STATE.value else self.survival_mortality_entry.insert(tk.END, '0.1')
        self.survival_mortality_entry.grid(row=2, column=1)


    @staticmethod
    def preprocessing_enums_for_option_menu(vals: list):
        new_vals = []
        for val in vals:
            val_ = val.lower().replace('_', ' ')
            val_ = val_[0].upper() + val_[1:]
            new_vals.append(val_)
        return new_vals

    @staticmethod
    def preprocessing_option_menu_for_enum(val: Enum):
        return val.lower().replace(' ', '_')

    def update_canvas(self, cities=None, path=None, fit=None, time=None):
        if cities is not None:
            self.canvas.update_cities(cities)
        if path is not None:
            self.canvas.update_paths(path)
        if fit is not None and time is not None:
            self.canvas.update_table(fit, time)

    def run(self):
        self.master_frame.mainloop()

def open_GUI(graph_name='Some country'):
    window = TestingGUI()
    window.run()