import torch
from .conversions import *

class GeneticAlgorithmBase:
    def __init__(self, config, parameterization_dict, parameterization_std_devs_dict, parameter_name_dict, network_class, network_args, device="cpu"):
        
        self.population_size = config.get("population_size", 10)
        self.num_generations = config.get("num_generations", 10)
        self.mutation_rate = config.get("mutation_rate", 0.1)
        self.population_size = config.get("population_size", 0.1)
        self.crossover_rate = config.get("crossover_rate", 0.1)
        self.device = device
        self.population = None
        self.fitness_scores = None
        self.best_individual = None
        self.best_fitness = None
        self.std_devs = parameterization_to_vector(parameterization_dict=parameterization_std_devs_dict, device=self.device)
        self.population = self.initialize_population(parameterization_dict=parameterization_dict, population_size=self.population_size, std_devs=self.std_devs)
        population = self.initialize_population(parameterization_dict=parameterization_dict, population_size=self.population_size, std_devs=self.std_devs)
        self.parameter_name_dict = parameter_name_dict
        self.network_class = network_class
        self.network_args = network_args
        self.input_projection = None
        self.target_projection = None
        self.readout_projection = None
        self.networks_array = self.create_networks(population=population, parameter_name_dict=self.parameter_name_dict, 
        network_class=self.network_class, network_args=self.network_args)
        self.elites = None
        self.elites_fitness_scores = torch.zeros(0, device=self.device)

        # projections
        self.input_projection = self.networks_array[0].input_projection
        self.target_projection = self.networks_array[0].target_projection
        self.readout_projection = self.networks_array[0].readout_projection
        for network in self.networks_array:
            network.input_projection = self.input_projection
            network.target_projection = self.target_projection
            network.readout_projection = self.readout_projection
        
      
    
    def mutate(self, individual):
        # Generate Gaussian noise with specific std deviation for each entry
        noise = torch.randn(individual.size(), device=self.device) * self.std_devs
        # Apply mutation to the individual
        mutated_individual = individual + noise * self.mutation_rate
        return mutated_individual
    
    def crossover(self, parent1, parent2):
        vector_dim = parent1.size()[0]
        children = torch.zeros((2, vector_dim), device=self.device)
        # Randomly select crossover points
        crossover_points = torch.rand(parent1.size(), device=self.device) < 0.5
        # Perform crossover
        child1 = parent1 * crossover_points + parent2 * ~crossover_points
        child2 = parent2 * crossover_points + parent1 * ~crossover_points
        children[0] = child1
        children[1] = child2
        return children
    
    def compute_fitness_scores(self, networks_array, evaluation_parameters):
        fitness_scores = torch.tensor([network.evaluation(evaluation_parameters=evaluation_parameters) for network in networks_array]).to(self.device)
        # update elites
        return fitness_scores
    
    # def select_parents(self, population, fitness_scores):
    #     # Select parents based on fitness scores, randomly selecting population_size parents
    #     parent_indices = torch.argsort(fitness_scores, descending=True)[:2]
    #     parent1 = population[parent_indices[0]]
    #     parent2 = population[parent_indices[1]]
    #     return parent1, parent2

    def select_parents(self, population, fitness_scores, num_return=None):
        """
        Select parents using softmax-normalized fitness scores for probabilistic selection.
        """
        # Ensure fitness scores are a tensor
        if isinstance(fitness_scores, list):
            fitness_scores = torch.tensor(fitness_scores, dtype=torch.float32)
        
        # check num_return
        if num_return is None:
            num_return = 2 * self.population_size
        # Apply softmax to compute selection probabilities
        selection_probabilities = torch.softmax(fitness_scores, dim=0)

        # Randomly select self.population_size parents based on probabilities
        parent_indices = torch.multinomial(selection_probabilities, num_return, replacement=True)

        # Select the parents from the population
        parents = population[parent_indices]

        return parents

    
    def get_next_generation(self, evaluation_parameters, generation_id=0, plot_spikes=False):
        fitness_scores = self.compute_fitness_scores(networks_array=self.networks_array, evaluation_parameters=evaluation_parameters)
        self.fitness_scores = fitness_scores
        self.compute_elites()
        print(f"best_fitness: {torch.max(fitness_scores)}, mean_fitness: {torch.mean(fitness_scores)}")
        if plot_spikes:
            best_network = self.networks_array[torch.argmax(fitness_scores)]
            sim_time = evaluation_parameters["sim_time"]
            dt = evaluation_parameters["dt"]
            external_input, x, y = best_network.generate_external_input(evaluation_parameters=evaluation_parameters)
            spike_history = best_network.forward(sim_time=sim_time, dt=dt, external_input=external_input)
            best_network.plot_spikes(spike_history, save_path=f"spike_plot_gen_{generation_id}.png")
            # plot output
            num_output_steps = evaluation_parameters.get("num_output_steps", 10)
            target = y[0]
            target = y[-1 * num_output_steps:]
            best_network.plot_output(spike_history=spike_history, evaluation_parameters=evaluation_parameters, target=target, save_path=f"output_plot_gen_{generation_id}.png")

        parents = self.select_parents(population=self.elites, fitness_scores=self.elites_fitness_scores, num_return=int(round(self.population_size * 2)))
        # group into pairs
        parent_pairs = parents.view(-1, 2, parents.size()[-1])
        num_parent_pairs = parent_pairs.size()[0]
        # crossover
        num_children = parents.size()[0]
        i = 0
        children = torch.zeros_like(parents, device=self.device)
        while i < num_parent_pairs: 
            double_i = 2 * i
            children[double_i:double_i + 2] = self.crossover(parent1=parent_pairs[i][0], parent2=parent_pairs[i][1])
            i += 2
        self.population = children
        # mutate
        self.population = torch.stack([self.mutate(individual) for individual in self.population])
        self.networks_array = self.create_networks(population=self.population, parameter_name_dict=self.parameter_name_dict, 
        network_class=self.network_class, network_args=self.network_args)
        
    def compute_elites(self):
        # find the top k from fitness scores and
        if self.elites is None:
            self.elites = self.population
            self.elites_fitness_scores = self.fitness_scores
            return
        total_population = torch.vstack((self.elites, self.population))
        total_fitness_scores = torch.cat((self.elites_fitness_scores, self.fitness_scores))
        top_k_indices = torch.argsort(total_fitness_scores, descending=True)[:self.population_size]
        self.elites = total_population[top_k_indices]
        self.elites_fitness_scores = total_fitness_scores[top_k_indices]
    def initialize_population(self, parameterization_dict, population_size, std_devs):
        # create initial vector
        vector = parameterization_to_vector(parameterization_dict=parameterization_dict, device=self.device)
        # create population
        # create mutations from std_devs
        mutations = torch.randn((population_size, vector.size()[0]), device=self.device) * std_devs
        # add mutations to vector
        population = vector + vector * mutations
        return population
    
    def create_networks(self, population, parameter_name_dict,  network_class, network_args):
        population_size = population.size()[0]
        parameterization_array = [vector_to_parameterization(vector=population[i], parameter_name_dict=parameter_name_dict) for i in range(population_size)]
        networks_array = [network_class(parameterization_dict=parameterization, network_args=network_args, device=self.device) for parameterization in parameterization_array]
        for network in networks_array:
            if self.input_projection is not None:
                network.input_projection = self.input_projection
            if self.target_projection is not None:
                network.target_projection = self.target_projection
            if self.readout_projection is not None:
                network.readout_projection = self.readout_projection
        return networks_array



        
        
    
