import torch
import matplotlib.pyplot as plt
from src import IzhikevichProbabilityNetwork
from src import LinearDataGenerator
from src import parameterization_to_vector, vector_to_parameterization
from src import GeneticAlgorithmBase

NUM_LINES = 20
NUM_POINTS = 100
SLOPE_BOUNDS = [-1, 1]
Y_INTERCEPT_BOUNDS = [-2, 2]
X_RANGE = [-10, 10]

data_generator = LinearDataGenerator(num_lines=NUM_LINES, num_points=NUM_POINTS, slope_bounds=SLOPE_BOUNDS, 
y_intercept_bounds=Y_INTERCEPT_BOUNDS, x_range=X_RANGE)
x, y = data_generator.get_data()


# Simulation parameters
n_neurons = 100  # Number of neurons in the network
sim_time = 10000  # Simulation time in milliseconds
dt = 1           # Time step (1 ms)
synaptic_strength = 10  # Synaptic connection strength for simplicity
density = 0.1  # Connection density (10% connectivity)
device = "cuda" if torch.cuda.is_available() else "cpu"
# Create an Izhikevich neuron network
num_neuon_types = 6
connectivity_probs = 0.1 * torch.ones(num_neuon_types ** 2, device=device)
synaptic_strength = 20.0 * torch.ones(num_neuon_types ** 2, device=device)
excitatory_probs = 0.6 * torch.ones(num_neuon_types ** 2, device=device)
parameterization_dict = dict(
    neuron_probs=torch.tensor([1, 1, 1, 1, 1, 1]),
    a=torch.tensor([0.02, 0.02, 0.02, 0.02, 0.02, 0.02]),
    b=torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
    c=torch.tensor([-65, -65, -65, -65, -65, -65]),
    d=torch.tensor([6, 6, 6, 6, 6, 6]),
    initial_v=torch.tensor([-65.0]),
    connectivity_probs=connectivity_probs,
    synaptic_strength=synaptic_strength,
    excitatory_probs=excitatory_probs
)

connectivity_probs_std_dev = 0.1 * torch.ones(num_neuon_types ** 2, device=device)
synaptic_strength_std_dev = 0.1 * torch.ones(num_neuon_types ** 2, device=device)
excitatory_probs_std_dev = 0.1 * torch.ones(num_neuon_types ** 2, device=device)
parameterization_std_devs_dict = dict(
    neuron_probs=torch.tensor([0.02, 0.02, 0.02, 0.02, 0.02, 0.02]),
    a=torch.tensor([0.02, 0.02, 0.02, 0.02, 0.02, 0.02]),
    b=torch.tensor([0.02, 0.02, 0.02, 0.02, 0.02, 0.02]),
    c=torch.tensor([1, 1, 1, 1, 1, 1]),
    d=torch.tensor([1, 1, 1, 1, 1, 1]),
    initial_v=torch.tensor([1]),
    connectivity_probs=connectivity_probs_std_dev,
    synaptic_strength=synaptic_strength_std_dev,
    excitatory_probs=excitatory_probs_std_dev
)
parameter_name_dict = dict(
    neuron_probs=6,
    a=6,
    b=6,
    c=6,
    d=6,
    initial_v=1,
    connectivity_probs=36,
    synaptic_strength=36,
    excitatory_probs=36
)

network_args = dict(
    min_spike_rate=500,
    max_spike_rate=2000,
    n_input_neurons=10,
    n_target_neurons=10,
    n_readout_neurons=10,
    n_neurons=n_neurons,
    n_neuron_types=6,
    results_dir="results",
)

genetic_algorithm_config = dict(
    population_size=10, 
    num_generations=10,
    mutation_rate=0.1, 
    crossover_rate=0.1
)
genetic_algorithm_base = GeneticAlgorithmBase(parameterization_dict=parameterization_dict, 
parameterization_std_devs_dict=parameterization_std_devs_dict,
parameter_name_dict=parameter_name_dict,
network_class=IzhikevichProbabilityNetwork,
network_args=network_args,
config=genetic_algorithm_config,
device=device)


evaluation_parameters = dict(
    sim_time=sim_time,
    dt=dt,
    mask_y=True,
    mask_pct=0.6,
    time_window=100,
    num_output_steps=5,
    data_generator=data_generator,
)

# networks_array = genetic_algorithm_base.networks_array
# fitness_scores = genetic_algorithm_base.compute_fitness_scores(networks_array=networks_array, evaluation_parameters=evaluation_parameters)
# print(f"fitness_scores: {fitness_scores}")
# parents = genetic_algorithm_base.select_parents(population=genetic_algorithm_base.population, fitness_scores=fitness_scores)
num_generations = 10
for i in range(num_generations):
    print(f"Generation {i + 1}")
    genetic_algorithm_base.get_next_generation(evaluation_parameters=evaluation_parameters, 
    generation_id=i, 
    plot_spikes=True)

print("done")

