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


print("did genetic algorithm")


network = IzhikevichProbabilityNetwork(parameterization_dict=parameterization_dict, 
network_args=network_args, device=device)

vector = parameterization_to_vector(parameterization_dict=network.get_parameterization(), device=device)
rate_coded_data = network.process_data(x=x, y=y, x_range=data_generator.get_x_bounds(), 
y_range=data_generator.get_y_bounds(), sim_time=20, dt=dt, mask_y=True, mask_pct=0.95)
num_steps_input = rate_coded_data.shape[0]
rate_coded_input_full = torch.zeros((n_neurons, num_steps_input), device=device)
input_neuron_count = rate_coded_data.shape[1]
rate_coded_input_full[:input_neuron_count, :] = rate_coded_data.T
rate_coded_input_full = rate_coded_input_full.T
external_input = rate_coded_input_full
# Plotting results (move data back to CPU for plotting)
# rate_coded_input_full_cpu = rate_coded_input_full.T.cpu().numpy()
# plt.figure(figsize=(12, 6))
# plt.imshow(rate_coded_input_full_cpu, aspect='auto', cmap='gray_r', interpolation='nearest', origin='lower')
# plt.xlabel("Time (ms)")
# plt.ylabel("Neuron")
# plt.title("Spike Raster Plot of Izhikevich Neuron Network")
# plt.grid(False)  # Remove grid lines if they interfere with the appearance
# # Save the plot as an image file
# plt.savefig("rate_coded_input_plot.png")  # Save as a PNG file; you can specify other formats like .pdf, .jpg, etc.
# plt.show()

spike_history = network.forward(sim_time=sim_time, dt=dt, 
external_input=external_input)
network.plot_spikes(spike_history)
output = network.get_output(spike_history, time_window=100, num_output_steps=5, 
x_range=data_generator.get_x_bounds(), y_range=data_generator.get_y_bounds())
print(f"output: {output}")
# target = torch.ones_like(output).to(device)
target = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]).to(device)
loss = network.compute_loss(output, target)
print(f"loss: {loss}")


evaluation_parameters = dict(
    sim_time=sim_time,
    dt=dt,
    mask_y=True,
    mask_pct=0.6,
    time_window=100,
    num_output_steps=5,
    x_range=data_generator.get_x_bounds(),
    y_range=data_generator.get_y_bounds(),
    external_input=external_input,
    target=target
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



# evaluation_loss = network.evaluation(evaluation_parameters=evaluation_parameters)
# print(f"evaluation_loss: {evaluation_loss}")

# parameters of network are connection density, connection pattern, synaptic strength, maybe synapse pattern
# environment parameters include how long to deliver stimulus how many points ,etc, change over generations
# assess genetic algorithm by trying to design network that causes neurons to pulse at given rate or have certain neurons off and some one
# multiple neuron classes with different dynamics and different probabilities of connecting to other neurons rather than prescribing architecture
# a, b, c, d, connectiivty matrix
# n_neurons is 100, 
# 4 * 100 + 100 * 100 = 104 * 100 = 10400
# would that be too big? 



# 6 neuron classes, each class has 4 parameters (a, b, c, d), each class has different probability of connecting to other classes,
# each class has different synapse strength for each other class, each class has different probability of occuring
# 6 * (4 + 6 + 6 + 1) = 6 * 17 = 102