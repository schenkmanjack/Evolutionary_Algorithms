import torch
import matplotlib.pyplot as plt
from src import IzhikevichNetwork
from src import LinearDataGenerator

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
synaptic_strength = 20  # Synaptic connection strength for simplicity
density = 0.1  # Connection density (10% connectivity)
device = "cuda" if torch.cuda.is_available() else "cpu"
# Create an Izhikevich neuron network
network = IzhikevichNetwork(n_neurons=n_neurons, density=density, synaptic_strength=synaptic_strength, device=device, 
min_spike_rate=500, max_spike_rate=2000)



# network.rate_code_sequence(x, y, data_generator.get_x_bounds(), data_generator.get_y_bounds(), sim_time, dt)
rate_coded_data = network.process_data(x=x, y=y, x_range=data_generator.get_x_bounds(), 
y_range=data_generator.get_y_bounds(), sim_time=20, dt=dt, mask_y=True, mask_pct=0.95)
print(rate_coded_data.shape)
num_steps_input = rate_coded_data.shape[0]
rate_coded_input_full = torch.zeros((n_neurons, num_steps_input), device=device)
input_neuron_count = rate_coded_data.shape[1]
rate_coded_input_full[:input_neuron_count, :] = rate_coded_data.T
rate_coded_input_full = rate_coded_input_full.T

# Plotting results (move data back to CPU for plotting)
rate_coded_input_full_cpu = rate_coded_input_full.T.cpu().numpy()
plt.figure(figsize=(12, 6))
plt.imshow(rate_coded_input_full_cpu, aspect='auto', cmap='gray_r', interpolation='nearest', origin='lower')
plt.xlabel("Time (ms)")
plt.ylabel("Neuron")
plt.title("Spike Raster Plot of Izhikevich Neuron Network")
plt.grid(False)  # Remove grid lines if they interfere with the appearance
# Save the plot as an image file
plt.savefig("rate_coded_input_plot.png")  # Save as a PNG file; you can specify other formats like .pdf, .jpg, etc.
plt.show()

spike_history = network.forward(sim_time=sim_time, dt=dt, 
external_input=rate_coded_input_full)
network.plot_spikes(spike_history)
output = network.get_output(spike_history, time_window=100, num_output_steps=5, 
x_range=data_generator.get_x_bounds(), y_range=data_generator.get_y_bounds())
print(f"output: {output}")


# parameters of network are connection density, connection pattern, synaptic strength, maybe synapse pattern
# environment parameters include how long to deliver stimulus how many points ,etc, change over generations
# assess genetic algorithm by trying to design network that causes neurons to pulse at given rate or have certain neurons off and some one
# multiple neuron classes with different dynamics and different probabilities of connecting to other neurons rather than prescribing architecture