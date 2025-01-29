import random
import torch
import matplotlib.pyplot as plt
from src.snn.izhikevich_network import IzhikevichNetwork
class IzhikevichProbabilityNetwork(IzhikevichNetwork):
    def __init__(self, parameterization_dict, network_args, device="cpu"):
        min_spike_rate = network_args.get("min_spike_rate", 500)
        max_spike_rate = network_args.get("max_spike_rate", 2000)
        n_input_neurons = network_args.get("n_input_neurons", 10)
        n_target_neurons = network_args.get("n_target_neurons", 10)
        n_readout_neurons = network_args.get("n_readout_neurons", 10)
        n_neuron_types = network_args.get("n_neuron_types", 6)
        n_neurons = network_args.get("n_neurons", 100)
        self.results_dir = network_args.get("results_dir", "results")
        super().__init__(n_neurons=n_neurons, a=0.02, b=0.2, c=-65, d=8, initial_v=-65.0, density=0.15, synaptic_strength=2.0,
        min_spike_rate=min_spike_rate, max_spike_rate=max_spike_rate, n_input_neurons=10, n_target_neurons=10, n_readout_neurons=10, device="cpu")
        self.parameterization_dict = parameterization_dict
        self.n_neurons = n_neurons
        self.device = device
        self.min_spike_rate = min_spike_rate
        self.max_spike_rate = max_spike_rate
        # input and target projections
        self.n_input_neurons = n_input_neurons
        self.n_target_neurons = n_target_neurons
        self.n_readout_neurons = n_readout_neurons
        self.input_projection = torch.rand((1, self.n_input_neurons), device=device) 
        self.target_projection = torch.rand((1, self.n_target_neurons), device=device)
        self.readout_projection = torch.rand((1, self.n_readout_neurons), device=device)
        # Network connectivity (randomly generated adjacency matrix)
        torch.manual_seed(0)
        # sample from neuron_probs to get a, b, c, d
        self.a = torch.full((n_neurons,), 1, device=device, dtype=torch.float32)
        self.b = torch.full((n_neurons,), 1, device=device, dtype=torch.float32)
        self.c = torch.full((n_neurons,), 1, device=device, dtype=torch.float32)
        self.d = torch.full((n_neurons,), 1, device=device, dtype=torch.float32)
        neuron_probs = parameterization_dict["neuron_probs"].float()
        # softmax neuron_probs
        # neuron_probs = torch.nn.functional.softmax(torch.tensor(neuron_probs), dim=0)
        if isinstance(neuron_probs, list):
            print("neuron_probs is a list")
        neuron_probs = torch.nn.functional.softmax(neuron_probs.clone().detach(), dim=0)
        # convert to list
        neuron_probs = neuron_probs.tolist()
        a = parameterization_dict["a"]
        b = parameterization_dict["b"]
        c = parameterization_dict["c"]
        d = parameterization_dict["d"]
        connectivity_probs = parameterization_dict["connectivity_probs"]
        connectivity_probs = connectivity_probs.reshape((n_neuron_types, n_neuron_types))
        synaptic_strength = parameterization_dict["synaptic_strength"]
        synaptic_strength = synaptic_strength.reshape((n_neuron_types, n_neuron_types))
        excitatory_probs = parameterization_dict["excitatory_probs"]
        excitatory_probs = excitatory_probs.reshape((n_neuron_types, n_neuron_types))
        initial_v = parameterization_dict["initial_v"][0]
        neuron_indices = list(range(len(neuron_probs)))  # Indices of each neuron
        neurons = random.choices(neuron_indices, weights=neuron_probs, k=n_neurons)
        neurons[:self.n_input_neurons] = [0] * self.n_input_neurons
        neurons[self.n_input_neurons:self.n_input_neurons + self.n_target_neurons] = [1] * self.n_target_neurons
        neurons[-1 * self.n_readout_neurons:] = [2] * self.n_readout_neurons
        for idx, neuron_type in enumerate(neurons):
           a_val = a[neuron_type]
           b_val = b[neuron_type]
           c_val = c[neuron_type]
           d_val = d[neuron_type]
           self.a[idx] = a_val
           self.b[idx] = b_val
           self.c[idx] = c_val
           self.d[idx] = d_val
           # add to connectivity matrix
        self.connectivity_matrix = torch.zeros(n_neurons, n_neurons, device=device)
        self.synaptic_strength_matrix = torch.zeros(n_neurons, n_neurons, device=device)
        for i in range(n_neurons):
            for j in range(n_neurons):
                if i != j:
                    prob = connectivity_probs[neurons[i]][neurons[j]]
                    self.synaptic_strength_matrix[i][j] = synaptic_strength[neurons[i]][neurons[j]]
                    self.connectivity_matrix[i][j] = torch.rand(1, device=device) < prob
                    prob_excitatory = excitatory_probs[neurons[i]][neurons[j]]
                    if torch.rand(1, device=device) > prob_excitatory: # make inhibitory
                        self.synaptic_strength_matrix[i][j] *= -1


        self.weights = torch.rand(n_neurons, n_neurons, device=device) * self.synaptic_strength_matrix * self.connectivity_matrix
        # State variables for all neurons (initialized as float tensors)
        self.v = torch.full((n_neurons,), initial_v, device=device, dtype=torch.float32)  # Membrane potential
        self.u = self.b * initial_v  # Recovery variable for all neurons (vectorized and set as float)

    def get_properties(self):
        properties = dict(
            a=self.a,
            b=self.b,
            c=self.c,
            d=self.d,
            connectivity_matrix=self.connectivity_matrix,
            weights=self.weights
        )
        return properties
    
    def set_properties(self, properties):
        self.a = properties["a"]
        self.b = properties["b"]
        self.c = properties["c"]
        self.d = properties["d"]
        self.connectivity_matrix = properties["connectivity_matrix"]
        self.weights = properties["weights"]

    def output_to_rate(self, spike_history, dt):
        # Calculate the firing rate of each neuron
        rates = spike_history.sum(dim=1) / spike_history.shape[1] * 1000 / dt
        return rates


    def rescale_data(self, x, y, x_range, y_range):
        x_min, x_max = x_range
        y_min, y_max = y_range
        min_spike_rate = self.min_spike_rate
        max_spike_rate = self.max_spike_rate
        # scale x and y to be within spike_range
        if x is not None:
            x = (x - x_min) / (x_max - x_min) * (max_spike_rate - min_spike_rate) + min_spike_rate
        if y is not None:
            y = (y - y_min) / (y_max - y_min) * (max_spike_rate - min_spike_rate) + min_spike_rate
        return x, y
    
    def unscale_data(self, x, y, x_range, y_range):
        x_min, x_max = x_range
        y_min, y_max = y_range
        min_spike_rate = self.min_spike_rate
        max_spike_rate = self.max_spike_rate
        # scale x and y to be within spike_range
        if x is not None:
            x = (x - min_spike_rate) / (max_spike_rate - min_spike_rate) * (x_max - x_min) + x_min
        if y is not None:
            y = (y - min_spike_rate) / (max_spike_rate - min_spike_rate) * (y_max - y_min) + y_min
        return x, y
    
    def rate_code_input(self, rates, sim_time, dt):
        """
        Converts input values (rates) into spike trains using rate coding.
        :param rates: tensor of input rates (in Hz) for each neuron
        :param sim_time: total simulation time in ms
        :param dt: time step in ms
        :return: rate-coded current input of shape (n_neurons, sim_time)
        """

        # spread rates over more neurons

        n_neurons = rates.shape[0]
        spike_train = torch.zeros((n_neurons, sim_time), device=self.device)
        # Convert rates to probabilities per timestep (Poisson process)
        spike_prob = rates * dt / 1000  # Probability of spike per ms
        for t in range(sim_time):
            spike_train[:, t] = (torch.rand(n_neurons, device=self.device) < spike_prob).float()
        # Each spike in spike_train becomes a current pulse (1.0 units here) at each spike time
        return spike_train
    
    def rate_code_sequence(self, x, y, x_range, y_range, sim_time, dt, mask_y=False, mask_pct=0.2):
        x, y = self.rescale_data(x, y, x_range, y_range)
        # randomly choose row of y
        y = y[torch.randint(0, y.shape[0], (1,))]
        x = x.unsqueeze(0)
        x = x.to(self.device)
        y = y.to(self.device)
        x_projected = x.T @ self.input_projection
        y_projected = y.T @ self.target_projection
        if mask_y:
            y_projected[-1 * int(mask_pct * y_projected.shape[0]):] = y_range[0] # had been -1
        # project x and y onto more neurons
        data = torch.cat((x_projected, y_projected), dim=1).to(self.device)
        # data = data.T
        rate_coded_data = torch.zeros((sim_time * data.shape[0], data.shape[1]), device=self.device)
        start_time = 0
        for data_slice in data:
            new_data = self.rate_code_input(data_slice, sim_time, dt)
            rate_coded_data[start_time:start_time + sim_time] = new_data.T
            start_time += sim_time
        return rate_coded_data

    def process_data(self, x, y, x_range, y_range, sim_time, dt, mask_y=False, mask_pct=0.2):   
        # x, y = self.rescale_data(x, y, x_range, y_range)
        # # randomly choose row of y
        # y = y[torch.randint(0, y.shape[0], (1,))]
        # data = torch.cat((x.unsqueeze(0), y), dim=0).to(self.device)
        # data = data.T
        rate_coded_data = 50 * self.rate_code_sequence(x=x, y=y, x_range=x_range, 
        y_range=y_range, sim_time=sim_time, dt=dt, mask_y=mask_y, mask_pct=mask_pct)
        return rate_coded_data
    
    def forward_step(self, I, dt=1.0):  # Ensure dt is a float
        # Izhikevich model equations for the entire network (vectorized)
        dv = 0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + I
        du = self.a * (self.b * self.v - self.u)
        self.v += dv * dt
        self.u += du * dt

        # Check for spikes and reset membrane potentials where v >= 30
        spiked = self.v >= 30
        self.v[spiked] = self.c[spiked]
        self.u[spiked] += self.d[spiked]
        
        return spiked  # Returns a tensor indicating if each neuron spiked
    
    def forward(self, sim_time=1000, dt=1.0, external_input=None):
         # Initialize input current and spike history
         n_neurons = self.n_neurons
         device = self.device
         input_current = torch.zeros(n_neurons, device=device)
         spike_history = torch.zeros((n_neurons, sim_time), device=device)
         # Initial external input to all neurons to start the network activity
         # Initial external input with slight randomness
         if external_input is None:
            external_input = torch.rand((50, n_neurons), device=device) * 20.0  # Random values between 0 and 2
         t_bound = external_input.shape[0]
         # external_input = torch.full((n_neurons,), 30.0, device=device)  # Adjust the current as needed
         constant_background_input = torch.full((n_neurons,), 10, device=device)  # Small constant input
         # Simulation loop
         for t in range(sim_time):
            # Add the external input at the beginning of the simulation (or periodically)
            if t < t_bound:  # Apply only for the first 50 timesteps as a starter pulse
                input_current = external_input[t]
            else:
                # Compute input current as weighted sum of spikes from connected neurons
                input_current = torch.matmul(self.weights, spike_history[:, t-1]) #+ constant_background_input

            # Update network state in parallel
            spiked = self.forward_step(input_current, dt)
            
            # Record spike events
            spike_history[:, t] = spiked.float()
         return spike_history
    
    def get_output(self, spike_history, time_window=100, num_output_steps=10, x_range=[-1, 1], y_range=[-1,1]):
        # Calculate the loss as the difference between the target and the spike history
        total_time_window = time_window * num_output_steps
        output = spike_history[-1 * self.n_readout_neurons:, -1 * total_time_window:]
        output = output.reshape((self.n_readout_neurons, num_output_steps * time_window))
        output = self.readout_projection @ output
        output = output.reshape((num_output_steps, time_window))
        # get rate
        output = self.output_to_rate(output, dt=1)
        _, output = self.unscale_data(x=None, y=output, x_range=x_range, y_range=y_range)
        return output
    
    def plot_output(self, spike_history, evaluation_parameters, target, save_path="output_plot.png"):
        time_window = evaluation_parameters["time_window"]
        num_output_steps = evaluation_parameters["num_output_steps"]
        data_generator = evaluation_parameters.get("data_generator", None)
        x_range = data_generator.get_x_bounds()
        y_range = data_generator.get_y_bounds()
        output = self.get_output(spike_history, time_window=time_window, num_output_steps=num_output_steps, 
        x_range=x_range, y_range=y_range)
        # target = evaluation_parameters["target"]
        output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        plt.figure(figsize=(12, 6))
        plt.plot(output, label="output")
        plt.plot(target, label="target")
        plt.legend()
        plt.savefig(f"{self.results_dir}/{save_path}")  # Save as a PNG file; you can specify other formats like .pdf, .jpg, etc.
        plt.show()
    
    def compute_loss(self, output, target):
        # calculate mse loss
        output = output.to(self.device)
        target = target.to(self.device)
        loss = torch.nn.functional.mse_loss(output, target)
        return loss
    
    def evaluation(self, evaluation_parameters):
        loss = 0
        for i in range(8):
            loss += self.evaluation_helper(evaluation_parameters)
        return loss 

    def evaluation_helper(self, evaluation_parameters):
        # perform simulation
        external_input, x, y = self.generate_external_input(evaluation_parameters)
        data_generator = evaluation_parameters.get("data_generator", None)
        x_range = data_generator.get_x_bounds()
        y_range = data_generator.get_y_bounds()
        sim_time = evaluation_parameters.get("sim_time", 1000)
        dt=evaluation_parameters.get("dt", 1.0)
        
        #
        spike_history = self.forward(sim_time=sim_time, dt=dt, external_input=external_input)
        # extract output from spike history
        time_window = evaluation_parameters.get("time_window", 100)
        num_output_steps = evaluation_parameters.get("num_output_steps", 10)
        output = self.get_output(spike_history=spike_history, time_window=time_window, num_output_steps=num_output_steps,
        x_range=x_range, y_range=y_range)
        # compute loss
        y = y[0]
        target = y[-1 * num_output_steps:]
        loss = -1 * self.compute_loss(output, target)
        return loss
    
    def generate_external_input(self, evaluation_parameters):
        sim_time = evaluation_parameters.get("sim_time", 1000)
        dt=evaluation_parameters.get("dt", 1.0)
        data_generator = evaluation_parameters.get("data_generator", None)
        x, y = data_generator.get_data()
        y = y[torch.randint(0, y.shape[0], (1,))]
        x_range = data_generator.get_x_bounds()
        y_range = data_generator.get_y_bounds()
        mask_y = evaluation_parameters.get("mask_y", False)
        mask_pct = evaluation_parameters.get("mask_pct", 0.2)

        # generate external input
        rate_coded_data = self.process_data(x=x, y=y, x_range=x_range, 
        y_range=y_range, sim_time=sim_time, dt=dt, mask_y=mask_y, mask_pct=mask_pct)
        num_steps_input = rate_coded_data.shape[0]
        rate_coded_input_full = torch.zeros((self.n_neurons, num_steps_input), device=self.device)
        input_neuron_count = rate_coded_data.shape[1]
        rate_coded_input_full[:input_neuron_count, :] = rate_coded_data.T
        rate_coded_input_full = rate_coded_input_full.T
        external_input = rate_coded_input_full
        return external_input, x, y

    def get_parameterization(self):
        return self.parameterization_dict

    def plot_spikes(self, spike_history, save_path="spike_raster_plot.png"):
        # Plotting results (move data back to CPU for plotting)
        spike_history_cpu = spike_history.cpu().numpy()
        plt.figure(figsize=(12, 6))
        plt.imshow(spike_history_cpu, aspect='auto', cmap='gray_r', interpolation='nearest', origin='lower')
        plt.xlabel("Time (ms)")
        plt.ylabel("Neuron")
        plt.title("Spike Raster Plot of Izhikevich Neuron Network")
        plt.grid(False)  # Remove grid lines if they interfere with the appearance
        # Save the plot as an image file
        plt.savefig(f"{self.results_dir}/{save_path}")  # Save as a PNG file; you can specify other formats like .pdf, .jpg, etc.
        plt.show()