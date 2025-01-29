import torch
import matplotlib.pyplot as plt
class IzhikevichNetwork:
    def __init__(self, n_neurons, a=0.02, b=0.2, c=-65, d=8, initial_v=-65.0, density=0.15, synaptic_strength=2.0,
    min_spike_rate=1.0, max_spike_rate=20.0, n_input_neurons=10, n_target_neurons=10, n_readout_neurons=10, device="cpu"):
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
        self.connectivity_matrix = (torch.rand(n_neurons, n_neurons, device=device) < density).float()  # 10% connectivity
        self.weights = torch.rand(n_neurons, n_neurons, device=device) * synaptic_strength * self.connectivity_matrix
        # Parameters for all neurons (ensuring they are float tensors)
        self.a = torch.full((n_neurons,), a, device=device, dtype=torch.float32)
        self.b = torch.full((n_neurons,), b, device=device, dtype=torch.float32)
        self.c = torch.full((n_neurons,), c, device=device, dtype=torch.float32)
        self.d = torch.full((n_neurons,), d, device=device, dtype=torch.float32)
        # State variables for all neurons (initialized as float tensors)
        self.v = torch.full((n_neurons,), initial_v, device=device, dtype=torch.float32)  # Membrane potential
        self.u = self.b * initial_v  # Recovery variable for all neurons (vectorized and set as float)

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
            y_projected[-1 * int(mask_pct * y_projected.shape[0]):] = -1
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
    
    def compute_loss(self, output, target):
        # calculate mse loss
        loss = torch.nn.functional.mse_loss(output, target)
        return loss

    def plot_spikes(self, spike_history):
        # Plotting results (move data back to CPU for plotting)
        spike_history_cpu = spike_history.cpu().numpy()
        plt.figure(figsize=(12, 6))
        plt.imshow(spike_history_cpu, aspect='auto', cmap='gray_r', interpolation='nearest', origin='lower')
        plt.xlabel("Time (ms)")
        plt.ylabel("Neuron")
        plt.title("Spike Raster Plot of Izhikevich Neuron Network")
        plt.grid(False)  # Remove grid lines if they interfere with the appearance
        # Save the plot as an image file
        plt.savefig("spike_raster_plot.png")  # Save as a PNG file; you can specify other formats like .pdf, .jpg, etc.
        plt.show()