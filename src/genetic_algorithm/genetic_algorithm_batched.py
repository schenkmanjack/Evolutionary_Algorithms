import torch
import math
from .genetic_algorithm_base import GeneticAlgorithmBase

class GeneticAlgorithmBatched(GeneticAlgorithmBase):
    def __init__(self, config, parameterization_dict, parameterization_std_devs_dict, parameter_name_dict, network_class, network_args, device="cpu", batch_config=dict()):
        super().__init__(config=config, parameterization_dict=parameterization_dict, 
        parameterization_std_devs_dict=parameterization_std_devs_dict, 
        parameter_name_dict=parameter_name_dict, 
        network_class=network_class, 
        network_args=network_args, 
        device=device)
        self.batch_config = batch_config

    def batch_networks(self, networks_array):
        # batch the networks
        networks_per_batch_sqrt = self.batch_config.get("networks_per_batch_sqrt", 10)
        num_batches = math.ceil(len(networks_array) / networks_per_batch_sqrt ** 2)
        # establish batched variables
        properties = networks_array[0].get_properties()
        batched_properties = dict()
        for key, value in properties.items():
            dims = value.shape
            if len(dims) == 1:
                property = torch.zeros((networks_per_batch_sqrt * networks_per_batch_sqrt * value.shape[0]), device=self.device)
            elif len(dims) == 2:
                property = torch.zeros((networks_per_batch_sqrt * networks_per_batch_sqrt * value.shape[0], networks_per_batch_sqrt * value.shape[1]), device=self.device)
            batched_properties[key] = property
        # create new network combining networks_array
        batch_id = 0
        networks_per_batch = networks_per_batch_sqrt ** 2
        for batch_id in range(num_batches):
            batch_networks = networks_array[batch_id*networks_per_batch:(batch_id+1)*networks_per_batch]
            for i, network in enumerate(batch_networks):
                properties = network.get_properties()
                for key, value in properties.items():
                    dims = value.shape
                    x = i % networks_per_batch_sqrt
                    y = i // networks_per_batch_sqrt
                    if len(dims) == 1:
                        batched_properties[key][i * networks_per_batch * value.shape[0]:(i+1)* networks_per_batch * value.shape[0]] = value
                    elif len(dims) == 2:
                        x_lower_bound = x * networks_per_batch_sqrt * value.shape[0]
                        x_upper_bound = (x+1) * networks_per_batch_sqrt * value.shape[0]
                        y_lower_bound = y * networks_per_batch_sqrt * value.shape[1]
                        y_upper_bound = (y+1) * networks_per_batch_sqrt * value.shape[1]    
                        batched_properties[key][x_lower_bound:x_upper_bound, y_lower_bound: y_upper_bound] = value
            # create new network
            batched_network = self.network_class(**self.network_args)
            batched_network.set_properties(properties=batched_properties)



    

        fitness_scores = torch.tensor([network.evaluation(evaluation_parameters=evaluation_parameters) for network in networks_array]).to(self.device)
        return fitness_scores