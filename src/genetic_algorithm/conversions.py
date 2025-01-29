import torch

def vector_to_parameterization(vector, parameter_name_dict):
    starting_index = 0
    parameterization_dict = {}
    for parameter_name, num_values in parameter_name_dict.items():
        vals = vector[starting_index:starting_index + num_values]
        parameterization_dict[parameter_name] = vals
        starting_index += num_values
    return parameterization_dict

def parameterization_to_vector(parameterization_dict, device):
    vector = torch.tensor([], device=device)
    for parameter_name, vals in parameterization_dict.items():
        vals = torch.tensor(vals, device=device) if isinstance(vals, list) else vals
        if len(vals.shape) > 1:
            vals = vals.flatten()
        vals = vals.to(device)
        if len(vector) == 0:
            vector = vals
        else:   
            vector = torch.cat((vector, vals))
    return vector

