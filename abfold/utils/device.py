import torch


def tensor_dict_to_device(tensor_dict: dict, device):
    new_dict = {}
    for key, val in tensor_dict.items():
        if isinstance(val, torch.Tensor):
            new_dict[key] = val.to(device, non_blocking=True)
        elif isinstance(val, dict):
            new_dict[key] = tensor_dict_to_device(val, device)
    return new_dict
