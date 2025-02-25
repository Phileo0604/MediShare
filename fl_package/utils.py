import json
import torch

def get_parameters(model):
    return [param.detach().cpu().numpy() for param in model.parameters()]

def set_parameters(model, parameters):
    with torch.no_grad():
        for param, value in zip(model.parameters(), parameters):
            param.copy_(torch.tensor(value, dtype=param.dtype))

def export_model_parameters(model, file_path="model_parameters.json"):
    parameters = get_parameters(model)
    with open(file_path, "w") as f:
        json.dump([param.tolist() for param in parameters], f)
    print(f"Model parameters saved to {file_path}")

def import_model_parameters(model, file_path="model_parameters.json"):
    with open(file_path, "r") as f:
        parameters = json.load(f)
    set_parameters(model, [torch.tensor(param) for param in parameters])
    print(f"Model parameters loaded from {file_path}")
