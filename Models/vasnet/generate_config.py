import json 
import os

def generate_json(save_path):
    parameter_dict = {"hidden_dim":2048}
    with open(save_path,'w') as f:
        json.dump(parameter_dict,f,indent = 2)


if __name__ == "__main__":
    directory = 'Configs/model_configs/vasnet'
    if not os.path.exists(directory):
        os.makedirs(directory,exist_ok=True)
    dataset = 'videoxum'
    save_name = f'vasnet_{dataset}.json'
    save_path = os.path.join(directory,save_name)
    generate_json(save_path)