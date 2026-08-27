import json 
import os

def generate_json(save_path):
    parameter_dict = {"d_model":1024, 
                      "nhead":1024, 
                      "ff_expansion_factor":10000}
    with open(save_path,'w') as f:
        json.dump(parameter_dict,f,indent = 2)


if __name__ == "__main__":
    directory = 'Configs/model_configs/mars'
    if not os.path.exists(directory):
        os.makedirs(directory,exist_ok=True)
    save_name = 'mars_tvsum'
    save_path = os.path.join(directory,save_name)
    generate_json(save_path)