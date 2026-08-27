import json 
import os

def generate_json(save_path):
    parameter_dict = {"input_size":1024, 
                      "output_size":1024, 
                      "block_size":10000,
                        "positional_encoding":None,
                 "num_segments":None, 
                 "heads":1, 
                 "fusion":None}
    with open(save_path,'w') as f:
        json.dump(parameter_dict,f,indent = 2)


if __name__ == "__main__":
    directory = 'Configs/model_configs/casum'
    if not os.path.exists(directory):
        os.makedirs(directory,exist_ok=True)
    save_name = 'casum_tvsum'
    save_path = os.path.join(directory,save_name)
    generate_json(save_path)