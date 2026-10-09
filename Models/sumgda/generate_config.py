import json 
import os

def generate_json(save_path):
    parameter_dict = {"input_size":1024, 
                      "mode":'Unsupervised', 
                      "positional_encoding":True}
    with open(save_path,'w') as f:
        json.dump(parameter_dict,f,indent = 2)


if __name__ == "__main__":
    directory = 'Configs/model_configs/sumgda'
    if not os.path.exists(directory):
        os.makedirs(directory,exist_ok=True)
    save_name = 'sumgda_tvsum,.json'
    save_path = os.path.join(directory,save_name)
    generate_json(save_path)