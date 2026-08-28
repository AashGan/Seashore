import json 
import os

def generate_json(save_path):
    parameter_dict = {"input_size":1024, 
                      "output_size":1024, 
                      "freq":10000,
                        "pos_enc":"absolute",
                 "num_segments":4, 
                 "heads":8, 
                 "fusion":"add"}
    with open(save_path,'w') as f:
        json.dump(parameter_dict,f,indent = 2)


if __name__ == "__main__":
    directory = 'Configs/model_configs/pgl_sum'
    if not os.path.exists(directory):
        os.makedirs(directory,exist_ok=True)
    save_name = 'pgl_sum_tvsum.json'
    save_path = os.path.join(directory,save_name)
    print(save_path)
    generate_json(save_path)