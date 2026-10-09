import json 
import os

def generate_json(save_path):
    parameter_dict = {"input_dim":1024, "model_dim":256, "nhead":4, "num_blocks":4,
                 "conv_kernel_sizes":(31,), "dropout": 0.1, "order":"att_first"}
    with open(save_path,'w') as f:
        json.dump(parameter_dict,f,indent = 2)


if __name__ == "__main__":
    directory = 'Configs/model_configs/mars'
    if not os.path.exists(directory):
        os.makedirs(directory,exist_ok=True)
    save_name = 'mars_tvsum'
    save_path = os.path.join(directory,save_name)
    generate_json(save_path)