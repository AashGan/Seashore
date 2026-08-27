import json 
import os

def generate_json(save_path):
    parameter_dict = {"input_dim":1024, 
                      "depth":1024, 
                      "heads":10000,
                        "mlp_dim":None,
                 "dropout_ratio":None}
    with open(save_path,'w') as f:
        json.dump(parameter_dict,f,indent = 2)


if __name__ == "__main__":
    directory = 'Configs/model_configs/sl_module'
    if not os.path.exists(directory):
        os.makedirs(directory,exist_ok=True)
    save_name = 'sl_module_tvsum'
    save_path = os.path.join(directory,save_name)
    generate_json(save_path)