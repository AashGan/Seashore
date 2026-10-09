import json 
import os

def generate_json(save_path):
    parameter_dict = {"input_dim":1024, 
                      "depth":3, 
                      "heads":8,
                        "mlp_dim":2048,
                 "dropout_ratio":0.5}
    with open(save_path,'w') as f:
        json.dump(parameter_dict,f,indent = 2)


if __name__ == "__main__":
    directory = 'Configs/model_configs/sl_module'
    if not os.path.exists(directory):
        os.makedirs(directory,exist_ok=True)
    save_name = 'sl_module_tvsum.json'
    save_path = os.path.join(directory,save_name)
    generate_json(save_path)