import json 
import os

def generate_json(save_path):
    parameter_dict = {"model_name":1024, 
                      "Scale":1024, 
                      "Softmax_axis":10000,
                        "Balance":None,
                 "Positional_encoding":None, 
                 "Positional_encoding_shape":1, 
                 "Positional_encoding_way":None,
                 "Dropout_on":None,
                 "Dropout_ratio":None,
                 "Classifier_on":None,
                 "CLS_on":None,
                 "CLS_mix":None,
                 "key_value_emb":None,
                 "Skip_connection":None,
                 "Layernorm":None,}
    
    with open(save_path,'w') as f:
        json.dump(parameter_dict,f,indent = 2)


if __name__ == "__main__":
    directory = 'Configs/model_configs/csta'
    if not os.path.exists(directory):
        os.makedirs(directory,exist_ok=True)
    save_name = 'csta_tvsum'
    save_path = os.path.join(directory,save_name)
    generate_json(save_path)