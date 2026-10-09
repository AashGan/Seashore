import json 
import os

def generate_json(save_path):
    parameter_dict = {"model_name":"GoogleNet_Attention", 
                      "Scale":1024, 
                      "Softmax_axis":"TD",
                        "Balance":None,
                 "Positional_encoding":"FPE", 
                 "Positional_encoding_shape":"TD", 
                 "Positional_encoding_way":'PGL_SUM',
                 "Dropout_on":True,
                 "Dropout_ratio":0.6,
                 "Classifier_on":True ,
                 "CLS_on":True,
                 "CLS_mix":'Final',
                 "key_value_emb":"kv",
                 "Skip_connection":"KC",
                 "Layernorm":True,}
    
    with open(save_path,'w') as f:
        json.dump(parameter_dict,f,indent = 2)


if __name__ == "__main__":
    directory = 'Configs/model_configs/csta'
    if not os.path.exists(directory):
        os.makedirs(directory,exist_ok=True)
    save_name = 'csta_tvsum'
    save_path = os.path.join(directory,save_name)
    generate_json(save_path)