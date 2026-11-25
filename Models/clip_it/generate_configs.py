import json
import os


def generate_json(save_path):
    parameter_dict = {
        'n_frame_features':768,
        'n_text_features':768,
        'n_head_language':4,
        'n_head_transformer':8,
        'n_transformer_layers':2,
        'transformer_droput':0.5
    }
    with open(save_path,'w') as f:
            json.dump(parameter_dict,f,indent = 2)

if __name__ == "__main__":
    directory = 'Configs/model_configs/clip_it'
    if not os.path.exists(directory):
        os.makedirs(directory,exist_ok=True)
    save_name = 'clip_it_tvsum.json'
    save_path = os.path.join(directory,save_name)
    generate_json(save_path)