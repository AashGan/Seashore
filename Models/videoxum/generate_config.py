import json 
import os

def generate_json(save_path):
    parameter_dict = {"dim":1024,'tt_depth':1,
                      'file_path':'ModelUtils/video_xum/model_base_capfilt_large.pth',
                      'med_config':'ModelUtils/video_xum/med_config.json'}
    with open(save_path,'w') as f:
        json.dump(parameter_dict,f,indent = 2)


if __name__ == "__main__":
    directory = 'Configs/model_configs/vasnet'
    if not os.path.exists(directory):
        os.makedirs(directory,exist_ok=True)
    save_name = 'vasnet_tvsum'
    save_path = os.path.join(directory,save_name)
    generate_json(save_path)