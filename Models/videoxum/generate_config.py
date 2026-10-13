import json 
import os

def generate_json(save_path):
    parameter_dict = {"dim":768,'tt_depth':1,"kernel_size":5,
                      'file_path':'Supplemental/video_xum_supp/model_base_capfilt_large.pth',
                      'med_config':'Supplemental/video_xum_supp/med_config.json'}
    with open(save_path,'w') as f:
        json.dump(parameter_dict,f,indent = 2)


if __name__ == "__main__":
    directory = 'Configs/model_configs/vtsum_blip_tt_ca'
    if not os.path.exists(directory):
        os.makedirs(directory,exist_ok=True)
    save_name = 'vtsum_blip_tt_ca'
    save_path = os.path.join(directory,save_name)
    generate_json(save_path)