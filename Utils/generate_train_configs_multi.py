
import json

#dataset_obj_dict = {'unimodal_loader':MultiH5Loader,'multimodal_loader':MultiModalh5Loader}
#collate_dict = {'visual_feat':batch_collate_fn,'visual_text_feat':batch_text_visual,'visual_feat_text_sum':batch_text_sum_visual,'multi_feat_text_sum':batch_text_sum_multi}
#model_dict = {'casum':CA_SUM,'ctv_raw':ctv_functional,'mars':MARs,'pgl_sum':PGL_SUM,'vasnet':VASNet,'sl_module':SL_module,'sumgda':SUM_GDA,'csta':set_model,'vtsum_blip_tt':VTSum_BLIP_TT,'vtsum_blip_tt_ca':VTSum_BLIP_TT_CA,'sd_vsum':SD_VSum}
def generate_json(config_path,model_name,dataset,split_type):
    data_configs= {
        'batch_size':15,
        'collate_type':'batch_text_sum_multi',
        'lr':1e-5,
        'epochs':30,
        'shuffle_params': {},
        'loader_config': {'name':'multimodal_loader',
                          'params':{'split_file':'Splits/script_videoxum_split_reduced_fixed.json','included_keys':['frame_features','text_summary'],
                                    'multi_obj':True,'file_name':'blip'}}
    }
    eval_configs  = {
    
        'eval_type':{dataset:'gtscore'},
        'post_process_dict':{'mrhisum':'none'},
        'eval_criterion': 'corr',
        'log_metric':'kendall',
        'eval_dataset': dataset,
        'split_type':split_type
        }
    model_configs = {
        'model_name':f'{model_name}',
        'model_parameter_path':f'Configs/model_configs/{model_name}/{model_name}_{dataset}.json',
        'loss_function':{'mse_multi':{"params":{}}},
        'leaderboard_id':f'{model_name}_orig'
    }
    meta_configs = {
        'save_top_k':0,
        'device':'cuda:0'
    }
    training_config = {'data_configs':data_configs,'eval_configs':eval_configs,'model_configs':model_configs,'meta_configs':meta_configs}
    with open(config_path,'w') as f:
        json.dump(training_config,f,indent=2)


if __name__ == "__main__":
    model_name = 'vtsum_blip_tt_ca'
    config_path = f'Configs/train_{model_name}_videoxum_can.json'
    generate_json(config_path,model_name)