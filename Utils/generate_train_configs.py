
import json
#dataset_obj_dict = {'unimodal_loader':MultiH5Loader,'multimodal_loader':MultiModalh5Loader}
#collate_dict = {'visual_feat':batch_collate_fn,'visual_text_feat':batch_text_visual,'visual_feat_text_sum':batch_text_sum_visual,'multi_feat_text_sum':batch_text_sum_multi}
#model_dict = {'casum':CA_SUM,'ctv_raw':ctv_functional,'mars':MARs,'pgl_sum':PGL_SUM,'vasnet':VASNet,'sl_module':SL_module,'sumgda':SUM_GDA,'csta':set_model,'vtsum_blip_tt':VTSum_BLIP_TT,'vtsum_blip_tt_ca':VTSum_BLIP_TT_CA,'sd_vsum':SD_VSum}
#eval target: gt, user_score, user_summary
# post_process type: upsample, summary_gen, binarize_top_k,summary_gen_no_upsample
def generate_json(config_path,model_name,dataset,feature_name,split_type,split_file):
    
    data_configs= {
        'batch_size':1,
        'collate_type':"visual_feat",
        'lr':5e-5,
        'epochs':30,
        'shuffle_params': {},
        'loader_config': {'name': "unimodal_loader",
                          'params':{'split_file':split_file,
                                    'multi_obj':False,'feature_name':feature_name}}
    }
    eval_configs  = {

    'eval_type':{dataset:'max'},
    'post_process_dict':{dataset:'summary_gen'},
    'eval_criterion': 'f1',
    'log_metric':'kendall',
    'eval_dataset': dataset,
    'split_type':split_type
    }
    model_configs = {
        'model_name':f'{model_name}',
        'model_parameter_path':f'Configs/model_configs/{model_name}/{model_name}_{dataset}.json',
        'loss_function':{'mse':{"params":{}}},
        'leaderboard_id':f'{model_name}_average_gt'
    }
    meta_configs = {
        'save_top_k':0,
        'device':'cuda:0'
    }
    training_config = {'data_configs':data_configs,'eval_configs':eval_configs,'model_configs':model_configs,'meta_configs':meta_configs}
    with open(config_path,'w') as f:
        json.dump(training_config,f,indent=2)


if __name__ == "__main__":
    models_batched = ['pgl_sum','sl_module','vasnet']
    models_unbatched = ['casum','sumgda','csta','mars']
    datasets = {'tvsum':['can','Splits/tvsum_can_1.json','googlenet'],'summe':['can','Splits/summe_can_1.json','googlenet'],
                'mrhisum':['orig','Splits/sm_mrhisum_split.json','inception_3'],'videoxum':['orig','Splits/script_videoxum_split_reduced_fixed.json','resnet']}

    # Do batched models first
    for model_name in models_batched:
        for dataset,params_list in datasets.items():
            split_type = params_list[0]
            split_file = params_list[1]
            feature_name = params_list[2]
            config_path = f'Configs/user_summary_upsample_batch_configs/train_{model_name}_{dataset}_{split_type}.json'
            generate_json(config_path,model_name,dataset,feature_name,split_type,split_file)