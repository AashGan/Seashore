import json


def generate_json(config_path):
    data_configs= {
        'data_split_path':'Splits/tvsum_can_1.json',
        'batch_size':1,
        'feature_name': 'googlenet',
        'train_split_name':'train',
        'val_split_name':'val'
    }
    eval_configs  = {

    'eval_type':{'tvsum':'user_score'},
    'post_process_dict':{'tvsum':'upsample'},
    'eval_criterion': 'corr',
    'log_metric':'kendall'
    }
    model_configs = {
        'model_name':'pgl_sum',
        'model_parameter_path':'Configs/model_configs/pgl_sum/pgl_sum_tvsum.json',
        'loss_function':'mse'
    }
    meta_configs = {
        'save_top_k':0,
        'device':'cuda:0'
    }
    training_config = {'data_configs':data_configs,'eval_configs':eval_configs,'model_configs':model_configs,'meta_configs':meta_configs}
    with open(config_path,'w') as f:
        json.dump(training_config,f,indent=2)


if __name__ == "__main__":
    config_path = 'Configs/train_pgl_sum_tvsum_can.json'
    generate_json(config_path)