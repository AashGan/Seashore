from Trainers.base_trainer import BaseTrainer
from Models.model_links import model_dict
import json
from Data.Datasets import dataset_obj_dict
from Data.Datasets import ShuffleComposer
from torch.utils.data import DataLoader
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pathlib import Path
from Models.losses.loss_functions import loss_dict_stacker,WeightedLossStack
from Data.Loaders.collators import collate_dict
import csv
import argparse
def parse_and_return_json(json_path):
    with open(json_path,'r') as f:
        return json.load(f)


def write_to_leaderboard(result,model_name,dataset_name,criterion,eval_type,post_process_type,split_type):
    """ Creating a leaderboard based on different models, datasets, criterion and eval types
    """
    path = Path(f'Leaderboards/model_ranking_{dataset_name}_{criterion}.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    with path.open("a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["model_name","eval_type","post_process_type",split_type, "result"])
        writer.writerow([model_name,eval_type, post_process_type,split_type,result])

def construct_loss_function(loss_function_dict):
    if len(loss_function_dict)<2:
        key = list(loss_function_dict.keys())[0]
        params = loss_function_dict[key]['params']
        #return loss_dict_stacker[key](**params)
        return WeightedLossStack([loss_dict_stacker[key](**params)],[1])
    else:
        loss_list = []
        weight_list = []
        for key in list(loss_function_dict.keys()):
            params = loss_function_dict[key]['params']
            weight = loss_function_dict[key]['weight']
            loss_list.append(loss_dict_stacker[key](**params))
            weight_list.append(weight)
        criterion = WeightedLossStack(loss_list,weight_list)
        return criterion

    
def run_train_base(config):
    config = parse_and_return_json(config)
    data_configs = config['data_configs'] # Contains the data-split, batch size
    model_configs = config['model_configs'] # Has the model names and the path to the model dicts
    eval_configs = config['eval_configs'] # Has the evaluation metrics, evaluation types and post process types
    meta_configs = config['meta_configs'] # Has the configurations for logging, checkpoint saving etc

    # Model Parameters, Data splits, batchsize
    model_parameters = parse_and_return_json(model_configs['model_parameter_path'])
    model_name = model_configs['model_name']
    leaderboard_id = model_configs.get('leaderboard_id',model_configs['model_name'])
    include_features = model_configs.get('include_features',False)
    # Data configs
    dataset_obj = dataset_obj_dict[data_configs['loader_config']['name']]
    loader_params  = data_configs['loader_config']['params']

    data_splits = parse_and_return_json(loader_params['split_file']) # This checks if we need to run val or cross val
    dataset_name,split_type = eval_configs['eval_dataset'],eval_configs['split_type']
    epochs = data_configs.get('epochs',15)
    learning_rate  = data_configs.get('lr',5e-5)
    shuffler_params = data_configs.get('shuffler_params',None)
    if shuffler_params is not None:
        shuffler_params = ShuffleComposer(shuffler_params)
    sampled_frames = loader_params.get('sampled_frames',None)
    collate_type = data_configs.get('collate_type',None) #TODO: add the collation function variants when you get them running 
    batch_collate_fn = None
    if collate_type is not None:
        batch_collate_fn = collate_dict[collate_type]

    batch_size = data_configs['batch_size']
    if collate_type is None and batch_size>1:
        if sampled_frames is not None:
            pass
        else:
            print('Cannot run batch size>1 without a collator function setting batch size to 1')
            batch_size = 1
    # Evaluation configs
    eval_type = eval_configs['eval_type']
    eval_criterion = eval_configs['eval_criterion']
    criterion = construct_loss_function(model_configs['loss_function'])
    post_process_dict = eval_configs['post_process_dict']
    log_metric = eval_configs['log_metric']
    # Meta configs
    save_top_k = meta_configs.get('save_top_k',0)
    device = meta_configs.get('device','cuda:0')
    print('')
    # declaring model from parameters, dataset objects 
    if len(data_splits)==1:
        print('Running Train Val Split')
        model = model_dict[model_name](**model_parameters)
        loader_params['cross_val_idx'] = 0
        loader_params['split_type'] = 'train'
        train_dataset = dataset_obj(**loader_params,shuffler_obj=shuffler_params)
        loader_params['split_type'] = 'val'
        val_dataset = dataset_obj(**loader_params)
        train_dataloader =  DataLoader(train_dataset,shuffle=True,collate_fn=batch_collate_fn,batch_size = batch_size)
        val_dataloader =  DataLoader(val_dataset,shuffle=False,batch_size = 1)
        #TODO: Adjust the lightning module to have the learning rate 
        lightning_module = BaseTrainer(model,train_dataset.included_dataset,eval_type,post_process_dict,learning_rate,criterion,include_features,eval_criterion)
        checkpoint = ModelCheckpoint(
                                        monitor=eval_criterion,
                                        mode="max",
                                        save_top_k=save_top_k,
                                        )
        trainer = pl.Trainer(max_epochs = epochs,  callbacks=[
        checkpoint
    ],devices=1) 
        trainer.fit(lightning_module,train_dataloader,val_dataloader)
        best_score = checkpoint.best_model_score.item()
        print(f"Best validation {eval_criterion}: {best_score:.4f}")
        result = best_score
    else:
        split_scores = []
        print('Running Cross Val')
        for i in range(len(data_splits)):
#TODO add a dict to make the multimodal dataset object work here  function here as well
            model = model_dict[model_name](**model_parameters)
            loader_params['cross_val_idx'] = i
            loader_params['split_type'] = 'train'
            train_dataset = dataset_obj(**loader_params)
            loader_params['split_type'] = 'val'
            val_dataset = dataset_obj(**loader_params)
            train_dataloader = DataLoader(train_dataset,shuffle=True,collate_fn=batch_collate_fn,batch_size = batch_size)
            val_dataloader   = DataLoader(val_dataset,shuffle=False,batch_size = 1)
            lightning_module = BaseTrainer(model,train_dataset.included_dataset,eval_type,post_process_dict,learning_rate,criterion,False,eval_criterion)
            checkpoint = ModelCheckpoint(
                                            monitor=log_metric,
                                            mode="max",
                                            save_top_k=save_top_k,
                                            )
            trainer = pl.Trainer(max_epochs = epochs,  callbacks=[
            checkpoint
        ],accelerator="gpu",
    devices=1) 
            trainer.fit(lightning_module,train_dataloader,val_dataloader)
            best_score = trainer.validate(lightning_module,val_dataloader)[0][log_metric]
            print(f"Best validation for Split: {i+1} {eval_criterion}: {best_score:.4f}")
            split_scores.append(best_score)
            lightning_module.metadata_stores.close()
        print(f'Average over all splits: {np.mean(split_scores)}')
        result = np.mean(split_scores)
    # This does single dataset leaderboard logging
    eval_type_dataset = eval_type[dataset_name]
    post_process_type_dataset = post_process_dict[dataset_name]

    write_to_leaderboard(result,leaderboard_id,dataset_name,log_metric,eval_type_dataset,post_process_type_dataset,split_type)
        




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="Provide the path to the configuration here")
    args = parser.parse_args()
    run_train_base(args.config_path)
        