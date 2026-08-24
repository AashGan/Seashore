from Trainers.base_trainer import BaseTrainer
from Models.model_links import model_dict
import torch
import json
from Data.Datasets import MultiH5Loader
from torch.utils.data import DataLoader
import numpy as np
import os
import pytorch_lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from pathlib import Path
# Base Trainer for Video Summarization, assuming it is uni-modal
#TODO: Add my shufflers 
#TODO: Add 
import csv

def parse_and_return_json(json_path):
    with open(json_path,'r') as f:
        return json.load(json_path)

def parse_config_filepath(json_path):
    """
    Returns: Dataset Name, Split Type 
    """
    processed_path = json_path.split(".")[0].split('/')[-1] # This should return the file path
    processed_path = processed_path.split("_")
    return processed_path[0] ,processed_path[1]

def write_to_leaderboard(result,model_name,dataset_name,criterion,eval_type,post_process_type):
    """ Creating a leaderboard based on different models, datasets, criterion and eval types
    """
    path = Path(f'Leaderboards/model_ranking_{dataset_name}_{criterion}.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    with path.open("a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["model_name","eval_type","post_process_type", "result"])
        writer.writerow([model_name,eval_type, post_process_type,result])


def run_train_base(config):
    data_configs = config['data_configs'] # Contains the data-split, batch size
    model_configs = config['model_configs'] # Has the model names and the path to the model dicts
    eval_configs = config['eval_configs'] # Has the evaluation metrics, evaluation types and post process types
    meta_configs = config['meta_configs'] # Has the configurations for logging, checkpoint saving etc

    # Model Parameters, Data splits, batchsize
    model_parameters = parse_and_return_json(model_configs['model_parameter_path'])
    model_name = model_parameters['model_name']
    # Data configs
    data_splits = parse_and_return_json(data_configs['data_split_path'])
    dataset_name,split_type = parse_config_filepath(data_configs['data_split_path'])
    epochs = data_configs.get('epochs',15)
    learning_rate  = data_configs.get('lr',1e-5)

    collate_type = data_configs.get('collate_type',None) #TODO: add the collation function variants when you get them running
    batch_collate_fn = collate_type

    batch_size = data_configs['batch_size']
    # Evaluation configs
    eval_type = eval_configs['eval_type']
    eval_criterion = eval_configs['eval_criterion']
    post_process_dict = eval_configs['post_process']
    log_metric = eval_configs['log_metric']
    # Meta configs
    save_top_k = meta_configs.get('save_top_k',0)

    # declaring model from parameters, dataset objects 
    if len(data_splits)==1:
        print('Running Train Val Split')
        model = model_dict[model_name](**model_parameters)
        train_dataset = MultiH5Loader(data_splits,data_configs['train_split_name'],0,data_configs['feature_name'])
        val_dataset = MultiH5Loader(data_splits,data_configs['val_split_name'],0,data_configs['feature_name'])
        train_dataloader =  DataLoader(train_dataset,shuffle=True,collate_fn=batch_collate_fn,batch_size = batch_size)
        val_dataloader =  DataLoader(val_dataset,shuffle=False,batch_size = 1)
        #TODO: Adjust the lightning module to have the learning rate 
        lightning_module = BaseTrainer(model,train_dataset.included_datasets,eval_type,post_process_dict)
        checkpoint = ModelCheckpoint(
                                        monitor=eval_criterion,
                                        mode="max",
                                        save_top_k=save_top_k,
                                        )
        trainer = pl.Trainer(max_epochs = epochs,  callbacks=[
        checkpoint
    ]) 
        trainer.fit(lightning_module,train_dataloader,val_dataloader)
        best_score = checkpoint.best_model_score.item()
        print(f"Best validation {eval_criterion}: {best_score:.4f}")
        result = best_score
    else:
        split_scores = []
        for i in range(len(data_splits)):
            model = model_dict[model_name](**model_parameters)
            train_dataset = MultiH5Loader(data_splits,data_configs['train_split_name'],0,data_configs['feature_name'])
            val_dataset = MultiH5Loader(data_splits,data_configs['val_split_name'],0,data_configs['feature_name'])
            train_dataloader =  DataLoader(train_dataset,shuffle=True,collate_fn=batch_collate_fn,batch_size = batch_size)
            val_dataloader =  DataLoader(val_dataset,shuffle=False,batch_size = 1)
            lightning_module = BaseTrainer(model,train_dataset.included_datasets,eval_type,post_process_dict)
            checkpoint = ModelCheckpoint(
                                            monitor=log_metric,
                                            mode="max",
                                            save_top_k=save_top_k,
                                            )
            trainer = pl.Trainer(max_epochs = epochs,  callbacks=[
            checkpoint
        ]) 
            trainer.fit(lightning_module,train_dataloader,val_dataloader)
            best_score = checkpoint.best_model_score.item()
            print(f"Best validation for Split: {i+1} {eval_criterion}: {best_score:.4f}")
            split_scores.append(best_score)
        print(f'Average over all splits: {np.mean(split_scores)}')
        result = np.mean(split_scores)
    write_to_leaderboard(result,model_name,dataset_name,log_metric)
        





        