from Data.Datasets import dataset_obj_dict
from Data.Datasets import ShuffleComposer
from torch.utils.data import DataLoader
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
from Models.losses.loss_functions import loss_dict_stacker,WeightedLossStack
from Data.Loaders.collators import collate_dict
import csv
from Trainers.base_trainer_multi import BaseTrainerMulti
from videoxum_trainer import parse_and_return_json,write_to_leaderboard,construct_loss_function
from Models.model_links import model_dict
import argparse



def run_train(config):
    config = parse_and_return_json(config)
    data_configs = config['data_configs'] # Contains the data-split, batch size
    model_configs = config['model_configs'] # Has the model names and the path to the model dicts
    eval_configs = config['eval_configs'] # Has the evaluation metrics, evaluation types and post process types
    meta_configs = config['meta_configs']
    # Model parameters
    model_parameters = parse_and_return_json(model_configs['model_parameter_path'])
    model_name = model_configs['model_name']
    loss_weights = model_configs.get('loss_weights',[17,1])
    leaderboard_id = model_configs.get('leaderboard_id',model_configs['model_name'])

    # Dataset stuff 
    dataset_obj = dataset_obj_dict[data_configs['loader_config']['name']]
    loader_params  = data_configs['loader_config']['params']
    data_splits = parse_and_return_json(loader_params['split_file']) # This checks if we need to run val or cross val
    dataset_name,split_type = eval_configs['eval_dataset'],eval_configs['split_type']

    # Learning stuff
    epochs = data_configs.get('epochs',15)
    learning_rate  = data_configs.get('lr',1e-5)
    shuffler_params = data_configs.get('shuffler_params',None)
    if shuffler_params is not None:
        shuffler_params = ShuffleComposer(shuffler_params)
    sampled_frames = data_configs.get('sampled_frames',None)
    collate_type = data_configs.get('collate_type',None) #TODO: add the collation function variants when you get them running 
    val_collate_type = data_configs.get('val_collate_type',None)
    batch_collate_fn = None
    val_batch_collate_fn= None
    if collate_type is not None:
        batch_collate_fn = collate_dict[collate_type]
    if val_collate_type is not None:
            val_batch_collate_fn = collate_dict[val_collate_type]
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
    device = meta_configs.get('device',[1])
    if len(data_splits)==1:
        print('Running Train Val Split')
        model = model_dict[model_name](**model_parameters)
        loader_params['cross_val_idx'] = 0
        loader_params['split_type'] = 'train'
        train_dataset = dataset_obj(**loader_params,shuffler_obj=shuffler_params)
        loader_params['split_type'] = 'val'
        loader_params['multi_obj'] = False
        val_dataset = dataset_obj(**loader_params)
        train_dataloader =  DataLoader(train_dataset,shuffle=True,collate_fn=batch_collate_fn,batch_size = batch_size)
        val_dataloader =  DataLoader(val_dataset,shuffle=False,collate_fn=val_batch_collate_fn,batch_size = 1)
        lightning_module = BaseTrainerMulti(model,train_dataset.included_dataset,eval_type,post_process_dict,
                                               lr=learning_rate,criterion=criterion,eval_criterion=eval_criterion)
        checkpoint = ModelCheckpoint(
                                                monitor=log_metric,
                                                mode="max",
                                                save_top_k=save_top_k,
                                                )
        early_stop = EarlyStopping(monitor=log_metric,patience=7,mode='max',strict = True,verbose=True)
        trainer = pl.Trainer(max_epochs = epochs,  callbacks=[early_stop,
        checkpoint
    ],devices=device) 
        trainer.fit(lightning_module,train_dataloader,val_dataloader)
        best_score =  early_stop.best_score.detach().to('cpu').item()
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
            val_dataloader   = DataLoader(val_dataset,shuffle=False,collate_fn=val_batch_collate_fn,batch_size = 1)
            lightning_module = BaseTrainerMulti(model,train_dataset.included_dataset,eval_type,post_process_dict,
                                               lr=learning_rate,criterion=criterion,eval_criterion=eval_criterion)
            checkpoint = ModelCheckpoint(
                                                    monitor=log_metric,
                                                    mode="max",
                                                    save_top_k=save_top_k,
                                                    )
            early_stop = EarlyStopping(monitor=log_metric,patience=7,mode='max',strict = True,verbose=True)
            trainer = pl.Trainer(max_epochs = epochs,  callbacks=[early_stop,
                    checkpoint
                ],devices=[1]) 
            
            trainer.fit(lightning_module,train_dataloader,val_dataloader)
            
            best_score =  early_stop.best_score.detach().to('cpu').item()
            print(f"Best validation for Split: {i+1} {eval_criterion}: {best_score:.4f}")
            split_scores.append(best_score)
            lightning_module.metadata_stores.close()
        print(f'Average over all splits: {np.mean(split_scores)}')
        result = np.mean(split_scores)
        eval_type_dataset = eval_type[dataset_name]
        post_process_type_dataset = post_process_dict[dataset_name]
    
        write_to_leaderboard(result,leaderboard_id,dataset_name,log_metric,eval_type_dataset,post_process_type_dataset,split_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="Provide the path to the configuration here")
    args = parser.parse_args()
    run_train(args.config_path)