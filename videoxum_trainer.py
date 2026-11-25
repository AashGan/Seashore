from Trainers.base_trainer_multi import BaseTrainerVideoXUM
from Models.model_links import model_dict
from Data.Datasets import MultiH5Loader,dataset_obj_dict
from Data.Datasets import ShuffleComposer
from torch.utils.data import DataLoader
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pathlib import Path
from Models.losses.loss_functions import loss_dict_stacker,WeightedLossStack
from Data.Loaders.collators import collate_dict
import csv
import json 
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
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
        return loss_dict_stacker[key](**params)
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

def parse_config_filepath(json_path):
    """
    Returns: Dataset Name, Split Type 
    """
    processed_path = json_path.split(".")[0].split('/')[-1] # This should return the file path
    processed_path = processed_path.split("_")
    return processed_path[0] ,processed_path[1]

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
        #TODO: Adjust the lightning module to have the learning rate 
        lightning_module = BaseTrainerVideoXUM(model,train_dataset.included_dataset,eval_type,post_process_dict,
                                               lr=learning_rate,criterion=criterion,eval_criterion=eval_criterion,
                                               loss_weights=loss_weights)
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
                val_dataloader   = DataLoader(val_dataset,shuffle=False,batch_size = 1)
                lightning_module = BaseTrainerVideoXUM(model,train_dataset.included_dataset,eval_type,post_process_dict,
                                               lr=learning_rate,criterion=criterion,eval_criterion=eval_criterion,
                                               loss_weights=loss_weights)
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