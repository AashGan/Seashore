import pytorch_lightning as pl
import torch
import wandb
import h5py
from Utils import process_and_route_single
import torch.nn as nn
import torch.nn.Functional as F


class MetadataStore:
    def __init__(self, datasets:list|dict):
        # The dataset dict paths should also allow you to override and add custom h5's incase the h5's deviate (different shot boundaries,fps sampling etc)

        self.datasets = datasets
        self.files = {}

    def open(self):
        if isinstance(self.datasets, dict):
            self.files = {dataset:h5py.File(paths) for dataset,paths in self.datasets.items()}
        elif isinstance(self.datasets, list):
            self.files = {
                dataset: h5py.File(
                    f"Data/Metadata/{dataset}_metadata.h5", "r"
                )
                for dataset in self.datasets
            }
        else:
            raise TypeError("provide a list of included datasets, or a set of file paths")

    def get(self, dataset, video_key):
        f = self.files[dataset]
        group = f[video_key]

        metadata = {
            "positions": group["positions"][...],
            "n_frames": int(group["n_frames"][...]),
            "shot_bounds": group["shot_bounds"][...]
        }


        return metadata

    def get_gt(self,dataset,video_key):
        f = self.files[dataset]
        group = f[video_key]

        metadata = {
            "user_score": group["user_score"][...],
            "user_summary": group["user_summary"][...]
        }


        return metadata


    def close(self):
        for f in self.files.values():
            f.close()
        self.files.clear()

class BaseTrainer(pl.LightningModule):

    def __init__(self,model:nn.Module,datasets:list|dict,eval_type:dict[str],post_process_dict:dict[str],criterion:Callable = F.mse_loss,eval_criterion='corr'):
        super().__init__()
        self.model = model
        self.eval_type = eval_type # Stores how each dataset has to be evaluated 
        self.metadata_stores = MetadataStore(datasets)
        self.metadata_stores.open()
        self.eval_criterion = eval_criterion
        self.post_process_dict = post_process_dict
        #TODO, add validation keys to exclude for hyper-parameter logging
        self.criterion = criterion

    def training_step(self, batch, batch_idx):
        x, y= batch['features'],batch['gtscore']
        mask = None
        if "mask" in batch.keys():
            mask = batch['mask']
        # Forward pass
        y_pred = self.model(x)
        
        # Calculate loss (MSE)
        loss = self.criterion(y_pred, y, mask) if mask is not None else self.criterion(y_pred, y)
        
        # Log the loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self,batch,batch_idx):
        x,y,video_key = batch['features'],batch['gtscore'],batch['data_point'] # This might need to be changed to a dict, check with collate function
        y_pred = self.model(x) #TODO: maybe change this to a dictionary output.
        y = y.to('cpu')
        dataset,video_index = video_key[0].split('/')
        metadata = self.metadata_stores.get(dataset,video_index)
        ground_truth_data = self.metadata_stores.get_gt(dataset,video_index)
        eval_type = self.eval_type[dataset]
        post_process = self.post_process_dict[dataset]
        if self.eval_criterion =='corr':
            result_dict = process_and_route_single(y_pred,y,metadata,ground_truth_data,eval_type,post_process,self.eval_criterion)
            self.log('kendall',result_dict['kendall'],prog_bar = True,on_epoch=True)
            self.log('spearman',result_dict['spearman'],prog_bar = True,on_epoch=True)
        elif self.eval_criterion == 'f1':
            f1 = process_and_route_single(y_pred,y,metadata,ground_truth_data,eval_type,post_process,self.eval_criterion)
            self.log('f1',f1)

    def on_train_end(self):
        self.metadata_stores.close()