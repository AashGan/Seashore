import pytorch_lightning as pl
import torch
import wandb
import h5py
from Utils import process_and_route_single
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Callable
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
        positions = self.get_key_and_return_array(group,"positions")
        n_frames = self.get_key_and_return_array(group,"n_frames")
        shot_bounds = self.get_key_and_return_array(group,"shot_bounds")
        metadata = {
            "positions": positions,
            "n_frames": n_frames,
            "shot_bounds": shot_bounds
        }


        return metadata
    @staticmethod
    def get_key_and_return_array(group,key):
        output = group.get(key,None)
        if output is not None:
            return output[...]
        return output
    def get_gt(self,dataset,video_key):
        f = self.files[dataset]
        group = f[video_key]
        if "user_score" in group.keys():
            user_score = group["user_score"][...]
        elif "gtscore_xum" in group.keys():
            user_score = group['gtscore_xum'][...]
        else:
            user_score = group["gtscore"][...]
        user_summary = group["user_summary"][...]
        if len(user_summary.shape)<2:
            user_summary = user_summary[None,:]
        metadata = {
            "user_score": user_score,
            "user_summary":user_summary
        }


        return metadata


    def close(self):
        for f in self.files.values():
            f.close()
        self.files.clear()

class BaseTrainer(pl.LightningModule):

    def __init__(self,model:nn.Module,datasets:list|dict,eval_type:dict[str],
                 post_process_dict:dict[str],lr:float=1e-5,criterion:Callable = F.mse_loss,include_features=False,
                 eval_criterion='corr'):
        super().__init__()
        self.model = model
        self.eval_type = eval_type # Stores how each dataset has to be evaluated 
        self.metadata_stores = MetadataStore(datasets)
        self.metadata_stores.open()
        self.eval_criterion = eval_criterion
        self.post_process_dict = post_process_dict
        #TODO, add validation keys to exclude for hyper-parameter logging
        self.criterion = criterion
        self.lr = lr
        self.include_feature = include_features

    def training_step(self, batch, batch_idx):
        x, y= batch['features'],batch['gtscore']
        mask = None
        if "mask" in batch.keys():
            mask = batch['mask']
        # Forward pass
        y_pred = self.model(x) # This output would be a dict
        
        if self.include_feature:
            y_pred['frame_features'] = x
        loss = self.criterion(**y_pred,keyframe_labels= y, keyframe_masks = mask) 
        
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
            result_dict = process_and_route_single(y_pred['model_predictions'],y,metadata,ground_truth_data,eval_type,post_process,self.eval_criterion)
            self.log('kendall',result_dict['kendall'],prog_bar = True,on_epoch=True)
            self.log('spearman',result_dict['spearman'],prog_bar = True,on_epoch=True)
        elif self.eval_criterion == 'f1':
            f1 = process_and_route_single(y_pred['model_predictions'],y,metadata,ground_truth_data,eval_type,post_process,self.eval_criterion)
            self.log('f1',f1)

    def on_train_end(self):
        pass
        # self.metadata_stores.close()

    def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            return optimizer