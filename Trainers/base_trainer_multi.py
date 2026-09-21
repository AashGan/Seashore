import pytorch_lightning as pl
import torch
import wandb
import h5py
from Utils import process_and_route_single
import torch.nn as nn
import torch.nn.functional as F
from .base_trainer import MetadataStore
from collections.abc import Callable



class BaseTrainerVideoXUM(pl.LightningModule):
    """
    A videoxum style module.
    We assume a Huggingface style text loss, where the loss is computed within text object

    """

    def __init__(self, model:nn.Module, datasets:list|dict, eval_type:dict[str],
                 post_process_dict:dict[str], lr:float=1e-5,criterion:Callable = F.mse_loss,
                 eval_criterion='corr',loss_weights = [0.5,0.5]):

        super().__init__()
        self.model = model
        self.eval_type = eval_type # Stores how each dataset has to be evaluated 
        self.metadata_stores = MetadataStore(datasets)
        self.metadata_stores.open()
        self.eval_criterion = eval_criterion
        self.post_process_dict = post_process_dict
        self.criterion = criterion
        self.lr = lr
        self.loss_weights = loss_weights 

    def training_step(self, batch, batch_idx):
        x_vision, y_text, y_scores = batch['visual_features'], batch['text_labels'], batch['gtscores']
        mask = None
        if "mask" in batch.keys():
            mask = batch['mask']
        y_output, y_text_output = self.model(x_vision,y_text,mask)
        text_loss = y_text_output.loss
        vision_loss = self.criterion(y_output,y_scores)
        total_loss = self.loss_weights[0]*vision_loss + self.loss_weights[1]*text_loss
        self.log('train_vision_loss',vision_loss)
        self.log('train_text_loss',text_loss)
        self.log('train_total_loss',total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x_vision, y_text, y,video_key = batch['visual_features'], batch['text_labels'], batch['gtscores'],batch['data_point'] 
        y_outputs, _ = self.model(x_vision,y_text)
        y = y.to('cpu')
        y_output = y_outputs['saliency_scores']
        dataset,video_index = video_key[0].split('/')
        metadata = self.metadata_stores.get(dataset,video_index)
        ground_truth_data = self.metadata_stores.get_gt(dataset,video_index)
        eval_type = self.eval_type[dataset]
        post_process = self.post_process_dict[dataset]
        if self.eval_criterion =='corr':
            result_dict = process_and_route_single(y_output,y,metadata,ground_truth_data,eval_type,post_process,self.eval_criterion)
            self.log('kendall',result_dict['kendall'],prog_bar = True,on_epoch=True)
            self.log('spearman',result_dict['spearman'],prog_bar = True,on_epoch=True)
        elif self.eval_criterion == 'f1':
            f1 = process_and_route_single(y_output,y,metadata,ground_truth_data,eval_type,post_process,self.eval_criterion)
            self.log('f1',f1)


    def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            return optimizer

        

class BaseTrainerMulti(pl.LightningModule):
    """ Trainer for CLIP-IT and SD-VSUM model strategies,
    Assumes that models receive visual and textual features as a batch
    """
    def __init__(self, model:nn.Module, datasets:list|dict, eval_type:dict[str],
                      post_process_dict:dict[str],lr:float=1e-5, criterion:Callable = F.mse_loss,
                 eval_criterion='corr'):
        super().__init__()
        self.model = model
        self.eval_type = eval_type # Stores how each dataset has to be evaluated 
        self.metadata_stores = MetadataStore(datasets)
        self.metadata_stores.open()
        self.eval_criterion = eval_criterion
        self.post_process_dict = post_process_dict
        self.criterion = criterion
        self.lr = lr  

    def training_step(self, batch, batch_idx):
        x_vision, x_text, y_scores = batch['visual_features'], batch['text_features'], batch['gtscores']
        visual_mask = None
        text_mask = None 
        if "frame_mask" in batch.keys():
            visual_mask = batch['visual_mask']
            text_mask = batch['text_mask']

        y_pred = self.model(x_vision,x_text,visual_mask,text_mask) # This might return embeddings, we need to account for this
        loss = self.criterion(y_pred, y_scores, visual_mask)
        self.log('train_loss', loss)
        return loss 
    def validation_step(self,batch,batch_idx):
        x_vision, x_text, y_scores, video_key = batch['visual_features'], batch['text_features'], batch['gtscores'], batch['data_point'] 
        y_pred = self.model(x_vision,x_text)
        y = y_scores[''].to('cpu')
        dataset,video_index = video_key[0].split('/')
        y_output = y_pred['saliency_scores']
        metadata = self.metadata_stores.get(dataset,video_index)
        ground_truth_data = self.metadata_stores.get_gt(dataset,video_index)
        eval_type = self.eval_type[dataset]
        post_process = self.post_process_dict[dataset]
        if self.eval_criterion =='corr':
            result_dict = process_and_route_single(y_output,y,metadata,ground_truth_data,eval_type,post_process,self.eval_criterion)
            self.log('kendall',result_dict['kendall'],prog_bar = True,on_epoch=True)
            self.log('spearman',result_dict['spearman'],prog_bar = True,on_epoch=True)
        elif self.eval_criterion == 'f1':
            f1 = process_and_route_single(y_output,y,metadata,ground_truth_data,eval_type,post_process,self.eval_criterion)
            self.log('f1',f1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer