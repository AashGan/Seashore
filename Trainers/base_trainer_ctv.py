# A CTV specific lightning module
import pytorch_lightning as pl
import torch
import wandb
import h5py
from Utils import process_and_route_single
import torch.nn as nn
import torch.nn.functional as F
from .base_trainer import MetadataStore
from collections.abc import Callable
from scipy.ndimage import gaussian_filter1d
import numpy as np

class BaseTrainerCTV(pl.LightningModule):
    """
    A CTV Style Video Summarizer 
    """
    def __init__(self,model:nn.Module,datasets:list|dict, eval_type:dict[str],
                 post_process_dict:dict[str],pred_criterion, use_unif,use_unq = False,alpha = 0.5,ratio_s =0 ,ratio_k1 = 0.1 , lr:float=1e-5,criterion:Callable = F.mse_loss,
                 eval_criterion='corr'):
        super().__init__()
        self.model = model
        self.eval_type = eval_type
        self.metadata_stores = MetadataStore(datasets)
        self.metadata_stores.open()
        self.eval_criterion = eval_criterion
        self.post_process_dict = post_process_dict
        self.criterion = criterion
        self.pred_criterion = pred_criterion
        self.lr = lr
        self.ratio_s = ratio_s
        self.ratio_k1 = ratio_k1 
        self.use_unq = use_unq
        self.alpha = alpha
        self.use_unif = use_unif
    def training_step(self,batch,batch_idx):
        x, _ = batch['features'] ,  batch['gtscores']
        proj, y_preds = self.model(x)
        laln, lunif, lunif_fv, lunq = self.criterion(x,proj,y_preds,self.ratio_s,self.ratio_k1)

        if self.use_unq:
            return laln.mean() + self.alpha * lunif.mean()  + 0.1 * lunif_fv.mean() + 0.1 * lunq
        return laln.mean() +self.alpha*lunif.mean()

    def validation_step(self, batch, batch_idx):
        x, y, video_key = batch['features'] ,  batch['gtscores'],batch['data_point']
        proj, y_preds = self.model(x)
        y_output = self.pred_criterion(x,proj,y_preds,self.ratio_s,self.ratio_k1,self.use_unif,self.use_unq)
        dataset,video_index = video_key[0].split('/')
        y_output = BaseTrainerCTV.post_process_ctv(y_output,dataset)
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
    @staticmethod
    def post_process_ctv(scores:torch.tensor,dataset='tvsum'):
        scores = gaussian_filter1d(scores.numpy(), 1)
        if dataset =='tvsum':
            scores = np.exp(scores - 1) 
        elif dataset =='summe':
            scores = scores + 0.05
        return scores 
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer