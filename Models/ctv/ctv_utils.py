
import torch 
import torch.nn.functional as F
from einops import rearrange
from scipy.ndimage import gaussian_filter1d
import numpy as np
ratio_s = 0
ratio_k1 = 0.1 
def ctv_functional(frame_features,use_unif=True):
  norm_raw = F.normalize(frame_features,p=2,dim=-1)
  xy_raw = torch.einsum('bmc, bnc -> bmn', norm_raw, norm_raw)
  norm_proj = F.normalize(frame_features,p=2,dim=-1)
  xy = torch.einsum('bmc, bnc -> bmn', norm_proj, norm_proj)
  sort_ids = torch.argsort(xy_raw, -1, descending=True)
  diff_mat = 2 - 2 * xy
  L = frame_features.shape[1]
  S = int(L * ratio_s) 
  K1 = int(L * ratio_k1)
  pos = torch.gather(diff_mat, -1, sort_ids[:,:,S:S+K1])
  laln = pos.mean(dim=-1)
  lunif = diff_mat.mul(-2).exp().mean(dim=-1).log()
  laln = (laln - laln.min()) / (laln.max() - laln.min())
  lunif = (lunif - lunif.min()) / (lunif.max() - lunif.min())
  scores = laln
  if use_unif:
    return scores*lunif
  return scores

def ctv_loss_model_train(frame_features,proj,scores):
  norm_raw = F.normalize(frame_features,p=2,dim=-1)
  xy_raw = torch.einsum('bmc, bnc -> bmn', norm_raw, norm_raw)
  norm_proj = F.normalize(proj,p=2,dim=-1)
  xy = torch.einsum('bmc, bnc -> bmn', norm_proj, norm_proj)
  sort_ids = torch.argsort(xy_raw, -1, descending=True)
  diff_mat = 2 - 2 * xy
  L = frame_features.shape[1]
  S = int(L * ratio_s) 
  K1 = int(L * ratio_k1)
  pos = torch.gather(diff_mat, -1, sort_ids[:,:,S:S+K1])
  laln = pos.mean(dim=-1)
  lunif = diff_mat.mul(-2).exp().mean(dim=-1).log()
  s = 20
  seg = rearrange(proj, 'b (s l) c -> (b s) l c', s=s)
  seg_feats = F.normalize(seg.mean(dim=1), dim=1) # (b s) c
  fv_xy = torch.einsum("bmc, nc -> bmn", norm_proj, seg_feats) # b m (b s)
  
  mask = torch.ones_like(fv_xy)
  # for i in range(len(mask)):
  #     mask[i,:,i * s : (i+1) * s] = 0
  lunif_fv = (fv_xy.mul(4).exp() * mask).sum(dim=-1) / mask.sum(dim=-1)
  lunif_fv = lunif_fv.log()
  unq_target = (lunif_fv - lunif_fv.min(dim=-1, keepdim=True)[0]) / (lunif_fv.max(dim=-1, keepdim=True)[0] - lunif_fv.min(dim=-1, keepdim=True)[0]).add(1e-9)
  unq_target = unq_target * 0.5 + 0.25
  lunq = F.binary_cross_entropy(scores, 1 - unq_target.detach())
  return laln, lunif, lunif_fv, lunq

def ctv_loss_model_pred(frame_features,proj,unq_scores,use_unif=True,use_unique=False):
  norm_raw = F.normalize(frame_features,p=2,dim=-1)
  xy_raw = torch.einsum('bmc, bnc -> bmn', norm_raw, norm_raw)
  norm_proj = F.normalize(frame_features,p=2,dim=-1)
  xy = torch.einsum('bmc, bnc -> bmn', norm_proj, norm_proj)
  sort_ids = torch.argsort(xy_raw, -1, descending=True)
  diff_mat = 2 - 2 * xy
  L = frame_features.shape[1]
  S = int(L * ratio_s) 
  K1 = int(L * ratio_k1)
  pos = torch.gather(diff_mat, -1, sort_ids[:,:,S:S+K1])
  laln = pos.mean(dim=-1)
  lunif = diff_mat.mul(-2).exp().mean(dim=-1).log()
  laln = laln.flatten().cpu()
  lunif = lunif.flatten().cpu()

  laln = (laln - laln.min()) / (laln.max() - laln.min())
  lunif = (lunif - lunif.min()) / (lunif.max() - lunif.min())

  unq_scores = unq_scores.cpu().flatten()
  unq_scores = (unq_scores - unq_scores.min()) / (unq_scores.max() - unq_scores.min())
  scores  = laln
  if use_unif:
    scores *= lunif
  if use_unique:
    scores *= unq_scores
  return scores

def post_process_ctv(scores:torch.tensor,dataset='tvsum'):
  scores = gaussian_filter1d(scores.numpy(), 1)
  if dataset =='tvsum':
    scores = np.exp(scores - 1) 
  elif dataset =='summe':
     scores = scores + 0.05
  return scores 