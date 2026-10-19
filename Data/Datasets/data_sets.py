import h5py
import json
from torch.utils.data import Dataset
import os 
import numpy as np
from .shufflers import ShuffleComposer
import torch.nn.functional as F
base_file_path = 'Data/h5Datasets'
#TODO: Adjust the multi-modal loaders in a more intelligent fashion.
def min_max_scale(arr):
    return (arr-arr.min(axis=1,keepdims=True))/(arr.max(axis=1,keepdims=True)-arr.min(axis=1,keepdims=True))
def return_multi_objective(h5_obj,dataset,video_index):
  if 'gtscore_xum' in h5_obj[dataset][video_index].keys():
      gtscore = h5_obj[dataset][video_index]['gtscore_xum'][...]
  else:
      if dataset =='tvsum':
        gtscore = h5_obj[dataset][video_index]["user_score"][...]
        positions = h5_obj[dataset][video_index]["positions"][...]
        gtscore = gtscore[:,positions]
        gtscore = min_max_scale(gtscore)
      elif dataset =='summe':
        gtscore = h5_obj[dataset][video_index]["user_summary"][...]
        positions = h5_obj[dataset][video_index]["positions"][...]
        gtscore = gtscore[:,positions]
  return gtscore
#TODO: adjust this to also adjust when the output is multi-dimensional
def subsample(features,gt,sampled_frames):
  length = len(features)
  if length > sampled_frames:
    ids = np.random.permutation(length)[:sampled_frames]
    ids = np.sort(ids)
  else:
      ids = np.arange(length).astype(np.float32)
      ids = np.interp(
          np.linspace(0, length - 1, sampled_frames),
          np.arange(length),
          ids
      ).astype(np.int64)
  features = features[ids]
  if len(gt.shape)>1:
    gt = gt[:,ids]
  else:
    gt = gt[ids]
  return features, gt

class MultiH5Loader(Dataset):
  def __init__(self,split_file,split_type,cross_val_idx,
               feature_name,multi_obj=False,
               shuffler_obj:ShuffleComposer = None,
               sampled_frames:int = None,normalize=True):
    with open(split_file,'r') as f:
      self.split_file = json.load(f)
    self.data_points = self.split_file[cross_val_idx][split_type]
    self.included_dataset = list({sample.split('/',1)[0] for sample in self.data_points})
    print(self.included_dataset)
    self.feature_name = feature_name
    self.sampled_frames = sampled_frames
    self._create_data_dict(self.included_dataset)
    self.shuffler_obj = shuffler_obj
    self.multi_obj = multi_obj
    self.normalize = normalize 
  def __len__(self):
    return len(self.data_points)

  def _create_data_dict(self,included_datasets):
    self.dataset_dict = {}
    for dataset in included_datasets:
      data_paths = os.path.join(base_file_path,f'{self.feature_name}',f'{self.feature_name}_{dataset}.h5')
      self.dataset_dict[dataset]= h5py.File(data_paths,'r')

  def get_positions(self,dataset_name,video_index):
     if 'positions' in self.dataset_dict[dataset_name][video_index].keys():
        
      return self.dataset_dict[dataset_name][video_index]['positions'][...]
     else:
        return None
  def __getitem__(self,idx):
    data_point = self.data_points[idx]
    dataset,video_index = data_point.split('/')
    features = self.dataset_dict[dataset][video_index]['frame_features'][...]
    # Check for multi-objective flag
    if self.normalize:
       features = features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-12)
    if self.multi_obj:

      gtscore = return_multi_objective(self.dataset_dict,dataset,video_index)
    else:
      gtscore = self.dataset_dict[dataset][video_index]['gtscore'][...]
    if self.sampled_frames:
      features,gtscore = subsample(features,gtscore,self.sampled_frames)
    if self.shuffler_obj is not None:
      features = self.shuffler_obj(features)

    return {'visual_features':features.astype(np.float32),'gtscore':gtscore.astype(np.float32),'data_point': data_point} # We return the data_name for the eval stuff



class SingleH5Loader(Dataset):

  def __init__(self,split_file,split_name,cross_val_idx,feature_name,dataset):

    with open(split_file,'r') as f:
      self.split_file = json.load(f)
    self.data_points = self.split_file[cross_val_idx][split_name]
    data_paths = os.path.join(base_file_path,{dataset},f'{dataset}_{feature_name}.h5')
    self.dataset_file = h5py.File(data_paths,'r')

  def __len__(self):
    return len(self.data_points)

  def return_keys(self):
    return [data_point.split('/')[-1] for data_point in self.data_points]
  
  def __getitem__(self,idx):
    video_index= self.data_points[idx]
    features = self.dataset_file[video_index]['features'][...]
    gtscore = self.dataset_file[video_index]['gtscore'][...]
    return features,gtscore

# TODO: Potentially make the shuffing a bit more intelligent
class MultiModalh5Loader(Dataset):
  """ Multi-Modal h5-loader, we assume a custom name assigned to the h5 file
  Alongside this, we can assume any kind of features included, the feature name can be included in the included_keys arg
  """
  def __init__(self,split_file,split_type,cross_val_idx,
               file_name,included_keys,
               multi_obj=False,
               shuffler_obj:ShuffleComposer = None,
               sampled_frames:int = None,normalize=False):
    with open(split_file,'r') as f:
          self.split_file = json.load(f)
    self.data_points = self.split_file[cross_val_idx][split_type]
    self.included_dataset = list({sample.split('/',1)[0] for sample in self.data_points})
    self.file_name = file_name
    self._create_data_dict(self.included_dataset)
    self.included_keys = included_keys
    self.sampled_frames = sampled_frames
    self.shuffler_obj = shuffler_obj
    self.multi_obj = multi_obj
    self.normalize = normalize


  def __len__(self):
      return len(self.data_points)
  
  def _create_data_dict(self,included_datasets):
      self.dataset_dict = {}
      for dataset in included_datasets:
        data_paths = os.path.join(base_file_path,f'multi_modal',f'{self.file_name}_{dataset}.h5')
        self.dataset_dict[dataset]= h5py.File(data_paths,'r')
  
  def get_positions(self,dataset_name,video_index):
       if 'positions' in self.dataset_dict[dataset_name][video_index].keys():
          
        return self.dataset_dict[dataset_name][video_index]['positions'][...]
       else:
          return None
  def __getitem__(self,idx):
    data_point = self.data_points[idx]
    dataset,video_index = data_point.split('/')
    # Construct the output dictionary here
    output_dict = {}
    if self.multi_obj:
        gtscore = return_multi_objective(self.dataset_dict,dataset,video_index)
    else:
        gtscore = self.dataset_dict[dataset][video_index]['gtscore'][...]
    output_dict['gtscore'] = gtscore
    output_dict['data_point'] = data_point
    for key in self.included_keys:
      if key =='frame_features':
               output_dict['visual_features'] = self.dataset_dict[dataset][video_index][key][...]
      if isinstance(self.dataset_dict[dataset][video_index][key][()], bytes):
        output_dict[key] = self.dataset_dict[dataset][video_index][key][()].decode("utf-8")
      else:
        output_dict[key] = self.dataset_dict[dataset][video_index][key][...]
      
    if self.normalize:
       for key in output_dict.keys():
          if key.endswith("_features"):
             output_dict[key] = output_dict[key] / (np.linalg.norm(output_dict[key]
                                                                   , axis=-1, keepdims=True) + 1e-12)
    if self.sampled_frames:
          output_dict['visual_features'],output_dict['gtscore'] = subsample(output_dict['frame_features'],
                                                                           output_dict['gtscore'],
                                                                           self.sampled_frames)
    if self.shuffler_obj is not None:
      output_dict['visual_features'] = self.shuffler_obj(output_dict['visual_features'])
    return output_dict


dataset_obj_dict = {'unimodal_loader':MultiH5Loader,'multimodal_loader':MultiModalh5Loader}