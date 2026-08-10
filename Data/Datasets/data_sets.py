import h5py
import json
from torch.utils.data import Dataset
import os 


base_file_path = 'Data/h5Datasets'
#TODO: Implement multi-modal data-loaders
class MultiH5Loader (Dataset):
  def __init__(self,split_file,split_name,cross_val_idx,feature_name,included_dataset):
    with open(split_file,'r') as f:
      self.split_file = json.load(f)
    self.data_points = self.split_file[cross_val_idx][split_name]
    self.feature_name = feature_name
    self._create_data_dict(included_dataset)

  def __len__(self):
    return len(self.data_points)

  def _create_data_dict(self,included_datasets):
    self.dataset_dict = {}
    for dataset in included_datasets:
      data_paths = os.path.join(base_file_path,f'{self.feature_name}',f'{self.feature_name}_{dataset}.h5')
      self.dataset_dict[dataset]= h5py.File(data_paths,'r')


  def __getitem__(self,idx):
    data_point = self.data_points[idx]
    dataset,video_index = data_point.split('/')
    features = self.dataset_dict[dataset][video_index]['features'][...]
    gtscore = self.dataset_dict[dataset][video_index]['gtscore'][...]
    return features,gtscore,data_point # We return the data_name for the eval stuff
  


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