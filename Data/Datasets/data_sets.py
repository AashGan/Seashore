import h5py
import json
from torch.utils.data import Dataset
import os 


base_file_path = 'Data/h5Datasets'
#TODO: Implement multi-modal data-loaders
class MultiH5Loader (Dataset):
  def __init__(self,split_file,split_name,cross_val_idx,feature_name):
    with open(split_file,'r') as f:
      self.split_file = json.load(f)
    self.data_points = self.split_file[cross_val_idx][split_name]
    self.included_dataset = list({sample.split('/',1)[0] for sample in self.data_points})
    print(self.included_dataset)
    self.feature_name = feature_name
    self._create_data_dict(self.included_dataset)

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


    return {'features':features,'gtscore':gtscore,'data_point': data_point} # We return the data_name for the eval stuff
  


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


class MultiModelh5Loader(Dataset):
  """ Multi-Modal h5-loader, we assume a custom name assigned to the h5 file
  Alongside this, we can assume any kind of features included, the feature name can be included in the included_keys arg
  """
  def __init__(self,split_file,split_name,cross_val_idx,file_name,included_keys):
    with open(split_file,'r') as f:
          self.split_file = json.load(f)
    self.data_points = self.split_file[cross_val_idx][split_name]
    self.included_dataset = list({sample.split('/',1)[0] for sample in self.data_points})
    self.file_name = file_name
    self._create_data_dict(self.included_dataset)
    self.included_keys = included_keys


  def __len__(self):
      return len(self.data_points)
  
  def _create_data_dict(self,included_datasets):
      self.dataset_dict = {}
      for dataset in included_datasets:
        data_paths = os.path.join(base_file_path,f'{self.file_name}',f'{self.file_name}_{dataset}.h5')
        self.dataset_dict[dataset]= h5py.File(data_paths,'r')

  def __getitem__(self,idx):
    data_point = self.data_points[idx]
    dataset,video_index = data_point.split('/')
    # Construct the output dictionary here
    output_dict = {}
    output_dict['gtscore'] = self.dataset_dict[dataset][video_index]['gtscore'][...]
    output_dict['data_point'] = data_point
    for key in self.included_keys:
      output_dict[key] = self.dataset_dict[dataset][video_index][key][...]
    return output_dict