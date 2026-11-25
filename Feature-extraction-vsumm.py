from Data.Preprocessing import *
import torch
import numpy as np
import h5py
import os
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import resnet50, ResNet50_Weights,googlenet,GoogLeNet_Weights
from transformers import AutoProcessor, CLIPVisionModel,Siglip2VisionModel

def min_max_scale(arr):
    return (arr-arr.min(axis=1,keepdims=True))/(arr.max(axis=1,keepdims=True)-arr.min(axis=1,keepdims=True))

def run_feature_extractor_base(dataset_name,video_base_path,save_name):
    """
    A function which demonstrates feature extraction 
    There are a few TODOs here that we'll need to do, but this is the base and the structure we'd use for all of the files
    """
    # Creation of the feature extraction class objects
    # For pytorch models, newer versions seem to have a feature extractor class that can be used for feature extraction, we can use those for some models
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = resnet50(weights=weights)
    feature_extractor = create_feature_extractor(model, return_nodes=['avgpool'])
    preprocess = weights.transforms()
    torch_extract = TorchVideoExtractor(fps = 1.5,model = feature_extractor,preprocessor = preprocess,
                                        key='avgpool',batch_size = 300,device='cuda:0')

    metadata_file = h5py.File(f'Data/Metadata/{dataset_name}_metadata.h5','r')
    all_videos = list(metadata_file.keys())
    with h5py.File(f'{save_name}.h5','w') as f:
        for video in all_videos:
            print(f'{video} out of {len(all_videos)}')
            video_path = os.path.join(video_base_path,f'{video}.mp4')
            features,positions = torch_extract(video_path)
            shot_bounds = metadata_file[video]['shot_bounds'][...] # TODO: Mathieu, adding another source for shot boundaries here could be nice
            n_frames = metadata_file[video]['n_frames'][()]
            user_summary = metadata_file[video]['user_summary'][...]
            user_score = np.mean(user_summary,axis=0)
            gt = user_score[positions]
            assert len(features) == len(gt)
            f.create_dataset(f'{video}/frame_features',data=features)
            f.create_dataset(f'{video}/gtscore',data=gt)
            f.create_dataset(f'{video}/shot_bounds',data=shot_bounds)
            f.create_dataset(f'{video}/positions',data=positions)
            f.create_dataset(f'{video}/n_frames',data=n_frames)
            f.create_dataset(f'{video}/user_summary',data=user_summary)
            f.create_dataset(f'{video}/user_score',data=user_score)

def run_feature_extractor_hf(dataset_name,video_base_path,save_name):
    model= Siglip2VisionModel.from_pretrained("google/siglip2-base-patch16-naflex")
    processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-naflex")
    torch_extract = HFVideoExtractor(fps = 1.5,model = model,preprocessor = processor,
                                    pooler_att='pooler_output',batch_size = 300,device='cuda')
    metadata_file = h5py.File(f'Data/Metadata/{dataset_name}_metadata.h5','r')
    all_videos = list(metadata_file.keys())
    with h5py.File(f'{save_name}.h5','w') as f:
        for video in all_videos:
            print(f'{video} out of {len(all_videos)}')
            video_path = os.path.join(video_base_path,f'{video}.mp4')
            features,positions = torch_extract(video_path)
            shot_bounds = metadata_file[video]['shot_bounds'][...] # TODO: Mathieu, adding another source for shot boundaries here could be nice
            n_frames = metadata_file[video]['n_frames'][()]
            user_summary = metadata_file[video]['user_summary'][...]
            user_score = np.mean(user_summary,axis=0)
            gt = user_score[positions]
            assert len(features) == len(gt)
            f.create_dataset(f'{video}/frame_features',data=features)
            f.create_dataset(f'{video}/gtscore',data=gt)
            f.create_dataset(f'{video}/shot_bounds',data=shot_bounds)
            f.create_dataset(f'{video}/positions',data=positions)
            f.create_dataset(f'{video}/n_frames',data=n_frames)
            f.create_dataset(f'{video}/user_summary',data=user_summary)
            f.create_dataset(f'{video}/user_score',data=user_score)