from Data.Preprocessing import *

import torch

import numpy as np
import h5py
import os
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoProcessor, CLIPVisionModel,Siglip2VisionModel
# This script is an example of the various feature extraction procedures we'd have to conduct for multiple datasets.


# The things I'll need to extract
# features:provided by the unction, user_summary: taken from metadata, user_score:taken from metadata, gtscore: generated from the downsampling factor, n_frames: taken from either metadata or from reading the video,
# shot_bounds: either taken from the metadata, or provided by another function, positions, provided by the function



def run_feature_extractor_base(dataset_name,video_base_path,save_name):
    """
    A function which demonstrates feature extraction 
    There are a few TODOs here that we'll need to do, but this is the base and the structure we'd use for all of the files
    """
    # Creation of the feature extraction class objects
    # For pytorch models, newer versions seem to have a feature extractor class that can be used for feature extraction, we can use those for some models
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    weights = ResNet50_Weights.DEFAULT
    feature_extractor = create_feature_extractor(model, return_nodes=['avgpool'])
    preprocess = weights.transforms()
    torch_extract = TorchVideoExtractor(fps = 2,model = feature_extractor,preprocessor = preprocess,
                                        key='avgpool',batch_size = 300,device='cuda')
    
    if dataset_name =='summe':
        metadata_file = h5py.File('Data/Metadata/summe_metadata.h5','r')
        all_videos = list(metadata_file.keys())
        with h5py.File(f'{save_name}.h5','w') as f:
            for video in all_videos:
                print(f'{video} out of {len(all_videos)}')
                video_path = os.path.join(video_base_path,f'{video}.mp4')
                features,positions = torch_extract(video_path)
                shot_bounds = metadata_file[video]['shot_bounds'][...] # TODO: Mathieu, adding another source for shot boundaries here could be nice
                n_frames =metadata_file[video]['n_frames'][()]
                user_summary = metadata_file[video]['user_summary'][...]
                user_score = np.mean(user_summary,axis=0)
                gt = user_score[positions]
                assert len(features) == len(gt)
                f.create_dataset(f'{features}/features',data=features)
                f.create_dataset(f'{features}/gtscore',data=features)
                f.create_dataset(f'{video}/shot_bounds',data=shot_bounds)
                f.create_dataset(f'{video}/positions',data=positions)
                f.create_dataset(f'{video}/n_frames',data=n_frames)
                f.create_dataset(f'{video}/user_summary',data=user_summary)
                f.create_dataset(f'{video}/user_score',data=user_score)
    elif dataset_name =='tvsum':
            metadata_file = h5py.File('Data/Metadata/tvsum_metadata.h5','r')
            all_videos = list(metadata_file.keys())
            with h5py.File(f'{save_name}.h5','w') as f:
                for video in all_videos:
                    video_path = os.path.join(video_base_path,f'{video}.mp4')
                    features,positions = torch_extract(video_path)
                    shot_bounds = metadata_file[video]['shot_bounds'][...] # TODO: Mathieu, adding another source for shot boundaries here could be nice
                    n_frames =metadata_file[video]['n_frames'][()]
                    user_summary = metadata_file[video]['user_summary'][...]
                    user_score = metadata_file[video]['user_score'][...]
                    gt = user_score.mean(axis=0)[positions]
                    assert len(features) == len(gt)
                    f.create_dataset(f'{features}/features',data=features)
                    f.create_dataset(f'{features}/gtscore',data=features)
                    f.create_dataset(f'{video}/shot_bounds',data=shot_bounds)
                    f.create_dataset(f'{video}/positions',data=positions)
                    f.create_dataset(f'{video}/n_frames',data=n_frames)
                    f.create_dataset(f'{video}/user_summary',data=user_summary)
                    f.create_dataset(f'{video}/user_score',data=user_score)
    else:
         raise NotImplementedError

def run_feature_extractor_hf(dataset_name,video_base_path,save_name):
    """
    A function which demonstrates feature extraction 
    There are a few TODOs here that we'll need to do, but this is the base and the structure we'd use for all of the files
    """
    # Creation of the feature extraction class objects
    # For pytorch models, newer versions seem to have a feature extractor class that can be used for feature extraction, we can use those for some models
    # model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    # processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model= Siglip2VisionModel.from_pretrained("google/siglip2-base-patch16-naflex")
    processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-naflex")
    torch_extract = HFVideoExtractor(fps = 2,model = model,preprocessor = processor,
                                    pooler_att='pooler_output',batch_size = 300,device='cuda')
    if dataset_name =='summe':
        metadata_file = h5py.File('Data/Metadata/summe_metadata.h5','r')
        all_videos = list(metadata_file.keys())
        with h5py.File(f'{save_name}.h5','w') as f:
            for video in all_videos:
                print(f'{video} out of {len(all_videos)}')
                video_path = os.path.join(video_base_path,f'{video}.mp4')
                features,positions = torch_extract(video_path)
                shot_bounds = metadata_file[video]['shot_bounds'][...] # TODO: Mathieu, adding another source for shot boundaries here could be nice
                n_frames =metadata_file[video]['n_frames'][()]
                user_summary = metadata_file[video]['user_summary'][...]
                user_score = np.mean(user_summary,axis=0)
                gt = user_score[positions]
                assert len(features) == len(gt)
                f.create_dataset(f'{features}/features',data=features)
                f.create_dataset(f'{features}/gtscore',data=features)
                f.create_dataset(f'{video}/shot_bounds',data=shot_bounds)
                f.create_dataset(f'{video}/positions',data=positions)
                f.create_dataset(f'{video}/n_frames',data=n_frames)
                f.create_dataset(f'{video}/user_summary',data=user_summary)
                f.create_dataset(f'{video}/user_score',data=user_score)
    elif dataset_name =='tvsum':
            metadata_file = h5py.File('Data/Metadata/tvsum_metadata.h5','r')
            all_videos = list(metadata_file.keys())
            with h5py.File(f'{save_name}.h5','w') as f:
                for video in all_videos:
                    video_path = os.path.join(video_base_path,f'{video}.mp4')
                    features,positions = torch_extract(video_path)
                    shot_bounds = metadata_file[video]['shot_bounds'][...] # TODO: Mathieu, adding another source for shot boundaries here could be nice
                    n_frames =metadata_file[video]['n_frames'][()]
                    user_summary = metadata_file[video]['user_summary'][...]
                    user_score = metadata_file[video]['user_score'][...]
                    gt = user_score.mean(axis=0)[positions]
                    assert len(features) == len(gt)
                    f.create_dataset(f'{features}/features',data=features)
                    f.create_dataset(f'{features}/gtscore',data=features)
                    f.create_dataset(f'{video}/shot_bounds',data=shot_bounds)
                    f.create_dataset(f'{video}/positions',data=positions)
                    f.create_dataset(f'{video}/n_frames',data=n_frames)
                    f.create_dataset(f'{video}/user_summary',data=user_summary)
                    f.create_dataset(f'{video}/user_score',data=user_score)
    else:
         raise NotImplementedError

if __name__ == "__main__":
    #run_feature_extractor_base('summe','/home/aash/Datasets/summe','resnet_summe')
    run_feature_extractor_hf('tvsum','/home/aash/Datasets/tvsum','siglip_tvsum')