import numpy as np
import h5py
import torch
import pandas as pd 
import cv2 
import os
from Data.Preprocessing import *
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoProcessor, CLIPVisionModel,Siglip2VisionModel
from tqdm import tqdm
videoxum_metadata = 'Data/Metadata/'
data_source = 'Data/videos/Activity_Videos'
# TODO: should track the rows which are val and test
def generate_metadata_h5(division_rate=10,video_xum=None):
    metadata_df = pd.concat([pd.read_json(f'{videoxum_metadata}{mode}_videoxum.json') for mode in ['train','val','test']],axis=0,ignore_index=True) 
    metadata_h5 = f'videoxum_metadata.h5'
    
    with h5py.File(metadata_h5,'w') as data_file:
        for index,row in metadata_df.iterrows():
            video_name = f'video_{index+1}'
            print(f'{os.path.join(data_source,row["video_id"])}.mp4')
            if os.path.exists(f'{os.path.join(data_source,row["video_id"])}.mp4'):
                f =  cv2.VideoCapture(f'{os.path.join(data_source,row["video_id"])}.mp4')
            elif os.path.exists(f'{os.path.join(data_source,row["video_id"])}.mkv'):
                f =  cv2.VideoCapture(f'{os.path.join(data_source,row["video_id"])}.mkv')
            else:
                raise "Error, no file found"
            fps = round(f.get(cv2.CAP_PROP_FPS))
            total_frames = int(f.get(cv2.CAP_PROP_FRAME_COUNT))
            f.release()
            # Making positions, and shot boundaries, may require some adjusting to get different shot boundaries
            positions = np.arange(0,int(total_frames/fps))*fps
            step = fps*division_rate
            shot_bounds = np.array([(start,min(start+step,total_frames)) for start in range(0,total_frames,step)])
            vsum_hot = np.array(row['vsum_onehot'])
            # Repeat the values to upsampled
            vsum_hot_upsample  = np.repeat(vsum_hot,fps,axis = 1) # The paper says its sampled at one FPS, so we replicate each of them at the FPS

            if vsum_hot_upsample.shape[1]<total_frames: # Check if the length is less than
                pad_length = total_frames-vsum_hot_upsample.shape[1]
                vsum_hot_upsample = np.pad(vsum_hot_upsample,((0,0),(0,pad_length)),mode='edge')
            if vsum_hot_upsample.shape[1]>total_frames:
                vsum_hot_upsample = vsum_hot_upsample[:total_frames]
            try:
                data_file.create_dataset(f'{video_name}/gtscore',data=vsum_hot.mean(axis=0))
                data_file.create_dataset(f'{video_name}/gtscore_xum',data=vsum_hot)
                data_file.create_dataset(f'{video_name}/shot_bounds',data=shot_bounds)
                data_file.create_dataset(f'{video_name}/n_frames',data=total_frames)
                data_file.create_dataset(f'{video_name}/positions',data=positions)
                data_file.create_dataset(f'{video_name}/user_summary',data=vsum_hot_upsample)
                data_file.create_dataset(f'{video_name}/user_score',data=vsum_hot_upsample.mean(axis=1))
                data_file.create_dataset(f'{video_name}/text_summary',data=row['tsum'])
                data_file.create_dataset(f'{video_name}/vsum_time_stamps',data=row['vsum'])
                data_file.create_dataset(f'{video_name}/activitynet_video_id',data=row['video_id'])
            except:
                print(f'Video name :{video_name} exists')

            
            


def rename_h5(h5_path,rename_dict):
    with h5py.File(h5_path, "r+") as f:
        for old_name, new_name in rename_dict.items():
            if old_name not in f:
                raise KeyError(f"'{old_name}' not found")

            if new_name in f:
                raise ValueError(f"'{new_name}' already exists")

            f.move(old_name, new_name)           
            


def run_feature_extractor_base(video_base_path,save_name):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    weights = ResNet50_Weights.DEFAULT
    feature_extractor = create_feature_extractor(model, return_nodes=['avgpool'])
    preprocess = weights.transforms()
    torch_extract = TorchVideoExtractor(fps = 1,model = feature_extractor,preprocessor = preprocess,
                                        key='avgpool',batch_size = 500,device='cuda',videoxum_flag=True)
    metadata_file = h5py.File('Data/Metadata/videoxum_metadata.h5','r')
    all_videos = list(metadata_file.keys())
    with h5py.File(f'{save_name}.h5','w') as f:
        for video in tqdm(all_videos,'processing videos'):
            file_path =metadata_file[video]["activitynet_video_id"][()].decode("utf-8")
            if os.path.exists(os.path.join(video_base_path,f'{file_path}.mp4')):
                video_path = os.path.join(video_base_path,f'{file_path}.mp4')
            elif os.path.exists(os.path.join(video_base_path,f'{file_path}.mkv')):
                video_path = os.path.join(video_base_path,f'{file_path}.mkv')
            else:
                raise "Error, no file found"
            features,positions = torch_extract(video_path)
            shot_bounds = metadata_file[video]['shot_bounds'][...] # TODO: Mathieu, adding another source for shot boundaries here could be nice
            n_frames = metadata_file[video]['n_frames'][()]
            user_summary = metadata_file[video]['user_summary'][...]
            f.create_dataset(f'{video}/features',data=features)
            f.create_dataset(f'{video}/gtscore',data=metadata_file[video]['gtscore'][...])
            f.create_dataset(f'{video}/gtscore_xum',data=metadata_file[video]['gtscore_xum'][...])
            f.create_dataset(f'{video}/shot_bounds',data=shot_bounds)
            f.create_dataset(f'{video}/n_frames',data=n_frames)
            f.create_dataset(f'{video}/positions',data=positions)
            f.create_dataset(f'{video}/user_summary',data=user_summary)
            f.create_dataset(f'{video}/user_score',data=metadata_file[video]['user_score'][...])
            f.create_dataset(f'{video}/text_summary',data=metadata_file[video]['text_summary'][:])
            f.create_dataset(f'{video}/vsum_time_stamps',data=metadata_file[video]['vsum_time_stamps'][...])
            f.create_dataset(f'{video}/activitynet_video_id',data=metadata_file[video]['activitynet_video_id'][()])


def run_feature_extractor_hf(video_base_path,save_name):
    model= Siglip2VisionModel.from_pretrained("google/siglip2-base-patch16-naflex")
    processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-naflex")
    torch_extract = HFVideoExtractor(fps = 1,model = model,preprocessor = processor,
                                    pooler_att='pooler_output',batch_size = 500,device='cuda')

    metadata_file = h5py.File('Data/Metadata/videoxum_metadata.h5','r')
    all_videos = list(metadata_file.keys())
    with h5py.File(f'{save_name}.h5','w') as f:
         for video in tqdm(all_videos,'processing videos'):
            file_path =metadata_file[video]["activitynet_video_id"][()].decode("utf-8")
            if os.path.exists(os.path.join(video_base_path,f'{file_path}.mp4')):
                video_path = os.path.join(video_base_path,f'{file_path}.mp4')
            elif os.path.exists(os.path.join(video_base_path,f'{file_path}.mkv')):
                video_path = os.path.join(video_base_path,f'{file_path}.mkv')
            else:
                raise "Error, no file found"
            features,positions = torch_extract(video_path)
            features,positions = torch_extract(video_path)
            shot_bounds = metadata_file[video]['shot_bounds'][...] # TODO: Mathieu, adding another source for shot boundaries here could be nice
            n_frames = metadata_file[video]['n_frames'][()]
            user_summary = metadata_file[video]['user_summary'][...]
            f.create_dataset(f'{video}/features',data=features)
            f.create_dataset(f'{video}/gtscore',data=metadata_file[video]['gtscore'][...])
            f.create_dataset(f'{video}/gtscore_xum',data=metadata_file[video]['gtscore_xum'][...])
            f.create_dataset(f'{video}/shot_bounds',data=shot_bounds)
            f.create_dataset(f'{video}/n_frames',data=n_frames)
            f.create_dataset(f'{video}/positions',data=positions)
            f.create_dataset(f'{video}/user_summary',data=user_summary)
            f.create_dataset(f'{video}/user_score',data=metadata_file[video]['user_score'][...])
            f.create_dataset(f'{video}/text_summary',data=metadata_file[video]['text_summary'][:])
            f.create_dataset(f'{video}/vsum_time_stamps',data=metadata_file[video]['vsum_time_stamps'][...])
            f.create_dataset(f'{video}/activitynet_video_id',data=metadata_file[video]['activitynet_video_id'][()])


if __name__ == "__main__":
    #generate_metadata_h5()
    #run_feature_extractor_hf(data_source,"siglip2_videoxum")
    run_feature_extractor_base(data_source,"resnet_videoxum")