from helpers.video_helpers import whole_video_encode_and_message, forward_and_decode
from helpers.captioner_utils import run_image_wise_description, run_base_video_caption
from transformers import AutoModelForMultimodalLM,AutoProcessor
import os
from helpers.messages import *
import json
import tqdm
def run_smolvlm(video_directory,model_name,save_directory,save_name = 'test',captioning_style = 'whole_video',truncation = 5, refine = False):
    all_videos = os.listdir(video_directory)
    video_save_names = [video_name.split('.')[0] for video_name in all_videos]
    all_videos = [os.path.join(video_directory,video_name) for video_name in all_videos]
    # Model loading
    print(all_videos)
    model = AutoModelForMultimodalLM.from_pretrained( # Changed from AutoModelForVision2Seq
        model_name,
        device_map="auto" # Automatically map model to available devices (GPU if present)
    )
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()
    if not os.path.exists(save_directory):
                os.makedirs(save_directory)

    if captioning_style =='whole_video':
            save_dict = {}
            for video_key,video_path in tqdm.tqdm(zip(video_save_names,all_videos)):
                generated_outputs = run_base_video_caption(video_path,model,processor,generic_video_message,1024)
                generated_outputs = generated_outputs[0].split('Assistant: ')[-1]
                save_dict[video_key] = generated_outputs
            with os.path.join(save_directory,f'{save_name}.json','w') as f:
                   json.dump(save_dict,f)
    else:
           raise NotImplementedError


if __name__ == "__main__":
       video_directory = '/home/aash/Datasets/tvsum'
       model_name = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
       save_directory = 'Captioners'
       save_name = 'test_tvsum'
       run_smolvlm(video_directory,model_name,save_directory,save_name)