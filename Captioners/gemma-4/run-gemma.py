from transformers import AutoModelForMultimodalLM
from transformers import AutoProcessor
import os 
from .captioner_utils import *
from .messages import *
import json
from helpers.text_refiners import *
def run_gemma(video_directory,model_name,save_directory,captioning_style='whole_video',truncation=5,refine=False):
    # Creating Video_paths
    all_videos = os.listdir(video_directory)
    video_save_names = [video_name.split('.')[0] for video_name in all_videos]
    all_videos = [os.path.join(video_directory,video_name) for video_name in all_videos]
    # Model loading
    model = AutoModelForMultimodalLM.from_pretrained( # Changed from AutoModelForVision2Seq
        model_name,
        device_map="auto" # Automatically map model to available devices (GPU if present)
    )
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()
    if not os.path.exist(save_directory):
            os.makedirs(save_directory)

    if captioning_style == "whole_video":
        for video_key,video_path in zip(video_save_names,all_videos):
            generated_outputs = run_video_caption_gemma(video_path,model,processor,chunked_video_message,truncation)
            if refine:
                refined_message,input_len = refine_caption(processor,generated_outputs)
                generated_outputs = forward_and_decode(model, refined_message, input_len,processor,768)
                
            with open(os.path.join(save_directory,f'{video_key}.json'),'w') as f:
                json.dump({'Captions':generated_outputs},f)

    if captioning_style == "indexed_frame_wise":
        for video_key,video_path in zip(video_save_names,all_videos):
              generated_outputs = run_frame_wise_description_gemma(video_path,model,framewise_message,processor,truncation)
        with open(os.path.join(save_directory,f'{video_key}.json'),'w') as f:
                        json.dump({'Captions':generated_outputs},f)
    


