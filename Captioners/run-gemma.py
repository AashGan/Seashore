from transformers import AutoModelForMultimodalLM
from transformers import AutoProcessor
import os 
from helpers.captioner_utils import *
from helpers.messages import *
import json
from helpers.text_refiners import *
from time import time
def run_gemma(video_directory,model_name,save_directory,
              save_name='test',captioning_style='whole_video',truncation=5,
              refine=False,return_positions = False):
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
    if not os.path.exists(save_directory):
            os.makedirs(save_directory)

    if captioning_style == "whole_video":
        caption_dict = {}
        for video_key,video_path in zip(video_save_names,all_videos):
            generated_outputs = run_video_caption_gemma(video_path,model,processor,chunked_video_message,truncation,100)
            if refine:
                refined_message,input_len = refine_caption(processor,generated_outputs)
                generated_outputs = forward_and_decode(model, refined_message, input_len,processor,768)
            caption_dict[video_key] = generated_outputs
        with open(os.path.join(save_directory,f'{video_key}.json'),'w') as f:
            json.dump(caption_dict,f,indent = 2)

    if captioning_style == "indexed_frame_wise":
        for video_key,video_path in zip(video_save_names,all_videos):
              generated_outputs = run_frame_wise_description_gemma(video_path,model,framewise_message,processor,truncation)
        with open(os.path.join(save_directory,f'{video_key}.json'),'w') as f:
                        json.dump({'Captions':generated_outputs},f)

    if captioning_style =='independent_frame_wise':
          caption_dict = {}
          for video_key,video_path in zip(video_save_names,all_videos):
                start = time()
                if not return_positions:
                    generated_outputs = run_image_wise_description(video_path,model,processor,independent_frame_message,truncation,75)
                    caption_dict[video_key] = generated_outputs
                else:
                      generated_outputs, positions = run_image_wise_description(video_path,model,processor,independent_frame_message,truncation,75,True)
                      output_dict = {'captions':generated_outputs,'positions':positions.tolist()}
                      caption_dict[video_key] = output_dict
                print(f'video :{video_key}, time taken: {time()-start}')
          with open(os.path.join(save_directory,f'{save_name}.json'),'w') as f:
                      json.dump(caption_dict,f,indent = 2)

def run_per_annotator_video_caption(model_name,base_video_directory,save_directory,save_name='per_annotator'):
    model = AutoModelForMultimodalLM.from_pretrained( # Changed from AutoModelForVision2Seq
            model_name,
            device_map="auto" # Automatically map model to available devices (GPU if present)
        )
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()
    if not os.path.exists(save_directory):
                os.makedirs(save_directory) 
    video_list = os.listdir(base_video_directory)
    all_captions = {}
    for video in video_list:
        video_captions = {}

        annotator_list = os.listdir(os.path.join(base_video_directory,video))
        for annotator in annotator_list:
            annotator_num = annotator.split('_')[1]
            video_path = os.path.join(base_video_directory,video,annotator)
            try:
                generated_outputs,_ = run_base_video_caption(video_path,model,processor,generic_video_message,1024)
                video_captions[f'Annotator_{annotator_num}'] = generated_outputs
            except:
                    print(f'Error on video: {annotator}')
        all_captions[video] = video_captions
    with open(os.path.join(save_directory,save_name),'w') as f:
          json.dump(all_captions,f)
if __name__ == "__main__":
    model_name = 'google/gemma-4-E2B-it'
    # video_directory = 'Supplemental/tvsum_and_summe_videos_per_annotation/Results/summe'
    # save_directory = 'Captioners/per_annotator/summe/'
    # run_per_annotator_video_caption(model_name,video_directory,save_directory,save_name='per_annotator.json')
    video_directory = '/home/aash/Datasets/tvsum'
    save_directory = 'Captioners/per_frame'
    run_gemma(video_directory,model_name,save_directory,'per_frame_tvsum_with_positions','independent_frame_wise',30,return_positions=True)
    video_directory = '/home/aash/Datasets/summe'
    save_directory = 'Captioners/per_frame'
    run_gemma(video_directory,model_name,save_directory,'per_frame_summe_with_positions','independent_frame_wise',30,return_positions=True)
    


