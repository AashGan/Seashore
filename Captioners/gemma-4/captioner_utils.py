import torch
import cv2
import numpy as np
from PIL import Image

from helpers.video_helpers import *

def run_image_wise_description_gemma(video_file_path,model,processor,truncation_rate = 5):
  frames,sampled_indices = read_video_and_batch(video_file_path)
  generated_outputs = []
  for i in range(0,len(frames),truncation_rate):
    sub_sampled_frames,indices = frames[i:i+truncation_rate],sampled_indices[i:i+truncation_rate]
    messages,lens = encode_batch_image_and_message(processor,sub_sampled_frames)
    generated_output = forward_and_batch_decode(model,messages,lens,processor)
    generated_outputs.append(generated_output)
  return generated_outputs

def run_frame_wise_description_gemma(video_file_path,model,processor,truncation_rate = 5):
  """ params: File_path; string of the video
      params: truncation_rate: the rate at which the frames are given to gemma 4

  """
  frames,sampled_indices = read_video_and_batch(video_file_path)
  generated_outputs = []
  for i in range(0,len(frames),truncation_rate):
    sub_sampled_frames,indices = frames[i:i+truncation_rate],sampled_indices[i:i+truncation_rate]
    messages,input_len = create_frame_prompt_and_return_message(processor,sub_sampled_frames,indices)
    generated_output = forward_and_decode(model,messages,input_len,processor)
    generated_outputs.append(generated_output)
  generated_outputs = "\n".join(generated_outputs)
  return generated_outputs

def run_video_caption_gemma(video_file_path,model,processor,message, truncation_rate = 5):
  frames,sampled_indices = read_video_and_batch(video_file_path)
  generated_outputs = []
  previous_context = None
  for i in range(0,len(frames),truncation_rate):
    sub_sampled_frames,indices = frames[i:i+truncation_rate],sampled_indices[i:i+truncation_rate]
    messages,input_len  = whole_video_prompt_message(processor, message ,sub_sampled_frames,indices,previous_context)
    generated_output = forward_and_decode(model,messages,input_len,processor)
    if previous_context is None:

      previous_context = "".join([generated_output])
    else:
      previous_context = "".join([previous_context,generated_output])
    generated_outputs.append(generated_output)
  generated_outputs = "\n".join(generated_outputs)
  return generated_outputs

def whole_video_prompt_message(processor,message,images,sampled_slices,previous_outputs = None):
  if previous_outputs is None:
    previous_outputs = ""
  content = []
  for i in range(len(images)):
    content.append({'type':"image", "image": Image.fromarray(images[i])})
  new_message = message.format(previous_outputs = previous_outputs)
  content.append({"type":"text","text":new_message})
  batch_message = [{"role":"user","content":content}]
  messages = processor.apply_chat_template(batch_message,
                                          tokenize=True,
                                          return_dict=True,
                                          return_tensors="pt",
                                          padding=True,
                                          add_generation_prompt=True,)
  input_len = messages["input_ids"].shape[-1]
  return messages,input_len
