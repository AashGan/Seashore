import cv2
import numpy as np
from PIL import Image
import torch 

def read_video_and_batch(file_path,sample_rate = 1):
  cap = cv2.VideoCapture(file_path)
  sampled_indices = np.arange(0, cap.get(cv2.CAP_PROP_FRAME_COUNT),
                            int(cap.get(cv2.CAP_PROP_FPS)/sample_rate))
  frames = []
  for i in sampled_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)

    ret, frame = cap.read()

    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
  frames = np.array(frames)
  return frames,sampled_indices

def create_frame_prompt_and_return_message(processor,message ,images,sampled_slices):
  content = []
  for i in range(len(images)):
    content.append({'type':"image", "image": Image.fromarray(images[i])})

  frame_info = "\n".join([f"frame_{int(sampled_slices[i])}" for i in range(len(sampled_slices))])
  new_message = message.format(frame_info = frame_info)
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

def encode_batch_image_and_message(processor, images, message):
    batch_messages = []

    for image in images:
        content = [
            {
                "type": "image",
                "image": Image.fromarray(image),
            },
            {
                "type": "text",
                "text": message,
            },
        ]

        batch_messages.append({
            "role": "user",
            "content": content,
        })

    batch = processor.apply_chat_template(
        batch_messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
    )

    input_lens = batch["attention_mask"].sum(dim=1)

    return batch, input_lens

def forward_and_decode(model,processed_message,input_len,processor,max_new_tokens = 75):
  with torch.no_grad():
    output_ids = model.generate(**processed_message.to(model.device),max_new_tokens = max_new_tokens)

  return processor.decode(output_ids[0][input_len:-1],skip_special_tokens = False)

def forward_and_batch_decode(model,processed_message,input_lens,processor,max_new_tokens = 75):
    with torch.no_grad():
        output_ids = model.generate(**processed_message.to(model.device),max_new_tokens = max_new_tokens)
    batch_decode = processor.batch_decode(output_ids,skip_special_tokens = False)
    batch_decode = [decoded_output[input_len:-1] for decoded_output,input_len in zip(batch_decode,input_lens)]
    return batch_decode