import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoModelForMultimodalLM
from transformers import AutoProcessor

message = """You are provided with a list of frames and their corresponding sampled indices as follows:
{frame_info}
Described each of the frames in one sentence. The description should be only based on the frame. Return your output as frame_<frame_index>:<description>. """

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

def create_prompt_and_return_message(processor,images,sampled_slices):
  content = []
  for i in range(len(images)):
    content.append({'type':"image", "image": Image.fromarray(images[i])})

  frame_info = "\n".join([f"frame_{int(sampled_slices[i])}" for i in range(len(sampled_slices))])
  new_message = message.format(frame_info = frame_info)
  print(new_message)
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


def encode_image_and_message(processor, images, message):
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
def forward_and_decode(model,processed_message,input_len,processor):
  with torch.no_grad():
    output_ids = model.generate(**processed_message.to(model.device),max_new_tokens = 256)

  return processor.decode(output_ids[0][input_len:-1],skip_special_tokens = False)

def run_frame_wise_description_gemma(video_file_path,model,processor,truncation_rate = 5):
  """ params: File_path; string of the video
      params: truncation_rate: the rate at which the frames are given to gemma 4

  """
  frames,sampled_indices = read_video_and_batch(video_file_path)
  generated_outputs = []
  for i in range(0,len(frames),truncation_rate):
    sub_sampled_frames,indices = frames[i:i+truncation_rate],sampled_indices[i:i+truncation_rate]
    messages,input_len = create_prompt_and_return_message(processor,sub_sampled_frames,indices)
    generated_output = forward_and_decode(model,messages,input_len,processor)
    generated_outputs.append(generated_output)
  generated_outputs = "\n".join(generated_outputs)
  return generated_outputs





def run(hf_gemma_model,truncation_rate = 6,sample_rate = 1):
   model_name = "google/gemma-4-E2B-it"

    model = AutoModelForMultimodalLM.from_pretrained( # Changed from AutoModelForVision2Seq
        model_name,
        torch_dtype=torch.bfloat16, # Use bfloat16 for potentially faster inference
        device_map="auto" # Automatically map model to available devices (GPU if present)
    )
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()
   