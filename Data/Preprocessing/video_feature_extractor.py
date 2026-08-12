
import ffmpeg 
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
import cv2

def video_sampler_indices(video_path, target_fps):
    """
    Downsamples a video to a target FPS and returns the downsampled frames
    and their original indices.

    Args:
        video_path (str): Path to the input video file.
        target_fps (int): Desired frames per second for downsampling.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Array of downsampled frames (dtype=uint8).
            - list: List of original frame indices that were sampled.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if original_fps == 0: # Handle case where original_fps might be zero
        print(f"Error: Original FPS is zero for video {video_path}.")
        cap.release()
        return None, None

    # Calculate the frame interval for downsampling
    if target_fps > original_fps:
        print(f"Warning: Target FPS ({target_fps}) is higher than original FPS ({original_fps}). No effective downsampling will occur.")
        frame_interval = 1
    elif target_fps <= 0:
        print("Error: Target FPS must be positive.")
        cap.release()
        return None, None
    else:
        frame_interval = int(original_fps / target_fps)

    target_frames = np.arange(0,original_frame_count,frame_interval).astype(int)

    downsampled_frames = []
    sampled_frame_indices = []
    frame_number = 0


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Select frames based on the calculated interval
        # Use a small epsilon for floating point comparison robustness
        if frame_number in target_frames:
            # cv2.read() already returns frames as numpy.ndarray with dtype=uint8
            downsampled_frames.append(frame)
            sampled_frame_indices.append(frame_number)

        frame_number += 1


    cap.release()

    if downsampled_frames:
        downsampled_frames_array = np.array(downsampled_frames)
        return downsampled_frames_array, sampled_frame_indices
    else:
        print("No frames were downsampled or extracted.")
        return None, None

class TorchVideoExtractor():
    """
    Video feature extractor using a PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        Model used for feature extraction.

    preprocessor : callable
        Preprocessing function applied to frames.

    batch_size : int
        Batch size used during inference.

    device : str, optional
        Device used for inference, by default "cpu".
    """
    def __init__(self,fps,model,preprocessor,batch_size,key=None,device='cpu'):
      self.model = model
      self.preprocessor = preprocessor
      self.batch_size = batch_size
      self.device = device
      self.fps = fps
      self.key = key
    def return_images(self,video_path):
      images,indices = video_sampler_indices(video_path,self.fps)
      return images,indices
    def process_and_return_embeddings(self,images):
      self.model.eval()
      self.model.to(self.device)
      print("Moved Model to Device")
      embeddings = []
      for i in range(0,len(images),self.batch_size):
        batch = images[i:i+self.batch_size].copy()
        print("Extracted Batch")
        inputs = self.preprocessor(torch.from_numpy(batch.transpose(0,3,1,2))) # Assumes that
        print("Ran Preprocessing")
        with torch.no_grad():
          outputs = self.model(inputs.to(self.device))
        print(f"Model ran sucessfully on batch{i}")
        if self.key is not None:
          embeddings.append(outputs[self.key].to('cpu').flatten(1).numpy())
        else:
          embeddings.append(outputs.to('cpu').flatten(1).squeeze().numpy())
      return embeddings
    def __call__(self,video_path):
        images,indices = self.return_images(video_path)
        print('Completed returning images from video')
        embeddings = self.process_and_return_embeddings(images)
        return np.concatenate(embeddings,axis = 0),indices
    

class HFVideoExtractor():
    def __init__(self,fps,model,preprocessor,batch_size,pooler_att='pooler_output',device='cpu'):
      self.model = model
      self.preprocessor = preprocessor
      self.batch_size = batch_size
      self.pooler_att = pooler_att
      self.fps = fps
      assert self.pooler_att in ['pooler_output','vision_pooler_output'], "Specify the pooler"
      self.device = device
    def return_images(self,video_path):
      images,indices = video_sampler_indices(video_path,self.fps)
      return images,indices
    def process_and_return_embeddings(self,images):
      self.model.eval()
      self.model.to(self.device)
      embedding_list = []

      for i in range(0,len(images),self.batch_size):
        batch = images[i:i+self.batch_size]
        inputs = self.preprocessor(images = batch,return_tensors = "pt")
        with torch.no_grad():
          outputs = self.model(**inputs.to(self.device))
        if self.pooler_att == 'pooler_output':
          embeddings = outputs.pooler_output.to('cpu').numpy()

        elif self.pooler_att == 'vision_pooler_output':
          embeddings = outputs.vision_pooler_output.to('cpu').numpy()
        embedding_list.append(embeddings)
      return embedding_list

    def __call__(self,video_path):
        images,indices= self.return_images(video_path)
        embeddings = self.process_and_return_embeddings(images)
        return np.concatenate(embeddings,axis = 0),indices