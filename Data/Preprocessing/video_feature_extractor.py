
import ffmpeg 
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
import cv2

#TODO: Let this return the 
def video_sampler(video_path,fps=2):
  """ An FFMPEG based uniform frame sampler using 
  args:
  video_path: path to video (str)
  fps: frame rate to sample from in the video (int)
  returns: bitwise array of frames
  """
  probe = ffmpeg.probe(video_path)
  video_stream = next((stream for stream in probe['streams']
                             if stream['codec_type'] == 'video'), None)
  width = int(video_stream['width'])
  height = int(video_stream['height'])
  cmd = (
                ffmpeg.input(video_path).filter('fps', fps=fps)
            )
  out, _ = (
                cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, quiet=True)
            )
  images = np.frombuffer(out,np.uint8).reshape([-1,height,width,3])
  return images

def return_frame_picks(video_path,fps=2):
  """ 
  Returns stuff for our positions variable 
  """
  cap = cv2.VideoCapture(video_path)
  orig_fps = round(cap.get(cv2.CAP_PROP_FPS))
  total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
  return np.arange(0,total_frames,orig_fps//fps) 


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
      images = video_sampler(video_path,self.fps)
      return images
    def process_and_return_embeddings(self,images):
      self.model.eval()
      self.model.to(self.device)
      embeddings = []
      for i in range(0,len(images),self.batch_size):
        batch = images[i:i+self.batch_size].copy()
        inputs = self.preprocessor(torch.from_numpy(batch.transpose(0,3,1,2))) 
        with torch.no_grad():
          outputs = self.model(inputs.to(self.device))
        if self.key is not None:
          embeddings.append(outputs[self.key].to('cpu').flatten(1).numpy())
        else:
          embeddings.append(outputs.to('cpu').flatten(1).squeeze().numpy())
      return embeddings
    def __call__(self,video_path):
        images = self.return_images(video_path)
        print('Completed returning images from video')
        embeddings = self.process_and_return_embeddings(images)
        return np.concatenate(embeddings,axis = 0)
    

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
      images = video_sampler(video_path,fps=self.fps)
      return images
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
        images = self.return_images(video_path)
        embeddings = self.process_and_return_embeddings(images)
        return np.concatenate(embeddings,axis = 0)