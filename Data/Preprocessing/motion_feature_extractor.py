from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from .video_feature_extractor import video_sampler_indices


def preprocess_image_for_motion(img: np.ndarray) -> np.ndarray:
    out = cv2.resize(img, (224, 224)).astype(np.float32)
    out = out - 127.5
    return out / 127.5


class MotionExtractor:
    def __init__(
        self,
        fps: int,
        model: Callable[[np.ndarray], np.ndarray],
        preprocessor: Optional[Callable[[np.ndarray], np.ndarray]],
        batch_size: int = 1,
        clip_length: int = 64,
        keep_every: int = 1,
    ) -> None:
        assert clip_length >= 1 and batch_size >= 1 and keep_every >= 1

        self.model = model
        self.batch_size = batch_size
        self.preprocessor = preprocessor
        self.fps = fps
        self.clip_length = clip_length
        self.keep_every = keep_every

    def return_images(self, video_path: str) -> Tuple[List[np.ndarray], List[int]]:
        images, indices = video_sampler_indices(video_path, self.fps)
        return images, indices

    def process_and_return_embeddings(
        self, images: List[np.ndarray], indices: List[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.asarray(indices)

        data = [
            self.preprocessor(img) if self.preprocessor is not None else img
            for img in images
        ]
        n = len(data)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=int)

        n_clips = -(-n // self.clip_length)
        pad = n_clips * self.clip_length - n
        if pad:
            data = np.concatenate([data, np.repeat(data[-1:], pad, axis=0)])
        clips = data.reshape(n_clips, self.clip_length, *data.shape[1:])

        outputs = []
        for start in range(0, n_clips, self.batch_size):
            out = np.asarray(self.model(clips[start : start + self.batch_size]))
            assert out.ndim == 3  # We want shape=(B, T_out, F)
            outputs.append(out.reshape(-1, out.shape[-1]))
        feats = np.concatenate(outputs, axis=0)

        t_out = len(feats) // n_clips
        step = self.clip_length // t_out

        n_valid = -(-n // step)
        feats = feats[: n_valid : self.keep_every]
        starts = np.arange(0, n_valid * step, step * self.keep_every)
        return feats, indices[starts]

    def __call__(self, video_path: str) -> Tuple[np.ndarray, np.ndarray]:
        images, indices = self.return_image(video_path)
        return self.process_and_return_embeddings(images, indices)
