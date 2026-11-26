import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from moviepy import VideoFileClip
from transformers import ASTFeatureExtractor

SAMPLING_RATE = 16000


def audio_sampler_indices(
    video_path: str, sampling_interval: int
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Loads and divides a video into specific interval length mono audio tracks.

    Args:
        video_path: Path to the input video file.
        sampling_interval: Desired length of audio slices, in seconds.
    Returns:
        Array of audio clips and list of starting times, in seconds.
    """
    video = VideoFileClip(video_path)
    is_stereo = video.audio.nchannels == 2

    audio = []
    timestamps = []

    for start in range(0, math.ceil(video.duration), sampling_interval):
        end = min(start + sampling_interval, video.duration)

        segment = video.audio.subclipped(start_time=start, end_time=end)
        sound_array = segment.to_soundarray(fps=SAMPLING_RATE)

        if is_stereo:
            sound_array = sound_array.mean(axis=1)
        else:
            sound_array = sound_array.flatten()

        audio.append(sound_array)
        timestamps.append(start)

    return audio, timestamps


class HFAudioExtractor:
    def __init__(
        self,
        interval: int,
        model: ASTFeatureExtractor,
        batch_size: int,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.interval = interval

    def return_segmented_audio(
        self, video_path: str
    ) -> Tuple[List[np.ndarray], List[int]]:
        audio, timestamps = audio_sampler_indices(video_path, self.interval)
        return audio, timestamps

    def process_and_return_embeddings(
        self, audio_clips: List[np.ndarray]
    ) -> List[List[np.ndarray]]:
        embedding_list = []

        for i in range(0, len(audio_clips), self.batch_size):
            batch = audio_clips[i : i + self.batch_size]

            with torch.no_grad():
                outputs = self.model(
                    batch, sampling_rate=SAMPLING_RATE, return_tensors="np"
                )

            embedding_list.append(outputs["input_values"])

        return embedding_list

    def __call__(self, video_path: str) -> Tuple[np.ndarray, List[int]]:
        audio_clips, indices = self.return_segmented_audio(video_path)
        embeddings = self.process_and_return_embeddings(audio_clips)
        return np.concatenate(embeddings, axis=0), indices
