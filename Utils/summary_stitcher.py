import cv2 
import ffmpeg
import os 
import h5py
from fractions import Fraction
import numpy as np
def create_annotator_summary(save_path, input_video, annotator_idx, fps,
                             frame_ranges):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i, (start_frame, end_frame) in enumerate(frame_ranges, 1):
        outfile = os.path.join(save_path, f"Annotator_{annotator_idx}_clip_{i:03d}.mp4")
        start = start_frame / fps
        duration = (end_frame - start_frame + 1) / fps
        (
            ffmpeg
            .input(input_video, ss=start, t=duration)
            .output(
                str(outfile),
                vcodec="libx264",
                acodec="aac",
                preset="fast",
                crf=18,
            )
            .overwrite_output()
            .run()
        )

def create_annotator_summary_joint(save_path,input_video,annotator_idx,fps,frame_ranges):
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    clips = []
    probe = ffmpeg.probe(input_video)
    has_audio = any(
    stream["codec_type"] == "audio"
    for stream in probe["streams"])
    outfile = os.path.join(save_path, f"Annotator_{annotator_idx}_complete.mp4")
    assert fps!=0, "Fps cannot be zero"
    for start_frame, end_frame in frame_ranges:
        start = start_frame / fps
        duration = (end_frame - start_frame + 1) / fps

        clip = ffmpeg.input(input_video, ss=start, t=duration)
        if has_audio:
            clips.append((clip.video, clip.audio))
        else:
            clips.append((clip.video,))

# Concatenate all selected video and audio streams
    try:
        streams = [stream for pair in clips for stream in pair]
        joined = ffmpeg.concat(*streams, v=1, a=1 if has_audio else 0)
        output_options = {
        "vcodec": "libx264",
        "preset": "fast",
        "crf": 18,
    }
        if has_audio:
            output_options["acodec"] = "aac"
        out,err = (
            joined
            .output(outfile,
                **output_options
            )
            .overwrite_output()
            .run()
        )
    except ffmpeg.Error as e:
        print("FFmpeg error:")
        print(e.stdout)
        print(e.stderr)
        print(lol)
# TODO: Add a notebook/script that allows you to do this a bit more easily.
def test_annotator_summary():
    video_name = 'video_1'
    video_path = f"C:\\Datasets\\summe\\{video_name}.mp4"
    probed_video = cv2.VideoCapture(video_path)
    fps = probed_video.get(cv2.CAP_PROP_FPS)
    dataset  = h5py.File('C:\\Final-Project-Refactor\\Data\\h5datasets\\eccv16_dataset_summe_google_pool5.h5')
    #all_shots = dataset[video_name]['change_points'][...]
    user_summary = dataset[video_name]['user_summary'][...][0]
    all_shots = convert_array_to_index_tuples(user_summary)
    create_annotator_summary('test',video_path,0,fps,all_shots)

def convert_array_to_index_tuples(arr):
    result = []
    start = None
    for i, x in enumerate(arr):
        if x == 1 and start is None:
            start = i
        elif x == 0 and start is not None:
            result.append((start, i - 1))
            start = None
    if start is not None:
        result.append((start, len(arr) - 1))
    return result


def run_video_anno_summe():
    dataset  = h5py.File('C:\\Final-Project-Refactor\\Data\\h5datasets\\eccv16_dataset_summe_google_pool5.h5')
    overall_path = 'SumMeSummaries'
    for video_name in list(dataset.keys()):
        video_path = f"C:\\Datasets\\summe\\{video_name}.mp4"
        probed_video = cv2.VideoCapture(video_path)
        fps = probed_video.get(cv2.CAP_PROP_FPS)
        save_path = f'{overall_path}/{video_name}'
        for i, user_summary in enumerate(dataset[video_name]['user_summary'][...]):
            all_shots = convert_array_to_index_tuples(user_summary)
            create_annotator_summary(save_path, video_path, i+1, fps, all_shots)


def select_top_k(score,rate=0.175):
    k = int((rate * len(score)))
    binary_score = np.zeros_like(score)
    binary_score[np.argsort(score)[-k:]] = 1
    return binary_score

def read_fps_with_ffmpeg(input_video):
    """read fps when cv2 fails to read"""
    probe = ffmpeg.probe(input_video)

    video_stream = next(
        s for s in probe["streams"]
        if s["codec_type"] == "video"
    )

    fps = float(Fraction(video_stream["avg_frame_rate"]))
    return fps
    
def run_video_tvsum(base_video_path,h5_path,overall_path,style='separate_clips'):
    dataset = h5py.File(h5_path)
    for video_name in list(dataset.keys()):
        print(f'video: {video_name}')
        video_path = os.path.join(base_video_path,f'{video_name}.mp4')
        print(f'doing video: {video_path}')
        fps = read_fps_with_ffmpeg(video_path)
        print(fps)
        
        save_path = f'{overall_path}/{video_name}'
        if style =='separate_clips':
            for i, user_summary in enumerate(dataset[video_name]['user_score'][...]):
                user_summary = select_top_k(user_summary)
                all_shots = convert_array_to_index_tuples(user_summary)
                create_annotator_summary(save_path, video_path, i+1, fps, all_shots)
        elif style =='complete':
            for i, user_summary in enumerate(dataset[video_name]['user_score'][...]):
                user_summary = select_top_k(user_summary)
                print(sum(user_summary))
                all_shots = convert_array_to_index_tuples(user_summary)
                print(all_shots)
                create_annotator_summary_joint(save_path, video_path, i+1, fps, all_shots)
def run_video_summe(base_video_path,h5_path,overall_path,style='separate_clips'):
    dataset = h5py.File(h5_path)

    for video_name in list(dataset.keys()):
        print(f'video: {video_name}')
        video_path = os.path.join(base_video_path,f'{video_name}.mp4')
        print(f'doing video: {video_path}')
        fps = read_fps_with_ffmpeg(video_path)
        print(fps)
        if os.path.exists(f'{overall_path}/{video_name}'):
            continue
        
        save_path = f'{overall_path}/{video_name}'
        if style =='separate_clips':
            for i, user_summary in enumerate(dataset[video_name]['user_summary'][...]):

                all_shots = convert_array_to_index_tuples(user_summary)
                create_annotator_summary(save_path, video_path, i+1, fps, all_shots)
        elif style =='complete':
            for i, user_summary in enumerate(dataset[video_name]['user_summary'][...]):
                print(sum(user_summary))
                all_shots = convert_array_to_index_tuples(user_summary)
                print(all_shots)
                create_annotator_summary_joint(save_path, video_path, i+1, fps, all_shots)
    
if __name__ == "__main__":
    #run_video_anno_summe()
    video_path = "C:/Datasets/summe"
    h5_path = "Data/Metadata/summe_metadata.h5"
    run_video_summe(video_path,h5_path,'Results/summe','complete')
    video_path = "C:/Datasets/tvsum"
    h5_path = "Data/Metadata/tvsum_metadata.h5"
    run_video_tvsum(video_path,h5_path,'Results/tvsum','complete')
    