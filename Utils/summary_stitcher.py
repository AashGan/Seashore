import cv2 
import ffmpeg
import os 
import h5py
def create_annotator_summary(save_path, input_video, annotator_idx, fps,
                             frame_ranges):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
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


    
if __name__ == "__main__":
    run_video_anno_summe()