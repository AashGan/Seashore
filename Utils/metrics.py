import numpy as np
from dicts import dataset_metadata_paths
from scipy.stats import spearmanr,kendalltau
from sklearn.metrics import f1_score
from post_process import knapSack, upsample
#--------------------------------------------Summary Generation-----------------------------------------------------------------------------
    

def generate_summary_single(shot_bound,score,n_frames,positions,return_shot_info = False):
    frame_init_scores = score
    frame_scores = np.zeros(n_frames, dtype=np.float32)
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])

    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i + 1]
        if i == len(frame_init_scores):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = frame_init_scores[i]

    # Compute shot-level importance scores by taking the average importance scores of all frames in the shot
    shot_imp_scores = []
    shot_lengths = []
    for shot in shot_bound:
        shot_lengths.append(shot[1] - shot[0] + 1)
        shot_imp_scores.append((frame_scores[shot[0]:shot[1] + 1].mean()).item())

    # Select the best shots using the knapsack implementation
    final_shot = shot_bound[-1]
    final_max_length = int((final_shot[1] + 1) * 0.15)

    selected = knapSack(final_max_length, shot_lengths, shot_imp_scores, len(shot_lengths))

    # Select all frames from each selected shot (by setting their value in the summary vector to 1)
    summary = np.zeros(final_shot[1] + 1, dtype=np.int8)
    for shot in selected:
        summary[shot_bound[shot][0]:shot_bound[shot][1] + 1] = 1
    if return_shot_info:
        return shot_lengths, shot_imp_scores,selected,summary
    return summary

#--------------------------------------------Statistical-tests---------------------------------------------------------------

def eval_kendall(preds:np.ndarray,gt:np.ndarray):

    return kendalltau(preds,gt)[0]

def eval_spearman(preds:np.ndarray,gt:np.ndarray):

    return spearmanr(preds,gt)[0]



def correlation_metric_wrappers(preds,gt,dataset_list:list,aggregation,post_process=False,video_list = None):
      """
      A wrapper for evaluation of video summarization, specifically for multiple possible evaluation strategies.

      """
      

      if post_process:
            assert video_list is not None, "Pass the video list to post-process"
            


def process_and_eval(preds,gts,video_key_list,dataset_list):
      """ 
      Function to process the model predictions before evaluating the Spearman Correlation Coefficient
      """
      assert len(preds) == len(gts) == len(video_key_list) == len(dataset_list), (
    "preds, gt, video_key_list, and dataset_list must all have the same length.")
    #TODO: Test to see if these return everything as intended
      shot_bounds = [ shot_bound
                     for i in range(len(video_key_list))
                     for video_key in video_key_list[i]
                     for shot_bound in dataset_metadata_paths[dataset_list[i]][video_key]["change_points"]
                    ] 
      n_frames_videos = [ n_frames
                     for i in range(len(video_key_list))
                     for video_key in video_key_list[i]
                     for n_frames in dataset_metadata_paths[dataset_list[i]][video_key]["n_frames"]
                    ]


      all_positions = [     positions
                     for i in range(len(video_key_list))
                     for video_key in video_key_list[i]
                     for positions in dataset_metadata_paths[dataset_list[i]][video_key]["positions"]
                    ]
      assert len(preds) == len(gts) == len(shot_bounds) == len(n_frames_videos), (
          "preds, gt, video_key_list, and dataset_list must all have the same length.")
    # Process the summary predictions 
      all_processed_outputs = [generate_summary_single(shot_bound,score,n_frames,positions) for shot_bound,score,n_frames,positions in zip(shot_bounds, preds, n_frames_videos, all_positions)]
    # Compute the Kendall and Spearman Correlation 

      all_kendalls = [np.mean([eval_kendall(processed_preds,gt_i) for gt_i in gt])for processed_preds,gt in zip(all_processed_outputs,gts)]
      all_spearmans = [np.mean([eval_spearman(processed_preds,gt_i) for gt_i in gt])for processed_preds,gt in zip(all_processed_outputs,gts)]

      return np.mean(all_kendalls), np.mean(all_spearmans)


def eval_direct(preds,gts,video_key_list,dataset_list):
     assert len(preds) == len(gts) == len(video_key_list) == len(dataset_list), (
         "preds, gt, video_key_list, and dataset_list must all have the same length.")
     n_frames_videos = [ n_frames
                          for i in range(len(video_key_list))
                          for video_key in video_key_list[i]
                          for n_frames in dataset_metadata_paths[dataset_list[i]][video_key]["n_frames"]
                         ]
     
     
     all_positions = [   positions
                          for i in range(len(video_key_list))
                          for video_key in video_key_list[i]
                          for positions in dataset_metadata_paths[dataset_list[i]][video_key]["positions"]
                         ]
     all_processed_outputs = [upsample(score,n_frames,positions) for score,n_frames,positions in zip(preds, n_frames_videos, all_positions)]

     all_kendalls = [np.mean([eval_kendall(processed_preds,gt_i) for gt_i in gt])for processed_preds,gt in zip(all_processed_outputs,gts)]
     all_spearmans = [np.mean([eval_spearman(processed_preds,gt_i) for gt_i in gt])for processed_preds,gt in zip(all_processed_outputs,gts)]

     return np.mean(all_kendalls), np.mean(all_spearmans)


def eval_average(preds,gts): 
    
    all_kendalls