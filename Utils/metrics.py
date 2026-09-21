import numpy as np
from .dicts import dataset_metadata_paths
from scipy.stats import spearmanr,kendalltau
from sklearn.metrics import f1_score
from .post_process import knapSack, upsample
import h5py
import torch
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
# Binarization from Source: https://github.com/IDT-ITI/SD-VSum/blob/main/model/utils/evaluation_metrics.py
def binarize_top_percent(score, top_percent=0.15):
    """
    Selects the top_percent frames of the video to be included in the summary, by binarizing the top percentage of scores into a binary indicator array.
    :param numpy.ndarray score: 1D array of scores for frames.
    :param float top_percent: Percentage of frames to be included in the summary. Defaults to 0.15.
    :return numpy.ndarray: Binary array with values 1.0 for the top_percent highest scores and 0.0 otherwise.
    """
    n = score.size
    k = int((top_percent * n))
    sorted_descending = np.sort(score)[::-1]
    threshold_val = sorted_descending[k]
    # Create binary array (values ≥ threshold become 1)
    binary_score = (score >= threshold_val).astype(np.float32)
    return binary_score
#--------------------------------------------Statistical-tests---------------------------------------------------------------

def eval_kendall(preds:np.ndarray,gt:np.ndarray):

    return kendalltau(preds,gt)[0]

def eval_spearman(preds:np.ndarray,gt:np.ndarray):

    return spearmanr(preds,gt)[0]


            





def eval_average(preds,gts): 
    
    all_kendalls = [eval_kendall(pred,gt) for pred,gt in zip(preds,gts)]
    all_spearmans = [eval_spearman(pred,gt) for pred,gt in zip(preds,gts)]
    return np.mean(all_kendalls), np.mean(all_spearmans)


# A function that routes different evaluation based on the inputs 

def post_process_preds(pred,metadata,post_process):
    positions = metadata['positions']
    n_frames = metadata['n_frames']
    shot_bound = metadata['shot_bounds']
    if post_process =='none':
        return pred
    elif post_process == "upsample":
        return upsample(pred,positions,n_frames)
    elif post_process == "summary_gen":
        return generate_summary_single(shot_bound,pred,n_frames,positions)
    elif post_process == "binarize_top_k":
        return binarize_top_percent(pred)
    else:
        raise ValueError('provide an eval type: [none,upsample,summary_gen]')

def process_and_route_single(pred:torch.tensor,gt:torch.tensor,metadata:dict,ground_truth_data:dict,eval_type:str,post_process:str = "upsample",metric='corr'):
    pred = pred.to('cpu').squeeze().numpy()
    gt = gt.to('cpu').squeeze().numpy()
    pred = post_process_preds(pred,metadata,post_process) # Does the post-procesing based on type
    if metric =="corr":
        # Returns both Kendall and Spearman Correlation
        return evaluate_correlation(pred,gt,ground_truth_data,eval_type)

    elif metric =="f1":
        assert post_process in ['summary_gen','binarize_top_k'], "For F1 evaluation Post processing must be summary_gen, or binarize_top_k"
        return evaluate_f1(pred,ground_truth_data,eval_type)
    else:
        raise ValueError('provide an eval type: [corr,f1]')
    

def evaluate_correlation(pred,gt,ground_truth_data,eval_type):
    if eval_type == 'gt':
        return {"kendall":eval_kendall(pred,gt),"spearman":eval_spearman(pred,gt)}
    if eval_type == "user_score":
        user_score = ground_truth_data['user_score']
        # We do list comprenehsion since tvsum has multiple user scores 
        all_kendalls = [eval_kendall(pred,gt_i) for gt_i in user_score]
        all_spearmans = [eval_spearman(pred,gt_i) for gt_i in user_score]

        return {"kendall":np.mean(all_kendalls),"spearman":np.mean(all_spearmans)}
    elif eval_type == "user_summary":
        user_score = ground_truth_data['user_summary']
        # We do list comprenehsion since all of them have multiple summaries
        all_kendalls = [eval_kendall(pred,gt_i) for gt_i in user_score]
        all_spearmans = [eval_spearman(pred,gt_i) for gt_i in user_score]

        return {"kendall":np.mean(all_kendalls),"spearman":np.mean(all_spearmans)}
    else:
        raise ValueError('provide an eval type: [gt,user_score,user_summary]')

    
#TODO: implement f1, which takes the eval_type to have the "max" and "best" eval thing from past research
def evaluate_f1(pred,ground_truth_data,eval_type):
    raise NotImplementedError
        
