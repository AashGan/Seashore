import numpy as np
from scipy.stats import spearmanr,kendalltau

#--------------------------------------------Post-Process-Utils----------------------------------------------------------------

def knapSack(W, wt, val, n):
	""" Maximize the value that a knapsack of capacity W can hold. You can either put the item or discard it, there is
	no concept of putting some part of item in the knapsack.

	:param int W: Maximum capacity -in frames- of the knapsack.
	:param list[int] wt: The weights (lengths -in frames-) of each video shot.
	:param list[float] val: The values (importance scores) of each video shot.
	:param int n: The number of the shots.
	:return: A list containing the indices of the selected shots.
	"""
	K = [[0 for _ in range(W + 1)] for _ in range(n + 1)]

	# Build table K[][] in bottom up manner
	for i in range(n + 1):
		for w in range(W + 1):
			if i == 0 or w == 0:
				K[i][w] = 0
			elif wt[i - 1] <= w:
				K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w])
			else:
				K[i][w] = K[i - 1][w]

	selected = []
	w = W
	for i in range(n, 0, -1):
		if K[i][w] != K[i - 1][w]:
			selected.insert(0, i - 1)
			w -= wt[i - 1]

	return selected

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



def correlation_metric_wrappers(preds,gt,aggregation,post_process=False,video_list = None):
      """
      A wrapper for evaluation of video summarization, specifically for multiple possible evaluation strategies.

      """
      

      if post_process:
            assert video_list is not None, "Pass the video list to post-process"
            

