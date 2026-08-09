import numpy as np 
import hdf5storage 
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

def upsample(score,positions,n_frames):
    " A function which upsamples the score based on the positions from which the video was downsampled from"
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
    return frame_init_scores
# TODO: Move this metadata to a Pure h5 file 
def load_tvsum_mat(filename='ydata-tvsum50.mat'):
    data = hdf5storage.loadmat(filename, variable_names=['tvsum50'])
    data = data['tvsum50'].ravel()
    
    data_list = []
    for item in data:
        video, category, title, length, nframes, user_anno, gt_score = item
        
        item_dict = {
        'video': video[0, 0],
        'category': category[0, 0],
        'title': title[0, 0],
        'length': length[0, 0],
        'nframes': nframes[0, 0],
        'user_anno': user_anno,
        'gt_score': gt_score
        }
        
        data_list.append((item_dict))
    return data_list