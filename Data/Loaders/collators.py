import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def batch_collate_fn(batch):
    """
    A function to process the h5 as a batch, padding video sequences
    Parameters
    ----------
    batch : list of tuples
        Each element in the list is a tuple of the form
        (features, gtscore, data_point)

        * features  : torch.Tensor of shape (seq_len, feat_dim)
        * gtscore  : torch.Tensor of shape (seq_len, score_dim)
        * data_point: any object you want to keep unchanged

    Returns
     -------
    padded_features : torch.Tensor
        Shape: (batch_size, max_seq_len, feat_dim)
    padded_gtscore : torch.Tensor
        Shape: (batch_size, max_seq_len, score_dim)
    data_points   : list
        The original data_point objects (unchanged)
    """
    features = [torch.from_numpy(x["features"]) for x in batch]
    gtscore = [torch.from_numpy(x["gtscore"]) for x in batch]
    data_points = [x["data_point"] for x in batch]
    lengths = torch.tensor([len(x) for x in gtscore])
    features = pad_sequence(features, batch_first=True, padding_value=0)
    gtscore = pad_sequence(gtscore, batch_first=True, padding_value=-1) # This is padded with -1 for safety
    mask = torch.arange(gtscore.size(1))[None, :]< lengths[:, None]
    

    return {"features":torch.tensor(features),"gtscore": torch.tensor(gtscore),"mask": mask,"data_points" : data_points}


def batch_text_visual(batch):
    """
    Similar to the above function, but it pads visual and textual features
    """

    visual_features = [torch.from_numpy(x["frame_features"]) for x in batch]
    textual_features = [torch.from_numpy(x["text_features"]) for x in batch]
    gtscore = [torch.from_numpy(x["gtscore"]) for x in batch]
    data_points = [x["data_point"] for x in batch]
    lengths = torch.tensor([len(x) for x in gtscore])
    text_lengths = torch.tensor([len(x) for x in textual_features])
    visual_features = pad_sequence(visual_features, batch_first=True, padding_value=0)
    textual_features = pad_sequence(textual_features,batch_first=True,padding_value=0)
    gtscore = pad_sequence(gtscore, batch_first=True, padding_value=-1) # This is padded with -1 for safety
    mask = torch.arange(gtscore.size(1))[None, :]< lengths[:, None]
    text_mask = torch.arange(textual_features.size(1))[None,:] < text_lengths[:,None]

    return {"visual_features":torch.tensor(visual_features), "text_features": torch.tensor(textual_features)
            ,"gtscore": gtscore,"visual_mask": mask,'text_mask':text_mask,"data_points" : data_points}


def batch_text_sum_visual(batch):
    """
    Similar to the above function, but it pads visual and textual features
    """

    visual_features = [torch.from_numpy(x["frame_features"]) for x in batch]
    textual_features = [(x["text_summary"]) for x in batch]
    gtscore = [torch.from_numpy(x["gtscore"]) for x in batch]
    data_points = [x["data_point"] for x in batch]
    lengths = torch.tensor([len(x) for x in gtscore])
    visual_features = pad_sequence(visual_features, batch_first=True, padding_value=0)
    gtscore = pad_sequence(gtscore, batch_first=True, padding_value=-1) # This is padded with -1 for safety
    mask = torch.arange(gtscore.size(1))[None, :]< lengths[:, None]

    return {"visual_features":torch.tensor(visual_features), "text_labels": textual_features
            ,"gtscore": gtscore,"mask": mask,"data_points" : data_points}
    