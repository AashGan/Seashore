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
    features = [torch.from_numpy(x["visual_features"]) for x in batch]
    gtscore = [torch.from_numpy(x["gtscore"]) for x in batch]
    data_points = [x["data_point"] for x in batch]
    lengths = torch.tensor([len(x) for x in gtscore])
    features = pad_sequence(features, batch_first=True, padding_value=0)
    gtscore = pad_sequence(gtscore, batch_first=True, padding_value=-1) # This is padded with -1 for safety
    mask = torch.arange(gtscore.size(1))[None, :]< lengths[:, None]
    

    return {"visual_features":torch.tensor(features),"gtscore": torch.tensor(gtscore),"mask": mask,"data_points" : data_points}


def batch_text_visual(batch):
    """
    Similar to the above function, but it pads visual and textual features
    """

    visual_features = [torch.from_numpy(x["visual_features"]) for x in batch]
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

def batch_multi_text_visual(batch):
    """
    Similar to the above function, but it pads visual and textual features, when there are multiple possible textual features
    """

    visual_features = [torch.from_numpy(x["visual_features"]) for x in batch]
    textual_features = [torch.from_numpy(x["text_features"]) for x in batch]
    gtscore = [torch.from_numpy(x["gtscore"]) for x in batch]
    data_points = [x["data_point"] for x in batch]
    lengths = torch.tensor([len(x) for x in gtscore])
    text_lengths = torch.tensor([len(x) for x in textual_features])
    textual_features = pad_sequence(textual_features,batch_first=True,padding_value=0)
    gtscore = pad_sequence(gtscore, batch_first=True, padding_value=-1) # This is padded with -1 for safety
    text_mask = torch.arange(textual_features.size(1))[None,:] < text_lengths[:,None]

    return {"visual_features":torch.tensor(visual_features), "text_features": torch.tensor(textual_features)
            ,"gtscore": gtscore,'text_mask':text_mask,"data_points" : data_points}
def batch_text_sum_visual(batch):
    """
    Similar to the above function, but it pads visual and textual features
    """

    visual_features = [torch.from_numpy(x["visual_features"]) for x in batch]
    textual_features = [x["text_summary"] for x in batch]
    gtscore = [torch.from_numpy(x["gtscore"]) for x in batch]
    data_points = [x["data_point"] for x in batch]
    lengths = torch.tensor([len(x) for x in gtscore])
    visual_features = pad_sequence(visual_features, batch_first=True, padding_value=0)
    gtscore = pad_sequence(gtscore, batch_first=True, padding_value=-1) # This is padded with -1 for safety
    mask = torch.arange(gtscore.size(1))[None, :]< lengths[:, None]

    return {"visual_features":torch.tensor(visual_features), "text_labels": textual_features
            ,"gtscore": gtscore,"mask": mask,"data_points" : data_points}

def batch_text_sum_multi(batch):
    visual_features = [torch.from_numpy(x["visual_features"]) for x in batch]
    textual_features = [b"".join(x["text_summary"]).decode('utf-8') for x in batch]
    gtscore_xum = [torch.from_numpy(x['gtscore']) for x in batch]
    data_points = [x["data_point"] for x in batch]
    lengths = torch.tensor([x.shape[1] for x in gtscore_xum])
    max_length = max(lengths)
    visual_features = pad_sequence(visual_features, batch_first=True, padding_value=0)
    gtscore_xum_pad = torch.stack([torch.nn.functional.pad(t,(0,max_length-t.size(1)),value=-1) for t in gtscore_xum])
    mask = torch.arange(gtscore_xum_pad.size(2))[None,:]<lengths[:,None]
    return {"visual_features":torch.tensor(visual_features), "text_labels": textual_features
                ,"gtscore": gtscore_xum_pad,"mask": mask,"data_points" : data_points}

def batch_text_sum_single(batch):
    """ The collator for VideoXUM basic feed without padding
    """
    visual_features = [torch.from_numpy(x["visual_features"]) for x in batch]
    textual_features = [b"".join(x["text_summary"]).decode('utf-8') for x in batch]
    gtscore_xum = [torch.from_numpy(x['gtscore']) for x in batch]
    data_points = [x["data_point"] for x in batch]
    visual_features = pad_sequence(visual_features, batch_first=True, padding_value=0)
    return {"visual_features":torch.tensor(visual_features), "text_labels": textual_features
                ,"gtscore": torch.stack(gtscore_xum),"data_points" : data_points}

def batch_visual_multi(batch):
    visual_features = [torch.from_numpy(x["visual_features"]) for x in batch]
    gtscore_xum = [torch.from_numpy(x['gtscore']) for x in batch]
    data_points = [x["data_point"] for x in batch]
    lengths = torch.tensor([x.shape[1] for x in gtscore_xum])
    max_length = max(lengths)
    visual_features = pad_sequence(visual_features, batch_first=True, padding_value=0)
    gtscore_xum_pad = torch.stack([torch.nn.functional.pad(t,(0,max_length-t.size(1)),value=-1) for t in gtscore_xum])
    mask = torch.arange(gtscore_xum_pad.size(2))[None,:]<lengths[:,None]
    return {"visual_features":torch.tensor(visual_features), "gtscore": gtscore_xum_pad,"mask": mask,"data_points" : data_points}

def batch_multi_gt_feat_text(batch):
    """
    Duplicate text and video features based on size of the GT
    """
    # Duplicate the inputs to match the size of the GT
    visual_features = [torch.from_numpy(batch[i]['visual_features']) for i in range(len(batch)) for _ in range(batch[i]['gtscore'].shape[0]) ]
    lengths = torch.tensor([len(x) for x in visual_features])
    # Find the text lengths 
    text_lengths = torch.tensor([len(x) for x in textual_features])
    # Then pad the 
    data_points = [x["data_point"] for x in batch]
    textual_features = [torch.from_numpy(batch[i]['text_features']) for i in range(len(batch)) for _ in range(batch[i]['gtscore'].shape[0])]
    gtscore = [torch.from_numpy(batch[i]["gtscore"]) for i in range(len(batch)) for _ in range(batch[i]['gtscore'].shape[0]) ]
    visual_features = pad_sequence(visual_features, batch_first=True, padding_value=0)
    textual_features = pad_sequence(textual_features,batch_first=True,padding_value=0)
    text_mask = torch.arange(textual_features.size(1))[None,:] < text_lengths[:,None]
    visual_mask = torch.arange(visual_features.size(1))[None,:] < lengths [:,None]
    return {"visual_features":torch.tensor(visual_features), "text_features": torch.tensor(textual_features)
            ,"gtscore": gtscore,"visual_mask": visual_mask,'text_mask':text_mask,"data_points" : data_points}

def custom_videoxum_collator(batch):
    """
    Specifically batch collator function for SD-VSUM since their released dataset pads 
    """
    # Duplicate the inputs to match the size of the GT
    visual_features = [torch.from_numpy(batch[i]['visual_features']) for i in range(len(batch)) for _ in range(batch[i]['text_features'].shape[0]) ]
    lengths = torch.tensor([len(x) for x in visual_features])
    # Find the text lengths 
    # Then pad the 
    data_points = [x["data_point"] for x in batch]
    textual_features = torch.tensor([x['text_features'] for x in batch]).squeeze().flatten(start_dim=0,end_dim=1) # As SD-VSUM specifically releases all features as batch
    gtscore = [torch.from_numpy(batch[i]["gtscore"]) for i in range(len(batch)) for _ in range(batch[i]['gtscore'].shape[0]) ]
    visual_features = pad_sequence(visual_features, batch_first=True, padding_value=0)
    text_mask = torch.where(textual_features.count_nonzero(dim=2) ==0,False,True)
    visual_mask = torch.arange(visual_features.size(1))[None,:] < lengths [:,None]
    return {"visual_features":torch.tensor(visual_features), "text_features": torch.tensor(textual_features)
            ,"gtscore": gtscore,"visual_mask": visual_mask,'text_mask':text_mask,"data_points" : data_points}

collate_dict = {'visual_feat':batch_collate_fn,'visual_text_feat':batch_text_visual,
                'visual_feat_text_sum':batch_text_sum_visual,'multi_feat_text_sum':batch_text_sum_multi,
                'custom_sd_sum_collator':custom_videoxum_collator,'visual_feat_multi':batch_visual_multi,
                'single_gt_text_sum':batch_text_sum_single}
