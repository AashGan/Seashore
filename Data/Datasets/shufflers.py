# Code to shuffle the input
import torch
class WholeShuffle(object):

    def __init__(self,**kwargs):

        self.probability = kwargs.get('probability')

    def shuffle(self,features,ground_truth,**kwargs):
        assert len(features) == len(ground_truth), 'Length mismatch'
        if torch.rand((1))>self.probability:
            perm_indices = torch.randperm(len(features))
            features = features[perm_indices]
            ground_truth = ground_truth[perm_indices]
        return features, ground_truth
    
class Flip(object):
    def __init__(self,**kwargs):

        self.probability = kwargs.get('probability')

    def shuffle(self,features,ground_truth,**kwargs):
        assert len(features) == len(ground_truth), 'Length mismatch'
        if torch.rand((1))>self.probability:
            return features.flip(dims = [0]),ground_truth.flip(dims=[0])
        else:
            return features, ground_truth
    
class Shufflebylength(object):

    def __init__(self,**kwargs):

        self.probability =  kwargs.get('probability')
        self.segment_length =  kwargs.get('segment_length')
    
    def shuffle(self,features,ground_truth,**kwargs):
        if torch.rand(1)> self.probability:
            list_of_segments = [[i,i+self.segment_length] for i in range(0,len(features),self.segment_length)]

            if list_of_segments[len(list_of_segments)-1][1]!= len(features): list_of_segments[len(list_of_segments)-1][1] = len(features)
            shot_order = torch.randperm(len(list_of_segments))
            # Features 
            ground_truth_new = torch.empty_like(ground_truth)

            a_index =0
            for shot in torch.asarray(list_of_segments)[shot_order]:
                ground_truth_new[a_index:a_index+(shot[1]-shot[0]).item()] = ground_truth[shot[0].item():shot[1].item()]
                a_index = a_index +(shot[1]-shot[0]).item()


            features_new = torch.empty_like(features)
            a_index =0
            for shot in torch.asarray(list_of_segments)[shot_order]:
                features_new[a_index:a_index+(shot[1]-shot[0]).item()] = features[shot[0].item():shot[1].item()]
                a_index = a_index +(shot[1]-shot[0]).item()
            return features_new,ground_truth_new
        else:
            return features,ground_truth


shuffle_dict = {'complete':WholeShuffle,'segments':Shufflebylength,'flip':Flip}