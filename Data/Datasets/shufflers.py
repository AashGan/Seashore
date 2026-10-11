# Code to shuffle the input
import torch
import numpy as np

class WholeShuffle(object):
    def __init__(self, **kwargs):
        self.probability = 1-kwargs.get('probability')
        self.multi_obj = kwargs.get('multi_obj', False)

    def __call__(self, features, ground_truth, **kwargs):
        
        if np.random.rand(1) > self.probability:
            if self.multi_obj:
                assert len(features) == len(ground_truth[0]), 'Length mismatch'
            perm_indices = np.random.permutation(len(features))
            features = features[perm_indices]
            if self.multi_obj:
                ground_truth = ground_truth[:,perm_indices]
            else:
                assert len(features) == len(ground_truth), 'Length mismatch'
                ground_truth = ground_truth[perm_indices]
        return features, ground_truth


class Flip(object):
    def __init__(self, **kwargs):
        self.probability = 1-kwargs.get('probability')
        self.multi_obj = kwargs.get('multi_obj',False)
    def __call__(self, features, ground_truth, **kwargs):
        
        if np.random.rand(1) > self.probability:
            if self.multi_obj:
                assert len(features) == len(ground_truth[0]), 'Length mismatch'
                return np.flip(features, axis=0), np.flip(ground_truth, axis=1)
            else:
                assert len(features) == len(ground_truth), 'Length mismatch'
                return np.flip(features, axis=0), np.flip(ground_truth, axis=0)
        else:
            return features, ground_truth


class Shufflebylength(object):
    def __init__(self, **kwargs):
        self.probability = 1-kwargs.get('probability')
        self.segment_length = kwargs.get('segment_length')
        self.multi_obj = kwargs.get('multi_obj',False)
    def __call__(self, features, ground_truth, **kwargs):
        if np.random.rand(1) > self.probability:
            list_of_segments = [[i, i + self.segment_length] for i in range(0, len(features), self.segment_length)]
            if list_of_segments[-1][1] != len(features):
                list_of_segments[-1][1] = len(features)
            shot_order = np.random.permutation(len(list_of_segments))

            ground_truth_new = np.empty_like(ground_truth)
            a_index = 0
            for shot in np.array(list_of_segments)[shot_order]:
                if self.multi_obj:
                    ground_truth_new[:,a_index:a_index + (shot[1] - shot[0])] = ground_truth[:,shot[0]:shot[1]]
                else:
                    ground_truth_new[a_index:a_index + (shot[1] - shot[0])] = ground_truth[shot[0]:shot[1]]
                a_index += (shot[1] - shot[0])

            features_new = np.empty_like(features)
            a_index = 0
            for shot in np.array(list_of_segments)[shot_order]:
                features_new[a_index:a_index + (shot[1] - shot[0])] = features[shot[0]:shot[1]]
                a_index += (shot[1] - shot[0])
            return features_new, ground_truth_new
        else:
            return features, ground_truth
class WholeShuffle_t(object):

    def __init__(self,**kwargs):

        self.probability = kwargs.get('probability')

    def shuffle(self,features,ground_truth,**kwargs):
        assert len(features) == len(ground_truth), 'Length mismatch'
        if torch.rand((1))>self.probability:
            perm_indices = torch.randperm(len(features))
            features = features[perm_indices]
            ground_truth = ground_truth[perm_indices]
        return features, ground_truth
    
class Flip_t(object):
    def __init__(self,**kwargs):

        self.probability = kwargs.get('probability')

    def shuffle(self,features,ground_truth,**kwargs):
        assert len(features) == len(ground_truth), 'Length mismatch'
        if torch.rand((1))>self.probability:
            return features.flip(dims = [0]),ground_truth.flip(dims=[0])
        else:
            return features, ground_truth
    
class Shufflebylength_t(object):

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


class ShuffleComposer():
    def __init__(self,shuffle_length_dict):
        self.shuffle_object = [shuffle_dict[key](**values) for key,values in shuffle_length_dict.items()]
    def __call__(self,x,y):
        for obj in self.shuffle_object:
            x,y = obj(x,y)
        return x,y
shuffle_dict = {'complete':WholeShuffle,'segments':Shufflebylength,'flip':Flip}