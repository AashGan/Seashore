
import torch
from Data.Preprocessing import SentenceTransformerWrapper,HFTextWrapper,InstructionTunerFeatures
import numpy as np
import h5py
import json
from transformers import AutoProcessor,SiglipTextModel
from Utils.h5_utils import copy_h5
def open_and_return_json(json_path):
    with open(json_path,'r') as f:
        return json.load(f)

def run_text_feature_extractor_base(json_path,sentence_transformer_name,save_name,device = 'cpu',metadata_source = None):
    # We would have to assume that the sentences are joined together
    sentence_feature_extractor = SentenceTransformerWrapper(sentence_transformer_name,device)
    caption_dict = open_and_return_json(json_path)
    embedding_dict = {}
    for key in caption_dict.keys():
        captions = caption_dict[key]
        embedding_dict[key] = sentence_feature_extractor(captions)
    with h5py.File(save_name,'w') as f1:
        if metadata_source is not None:
            with h5py.File(metadata_source, 'r') as f2:
                copy_h5(f2,f1)
        for key in embedding_dict.keys():
            f1.create_dataset(f'{key}/text_features',data = embedding_dict[key])


def run_text_feature_extractor_hf(json_path,hf_model,processor,save_name,target_key,processor_kwargs,device = 'cpu',metadata_source = None):
    hf_model.to(device)
    sentence_feature_extractor = HFTextWrapper(hf_model,processor,target_key)
    caption_dict = open_and_return_json(json_path)
    embedding_dict = {}
    processor_kwargs = processor_kwargs
    for key in caption_dict.keys():
        captions = caption_dict[key]
        embedding_dict[key] = sentence_feature_extractor(captions,processor_kwargs=processor_kwargs).numpy()
    with h5py.File(save_name,'w') as f1:
        if metadata_source is not None:
            with h5py.File(metadata_source, 'r') as f2:
                copy_h5(f2,f1)
        for key in embedding_dict.keys():
            f1.create_dataset(f'{key}/text_features',data = embedding_dict[key])

    
    


if __name__ == "__main__":
    json_path = 'Captioners/test_summe_2.json'
    model_name = "google/siglip2-base-patch16-224"
    save_name = 'summe_siglip_text.h5'
    processor_kwargs = {'return_tensors':'pt','padding':True}
    processor = AutoProcessor.from_pretrained(model_name)
    model = SiglipTextModel.from_pretrained(model_name)
    run_text_feature_extractor_hf(json_path,model,processor,save_name,'pooler_output',processor_kwargs,device = 'cpu')


