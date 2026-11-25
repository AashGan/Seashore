
import torch
from Data.Preprocessing import SentenceTransformerWrapper,HFTextWrapper,InstructionTunerFeatures
import numpy as np
import h5py
import json
from transformers import AutoProcessor,SiglipTextModel,AutoModelForMultimodalLM
from Utils.h5_utils import copy_h5
from Data.Preprocessing.system_instructions import *
def open_and_return_json(json_path):
    with open(json_path,'r') as f:
        return json.load(f)

def run_text_feature_extractor_base(json_path,sentence_transformer_name
                                    ,save_name,device = 'cpu',metadata_source = None):
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

    
    
def run_llmvs_feature_extractor(all_caption_path,hf_model,processor,save_name,WINDOW_SIZE=7):
    HALF_WINDOW = WINDOW_SIZE // 2
    instruct_tuner = InstructionTunerFeatures(model = hf_model,processor = processor,
                                              system_instruction=system_instructions,example_text=example)
    captions_all = open_and_return_json(all_caption_path)
    embedding_dict = {}
    positions_dict = {}
    for key in list(captions_all.keys()):
        all_keys_in_video = list(captions_all[key]['captions'].keys())
        all_features = []
        captions = captions_all[key]['captions']
        positions_dict[key] = captions_all[key]['positions']
        print(f'Processing {key} out of {len(captions_all)}')
        for pos, frame_idx in enumerate(captions):
            start = max(0, pos - HALF_WINDOW)
            end = min(len(captions), pos + HALF_WINDOW + 1)
            window_indices = all_keys_in_video[start:end]
            target_idx = pos - start + 1
            window_captions = "\n".join(
                    f"#{int(idx.split('_')[1])+1}: {captions[idx]}"
                    for i, idx in enumerate(window_indices))
            query = (
                    f"Please evaluate the importance score of the central frame "
                    f"#{target_idx+1} in the following {len(window_indices)} frames:\n\n"
                    f"{window_captions}"
                )
            features =instruct_tuner(query)
            all_features.append(features.detach().to('cpu').float().numpy())
        embedding_dict[key] = np.stack(all_features,axis=0)

    with h5py.File(save_name,'w') as f:
        for key in embedding_dict.keys():
            f.create_dataset(f'{key}/text_features',data = embedding_dict[key])
            f.create_dataset(f'{key}/positions',data = positions_dict[key])


if __name__ == "__main__":
    json_path = 'Captioners/refined_captions/refined_complete_tvsum.json'
    model_name = "google/siglip2-base-patch16-224"
    save_name = 'tvsum_siglip_text.h5'
    processor_kwargs = {'return_tensors':'pt','padding':True}
    processor = AutoProcessor.from_pretrained(model_name)
    model = SiglipTextModel.from_pretrained(model_name)
    run_text_feature_extractor_hf(json_path,model,processor,save_name,'pooler_output',processor_kwargs,device = 'cpu')
    json_path = 'Captioners/refined_captions/refined_complete_summe.json'
    model_name = "google/siglip2-base-patch16-224"
    save_name = 'summe_siglip_text.h5'
    run_text_feature_extractor_hf(json_path,model,processor,save_name,'pooler_output',processor_kwargs,device = 'cpu')

    # model_id = "Qwen/Qwen3.5-4B"

    # processor = AutoProcessor.from_pretrained(model_id)

    # model = AutoModelForMultimodalLM.from_pretrained(
    #     model_id,
    #     torch_dtype=torch.bfloat16,
    #     device_map="cuda:0",
    # ).eval()
    # json_path = 'Captioners/per_frame/per_frame_tvsum_with_positions.json'
    # save_name = 'llmvs_qwen_tvsum.h5'
    # run_llmvs_feature_extractor(json_path,model,processor,save_name,7)


