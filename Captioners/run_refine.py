import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
import json
from helpers.text_refiners import refine_caption
from helpers.video_helpers import forward_and_decode
import gc
def run_refine(caption_json,model_name,save_name,max_tokens = 128):
    with open(caption_json,'r') as f:
        caption_dict= json.load(f)

    model_name = model_name
    model = AutoModelForMultimodalLM.from_pretrained( # Changed from AutoModelForVision2Seq
                model_name,
                device_map="auto" # Automatically map model to available devices (GPU if present)
            )
    processor = AutoProcessor.from_pretrained(model_name)

    refined_caption_dict = {}

    for key in list(caption_dict.keys()):
        captions = caption_dict[key]
        if isinstance(captions,dict):
            # This needs to refine captions per annotator
            annotator_dict= {}
            for i in range(len(captions)):
                per_anno_caption = captions[f'Annotator_{i+1}']
                messages,input_lens = refine_caption(processor,per_anno_caption)
                with torch.inference_mode():
                    outputs = forward_and_decode(model,messages,input_lens,processor,max_tokens,True)
                annotator_dict[f'Annotator_{i+1}'] = outputs
            refined_caption_dict[key] = annotator_dict
        else:
            messages,input_lens = refine_caption(processor,captions)
            with torch.inference_mode():
                outputs = forward_and_decode(model,messages,input_lens,processor,max_tokens,True)
            print(outputs)
            refined_caption_dict[key] = outputs
    with open(f'{save_name}.json','w') as f:
        json.dump(refined_caption_dict,f,indent = 2)
    del model
    del processor
    torch.cuda.empty_cache()
    gc.collect()
if __name__ == "__main__":
    model_name = 'google/gemma-4-12B-it'
    caption_json = 'Captioners/whole_video/tvsum_complete_qwen.json'
    save_name = 'refined_complete_tvsum'
    run_refine(caption_json,model_name,save_name)
    caption_json = 'Captioners/whole_video/summe_complete_qwen.json'
    save_name = 'refined_complete_summe'
    run_refine(caption_json,model_name,save_name)