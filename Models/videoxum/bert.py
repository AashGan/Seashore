from transformers import BertTokenizer, BertConfig, BertLMHeadModel
import torch
def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'}) # add bos_token for prompt_generate
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']}) # add enc_token for image features
    tokenizer.enc_token_id = tokenizer.convert_tokens_to_ids("[ENC]")
    return tokenizer
def load_checkpoint(model, filename):

  checkpoint = torch.load(filename, map_location='cpu')
  state_dict = checkpoint['model']
  msg = model.load_state_dict(state_dict, strict=False)
  print(f'load checkpoint from {filename}')
  return model, msg

def load_blip_pretrained_checkpoint(model, filename):
  """ This loads the pre-trained checkpoint for BERT from BLIP
  """
  checkpoint = torch.load(
        filename,
        map_location="cpu",
    )

  state_dict = checkpoint["model"]

    # Remove the VTSum/BLIP text_decoder prefix.
  bert_state_dict = {
        key.replace("text_decoder.", "", 1): value
        for key, value in state_dict.items()
        if key.startswith("text_decoder.")
    }

  msg = model.load_state_dict(
        bert_state_dict,
        strict=False,
    )

  print(f"Loaded BERT weights from {filename}")
  print("Missing keys:", msg.missing_keys)
  return model

def create_bert_and_tokenizer(filename,config_path):
  """ This should return the BERTLM model, and also loads the weights from the filename
  """
  tokenizer = init_tokenizer()
  med_config = BertConfig.from_json_file(config_path)
  med_config.is_decoder = True  # Add this line to set is_decoder to True
  model = BertLMHeadModel(config=med_config)
  model = load_blip_pretrained_checkpoint(model, filename)
  return model, tokenizer


if __name__ == "__main__":
   path_to_pretrained_model = '/content/model_base_capfilt_large.pth'
   path_to_config = '/content/med_config.json'
   vmodel,tokenizer = create_bert_and_tokenizer(path_to_pretrained_model,path_to_config)