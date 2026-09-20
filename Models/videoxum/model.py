import torch
import torch.nn as nn
from .attention import create_tt,LocalAttenModule
from .bert import create_bert_and_tokenizer
class VTSum_BLIP_TT(nn.Module):
    def __init__(self,
                 dim,
                 tt_depth=1,
                 file_path = 'model_base_capfilt_large.pth',
                 med_config='med_config.json',
                 prompt='a video of ',
                 max_text_length=128):
        """ VTSum_BLIP model with Temporal Transformer (TT)
        But without loss computed in the nn.Module
        Also Saliency scores returns sigmoid output
        And added masking strategy
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            vit (str): model size of vision transformer
            prompt (str): the prompt for the decoder

        """
        super().__init__()
        self.vision_width = dim
        self.position_embeddings = nn.Embedding(512, self.vision_width)
        self.tt, tt_width = create_tt(
            self.vision_width, depth=tt_depth)
        assert tt_width == self.vision_width
        self.vsum_head = nn.Sequential(nn.Linear(self.vision_width, 1))
        nn.init.normal_(self.vsum_head[-1].weight, std=.02)
        # tsum decoder
        self.text_decoder,self.tokenizer= create_bert_and_tokenizer(file_path,med_config)

        self.max_text_length = max_text_length
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1


    def _interpolate_pos_embed(self, pos_embed, video_length):
        if video_length > 512:
            pos_embed = torch.nn.functional.interpolate(
                pos_embed[:, None, None, :].permute(1, 3, 0, 2),
                size=(video_length, 1), mode='bicubic', align_corners=False)[0, :, :, 0].permute(1, 0)
        else:
            pos_embed = pos_embed[:video_length]
        return pos_embed

    def forward(self,video_embeddings,tsum_labels,video_mask = None):
      """ Passing TSUM labels j
      """
      position_embeddings = self._interpolate_pos_embed(self.position_embeddings.weight, video_embeddings.size(1))
      video_embeddings = video_embeddings + position_embeddings

      # temporal transformer
      video_mask_expand = None
      if video_mask is not None:
        video_mask_expand = video_mask[:, None, None, :]

      video_embeddings = self.tt(video_embeddings, video_mask_expand)
      video_embeddings = video_embeddings + position_embeddings
      if video_mask is not None:
        del video_mask_expand
      # vsum decoder
      saliency_scores = self.vsum_head(video_embeddings)
      text = self.tokenizer(
            tsum_labels, padding='longest',
            truncation=True, max_length=self.max_text_length,
            return_tensors="pt").to(video_embeddings.device)

      text.input_ids[:, 0] = self.tokenizer.bos_token_id

      decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)
      decoder_targets[:, :self.prompt_length] = -100
      if video_mask is not None:
        video_mask = video_mask.long()
      else:
          video_mask = torch.ones_like(saliency_scores).to(video_embeddings.device)
      decoder_output = self.text_decoder(
          input_ids=text.input_ids,
          attention_mask=text.attention_mask,
          encoder_hidden_states=video_embeddings,
          encoder_attention_mask=video_mask,
          labels=decoder_targets,
          return_dict=True)
      return saliency_scores.sigmoid(),decoder_output

    def generate(self, video_embeddings, video_mask=None, sample=False,
                 num_beams=3, max_length=30, min_length=10, top_p=0.9,
                 repetition_penalty=1.0):
        device = video_embeddings.device
        batch_size = video_embeddings.size(0)

        # interpolate position embeddings
        position_embeddings = self._interpolate_pos_embed(self.position_embeddings.weight, video_embeddings.size(1))
        video_embeddings = video_embeddings + position_embeddings

        # temporal transformer
        video_mask_expand = None
        if video_mask is not None:
          video_mask_expand = video_mask[:, None, None, :]
        video_embeddings = self.tt(video_embeddings, video_mask_expand)
        video_embeddings = video_embeddings + position_embeddings
        del video_mask_expand
        # vsum decoder
        saliency_scores = self.vsum_head(video_embeddings)

        # tsum decoder
        if not sample:
            # video_embeddings = video_embeddings.repeat_interleave(num_beams, dim=0)
            pass
        if video_mask is not None:
          video_mask = video_mask.long()
        else:
          video_mask = torch.ones_like(saliency_scores).squeeze(-1)
        
        model_kwargs = {"encoder_hidden_states": video_embeddings,
                        "encoder_attention_mask": video_mask} # Removed repeat interleave
        print(model_kwargs["encoder_hidden_states"].shape)
        print(model_kwargs["encoder_attention_mask"].shape)
        prompt = [self.prompt] * batch_size
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(video_embeddings.device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        if sample:
            # nucleus sampling
            outputs = self.text_decoder.generate(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                top_p=top_p,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.1,
                use_cache=False, # Added this line
                **model_kwargs)
        else:
            # beam search
            outputs = self.text_decoder.generate(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                use_cache=False, # Added this line
                **model_kwargs)

        tsum_labels = []
        for output in outputs:
            tsum_label = self.tokenizer.decode(output, skip_special_tokens=True)
            tsum_labels.append(tsum_label[len(self.prompt):])
        return saliency_scores.sigmoid(), tsum_labels

class VTSum_BLIP_TT_CA(VTSum_BLIP_TT):
  def __init__(self,
                kernel_size=5,
                *args, **kwargs):
    super(VTSum_BLIP_TT_CA, self).__init__(*args, **kwargs)
    """ 
    Altered original videoxum code to dynamically add local attention mask
    """
    self.kernel_size = kernel_size
    self.local_atten_head = LocalAttenModule(
        dim=self.vision_width)
  def forward(self, video_embeddings, tsum_labels,video_mask = None):
    position_embeddings = self._interpolate_pos_embed(self.position_embeddings.weight, video_embeddings.size(1))
    video_embeddings = video_embeddings + position_embeddings

    # temporal transformer
    video_mask_expand = None
    if video_mask is not None:
      video_mask_expand = video_mask[:, None, None, :]
    video_embeddings = self.tt(video_embeddings, video_mask_expand)
    video_embeddings = video_embeddings + position_embeddings
    del video_mask_expand

    # local attention
    video_len = video_embeddings.size(1)
    idx = torch.arange(video_len)
    local_mask = (
        (idx[:, None] - idx[None, :]).abs()
        <= self.kernel_size
    ).to(video_embeddings.device)
    if video_mask is None:
      video_mask = torch.ones(video_embeddings.shape[:2], dtype=torch.bool)

    combined_mask = (
    video_mask[:, None, None, :]
    & local_mask[None, None, :, :]
    )

    video_embeddings = self.local_atten_head(video_embeddings, combined_mask)

    # vsum decoder
    saliency_scores = self.vsum_head(video_embeddings)
    num_frame = saliency_scores.size(-1)
    saliency_scores = saliency_scores.reshape(-1, num_frame)
    # text decoder
    text = self.tokenizer(
        tsum_labels, padding='longest',
        truncation=True, max_length=self.max_text_length,
        return_tensors="pt").to(video_embeddings.device)

    text.input_ids[:, 0] = self.tokenizer.bos_token_id

    decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)
    decoder_targets[:, :self.prompt_length] = -100

    decoder_output = self.text_decoder(
        input_ids=text.input_ids,
        attention_mask=text.attention_mask,
        encoder_hidden_states=video_embeddings,
        encoder_attention_mask=video_mask,
        labels=decoder_targets,
        return_dict=True)
    return saliency_scores.sigmoid(),decoder_output
  def generate(self, video_embeddings, video_mask=None, sample=False,
                 num_beams=3, max_length=30, min_length=10, top_p=0.9,
                 repetition_penalty=1.0 ):
        device = video_embeddings.device
        batch_size = video_embeddings.size(0)

        # interpolate position embeddings
        position_embeddings = self._interpolate_pos_embed(self.position_embeddings.weight, video_embeddings.size(1))
        video_embeddings = video_embeddings + position_embeddings

        # temporal transformer
        video_mask_expand = None
        if video_mask is not None:
          video_mask_expand = video_mask[:, None, None, :]
        video_embeddings = self.tt(video_embeddings, video_mask_expand)
        video_embeddings = video_embeddings + position_embeddings
        del video_mask_expand
        if video_mask is None:
          video_mask = torch.ones(video_embeddings.shape[:2], dtype=torch.bool)
        video_len = video_embeddings.size(1)
        idx = torch.arange(video_len)
        local_mask = (
                (idx[:, None] - idx[None, :]).abs()
                <= self.kernel_size
            ).to(video_embeddings.device)
        combined_mask = (
                        video_mask[:, None, None, :]
                        & local_mask[None, None, :, :]
                        )
        video_embeddings = self.local_atten_head(video_embeddings, combined_mask)
        saliency_scores = self.vsum_head(video_embeddings)
        model_kwargs = {"encoder_hidden_states": video_embeddings,
                        "encoder_attention_mask": video_mask}
        print(model_kwargs["encoder_hidden_states"].shape)
        print(model_kwargs["encoder_attention_mask"].shape)
        prompt = [self.prompt] * batch_size
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(video_embeddings.device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]
        if sample:
            # nucleus sampling
            outputs = self.text_decoder.generate(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                top_p=top_p,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.1,
                use_cache=False, # Added this line
                **model_kwargs)
        else:
            # beam search
            outputs = self.text_decoder.generate(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                use_cache=False, # Added this line
                **model_kwargs)

        tsum_labels = []
        for output in outputs:
            tsum_label = self.tokenizer.decode(output, skip_special_tokens=True)
            tsum_labels.append(tsum_label[len(self.prompt):])
        return saliency_scores.sigmoid(), tsum_labels
