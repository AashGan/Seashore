def refine_caption(processor,generated_captions):
  content = [{'type':'text','text':generated_captions}]
  new_message = 'You are provided with automatically generated captions. Please refine these captions by removing redundant information and including all details while ensuring the final description is concise. Return only the reformulated captions'
  content.append({"type":"text","text":new_message})
  batch_message = [{"role":"user","content":content}]
  messages = processor.apply_chat_template(batch_message,
                                          tokenize=True,
                                          return_dict=True,
                                          return_tensors="pt",
                                          padding=True,
                                          add_generation_prompt=True,)
  input_len = messages["input_ids"].shape[-1]
  return messages,input_len

def summarize_video_and_transcription(processor,generated_captions,transcription):
    content = [{'type':'text','text':f" Video Captions: {generated_captions}"},{"type":"text","text":f"Audio Transcription: {transcription}"}]
    new_message = 'You are provided with automatically generated video captions and audio transcriptions. Compile and write a concise summary of the video considering the information of both the video and audio captions'
    content.append({"type":"text","text":new_message})
    batch_message = [{"role":"user","content":content}]
    messages = processor.apply_chat_template(batch_message,
                                            tokenize=True,
                                            return_dict=True,
                                            return_tensors="pt",
                                            padding=True,
                                            add_generation_prompt=True,)
    input_len = messages["input_ids"].shape[-1]
    return messages,input_len