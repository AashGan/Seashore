def refine_caption(processor,generated_captions,processor_kwargs= {}):
  new_message = f"""You are provided with automatically generated captions from a video.
  Please do the following:
  Refine the cations: Remove unnecessary formatting details, such as **title cards**.
  Keep important and relevant details, especially details unique to the description, such as important people, places logos, etc.
  Remove repeating information and timestamps/timecodes(i.e 00:00,)
    Keep the final captions concise,reducing each sentence to be 15 words long
  Rephrase sentences like: "The video shows" into something that describes the scene
  Return only the rewritten captions
  Here are the captions you must rewrite: {generated_captions}"""
  content = [{"type":"text","text":new_message}]
  batch_message = [{"role":"user","content":content}]
  messages = processor.apply_chat_template(batch_message,
                                          tokenize=True,
                                          return_dict=True,
                                          return_tensors="pt",
                                          padding=True,
                                          add_generation_prompt=True, **processor_kwargs)
  input_len = messages["input_ids"].shape[-1]
  return messages,input_len

def summarize_video_and_transcription(processor,generated_captions,transcription,processor_kwargs= {}):
    content = [{'type':'text','text':f" Video Captions: {generated_captions}"},{"type":"text","text":f"Audio Transcription: {transcription}"}]
    new_message = 'You are provided with automatically generated video captions and audio transcriptions. Compile and write a concise summary of the video considering the information of both the video and audio captions'
    content.append({"type":"text","text":new_message})
    batch_message = [{"role":"user","content":content}]
    messages = processor.apply_chat_template(batch_message,
                                            tokenize=True,
                                            return_dict=True,
                                            return_tensors="pt",
                                            padding=True,
                                            add_generation_prompt=True,**processor_kwargs)
    input_len = messages["input_ids"].shape[-1]
    return messages,input_len