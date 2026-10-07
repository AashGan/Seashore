# Scripts for sentence level feature extraction
# The approach is largely driven by transformers/sentence transformers for convenience



import torch
from sentence_transformers import SentenceTransformer


class SentenceTransformerWrapper():
    def __init__(self,sentence_transformer_name,device = 'cpu'):

        self.model = SentenceTransformer(sentence_transformer_name)
        self.device = device
        self.model.to(device)

    def __call__(self,input_text):
        """ 
        The call assumes that input_text is structured as a paragraph: "This is something I've written. Written in this manner"
        """
        input_text_batch = input_text.split('.')
        return self.model.encode(input_text_batch.to(self.device))


class HFTextWrapper():
    """ Wrapper class for HF style model, for visually/omni grounded textual extraction"""

    def __init__(self, model,processor,target_key = None): 
        self.model = model # This should either be the model or it should be a wrapper function
        self.processor = processor # Standard HF wrapper 
        self.device =  model.device 
        self.target_key =target_key # This would assume that the model output is a dict or not, would depend on the model implementation
    def __call__(self,input_text:str,**kwargs):
        """
        param: input_text: An input string where each sentence is separated by full stops
        """
        input_text_batch = input_text.split('.')
        processor_kwargs = kwargs.get('processor_kwargs',{})
        processed_inputs = self.processor(text = input_text_batch,**processor_kwargs)
        with torch.no_grad():
            outputs = self.model(**processed_inputs.to(self.device))
            if self.target_key is not None:
                return outputs[self.target_key]
        return outputs  


class InstructionTunerFeatures():
    """ 
    Class for instruction tuner style decoder (LLMVS), assume a Jinja style formatting and input
    """
    def __init__(self,model, processor, system_instruction,example_text=None,embedding_style ="query_output_maxpool", max_tokens = 77):
        self.model = model
        self.processor = processor 
        self.embedding_style =  embedding_style 
        self.system_prompt = self._create_system_message(system_instruction,example_text)
        self.max_tokens = max_tokens
    def _create_system_message(self,system_message,example_text):
        system_messages = [{"role":"system","content":[{"type": "text", "text": system_message}]}]
        if example_text is not None:
            system_message.append({"role":"user","content":[{"type": "text", "text": example_text}]})
        return system_messages

    def _compute_exclusion_length(self,prompt):
        inputs = self.processor.apply_chat_template(prompt,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=False,)
        exclusion_length = inputs["input_ids"].shape[-1]
        del inputs
        return exclusion_length

    def __call__(self, query):
        final_query = self.system_prompt.append({"role":"user","content":[{"type":"text","text":query}]})
        if self.embedding_style == "query_output_maxpool":
            exclusion_length = self._compute_exclusion_length(self.system_prompt)
        elif self.embedding_style == "output_maxpool":
            exclusion_length = None
        final_query = self.processor.apply_chat_template(final_query,tokenize=True,
                return_tensors="pt",
                add_generation_prompt=True,)
        if exclusion_length is None:
            exclusion_length = final_query["input_ids"].shape[-1]
        with torch.inference_mode():
            generated_output = self.model.generate(**final_query.to(self.model.device), max_new_tokens = self.max_tokens, do_sample=False)
        output_embeddings = generated_output.hidden_state






