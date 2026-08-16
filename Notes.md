

# Structing the Repo


1. A single lightning module if the models are optimized with a single loss function
2. For models with custom loss functions, make their own lightning module 



# Current notes

Tests needs to be implemented
The dataset creation needs to be carried out
VideoXUM videos need to be found (completed)


# Main flow and challenges

## Different experiments 
1. Different feature evaluation across multiple models and datasets:
    Visual:
    1. GoogleNet (original for many papers)
    2. ResNet
    3. Clip
    4. Siglipv2
    Textual:
    1. DistilBERT
    2. Qwen
    3. Misc others 
    Auditory:
    1. AST
2. Different evaluation strategies:
    1. Multiple seeds
    2. Multiple dataset configurations 
    3. Different evaluation conditions
3. Model type eval
    1. Supervised
        1. Uni-Modal
        2. Multi-modal
    2. Un-Supervised
    3. Self-Supervised
    4. LLM
4. Different optimization techniques:
    1. Learn from the GT created by averaging.
    2. Learn from all GT simultaneously.(A.K.A VideoXum Methods)


## Challenges 

1. VideoXUM: Processing the 
2. MrHIsum: Getting youtube videos without being sued

# Where does Mathieu come in?

1. Developing the nn.module for various models:
    1. Return model prediction (NxD vector)
    2. Take dictionary input in the model
    3. Alter the lightning module for the specific model
    4. Add a dataset object which returns the batch in the format you require 
2. Some pre-processing which isn't visual:
    1. Multiple types of captioners
    2. Feature extraction for text
3. Running modules on different python versions:
    1. For shot boundary detection
    2. Miscellaneous task

