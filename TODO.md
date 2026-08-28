# TODOs per Section

## Models


### UniModal
- [ ] PGLSum
- [ ] VasNet
- [ ] Generic Transformer

### MultiModal
- [ ] CLIP-IT

## Config

### Data

- [ ] Data parameters
- [ ] Batch size

### Model
- [ ] Function to generate config files 



## Data
### Datasets
- [ ] Migrate datasets (based on repo) to specifc 
### Pre Processing

- [ ] Creation of H5 Datasets
 - [ ] TVSUM
 - [ ] SumMe
 - [ ] VideoXUM
- [ ] Integration of additional dataset
    - [ ] MrHiSum
    - [ ] TVSum's additional labels 
- [ ] Unify the names of the ground truth for all the h5 files (gtscore)
- [ ] Integrate additional shot-boundary detection algorithms
 - [ ] GPU KTS
 - [ ] Change point detectors
 - [ ] TransNetV2 
- [ ] Run the scripts for the shot-boundary detection

### Loaders
- [ ] Integrate existing loader to the repo
- [ ] Integrate text-based data-loader
- [ ] Integrate multi-modal loaders (assuming keys)


## Visualization

- [ ] Add Summary creation script 
- [ ] Add MoviePy automatic Editor
## Scripts

### H5 Dataset Creation

- [ ] Dataset Creation Script 
### Training

- [ ] Develop a torch lightning train script
    - [ ] Add the WandB result tracking
    - [ ] Adding sweeps for hyper-parameter searches

### Evaluation

- [ ] Add evaluation script for inference 
- [ ] Add F1 score evaluation 
- [ ] Add other metrics from the literature 


### Testing
- [ ] Start implementing Pytests for different functions
     - [ ] Evaluation Functions
     - [ ] Dataloaders
     - [ ] Model Predictions 
     - [ ] Feature Extractors
     - [ ] 


# Task List: Mathieu

- [ ] Implement the paper: CLIP-IT!
    - [ ] Model architecture (as torch module)
    - [ ] Loss functions (as separate function)
- [ ] Add batch collate function for sequential inputs (basically something that pads in the all the input vectors to the same length)
- [ ] Go through Otani's paper: Rethinking evaluation in Video Summarization
- [ ] Loss function
  - [ ] Consolidate the loss functions found across the literature
  - [ ] Create a class that composes the multiple losses allowing them to be weighed (when possible)
  - [ ] Implement the functions and add them to Models/losses/loss_function.py
  - [ ] Mark the ones that are too complex
- [ ] Captioners
    - [ ] Read through three paper (SD-VSUM, LLMVS, PDL ) and describe how they caption the videos (if the core details are mentioned)
    - [ ] Implement the captioners (extending with SmolVLM)
    - [ ] Mark the ones which can't be tested due to lack of details or compute issues 