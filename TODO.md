# TODOs per Section

## Models


### UniModal
- [X] PGLSum
- [X] VasNet
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

- [] Creation of H5 Datasets
 - [X] TVSUM
 - [X] SumMe
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