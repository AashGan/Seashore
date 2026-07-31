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
- [ ] Add MoviePy automatic Edito
## Scripts

### Training

- [ ] Develop a torch lightning train script
    - [ ] Add the WandB result tracking
    - [ ] Adding sweeps for hyper-parameter searches

### Evaluation

- [ ] Add evaluation script for inference 