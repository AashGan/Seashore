

# Structing the Repo


1. A single lightning module if the models are optimized with a single loss function
2. For models with custom loss functions, make their own lightning module 



# Current notes

Tests needs to be implemented
The dataset creation needs to be carried out
VideoXUM videos need to be found



For multi-data loading and eval
The json would contain the dataset and video index per video.
Therefore, the config needs to have the eval type per dataset

An important point: For evaluation, when I do the upsampling, it might require some changes if the meta-parameters for sampling specific indices, shot boundaires change.
