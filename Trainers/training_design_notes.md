# Notes

1. We always assume validation steps have a batch size of 1, as most evaluation is done in this manner.
2. The model outputs should be a dictionary and we parse the saliency scores in the validaton step of data-loader