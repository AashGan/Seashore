This code needs major fixes

1. Adding a function for the raw losses instead of having it as a function: Done 
2. I need to figure out sequence length divisions in case the batch size is uneven: Done
3. This also requires a modification of the batch size : Done
4. Need to make a custom dataloader which samples frames from the video which samples frames
5. Need to make a custom torch lightning script. 