# MAAM

## Expectation

input_frames = 50
input_dim =  512
Num_annotators = 5 
number_of_videos = 2
k = iterator over number of videos
l = length of video
output_model = torch.float([2,50,1])
labels_video = torch.float([2,50,5])
expectation_loss (1/$ lamda$ _{j} ) = 1/(2*50)sum _{k}sum{l}(mse(label_video[k,l,j]-output_model[k,l,:])) -> lambda torch.float([5])

maximization_loss = average(dot(lambda,(mse(label_video - output_model.stack(shape_like(labels))))))

maximization_loss.backward()