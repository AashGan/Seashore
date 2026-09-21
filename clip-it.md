# How it works
N = 100
D_frames = 1024
D_text = 768
N_sent = 10
inputs_to_model = np.array([100,1024])
text_input_to_model = np.array([10,768])
W_q = np.array([1024,768]) # Projecting the input vector to the same space as the text
query_vector = W_q*inputs_to_model; np.array([100,768]) 
W_k = np.array([768,768]) # Projecting the input vector to the same space as the video input
key_vector = W_k*text_inputs to the model; np.array([10,768])
logits = q*k^T/d^1/2 # [100,10]
attention_weights = softmax(logits) #[100,10]
W_v = np.array([768,768])
value_vector = W_v*text_inputs # np.array([10,768])
attended_frames = attention*value_vector # np.array([100,768])
