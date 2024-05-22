import numpy as np

# Load the npy file
data = np.load('/storage/siddharath/Foley/AudioLDM-training-finetuning/esc50_custom_caption_with_traffic.npy',allow_pickle=True)

# Print the content
print(data)
