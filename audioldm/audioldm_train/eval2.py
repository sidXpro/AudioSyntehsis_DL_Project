import os
import yaml
import torch
from audioldm_eval import EvaluationHelper

SAMPLE_RATE = 16000
device = torch.device(f"cuda")
evaluator = EvaluationHelper(SAMPLE_RATE, device)

# test_audio_data_folder = os.path.join(test_audio_path, test_dataset)

print("-------------------------------dog-----------------------------")
folder1 = "/storage/siddharath/Foley/AudioLDM-training-finetuning/log1/latent_diffusion/2023_08_23_reproduce_audioldm/audioldm_original1/folley/dog"
test_audio_data_folder = "/storage/siddharath/Foley/AudioLDM-training-finetuning/log2/testset_data/folley"

evaluator.main(folder1, test_audio_data_folder)

print("--------------------------------footstep---------------------------------------")
folder2 = "/storage/siddharath/Foley/AudioLDM-training-finetuning/log1/latent_diffusion/2023_08_23_reproduce_audioldm/audioldm_original1/folley/footstep"
test_audio_data_folder = "/storage/siddharath/Foley/AudioLDM-training-finetuning/log2/testset_data/folley"

evaluator.main(folder2, test_audio_data_folder)

print("-------------------------------------gunshot-------------------------------")
folder3 = "/storage/siddharath/Foley/AudioLDM-training-finetuning/log1/latent_diffusion/2023_08_23_reproduce_audioldm/audioldm_original1/folley/gunshot"
test_audio_data_folder = "/storage/siddharath/Foley/AudioLDM-training-finetuning/log2/testset_data/folley"

evaluator.main(folder3, test_audio_data_folder)


print("------------------------------keyboard-------------------------")
folder4 = "/storage/siddharath/Foley/AudioLDM-training-finetuning/log1/latent_diffusion/2023_08_23_reproduce_audioldm/audioldm_original1/folley/keyboard"
test_audio_data_folder = "/storage/siddharath/Foley/AudioLDM-training-finetuning/log2/testset_data/folley"

evaluator.main(folder4, test_audio_data_folder)


print(------------------------"motor--------------------------------------")
folder5 = "/storage/siddharath/Foley/AudioLDM-training-finetuning/log1/latent_diffusion/2023_08_23_reproduce_audioldm/audioldm_original1/folley/motor_vehicle"
test_audio_data_folder = "/storage/siddharath/Foley/AudioLDM-training-finetuning/log2/testset_data/folley"

evaluator.main(folder5, test_audio_data_folder)

print("---------------------------------rain----------------------------------")
folder6 = "/storage/siddharath/Foley/AudioLDM-training-finetuning/log1/latent_diffusion/2023_08_23_reproduce_audioldm/audioldm_original1/folley/rain"
test_audio_data_folder = "/storage/siddharath/Foley/AudioLDM-training-finetuning/log2/testset_data/folley"

evaluator.main(folder6, test_audio_data_folder)

print("-------------------------------------sneeze-------------------------")
folder7 = "/storage/siddharath/Foley/AudioLDM-training-finetuning/log1/latent_diffusion/2023_08_23_reproduce_audioldm/audioldm_original1/folley/sneeze"
test_audio_data_folder = "/storage/siddharath/Foley/AudioLDM-training-finetuning/log2/testset_data/folley"

evaluator.main(folder7, test_audio_data_folder)

