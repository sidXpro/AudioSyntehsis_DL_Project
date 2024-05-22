# Evaluate all existing generated folder
python3 audioldm_train/eval.py --log_path all

# Evaluate only a specific experiment folder
# python3 eval.py --log_path <path-to-the-experiment-folder> # e.g., python3 eval.py -l log/2023_08_23_reproduce
# python3 audioldm_train/eval.py -l log/latent_diffusion/2023_08_23_reproduce_audioldm