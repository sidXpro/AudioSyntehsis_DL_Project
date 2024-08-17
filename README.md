Secured 4th Global Rank in Dcase 2024 task7 challenge " Sound Scene Synthesis"

Results :  https://dcase.community/challenge2024/task-sound-scene-synthesis-results

We have made our custom dataset by mixing various audio dataset. Some audios are put as foreground while some are put in background.

for. ex. : baby crying in train
baby crying has been put in foreground and train sound out in background 
using following relation:

new_sound = 2 * foreground sound + 0.5*background sound 

here, amplitude for the background sound is scaled down while the amplitude for the foreground sound is scaled up

for mixing ESC50, AudioCaps and Acoustic Scene Dataset is used

Follwing is the implementation workflow:

1) Download ESC50 dataset and place it in a folder
2) Download DCASE Acoustic Scene Dataset and place it in a separate folder
3) Run the jupyer notebook script named 'Custom_Dataset_New.ipynb' to mix audios from the two folders
4) Follow the steps in the notebook to create the json file for training the audioldm
5) The dataset structure is similar to AudioCaps dataset, which can be downloaded from here "https://drive.google.com/file/d/16J1CVu7EZPD_22FxitZ0TpOd__FwzOmx/view?usp=drive_link"
6) The custom data is placed inside DATA2 folder inside audioldm
7) To train the audioldm, follow the readme file inside the audioldm folder
8) To train the vq-vae model for foley sound generation, follow the readme inside the foley folder
9) Moreover a Web-UI for generating music is created based on facial expression, age and dominant-colours in the photo. 
10) Open the Web-UI folder and follow the readme to replicate the project UI
