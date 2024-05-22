1) Download ESC50 dataset and place it in a folder
2) Download DCASE Acoustic Scene Dataset and place it in a separate folder
3) run the jupyer notebook script to mix audios from the two folders
4) Follow the steps in the notebook to create the json file for training the audioldm
5) The dataset structure is similar to AudioCaps dataset, which can be downloaded from here "https://drive.google.com/file/d/16J1CVu7EZPD_22FxitZ0TpOd__FwzOmx/view?usp=drive_link"
6) The custom data is placed inside DATA2 folder inside audioldm
7) To train the audioldm, follow the readme file inside the audioldm folder
8) To train the vq-vae model for foley sound generation, follow the readme inside the foley folder
9) Moreover a Web-UI for generating music is created based on facial expression, age and dominant-colours in the photo. 
10) Open the Web-UI folder and follow the readme to replicate the project UI