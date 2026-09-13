import os
from tqdm import tqdm
import subprocess


final_folder = "./FINAL_AD"

#apply n4 filter just for mri files
for subject in tqdm(os.listdir(final_folder)):
    subject_path = os.path.join(final_folder, subject)
    for name in os.listdir(subject_path):
        if name.endswith("_mri_n4.nii.gz"):
            mri_path = os.path.join(subject_path, name)
 
    brain_path = mri_path[:-7] + "_brain.nii.gz"
    subprocess.run(f"hd-bet -i {mri_path} -o {brain_path} --save_bet_mask --disable_tta -device cuda:0", shell=True, check=True)    