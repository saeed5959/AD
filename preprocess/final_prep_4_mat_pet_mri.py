import os
from tqdm import tqdm
import ants

final_folder = "./FINAL_CN"

for subject in tqdm(os.listdir(final_folder)):
    subject_path = os.path.join(final_folder, subject)
    for name in os.listdir(subject_path):
        if name.endswith("_mri_n4_brain.nii.gz"):
            mri_path = os.path.join(subject_path, name)

        elif name.endswith("pet.nii.gz"):
             pet_path = os.path.join(subject_path, name)

    fixed = ants.image_read(mri_path)
    moving = ants.image_read(pet_path)

    reg = ants.registration(
        fixed, moving,
        type_of_transform="Rigid",
        aff_metric="mattes",
        outprefix=os.path.join(subject_path, subject + "_pet2mri_"),
    )
