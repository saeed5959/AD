import os
from tqdm import tqdm
import ants

final_folder = "./FINAL_CN"
template_brain_path = "./template_mni/MNI152_T1_1.5mm_brain.nii.gz"

#apply n4 filter just for mri files
for subject in tqdm(os.listdir(final_folder)):
    subject_path = os.path.join(final_folder, subject)
    for name in os.listdir(subject_path):
        if name.endswith("_mri_n4_brain.nii.gz"):
            mri_path = os.path.join(subject_path, name)

    fixed = ants.image_read(template_brain_path)
    moving = ants.image_read(mri_path)

    reg = ants.registration(
        fixed, moving,
        type_of_transform="SyN",
        outprefix=os.path.join(subject_path, subject + "_mri2mni_"),
    )
