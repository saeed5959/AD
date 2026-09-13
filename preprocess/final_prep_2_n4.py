import os
from tqdm import tqdm
import ants

final_folder = "./FINAL_CN"

#apply n4 filter just for mri files
for subject in tqdm(os.listdir(final_folder)):
    subject_path = os.path.join(final_folder, subject)
    for name in os.listdir(subject_path):
        if "mri" in name:
            mri_path = os.path.join(subject_path, name)
 
    img = ants.image_read(mri_path)
    mask = ants.get_mask(img, cleanup=2)
    out = ants.n4_bias_field_correction(img, mask=mask, shrink_factor=4,
                                        convergence={"iters": [50]*4, "tol": 1e-7})
    ants.image_write(out, os.path.join(subject_path, subject + "_mri_n4.nii.gz"))