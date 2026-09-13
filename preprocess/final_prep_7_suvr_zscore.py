import os
from tqdm import tqdm
import ants
import numpy as np
from scipy.ndimage import binary_erosion

final_folder = "./FINAL_CN"
template_brain_mask_path = "./template_mni/MNI152_T1_1.5mm_brain_mask.nii.gz"
template_suvr_pons = "template_suvr/pons_mask_1.5mm.nii.gz"

for subject in tqdm(os.listdir(final_folder)):
    subject_path = os.path.join(final_folder, subject)

    pet    = ants.image_read(os.path.join(subject_path, subject + "_pet_mni.nii.gz")).numpy()
    t1     = ants.image_read(os.path.join(subject_path, subject + "_mri_mni.nii.gz")).numpy()
    brain  = ants.image_read(template_brain_mask_path).numpy() > 0.5
    shared = ants.image_read(os.path.join(subject_path, subject + "_shared_mask.nii.gz")).numpy() > 0.5      # brain & PET FOV
    pons   = ants.image_read(template_suvr_pons).numpy() > 0.5

    # erode 2 voxels -- avoids partial-volume bleed from adjacent tissue
    ref = binary_erosion(pons, iterations=2) & shared

    er = binary_erosion(pons, iterations=2)
    print(f"pons after erosion: {er.sum()} vox ≈ {er.sum() * 3.375e-3:.1f} mL")

    assert ref.sum() > 50, f"pons ROI too small after erosion/FOV: {ref.sum()} voxels"
    assert (binary_erosion(pons, iterations=2) & shared).sum() / binary_erosion(pons, iterations=2).sum() > 0.95, \
        "pons partly outside PET FOV -- SUVR denominator unreliable"

    suvr = pet / pet[ref].mean()
    suvr = suvr * shared                       # zero outside valid region

    t1_z = np.zeros_like(t1)
    t1_z[brain] = (t1[brain] - t1[brain].mean()) / t1[brain].std()

    ref_img = ants.image_read(os.path.join(subject_path, subject + "_pet_mni.nii.gz"))   # for geometry
    ants.image_write(ref_img.new_image_like(suvr.astype("float32")), os.path.join(subject_path, subject + "_pet_suvr.nii.gz"))
    ants.image_write(ref_img.new_image_like(t1_z.astype("float32")), os.path.join(subject_path, subject + "_mri_zscore.nii.gz"))