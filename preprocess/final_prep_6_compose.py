import os
from tqdm import tqdm
import ants
import numpy as np


final_folder = "./FINAL_CN"
template_brain_path = "./template_mni/MNI152_T1_1.5mm_brain.nii.gz"
template_brain_mask_path = "./template_mni/MNI152_T1_1.5mm_brain_mask.nii.gz"

#apply n4 filter just for mri files
for subject in tqdm(os.listdir(final_folder)):
    subject_path = os.path.join(final_folder, subject)
    for name in os.listdir(subject_path):
        if name.endswith("_mri_n4_brain.nii.gz"):
            mri_path = os.path.join(subject_path, name)

        elif name.endswith("pet.nii.gz"):
             pet_path = os.path.join(subject_path, name)

    mni  = ants.image_read(template_brain_path)
    pet  = ants.image_read(pet_path)
    t1   = ants.image_read(mri_path)

    # right-to-left: rigid applied first, then affine, then warp
    t1_mni_1Warp = os.path.join(subject_path, subject + "_mri2mni_1Warp.nii.gz")
    t1_mni_GAF = os.path.join(subject_path, subject + "_mri2mni_0GenericAffine.mat")
    pet_t1_mat = os.path.join(subject_path, subject + "_pet2mri_0GenericAffine.mat")
    pet_tlist = [t1_mni_1Warp, t1_mni_GAF, pet_t1_mat]
    t1_tlist  = [t1_mni_1Warp, t1_mni_GAF]

    pet_mni = ants.apply_transforms(mni, pet, pet_tlist, interpolator="lanczosWindowedSinc")
    t1_mni  = ants.apply_transforms(mni, t1,  t1_tlist,  interpolator="lanczosWindowedSinc")

    # FOV validity — where PET actually has data
    ones = pet.new_image_like(np.ones(pet.shape, dtype="float32"))
    fov  = ants.apply_transforms(mni, ones, pet_tlist, interpolator="nearestNeighbor").numpy() > 0.5

    brain  = ants.image_read(template_brain_mask_path).numpy() > 0.5
    shared = brain & fov

    pet_arr = np.clip(pet_mni.numpy(), 0, None) * shared    # clip: sinc undershoots below 0
    t1_arr  = t1_mni.numpy() * brain

    ants.image_write(pet_mni.new_image_like(pet_arr.astype("float32")), os.path.join(subject_path, subject + "_pet_mni.nii.gz"))
    ants.image_write(t1_mni.new_image_like(t1_arr.astype("float32")),  os.path.join(subject_path, subject + "_mri_mni.nii.gz"))
    ants.image_write(pet_mni.new_image_like(shared.astype("float32")), os.path.join(subject_path, subject + "_shared_mask.nii.gz"))

