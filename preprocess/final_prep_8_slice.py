import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm

final_folder = "./FINAL_AD"
out_root     = "../AD_train/preprocess/AD"
SIZE         = 224
PLANES = ["axial", "coronal", "sagittal"]

CENTERS_MRI = [(-20,0,20),(-10,-25,-40),(-10,-25,-40)]
DELTAS_MRI = [0, 4, -4, 8, -8, 14, -14]

CENTERS_PET = [(20, 32, 42), (-40, -55, -70), (-6, -25, -45)]
DELTAS_PET  = [0, 6, -6, 10, -10, 16, -16]


def mm_to_vox(affine, x, y, z):
    return np.round(np.linalg.inv(affine) @ np.array([x, y, z, 1.0]))[:3].astype(int)


def resize2d(sl, size=SIZE):
    return zoom(sl, (size / sl.shape[0], size / sl.shape[1]), order=1).astype("float32")


for subject in tqdm(sorted(os.listdir(final_folder))):
    sp = os.path.join(final_folder, subject)

    sub_out = os.path.join(out_root, subject)
    os.makedirs(sub_out, exist_ok=True)

    pet_img = nib.load(os.path.join(sp, f"{subject}_pet_suvr.nii.gz"))
    mri_img = nib.load(os.path.join(sp, f"{subject}_mri_zscore.nii.gz"))
    pet = pet_img.get_fdata().astype("float32")
    mri = mri_img.get_fdata().astype("float32")

    for center in CENTERS_MRI[0]:
        for delta in DELTAS_MRI:
            z_mri = mm_to_vox(pet_img.affine, 0, 0, center+delta)[2]     # axial   — posterior cingulate / precuneus

            np.save(os.path.join(sub_out, f"{subject}_mri_axial_{center}_{delta}.npy"), resize2d(mri[:,:,z_mri]))

    for center in CENTERS_MRI[1]:
        for delta in DELTAS_MRI:
            y_mri = mm_to_vox(pet_img.affine, 0, center+delta, 0)[1]    # coronal — hippocampal body

            np.save(os.path.join(sub_out, f"{subject}_mri_coronal_{center}_{delta}.npy"), resize2d(mri[:,y_mri,:]))

    for center in CENTERS_MRI[2]:
        for delta in DELTAS_MRI:
            x_mri = mm_to_vox(pet_img.affine, center+delta, 0, 0)[0]     # sagittal — posterior cingulate in profile

            np.save(os.path.join(sub_out, f"{subject}_mri_sagittal_{center}_{delta}.npy"), resize2d(mri[x_mri,:,:]))


for subject in tqdm(sorted(os.listdir(final_folder))):
    sp = os.path.join(final_folder, subject)

    sub_out = os.path.join(out_root, subject)
    os.makedirs(sub_out, exist_ok=True)

    pet_img = nib.load(os.path.join(sp, f"{subject}_pet_suvr.nii.gz"))
    mri_img = nib.load(os.path.join(sp, f"{subject}_mri_zscore.nii.gz"))
    pet = pet_img.get_fdata().astype("float32")
    mri = mri_img.get_fdata().astype("float32")

    for center in CENTERS_PET[0]:
        for delta in DELTAS_PET:
            z_pet = mm_to_vox(pet_img.affine, 0, 0, center+delta)[2]     # axial   — posterior cingulate / precuneus

            np.save(os.path.join(sub_out, f"{subject}_pet_axial_{center}_{delta}.npy"), resize2d(pet[:,:,z_pet]))

    for center in CENTERS_PET[1]:
        for delta in DELTAS_PET:
            y_pet = mm_to_vox(pet_img.affine, 0, center+delta, 0)[1]    # coronal — hippocampal body

            np.save(os.path.join(sub_out, f"{subject}_pet_coronal_{center}_{delta}.npy"), resize2d(pet[:,y_pet,:]))

    for center in CENTERS_PET[2]:
        for delta in DELTAS_PET:
            x_pet = mm_to_vox(pet_img.affine, center+delta, 0, 0)[0]     # sagittal — posterior cingulate in profile

            np.save(os.path.join(sub_out, f"{subject}_pet_sagittal_{center}_{delta}.npy"), resize2d(pet[x_pet,:,:]))
