import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm
import matplotlib.pyplot as plt


final_folder = "./FINAL_CN"
out_root     = "./preprocess_visualize/CN"
SIZE         = 224

def mm_to_vox(affine, x, y, z):
    return np.round(np.linalg.inv(affine) @ np.array([x, y, z, 1.0]))[:3].astype(int)


def resize2d(sl, size=SIZE):
    return zoom(sl, (size / sl.shape[0], size / sl.shape[1]), order=1).astype("float32")


def save_jpg(sl, path, cmap):
    lo, hi = np.percentile(sl[sl != 0], [1, 99]) if (sl != 0).any() else (0, 1)
    plt.imsave(path, np.flipud(sl.T), cmap=cmap, vmin=lo, vmax=hi, format="jpg")


for subject in tqdm(sorted(os.listdir(final_folder))):
    sp = os.path.join(final_folder, subject)

    pet_img = nib.load(os.path.join(sp, f"{subject}_pet_suvr.nii.gz"))
    mri_img = nib.load(os.path.join(sp, f"{subject}_mri_zscore.nii.gz"))
    pet = pet_img.get_fdata().astype("float32")
    mri = mri_img.get_fdata().astype("float32")

    z = mm_to_vox(pet_img.affine, 0, 0, -20)[2]     # axial   — posterior cingulate / precuneus
    y = mm_to_vox(pet_img.affine, 0, -10, 0)[1]    # coronal — hippocampal body
    x = mm_to_vox(pet_img.affine, -25, 0, 0)[0]     # sagittal — posterior cingulate in profile

    save_jpg(resize2d(pet[x,:,:]), os.path.join(out_root, f"{subject}_pet_sagittal.jpg"), "hot")
    save_jpg(resize2d(mri[x,:,:]), os.path.join(out_root, f"{subject}_mri_sagittal.jpg"), "grey")

    save_jpg(resize2d(pet[:,y,:]), os.path.join(out_root, f"{subject}_pet_coronal.jpg"), "hot")
    save_jpg(resize2d(mri[:,y,:]), os.path.join(out_root, f"{subject}_mri_coronal.jpg"), "grey")

    save_jpg(resize2d(pet[:,:,z]), os.path.join(out_root, f"{subject}_pet_axial.jpg"), "hot")
    save_jpg(resize2d(mri[:,:,z]), os.path.join(out_root, f"{subject}_mri_axial.jpg"), "grey")