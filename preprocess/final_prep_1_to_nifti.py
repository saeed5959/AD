import os
import subprocess
from tqdm import tqdm


mri_folder = "./DATASET/AD_MRI"
pet_folder = "./DATASET/AD_PET"
final_folder = "./FINAL_AD"

#copy nifti mri fime to out folder
for subject in tqdm(os.listdir(mri_folder)):
    path_1 = os.path.join(mri_folder, subject)
    path_2 = os.path.join(path_1, os.listdir(path_1)[0])
    path_3 = os.path.join(path_2, os.listdir(path_2)[0])
    path_4 = os.path.join(path_3, os.listdir(path_3)[0])
    mri_path = os.path.join(path_4, os.listdir(path_4)[0])

    final_subject_path = os.path.join(final_folder, subject)
    os.makedirs(final_subject_path, exist_ok=True)
    subprocess.run(["cp", mri_path, os.path.join(final_subject_path, subject + "_mri.nii")], check=True)


#convert dicom pet files to nifti files and then save it in out folder
for subject in tqdm(os.listdir(pet_folder)):
    path_1 = os.path.join(pet_folder, subject)
    path_2 = os.path.join(path_1, os.listdir(path_1)[0])
    path_3 = os.path.join(path_2, os.listdir(path_2)[0])
    pet_path = os.path.join(path_3, os.listdir(path_3)[0])

    final_subject_path = os.path.join(final_folder, subject)
    os.makedirs(final_subject_path, exist_ok=True)
    subprocess.run(["dcm2niix", "-z", "y", "-o", final_subject_path, pet_path], check=True)

    for name in os.listdir(final_subject_path):
        if name.endswith(".gz"):
            pet_path = os.path.join(final_subject_path, name)
            rename_path = os.path.join(final_subject_path, subject + "_pet.nii.gz")

    #rename pet name
    subprocess.run(f"mv {pet_path} {rename_path}",shell=True, check=True)

    #remove json files that are extra
    subprocess.run(f"rm {final_subject_path}/*.json",shell=True, check=True)


#remove subject that does not have pet or mri
for subject in tqdm(os.listdir(final_folder)):
    subject_path = os.path.join(final_folder, subject)

    if len(os.listdir(subject_path)) < 2:
        subprocess.run(f"rm -r {subject_path}",shell=True, check=True)
