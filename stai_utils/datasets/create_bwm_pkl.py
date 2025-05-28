from glob import glob
import pickle
import os


def save_pkl(data, output_path):
    # Save the dictionary to a pickle file
    with open(output_path, "wb") as f:
        pickle.dump(data, f)


def get_hcpag_paths_and_metadata(root_dir):
    res = []

    root_data_dir = os.path.join(root_dir, "DataSets/")
    hcpa_root_dir = os.path.join(root_data_dir, "HCP-Aging")
    registered_dir = os.path.join(hcpa_root_dir, "processed/Structural/registration")

    # Dictionary to store paths by subject ID
    subject_paths = {}

    # Get only immediate subdirectories (one level)
    subject_dirs = [
        d
        for d in os.listdir(registered_dir)
        if os.path.isdir(os.path.join(registered_dir, d)) and d.startswith("HCA")
    ]

    for subject_dir in subject_dirs:
        subject_id = subject_dir
        subject_path = os.path.join(registered_dir, subject_dir)

        # Look for T1 and T2 files in the T1w directory
        t1w_dir = os.path.join(subject_path, "T1w")
        assert os.path.exists(
            t1w_dir
        ), f"HCP-Aging: T1w directory does not exist for subject {subject_id}"

        t1_file = os.path.join(t1w_dir, "T1w_acpc_dc_restore_brain_mni_warped.nii.gz")
        t2_file = os.path.join(t1w_dir, "T2w_acpc_dc_restore_brain_mni_warped.nii.gz")

        subject_paths = {}
        if os.path.exists(t1_file):
            subject_paths["t1_path"] = t1_file[
                len(root_dir) :
            ]  # remove root_dir from path
        else:
            print(f"HCP-Aging: T1w file does not exist {t1_file}")
        if os.path.exists(t2_file):
            subject_paths["t2_path"] = t2_file[
                len(root_dir) :
            ]  # remove root_dir from path
        else:
            print(f"HCP-Aging: T2w file does not exist {t2_file}")

        res.append(subject_paths)

    for d in res:
        d["age"] = 0
        d["sex"] = 0
    return res


def get_hcpdev_paths_and_metadata(root_dir):
    res = []

    root_data_dir = os.path.join(root_dir, "DataSets/")
    hcpdev_root_dir = os.path.join(root_data_dir, "HCP-Development")
    registered_dir = os.path.join(hcpdev_root_dir, "processed/Structural/registration")

    # Dictionary to store paths by subject ID
    subject_paths = {}

    # Get only immediate subdirectories (one level)
    subject_dirs = [
        d
        for d in os.listdir(registered_dir)
        if os.path.isdir(os.path.join(registered_dir, d)) and d.startswith("HCD")
    ]

    for subject_dir in subject_dirs:
        subject_id = subject_dir
        subject_path = os.path.join(registered_dir, subject_dir)

        # Look for T1 and T2 files in the T1w directory
        t1w_dir = os.path.join(subject_path, "T1w")
        assert os.path.exists(
            t1w_dir
        ), f"HCP-Development: T1w directory does not exist for subject {subject_id}"

        t1_file = os.path.join(t1w_dir, "T1w_acpc_dc_restore_brain_mni_warped.nii.gz")
        t2_file = os.path.join(t1w_dir, "T2w_acpc_dc_restore_brain_mni_warped.nii.gz")

        subject_paths = {}
        if os.path.exists(t1_file):
            subject_paths["t1_path"] = t1_file[
                len(root_dir) :
            ]  # remove root_dir from path
        else:
            print(f"HCP-Development: T1w file does not exist {t1_file}")
        if os.path.exists(t2_file):
            subject_paths["t2_path"] = t2_file[
                len(root_dir) :
            ]  # remove root_dir from path
        else:
            print(f"HCP-Development: T2w file does not exist {t2_file}")

        res.append(subject_paths)

    for d in res:
        d["age"] = 0
        d["sex"] = 0
    return res


def get_hcpya_paths_and_metadata(root_dir):
    res = []

    t1_sequence = "T1w_MPR1"  # TODO: average T1w_MPR1 and T1w_MPR2
    t2_sequence = "T2w_SPC1"

    root_data_dir = os.path.join(root_dir, "DataSets/")
    hcpya_root_dir = os.path.join(root_data_dir, "HCP-YA")
    registered_dir = os.path.join(hcpya_root_dir, "processed/Structural/registration")

    # Dictionary to store paths by subject ID
    subject_paths = {}

    # Get only immediate subdirectories (one level)
    subject_dirs = [
        d
        for d in os.listdir(registered_dir)
        if os.path.isdir(os.path.join(registered_dir, d))
    ]

    for subject_dir in subject_dirs:
        subject_id = subject_dir.split("_")[0]
        field_strength = subject_dir.split("_")[1]
        subject_path = os.path.join(registered_dir, subject_dir)

        # Look for T1 and T2 files in the T1w directory
        img_dir = os.path.join(
            subject_path, subject_id, f"unprocessed/{field_strength}/{t1_sequence}"
        )

        t1_file = os.path.join(
            img_dir,
            f"{subject_id}_{field_strength}_{t1_sequence}_brain_mni_warped.nii.gz",
        )
        t2_file = os.path.join(
            img_dir,
            f"{subject_id}_{field_strength}_{t2_sequence}_brain_mni_warped.nii.gz",
        )
        # print(t1_file)

        subject_paths = {}
        if os.path.exists(t1_file):
            subject_paths["t1_path"] = t1_file[
                len(root_dir) :
            ]  # remove root_dir from path
        else:
            print(f"HCP-YA: T1w file does not exist {t1_file}")
        if os.path.exists(t2_file):
            subject_paths["t2_path"] = t2_file[
                len(root_dir) :
            ]  # remove root_dir from path
        else:
            print(f"HCP-YA: T2w file does not exist {t2_file}")

        res.append(subject_paths)

    for d in res:
        d["age"] = 0
        d["sex"] = 0
    return res


def get_openneuro_paths_and_metadata(root_dir):
    # /simurgh/group/BWM/DataSets/OpenNeuro/processed/Structural/registration/sub-ON00400/ses-01/anat
    res = []

    root_data_dir = os.path.join(root_dir, "DataSets/")
    openneuro_root_dir = os.path.join(root_data_dir, "OpenNeuro")
    registered_dir = os.path.join(
        openneuro_root_dir, "processed/Structural/registration"
    )

    # Dictionary to store paths by subject ID
    subject_paths = {}

    # Get only immediate subdirectories (one level)
    subject_dirs = [
        d
        for d in os.listdir(registered_dir)
        if os.path.isdir(os.path.join(registered_dir, d)) and d.startswith("sub")
    ]

    for subject_dir in subject_dirs:
        subject_path = os.path.join(registered_dir, subject_dir)
        for ses_dir in os.listdir(subject_path):

            ses_path = os.path.join(subject_path, ses_dir)

            img_dir = os.path.join(ses_path, "anat")

            t1_file = os.path.join(
                img_dir,
                f"{subject_dir}_{ses_dir}_acq-MPRAGE_T1w_brain_mni_warped.nii.gz",
            )
            t2_file = os.path.join(
                img_dir, f"{subject_dir}_{ses_dir}_acq-CUBE_T2w_brain_mni_warped.nii.gz"
            )

            subject_paths = {}
            if os.path.exists(t1_file):
                subject_paths["t1_path"] = t1_file[
                    len(root_dir) :
                ]  # remove root_dir from path
            else:
                print(f"OpenNeuro: T1w file does not exist {t1_file}")
            if os.path.exists(t2_file):
                subject_paths["t2_path"] = t2_file[
                    len(root_dir) :
                ]  # remove root_dir from path
            else:
                print(f"OpenNeuro: T2w file does not exist {t2_file}")

        res.append(subject_paths)

    for d in res:
        d["age"] = 0
        d["sex"] = 0
    return res


def get_abcd_paths_and_metadata(root_dir):
    # /hai/scratch/alanqw/BWM/DataSets/ABCD/processed/Structural/registration/Unzip/sub-NDARINV01AJ15N9/ses-baselineYear1Arm1/anat/sub-NDARINV01AJ15N9_ses-baselineYear1Arm1_run-01_T1w_brain_mni_warped.nii.gz
    res = []

    root_data_dir = os.path.join(root_dir, "DataSets/")
    abcd_root_dir = os.path.join(root_data_dir, "ABCD")
    registered_dir = os.path.join(
        abcd_root_dir, "processed/Structural/registration/Unzip"
    )

    # Dictionary to store paths by subject ID
    subject_paths = {}

    # Get only immediate subdirectories (one level)
    subject_dirs = [
        d
        for d in os.listdir(registered_dir)
        if os.path.isdir(os.path.join(registered_dir, d)) and d.startswith("sub")
    ]

    for subject_name in subject_dirs:
        subject_dir = os.path.join(registered_dir, subject_name)
        for ses_name in os.listdir(subject_dir):
            ses_dir = os.path.join(subject_dir, ses_name)

            img_dir = os.path.join(ses_dir, "anat")

            t1_file = os.path.join(
                img_dir,
                f"{subject_name}_{ses_name}_run-01_T1w_brain_mni_warped.nii.gz",
            )
            t2_file = os.path.join(
                img_dir,
                f"{subject_name}_{ses_name}_run-01_T2w_brain_mni_warped.nii.gz",
            )

            subject_paths = {}
            if os.path.exists(t1_file):
                subject_paths["t1_path"] = t1_file[
                    len(root_dir) :
                ]  # remove root_dir from path
            else:
                print(f"ABCD: T1w file does not exist {t1_file}")
            if os.path.exists(t2_file):
                subject_paths["t2_path"] = t2_file[
                    len(root_dir) :
                ]  # remove root_dir from path
            else:
                print(f"ABCD: T2w file does not exist {t2_file}")

        res.append(subject_paths)

    for d in res:
        d["age"] = 0
        d["sex"] = 0
    return res


def main():
    root_dir = "/simurgh/group/BWM/"
    root_data_dir = os.path.join(root_dir, "DataSets/")
    save_dir = "/simurgh/u/alanqw/BWM/"

    hcpya_root_dir = os.path.join(root_data_dir, "HCP-YA")
    hcpdev_root_dir = os.path.join(root_data_dir, "HCP-Development")
    hcpa_root_dir = os.path.join(root_data_dir, "HCP-Aging")
    openneuro_root_dir = os.path.join(root_data_dir, "OpenNeuro")

    hcpya_metadata_dir = os.path.join(hcpya_root_dir, "Metadata")
    hcpdev_metadata_dir = os.path.join(hcpdev_root_dir, "Metadata")
    hcpa_metadata_dir = os.path.join(hcpa_root_dir, "Metadata")
    openneuro_metadata_dir = os.path.join(openneuro_root_dir, "Metadata")

    # HCP-YA
    hcpya_paths = get_hcpya_paths_and_metadata(root_dir)
    print(len(hcpya_paths))

    # HCP-Development
    hcpdev_paths = get_hcpdev_paths_and_metadata(root_dir)
    print(len(hcpdev_paths))

    # HCP-Aging
    hcpag_paths = get_hcpag_paths_and_metadata(root_dir)
    print(len(hcpag_paths))

    # OpenNeuro
    openneuro_paths = get_openneuro_paths_and_metadata(root_dir)
    print(len(openneuro_paths))

    # ABCD
    abcd_paths = get_abcd_paths_and_metadata(root_dir)
    print(len(abcd_paths))

    # Save the pickle files
    save_pkl(hcpya_paths, os.path.join(save_dir, "hcpya_relpaths_and_metadata.pkl"))
    save_pkl(hcpdev_paths, os.path.join(save_dir, "hcpdev_relpaths_and_metadata.pkl"))
    save_pkl(hcpag_paths, os.path.join(save_dir, "hcpag_relpaths_and_metadata.pkl"))
    save_pkl(
        openneuro_paths, os.path.join(save_dir, "openneuro_relpaths_and_metadata.pkl")
    )
    save_pkl(abcd_paths, os.path.join(save_dir, "abcd_relpaths_and_metadata.pkl"))

    # Load the pickle file
    with open(os.path.join(save_dir, "hcpya_relpaths_and_metadata.pkl"), "rb") as f:
        data = pickle.load(f)
    print(data)


if __name__ == "__main__":
    main()
