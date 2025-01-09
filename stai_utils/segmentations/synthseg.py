import os
import subprocess
import numpy as np
from tqdm import tqdm


def shell_command(command):
    print("RUNNING", command)
    subprocess.run(command, shell=True)


def get_all_file_paths(directory):
    # List to store file paths
    file_paths = []

    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))

    return file_paths


def run_synthseg(input_dir, output_dir, skip_existing=True):
    # Create a new list of output paths where the file name is prepended with 'synthseg_'
    input_paths = get_all_file_paths(input_dir)
    output_paths = []
    for path in input_paths:
        file_name = os.path.basename(path)
        new_file_name = f"synthseg_{file_name}"
        new_path = os.path.join(output_dir, new_file_name)
        output_paths.append(new_path)

    if skip_existing:
        # Filter out paths where the output file already exists
        filtered_input_paths = []
        filtered_output_paths = []

        for i, (in_path, out_path) in enumerate(zip(input_paths, output_paths)):
            if not os.path.exists(out_path):
                filtered_input_paths.append(in_path)
                filtered_output_paths.append(out_path)
                print("appending", i)
            else:
                print(f"Output file already exists and will be skipped: {out_path}")

        input_paths = filtered_input_paths
        output_paths = filtered_output_paths

    # Dump input/output paths to txt files
    input_filename = os.path.join(input_dir, "synthseg_input_paths.txt")
    output_filename = os.path.join(input_dir, "synthseg_output_paths.txt")
    with open(input_filename, "w") as f:
        for path in input_paths:
            f.write(f"{path}\n")
    with open(output_filename, "w") as f:
        for path in output_paths:
            f.write(f"{path}\n")
    print(f"Input paths saved to {input_filename}")
    print(f"Output paths saved to {output_filename}")

    # Run SynthSeg command
    failed = []
    synthseg_cmd = f"python /afs/cs.stanford.edu/u/alanqw/SynthSeg/scripts/commands/SynthSeg_predict.py --i {input_filename} --o {output_filename}"
    try:
        shell_command(synthseg_cmd)
    except Exception as e:
        print("Error synthseging:")
        print(e)

    # Clean up
    os.remove(input_filename)
    os.remove(output_filename)

    return failed


def get_frequency_dict_in_dir(directory, cap=None):
    """
    Processes all .npz files in a directory and computes the frequency
    dictionary for the numpy array with the key 'vol_data' inside them.

    Parameters:
        directory (str): Path to the directory containing .npz files.

    Returns:
        list: A list of dictionaries with 'filename' and 'frequencies' keys.
    """
    results = {}

    count = 0
    for filename in tqdm(
        os.listdir(directory),
        desc="Getting frequency",
        total=cap if cap is not None else len(os.listdir(directory)),
    ):
        if filename.endswith(".npz"):
            count += 1
            if cap is not None and count > cap:
                break
            file_path = os.path.join(directory, filename)
            # Load the .npz file
            with np.load(file_path) as data:
                # Check for 'vol_data' key
                if "vol_data" in data.files:
                    array = data["vol_data"]
                else:
                    raise KeyError(f"'vol_data' key not found in {filename}")

                # Compute the frequency dictionary
                frequency_dict = get_frequency_dict(array)
                frequency_dict = {
                    int(key): value for key, value in frequency_dict.items()
                }

                # Append result as a dictionary
                results[filename] = frequency_dict

    return results


def get_frequency_dict(array):
    """
    Computes the frequencies of each unique integer in a numpy array.

    Parameters:
        array (numpy.ndarray): Input numpy array.

    Returns:
        dict: A dictionary with unique integers as keys and their frequencies as values.
    """
    unique_values, frequencies = np.unique(array, return_counts=True)
    unique_values = unique_values.tolist()
    frequencies = frequencies.tolist()
    return dict(zip(unique_values, frequencies))
