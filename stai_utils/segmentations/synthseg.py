import os
import subprocess


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
    output_filename = os.path.join(output_dir, "synthseg_output_paths.txt")
    with open(input_filename, "w") as f:
        for path in filtered_input_paths:
            f.write(f"{path}\n")
    with open(output_filename, "w") as f:
        for path in filtered_output_paths:
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
