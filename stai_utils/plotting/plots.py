import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def boxplot_loss_vs_agebins(models, bins_to_ignore=None, box_widths=0.2):
    """
    Plot side-by-side boxplots of loss binned by decades for multiple models.

    Args:
        models (dict): A dictionary where keys are model names and values are JSON file paths.
    """
    # Initialize a list to hold the data for all models
    all_data = []

    # Iterate over the models to load and process the data
    for model_name, json_path in models.items():
        # Load the JSON data
        with open(json_path, "r") as f:
            data = json.load(f)

        # Convert to DataFrame
        df = pd.DataFrame(data)
        # Create bins for decades
        df["decade"] = (df["label"] // 10 * 10).astype(int)
        # Add a column for the model name
        df["model"] = model_name

        # Append the DataFrame to the list
        all_data.append(df)

    # Concatenate all data into a single DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)

    # Remove rows corresponding to decades in bins_to_ignore
    if bins_to_ignore:
        combined_df = combined_df[~combined_df["decade"].isin(bins_to_ignore)]

    # Prepare the data for manual plotting
    grouped = (
        combined_df.groupby(["decade", "model"])["loss"]
        .apply(list)
        .unstack(fill_value=[])
    )
    decades = grouped.index
    models_list = list(models.keys())

    # Calculate positions
    num_models = len(models_list)
    total_width = num_models * box_widths
    x_positions = np.arange(len(decades))

    # Plot the boxplots
    plt.figure(figsize=(15, 6))
    for i, model_name in enumerate(models_list):
        model_positions = (
            x_positions - total_width / 2 + i * box_widths + box_widths / 2
        )
        plt.boxplot(
            grouped[model_name],
            positions=model_positions,
            widths=box_widths,
            patch_artist=True,
            boxprops=dict(facecolor=f"C{i}"),
            medianprops=dict(color="black"),
            label=model_name,
        )

    # Set x-axis ticks and labels
    interval_labels = [f"[{decade}, {decade + 10})" for decade in decades]
    plt.xticks(ticks=x_positions, labels=interval_labels, rotation=0)
    plt.xlabel("Age Decade")
    plt.ylabel("MAE")
    # plt.title("Loss Values Binned by Decades Across Models")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Show the plot
    plt.show()

    # # Create the boxplot
    # plt.figure(figsize=(15, 6))
    # sns.boxplot(
    #     x="decade",
    #     y="loss",
    #     hue="model",
    #     data=combined_df,
    #     dodge=True,  # Ensures models are side-by-side within the same decade
    #     palette="Set2",  # Different colors for each model
    #     widths=box_widths,
    # )

    # # Add labels, legend, and title
    # plt.title("Loss Values Binned by Decades Across Models")
    # plt.xlabel("Decade")
    # plt.ylabel("Loss")
    # plt.legend(title="Model")
    # plt.grid(axis="y", linestyle="--", alpha=0.7)
    # plt.tight_layout()

    # # Show the plot
    # plt.show()


def boxplot_accuracy_vs_sex(json_file):
    # Load the JSON data
    with open(json_file, "r") as f:
        data = json.load(f)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Ensure the label is treated as a binary category
    df["label"] = df["label"].astype(int)

    # Create the boxplot
    plt.figure(figsize=(8, 6))
    df.boxplot(column="acc", by="label", grid=False)

    # Add labels and title
    plt.title("Accuracy by Label")
    plt.suptitle("")  # Suppress the default title
    plt.xlabel("Label (0 or 1)")
    plt.ylabel("Accuracy")
    plt.xticks([1, 2], ["0", "1"])  # Ensure correct labeling
    plt.tight_layout()

    # Show the plot
    plt.show()


def lineplot_accuracy_vs_sex(json_file):
    # Load the JSON data
    with open(json_file, "r") as f:
        data = json.load(f)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Ensure the label is treated as a binary category
    df["label"] = df["label"].astype(int)

    # Calculate the average accuracy for each bin (label)
    avg_accuracy = df.groupby("label")["acc"].mean()

    # Plot the lineplot
    plt.figure(figsize=(8, 6))
    plt.plot(
        avg_accuracy.index,
        avg_accuracy.values,
        marker="o",
        linestyle="-",
        label="Average Accuracy",
    )

    # Add labels, title, and grid
    plt.title("Average Accuracy by Label")
    plt.xlabel("Label (0 or 1)")
    plt.ylabel("Average Accuracy")
    plt.xticks([0, 1])  # Ensure correct x-axis labels
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Show the plot
    plt.show()


import json
import matplotlib.pyplot as plt
import numpy as np


def boxplot_voxel_distributions(models_dict, index_to_name, indices_to_ignore=None):
    """
    Generate grouped boxplots for voxel distributions across brain structures from multiple models,
    using structure names instead of indices on the x-axis.

    Parameters:
        models_dict (dict): A dictionary where keys are model names and values are paths to the corresponding JSON files.
        index_to_name (dict): A dictionary mapping structure indices to structure names.
    """
    # Prepare data for the boxplot
    all_data = {}  # Stores data for all models

    for model_name, json_path in models_dict.items():
        # Load the JSON data
        with open(json_path, "r") as file:
            data = json.load(file)

        # Collect voxel counts for each brain structure index
        brain_structure_voxels = {}
        for filename, structures in data.items():
            for structure_index, voxel_count in structures.items():
                if indices_to_ignore and int(structure_index) in indices_to_ignore:
                    continue
                structure_index = int(structure_index)  # Ensure indices are integers
                if structure_index not in brain_structure_voxels:
                    brain_structure_voxels[structure_index] = []
                brain_structure_voxels[structure_index].append(voxel_count)

        # Add data to all_data for this model
        all_data[model_name] = brain_structure_voxels

    # Get all unique brain structure indices across all models
    all_indices = sorted(
        set(index for model_data in all_data.values() for index in model_data.keys())
    )

    # Map indices to names for x-axis labels
    x_labels = [index_to_name.get(index, str(index)) for index in all_indices]

    # Prepare data for grouped boxplots
    grouped_data = {model_name: [] for model_name in models_dict.keys()}
    for index in all_indices:
        for model_name in models_dict.keys():
            grouped_data[model_name].append(all_data.get(model_name, {}).get(index, []))

    # Plot the grouped boxplots
    plt.figure(figsize=(18, 8))
    num_models = len(models_dict)
    width = 0.6 / num_models  # Width of each box group
    x_positions = np.arange(len(all_indices))  # Shared x-axis positions

    for i, (model_name, data) in enumerate(grouped_data.items()):
        positions = (
            x_positions + (i - num_models / 2) * width + width / 2
        )  # Offset positions
        plt.boxplot(
            data,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor=f"C{i}"),
            label=model_name,
        )

    # Configure the plot
    plt.xticks(ticks=x_positions, labels=x_labels, rotation=90)
    plt.xlabel("Brain Structure")
    plt.ylabel("Voxel Count")
    plt.title("Voxel Count Distribution by Brain Structure Across Models")
    plt.legend()
    plt.tight_layout()
    plt.show()


import json
import matplotlib.pyplot as plt
import numpy as np


def boxplot_grouped_voxel_distributions(
    models_dict, structure_groups, ignore_indices, ylim=None
):
    """
    Generate grouped boxplots for voxel distributions across brain structures from multiple models,
    grouping left and right structures together and ignoring specified indices.

    Parameters:
        models_dict (dict): A dictionary where keys are model names and values are paths to the corresponding JSON files.
        index_to_name (dict): A dictionary mapping structure indices to structure names.
        structure_groups (dict): A dictionary mapping group names to a list of structure indices to combine.
        ignore_indices (set): A set of structure indices to ignore in the plot.
    """
    # Prepare data for the boxplot
    all_data = {}  # Stores data for all models

    for model_name, json_path in models_dict.items():
        # Load the JSON data
        with open(json_path, "r") as file:
            data = json.load(file)

        # Collect voxel counts for each brain structure index
        brain_structure_voxels = {}
        for filename, structures in data.items():
            for structure_index, voxel_count in structures.items():
                structure_index = int(structure_index)  # Ensure indices are integers
                if structure_index not in ignore_indices:  # Skip ignored indices
                    if structure_index not in brain_structure_voxels:
                        brain_structure_voxels[structure_index] = []
                    brain_structure_voxels[structure_index].append(voxel_count)

        # Add data to all_data for this model
        all_data[model_name] = brain_structure_voxels

    # Prepare grouped data
    grouped_data = {
        model_name: {group: [] for group in structure_groups.keys()}
        for model_name in models_dict.keys()
    }
    for group_name, indices in structure_groups.items():
        for model_name in models_dict.keys():
            group_voxels = []
            for index in indices:
                group_voxels.extend(all_data.get(model_name, {}).get(index, []))
            grouped_data[model_name][group_name] = group_voxels

    # Prepare data for plotting
    group_names = list(structure_groups.keys())
    num_groups = len(group_names)
    num_models = len(models_dict)
    x_positions = np.arange(num_groups)
    width = 0.6 / num_models  # Width of each box group

    # Plot the grouped boxplots
    plt.figure(figsize=(15, 6))
    for i, (model_name, model_data) in enumerate(grouped_data.items()):
        positions = x_positions + (i - num_models / 2) * width + width / 2
        plt.boxplot(
            [model_data[group] for group in group_names],
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor=f"C{i}"),
            label=model_name,
        )

    # Configure the plot
    plt.xticks(ticks=x_positions, labels=group_names, rotation=45)
    plt.xlabel("Brain Structure Group")
    plt.ylabel("Voxel Count")
    plt.yscale("log")
    if ylim is not None:
        plt.ylim(ylim)
    plt.title("Voxel Count Distribution by Brain Structure Group Across Models")
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()
