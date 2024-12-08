import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def boxplot_loss_vs_agebins(models):
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

    # Create the boxplot
    plt.figure(figsize=(12, 8))
    sns.boxplot(
        x="decade",
        y="loss",
        hue="model",
        data=combined_df,
        dodge=True,  # Ensures models are side-by-side within the same decade
        palette="Set2",  # Different colors for each model
    )

    # Add labels, legend, and title
    plt.title("Loss Values Binned by Decades Across Models")
    plt.xlabel("Decade")
    plt.ylabel("Loss")
    plt.legend(title="Model")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Show the plot
    plt.show()


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
