import pandas as pd


def generate_latex_table(model_dict, decimal_places=3, metric_name_remap=None):
    """
    Given a dictionary where:
      - key: model name (str)
      - value: path to a CSV file containing metrics for that model

    Returns a LaTeX-formatted table (string).

    Assumptions:
      - Each CSV has exactly one row of metric values, and the first row
        of the CSV contains the metric names.
      - The same set (or a compatible subset) of metrics appears across the CSVs.
      - Example format for each CSV:
            f1,accuracy,precision
            0.5,0.5,0.5
      - metric_name_remap (dict) is optional. If provided, its keys are
        the original metric names (as found in the CSV header), and its
        values are the desired renamed metrics. Columns not in the dictionary
        remain unchanged.
    """

    frames = []
    for model_name, csv_path in model_dict.items():
        # Read the CSV (header=0 uses the first row as column names)
        df = pd.read_csv(csv_path, header=0)

        # We assume there's exactly one row of values in each CSV.
        # Rename that row's index to be the model's name:
        df.index = [model_name]

        # Append to the list of frames for later concatenation
        frames.append(df)

    # Combine them into one DataFrame (one row per model)
    combined_df = pd.concat(frames, axis=0)

    # Optionally rename metric names (i.e., columns) using the remap dictionary
    if metric_name_remap is not None:
        combined_df.rename(columns=metric_name_remap, inplace=True)

    # (Optional) Round numeric columns to desired decimal places
    combined_df = combined_df.round(decimal_places)

    # Convert to a LaTeX table
    latex_str = combined_df.to_latex(
        index=True,
        float_format=f"%.{decimal_places}f",  # Format floats
        na_rep="N/A",
        caption="Model Comparison Table",
        label="tab:model-comparison",
    )

    return latex_str


import pandas as pd
import re


def generate_latex_table_real_synthetic(
    model_dict, decimal_places=3, metric_name_remap=None
):
    """
    Given a dictionary where:
      - key: model name (str)
      - value: path to a CSV file containing metrics for that model

    This function returns a LaTeX-formatted table (string).

    Features:
      1) Each CSV is assumed to have exactly one row of values (header row + data row).
      2) Columns that end with '_real', '_synthetic', or '_syn' will be mapped to
         "Real" or "Synthetic" rows, respectively, in the final table.
      3) The suffix is stripped from the column name for the final column header.
      4) If a CSV has both real and synthetic columns, we produce two rows:
           <ModelName (Real)> and <ModelName (Synthetic)>
      5) If metric_name_remap is provided, it is a dict that maps
         original column names (after stripping) to renamed metric names.

    Example CSV format (one row):
      sex_accuracy_synthetic,age_mae_synthetic,fid_2d_syn,fid_3d_syn,ms_ssim_syn
      0.81,18.9,231.26,15.98,0.816

    Another CSV might have columns ending with _real.

    We'll produce a final DataFrame with up to 2 rows per model:
      - One row for (ModelName (Synthetic))
      - One row for (ModelName (Real))
    """

    def parse_suffix(col_name):
        """
        Returns (base_metric, subset_label) by examining the column name.
        Possible subset_label = "Real", "Synthetic".
        We unify '_synthetic' and '_syn' as "Synthetic".
        We unify '_real' as "Real".
        If none of these suffixes exist, return (col_name, None) or handle as needed.
        """
        # Check for _real
        if col_name.endswith("_real"):
            base_metric = col_name[:-5]  # remove trailing "_real"
            return base_metric, "Real"
        # Check for _synthetic
        elif col_name.endswith("_synthetic"):
            base_metric = col_name[:-10]  # remove trailing "_synthetic"
            return base_metric, "Synthetic"
        # Check for _syn
        elif col_name.endswith("_syn"):
            base_metric = col_name[:-4]  # remove trailing "_syn"
            return base_metric, "Synthetic"
        else:
            # If you want to handle columns that don't match these suffixes,
            # you could return (col_name, None).
            # For now, let's treat them as "Synthetic" or skip them.
            # This depends on how you want to handle un-suffixed columns.
            return col_name, None

    all_rows = []  # We will accumulate DataFrame rows here

    for model_name, csv_path in model_dict.items():
        # Read the CSV (header=0 uses the first row as column names)
        df = pd.read_csv(csv_path, header=0)

        # We assume exactly one row of values in each CSV.
        if len(df) != 1:
            raise ValueError(
                f"CSV {csv_path} expected to have exactly 1 data row, found {len(df)}."
            )

        # Extract that single row as a Series for easy column-value access
        row_series = df.iloc[0]

        # Dictionary for each subset label (Real or Synthetic)
        # We'll build up the row of values here.
        # Example: data_by_subset["Real"] = {"sex_accuracy": 0.81, ...}
        data_by_subset = {}

        # Loop over columns to decide whether they belong to Real or Synthetic
        for col_name in row_series.index:
            value = row_series[col_name]
            base_metric, subset_label = parse_suffix(col_name)

            if subset_label is None:
                # If you want to skip or treat them differently, do so here.
                # Let's assume if there's no suffix, we skip or treat as Synthetic.
                # (Comment/uncomment as needed.)
                # continue
                subset_label = "Synthetic"

            # Initialize the dictionary for that subset if not present
            if subset_label not in data_by_subset:
                data_by_subset[subset_label] = {}

            # Assign the value to the base metric
            data_by_subset[subset_label][base_metric] = value

        # Now create a single-row DataFrame for each subset_label
        # For example, "Real" => one row, "Synthetic" => one row
        for subset_label, metric_values in data_by_subset.items():
            # If you want the row index to say: "ModelName (Real)" or "ModelName (Synthetic)"
            row_label = f"{model_name} ({subset_label})"

            # Create a 1-row DF with columns = base metrics
            subset_df = pd.DataFrame([metric_values], index=[row_label])

            # Collect it
            all_rows.append(subset_df)

    # Concatenate all subset DFs.
    # This yields one large DataFrame with multiple rows
    # (some for Real, some for Synthetic).
    combined_df = pd.concat(all_rows, axis=0)

    # Rename columns (the "base metric" part) if a remap dict is provided
    # NOTE: The keys in metric_name_remap should match the *base metric*
    #       after we stripped off _real/_synthetic/_syn.
    if metric_name_remap is not None:
        combined_df.rename(columns=metric_name_remap, inplace=True)

    # Round numeric columns
    combined_df = combined_df.round(decimal_places)

    # Generate LaTeX
    latex_str = combined_df.to_latex(
        index=True,
        float_format=f"%.{decimal_places}f",
        na_rep="N/A",
        caption="Model Comparison Table (Real vs. Synthetic)",
        label="tab:model-comparison",
    )

    return latex_str


if __name__ == "__main__":
    # Example usage:
    model_paths = {
        "ModelA": "model_a_metrics.csv",  # each CSV has 1 row of data
        "ModelB": "model_b_metrics.csv",
        "ModelC": "model_c_metrics.csv",
    }

    # Example of a remap dictionary to rename CSV columns
    metric_remap = {
        "f1": "F1-Score",
        "accuracy": "Accuracy (%)",
        "precision": "Precision (%)",
    }

    latex_table = generate_latex_table(
        model_dict=model_paths, decimal_places=2, metric_name_remap=metric_remap
    )
    print(latex_table)
