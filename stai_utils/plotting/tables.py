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

import pandas as pd


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
         Only metrics specified in this dictionary will be included in the final table.

    Example CSV format (one row):
      sex_accuracy_synthetic,age_mae_synthetic,fid_2d_syn,fid_3d_syn,ms_ssim_syn
      0.81,18.9,231.26,15.98,0.816
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
            # For now, we'll treat them as "Synthetic".
            return col_name, "Synthetic"

    all_rows = []  # We'll accumulate DataFrame rows here

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
        data_by_subset = {}

        # Loop over columns to decide whether they belong to Real or Synthetic
        for col_name in row_series.index:
            value = row_series[col_name]
            base_metric, subset_label = parse_suffix(col_name)

            # Initialize the dictionary for that subset if not present
            if subset_label not in data_by_subset:
                data_by_subset[subset_label] = {}

            # Assign the value to the base metric
            data_by_subset[subset_label][base_metric] = value

        # Now create a single-row DataFrame for each subset_label
        for subset_label, metric_values in data_by_subset.items():
            # Row label: "ModelName (Real)" or "ModelName (Synthetic)"
            row_label = f"{model_name} ({subset_label})"
            subset_df = pd.DataFrame([metric_values], index=[row_label])
            all_rows.append(subset_df)

    # Concatenate all subset DataFrames into one large DataFrame.
    combined_df = pd.concat(all_rows, axis=0)

    # If a metric_name_remap is provided, filter to only those columns and rename them.
    if metric_name_remap is not None:
        # Only include columns (base metric names) that are present in the remap dictionary.
        # Also, preserve the order as specified in metric_name_remap.
        cols_to_include = [
            metric
            for metric in metric_name_remap.keys()
            if metric in combined_df.columns
        ]
        combined_df = combined_df[cols_to_include]
        combined_df.rename(columns=metric_name_remap, inplace=True)

    # Round numeric columns to the specified number of decimal places.
    combined_df = combined_df.round(decimal_places)

    # Generate LaTeX table string.
    latex_str = combined_df.to_latex(
        index=True,
        float_format=f"%.{decimal_places}f",
        na_rep="N/A",
        caption="Model Comparison Table (Real vs. Synthetic)",
        label="tab:model-comparison",
    )

    return latex_str


import pandas as pd


def generate_latex_table_real_synthetic_with_std(
    model_dict, decimal_places=3, metric_name_remap=None, with_std=False
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
         Only metrics specified in this dictionary will be included in the final table.
      6) If with_std is True, each table entry is formatted as "mean $\pm$ std".
         It is assumed that for each metric there is a corresponding column
         with "_std" in its name (e.g. "sex_accuracy_std_synthetic") representing the standard deviation.
    """

    def parse_suffix(col_name):
        """
        Returns (base_metric, subset_label) by examining the column name.
        Possible subset_label = "Real", "Synthetic".
        We unify '_synthetic' and '_syn' as "Synthetic".
        We unify '_real' as "Real".
        If none of these suffixes exist, return (col_name, "Synthetic").
        """
        if col_name.endswith("_real"):
            base_metric = col_name[:-5]  # remove trailing "_real"
            return base_metric, "Real"
        elif col_name.endswith("_synthetic"):
            base_metric = col_name[:-10]  # remove trailing "_synthetic"
            return base_metric, "Synthetic"
        elif col_name.endswith("_syn"):
            base_metric = col_name[:-4]  # remove trailing "_syn"
            return base_metric, "Synthetic"
        else:
            return col_name, "Synthetic"

    all_rows = []

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

        # Dictionaries for mean values and (if with_std) standard deviations
        data_by_subset = {}
        std_by_subset = {} if with_std else None

        # Loop over columns in the CSV row
        for col_name in row_series.index:
            value = row_series[col_name]
            # If we are including standard deviations and this column is a std column...
            if with_std and "_std" in col_name:
                # Remove the '_std' part to get the corresponding metric name
                new_col = col_name.replace("_std", "")
                base_metric, subset_label = parse_suffix(new_col)
                if subset_label not in std_by_subset:
                    std_by_subset[subset_label] = {}
                std_by_subset[subset_label][base_metric] = value
            # Otherwise, treat as the mean value column.
            elif not (with_std and "_std" in col_name):
                base_metric, subset_label = parse_suffix(col_name)
                if subset_label not in data_by_subset:
                    data_by_subset[subset_label] = {}
                data_by_subset[subset_label][base_metric] = value

        # If with_std is True, combine mean and std into a formatted string.
        if with_std:
            for subset_label in data_by_subset:
                for metric, mean_val in data_by_subset[subset_label].items():
                    std_val = std_by_subset.get(subset_label, {}).get(metric, None)
                    if std_val is not None:
                        # Format as "mean ± std" using LaTeX's \pm
                        data_by_subset[subset_label][
                            metric
                        ] = f"{mean_val:.{decimal_places}f} $\\pm$ {std_val:.{decimal_places}f}"
                    else:
                        data_by_subset[subset_label][
                            metric
                        ] = f"{mean_val:.{decimal_places}f}"
        else:
            # If not including std, just round the numeric values.
            for subset_label in data_by_subset:
                for metric, mean_val in data_by_subset[subset_label].items():
                    try:
                        data_by_subset[subset_label][metric] = round(
                            mean_val, decimal_places
                        )
                    except Exception:
                        pass

        # Create a single-row DataFrame for each subset (Real or Synthetic)
        for subset_label, metric_values in data_by_subset.items():
            row_label = f"{model_name} ({subset_label})"
            subset_df = pd.DataFrame([metric_values], index=[row_label])
            all_rows.append(subset_df)

    # Concatenate all subset DataFrames into one large DataFrame.
    combined_df = pd.concat(all_rows, axis=0)

    # If a metric_name_remap is provided, filter to only those columns and rename them.
    if metric_name_remap is not None:
        cols_to_include = [
            metric
            for metric in metric_name_remap.keys()
            if metric in combined_df.columns
        ]
        combined_df = combined_df[cols_to_include]
        combined_df.rename(columns=metric_name_remap, inplace=True)

    # Only round the DataFrame if we haven't already formatted cells with std.
    if not with_std:
        combined_df = combined_df.round(decimal_places)

    # Generate LaTeX table string.
    latex_str = combined_df.to_latex(
        index=True,
        float_format=f"%.{decimal_places}f",
        na_rep="N/A",
        caption="Model Comparison Table (Real vs. Synthetic)",
        label="tab:model-comparison",
    )

    return latex_str


def generate_latex_table_real_synthetic_separated(
    model_dict, decimal_places=3, metric_name_remap=None
):
    """
    Given a dictionary where:
      - key: model name (str)
      - value: path to a CSV file containing metrics for that model

    This function returns a LaTeX-formatted table (string).

    Features:
      1) Each CSV is assumed to have exactly one row of values (header row + data row).
      2) Columns ending with '_real', '_synthetic', or '_syn' are mapped to Real or Synthetic,
         with the suffix stripped to leave only the base metric name.
      3) If a CSV has both real and synthetic columns, two rows are produced:
           - One for the real metrics (row label "Real")
           - One for the synthetic metrics (row label is the model name)
      4) Only one Real row is kept (if multiple CSVs supply a real row, only the first is kept).
      5) If metric_name_remap is provided, only the specified metrics are included in the final table
         and their column names are remapped accordingly.
      6) In the final LaTeX table the Real row appears first and a \midrule is inserted after it.
    """

    def parse_suffix(col_name):
        """
        Returns (base_metric, subset_label) by examining the column name.
        Interprets:
          - Suffix "_real" as indicating Real.
          - Suffix "_synthetic" or "_syn" as indicating Synthetic.
        If no suffix is found, defaults to Synthetic.
        """
        if col_name.endswith("_real"):
            return col_name[:-5], "Real"
        elif col_name.endswith("_synthetic"):
            return col_name[:-10], "Synthetic"
        elif col_name.endswith("_syn"):
            return col_name[:-4], "Synthetic"
        elif "synthetic" in col_name:
            return col_name.replace("synthetic", ""), "Synthetic"
        elif "real" in col_name:
            return col_name.replace("real", ""), "Real"
        else:
            return col_name, "Synthetic"

    all_rows = []  # To accumulate one-row DataFrames

    # Process each model/CSV pair.
    for model_name, csv_path in model_dict.items():
        df = pd.read_csv(csv_path, header=0)
        print(df)
        if len(df) != 1:
            raise ValueError(
                f"CSV {csv_path} expected to have exactly 1 data row, found {len(df)}."
            )
        row_series = df.iloc[0]
        data_by_subset = {}  # Group metrics by subset (Real or Synthetic)

        # Process each column, grouping its value by the subset label.
        for col_name in row_series.index:
            value = row_series[col_name]
            base_metric, subset_label = parse_suffix(col_name)
            if subset_label not in data_by_subset:
                data_by_subset[subset_label] = {}
            data_by_subset[subset_label][base_metric] = value

        # For each subset, create a one-row DataFrame.
        for subset_label, metric_values in data_by_subset.items():
            # Set the row label: if it's Real, use "Real", otherwise use the model name.
            if subset_label == "Real":
                row_label = "Real"
            else:  # Synthetic
                row_label = model_name
            subset_df = pd.DataFrame([metric_values], index=[row_label])
            all_rows.append(subset_df)

    # Combine all the one-row DataFrames.
    combined_df = pd.concat(all_rows, axis=0)

    # If a metric_name_remap is provided, keep only those columns and rename them.
    if metric_name_remap is not None:
        cols_to_include = [
            metric
            for metric in metric_name_remap.keys()
            if metric in combined_df.columns
        ]
        combined_df = combined_df[cols_to_include]
        combined_df.rename(columns=metric_name_remap, inplace=True)

    # Round all numeric values.
    combined_df = combined_df.round(decimal_places)

    # Remove duplicate Real rows, keeping only the first.
    real_rows = [idx for idx in combined_df.index if idx == "Real"]
    if len(real_rows) > 1:
        # Drop all but the first occurrence of "Real"
        combined_df = combined_df[
            ~(
                (combined_df.index == "Real")
                & (combined_df.index.duplicated(keep="first"))
            )
        ]

    # Reorder rows: the Real row first, then all other rows.
    real_idx = [idx for idx in combined_df.index if idx == "Real"]
    synthetic_idx = [idx for idx in combined_df.index if idx != "Real"]
    ordered_idx = real_idx + synthetic_idx
    combined_df = combined_df.loc[ordered_idx]

    # Generate the LaTeX table.
    latex_str = combined_df.to_latex(
        index=True,
        float_format=f"%.{decimal_places}f",
        na_rep="N/A",
        caption="Model Comparison Table (Real vs. Synthetic)",
        label="tab:model-comparison",
    )

    # Insert an extra \midrule after the Real row.
    # We locate the line that begins with "Real" and then insert a midrule immediately after.
    lines = latex_str.splitlines()
    real_row_index = None
    for i, line in enumerate(lines):
        # Strip leading whitespace and check if the line begins with "Real" (the row label).
        if line.lstrip().startswith("Real"):
            real_row_index = i
            break
    if real_row_index is not None and real_row_index + 1 < len(lines):
        # Insert a midrule if not already present.
        if r"\midrule" not in lines[real_row_index + 1]:
            lines.insert(real_row_index + 1, r"\midrule")
    latex_str = "\n".join(lines)

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
