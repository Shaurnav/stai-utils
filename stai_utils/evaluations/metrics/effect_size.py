import math
import numpy as np


def cohens_d(x1, x2):
    """
    Compute Cohen’s d between two 1D lists (or arrays) of values.

    Parameters
    ----------
    x1 : array-like
        1D array or list of numerical values.
    x2 : array-like
        1D array or list of numerical values.

    Returns
    -------
    float
        The Cohen’s d value comparing the two samples.
        If the pooled standard deviation is 0, returns 0.0.
    """
    x1 = np.array(x1, dtype=float)
    x2 = np.array(x2, dtype=float)

    # Compute means, standard deviations, and sizes
    mean1, mean2 = x1.mean(), x2.mean()
    sd1, sd2 = x1.std(ddof=1), x2.std(ddof=1)
    n1, n2 = len(x1), len(x2)

    # Pooled standard deviation
    pooled_sd = math.sqrt(((n1 - 1) * (sd1**2) + (n2 - 1) * (sd2**2)) / (n1 + n2 - 2))

    if pooled_sd == 0:
        # If there's zero variance, define Cohen's d as 0 (or NaN if you prefer)
        raise ValueError(
            "The pooled standard deviation is zero. Cohen's d is undefined."
        )

    return (mean1 - mean2) / pooled_sd


def compute_cohens_d(dict1, dict2):
    """
    Compute Cohen’s d for each brain structure across two dictionaries of segmentations.

    Parameters
    ----------
    dict1 : dict
        A dictionary of the form:
            {
                "seg_path_A": {structure_id_1: voxel_count_1, structure_id_2: voxel_count_2, ...},
                "seg_path_B": {...},
                ...
            }
    dict2 : dict
        Another dictionary of the same format as dict1.

    Returns
    -------
    dict
        A dictionary mapping each structure_id to the Cohen’s d value
        computed between the voxel counts from dict1 and dict2.
    """
    # 1. Gather all unique structure IDs from both dictionaries
    all_structures = set()
    for seg_path, struct_dict in dict1.items():
        all_structures.update(struct_dict.keys())
    for seg_path, struct_dict in dict2.items():
        all_structures.update(struct_dict.keys())

    # 2. For each structure, compute Cohen’s d
    results = {}
    for structure_id in sorted(all_structures):
        # Gather voxel counts for this structure from dict1
        x1 = []
        for seg_path, struct_dict in dict1.items():
            if structure_id in struct_dict:
                x1.append(struct_dict[structure_id])

        # Gather voxel counts for this structure from dict2
        x2 = []
        for seg_path, struct_dict in dict2.items():
            if structure_id in struct_dict:
                x2.append(struct_dict[structure_id])

        # Ensure we have at least one value in each list
        if len(x1) == 0 or len(x2) == 0:
            # If either set is empty, define Cohen's d as NaN or skip
            results[structure_id] = float("nan")
            continue

        # Use our separate cohens_d function
        d_value = cohens_d(x1, x2)
        results[structure_id] = d_value

    return results


# ------------------------
# Example Usage
# ------------------------
if __name__ == "__main__":
    dict1 = {
        "seg_path_1": {0: 100, 1: 200, 2: 300},
        "seg_path_2": {0: 110, 1: 210, 2: 290},
        "seg_path_3": {0: 120, 1: 220, 2: 310},
    }

    dict2 = {
        "seg_path_4": {0: 95, 1: 205, 2: 285},
        "seg_path_5": {0: 108, 1: 210, 2: 295},
    }

    result = compute_cohens_d(dict1, dict2)
    print(result)
    # Example output (structure_id : cohen's d)
    # {0: ..., 1: ..., 2: ...}
