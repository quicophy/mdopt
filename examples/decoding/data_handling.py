"""Helper functions for handling decoding data."""

import pickle


def load_data(file_key):
    """Load the experiment data from a pickle file."""
    with open(file_key, "rb") as pickle_file:
        data = pickle.load(pickle_file)
    return data


def check_error_consistency(errors_dict, lattice_sizes, max_bond_dims, error_rate):
    """
    Check if errors are consistent across chi_max for each lattice_size at a fixed error rate,
    ignoring the order of errors.

    Args:
        errors_dict (dict): Dict. containing errors for each (lattice_size, chi_max, error_rate).
        lattice_sizes (list): List of lattice sizes to check.
        max_bond_dims (list): List of bond dimensions to check.
        error_rate (float): The error rate to check for consistency.

    Returns:
        dict: The total number of inconsistencies and the inconsistent lattice sizes.
    """
    total_inconsistencies = 0
    total_checked = 0

    print(
        f"\nChecking error consistency for error rate {error_rate} (ignoring order):\n"
    )

    for lattice_size in lattice_sizes:
        # Collect errors for all chi_max for the given lattice_size and error_rate
        errors_by_chi_max = [
            errors_dict[(lattice_size, chi_max, error_rate)]
            for chi_max in max_bond_dims
            if (lattice_size, chi_max, error_rate) in errors_dict
        ]

        if len(errors_by_chi_max) > 1:
            reference_errors = set(errors_by_chi_max[0])  # Use set to ignore order
            num_total = len(reference_errors)  # Total number of unique errors
            print(f"Lattice size {lattice_size}:")

            for chi_max, errors in zip(max_bond_dims, errors_by_chi_max):
                current_errors = set(errors)  # Convert current list to set
                if current_errors == reference_errors:
                    print(f"  chi_max={chi_max}: Consistent")
                else:
                    num_inconsistent = len(
                        reference_errors.symmetric_difference(current_errors)
                    )
                    total_inconsistencies += num_inconsistent
                    total_checked += num_total
                    print(
                        f"  chi_max={chi_max}: Inconsistent ({num_inconsistent}/{num_total})"
                    )

    return {
        "total_inconsistencies": total_inconsistencies,
        "total_checked": total_checked,
    }
