"""Helper functions for handling decoding data."""

import os
import re
import pickle
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from scipy.stats import sem
from scipy.optimize import minimize


plt.rcParams["text.usetex"] = True  # Enable LaTeX in matplotlib
plt.rcParams["font.family"] = "serif"  # Optional: sets font family to serif


def load_data(file_key: str):
    """Load the experiment data from a pickle file."""
    with open(file_key, "rb") as pickle_file:
        data = pickle.load(pickle_file)
    return data


def process_failure_statistics(
    lattice_sizes: list[int], max_bond_dims: list[int], error_model: str, directory: str
):
    """
    Processes failure statistics for a given set of lattice sizes, bond dimensions,
    and an error model from a specified directory of data files.

    Parameters
    ----------
    lattice_sizes : list of int
        List of lattice sizes to process.
    max_bond_dims : list of int
        List of maximum bond dimensions to consider.
    error_model : str
        Name of the error model to filter files.
    directory : str
        Path to the directory containing data files.

    Returns
    -------
    error_rates_dict : dict
        Dictionary mapping `(lattice_size, chi_max)` tuples to sorted lists of error rates.
    failure_rates : dict
        Dictionary mapping `(lattice_size, chi_max, error_rate)` tuples to mean failure rates.
    error_bars : dict
        Dictionary mapping `(lattice_size, chi_max, error_rate)` tuples to
        error bars (standard error of the mean).
    errors_dict : dict
        Dictionary mapping `(lattice_size, chi_max, error_rate)` tuples to lists of errors.
    sorted_unique_error_rates : list
        A sorted list of all unique error rates found across all lattice sizes and bond dimensions.

    Notes
    -----
    - The function assumes that data files in the directory are named in a specific
      format, including lattice size, bond dimension, error rate, and error model.
    - It filters files using the provided `error_model` string.
    """
    error_rates_dict = {}
    failure_rates = {}
    error_bars = {}
    errors_dict = {}
    all_unique_error_rates = set()  # Set to store unique error rates

    for lattice_size in lattice_sizes:
        for chi_max in max_bond_dims:
            # Create a regex pattern to match the desired file format
            pattern = rf"^latticesize{lattice_size}_bonddim{chi_max}_errorrate[0-9\.]+_errormodel{error_model}_seed\d+\.pkl$"

            all_failures_statistics = {}
            all_errors_statistics = {}  # Dictionary to store errors for each error rate
            error_rates = set()  # Use a set to avoid duplicates

            for file_name in os.listdir(directory):
                if re.match(pattern, file_name):
                    data = load_data(os.path.join(directory, file_name))

                    failures_statistics = data["failures"]
                    file_errors = data["errors"]
                    file_error_rate = round(data["error_rate"], 5)

                    if file_error_rate not in all_failures_statistics:
                        all_failures_statistics[file_error_rate] = []
                    all_failures_statistics[file_error_rate].extend(failures_statistics)

                    if file_error_rate not in all_errors_statistics:
                        all_errors_statistics[file_error_rate] = []
                    all_errors_statistics[file_error_rate].extend(file_errors)

                    # Add the error rate to the sets
                    error_rates.add(file_error_rate)
                    all_unique_error_rates.add(file_error_rate)

            # Sort and store the error rates
            sorted_error_rates = sorted(error_rates)
            error_rates_dict[(lattice_size, chi_max)] = sorted_error_rates

            # Calculate mean failure rates, error bars, and store errors
            for error_rate in sorted_error_rates:
                failures_statistics = all_failures_statistics[error_rate]
                errors_statistics = all_errors_statistics[error_rate]

                if failures_statistics:
                    # Calculate mean failure rate
                    failure_rates[(lattice_size, chi_max, error_rate)] = np.mean(
                        failures_statistics
                    )

                    # Calculate standard error of the mean (error bar)
                    error_bars[(lattice_size, chi_max, error_rate)] = sem(
                        failures_statistics
                    )

                    # Store the errors
                    errors_dict[(lattice_size, chi_max, error_rate)] = errors_statistics
                else:
                    print(
                        f"No data for lattice_size={lattice_size}, chi_max={chi_max}, error_rate={error_rate}"
                    )

    return (
        error_rates_dict,
        failure_rates,
        error_bars,
        errors_dict,
        sorted(all_unique_error_rates),
    )


def check_error_consistency(
    errors_dict: dict,
    lattice_sizes: list[int],
    max_bond_dims: list[int],
    error_rate: float,
):
    """
    Check if errors are the same across `chi_max` for each `lattice_size` at a fixed error rate,
    ignoring the order of errors,
    this is to ensure the correct evaluation of the decoder's performance.

    Parameters
    ----------
    errors_dict : dict
        Dictionary containing errors for each `(lattice_size, chi_max, error_rate)` tuple.
    lattice_sizes : list of int
        List of lattice sizes to check for consistency.
    max_bond_dims : list of int
        List of bond dimensions to check for consistency.
    error_rate : float
        The error rate to check for consistency.

    Returns
    -------
    dict
        A dictionary containing:
        - 'total_inconsistencies' (int): The total number of inconsistencies found.
        - 'inconsistent_lattice_sizes' (list): List of lattice sizes where inconsistencies were found.

    Notes
    -----
    This function compares errors across different bond dimensions (`chi_max`) for a given lattice size
    and error rate. The order of errors is ignored during the comparison.
    """

    total_inconsistencies = 0
    total_checked = 0

    # print(
    #    f"\nChecking error consistency for error rate {error_rate} (ignoring order):\n"
    # )

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
            # print(f"Lattice size {lattice_size}:")

            for chi_max, errors in zip(max_bond_dims, errors_by_chi_max):
                current_errors = set(errors)  # Convert current list to set
                if current_errors == reference_errors:
                    # print(f"  chi_max={chi_max}: Consistent")
                    continue
                else:
                    num_inconsistent = len(
                        reference_errors.symmetric_difference(current_errors)
                    )
                    total_inconsistencies += num_inconsistent
                    total_checked += num_total
                    print(
                        f"  chi_max={chi_max}: Inconsistent ({num_inconsistent}/{num_total})"
                    )

    if total_inconsistencies == 0:
        print("No inconsistencies found.")

    return {
        "total_inconsistencies": total_inconsistencies,
        "total_checked": total_checked,
    }


def plot_failure_statistics(
    error_rates_dict: dict,
    failure_rates: dict,
    error_bars: dict,
    lattice_sizes: list[int],
    max_bond_dims: list[int],
    mode: str = "lattice_size",
    xscale: str = "linear",
    yscale: str = "linear",
):
    """
    Plot failure rates with error bars for either varying bond dimensions at all lattice sizes,
    or varying lattice sizes at all bond dimensions.

    Parameters
    ----------
    error_rates_dict : dict
        Dictionary mapping `(lattice_size, chi_max)` tuples to lists of error rates.
    failure_rates : dict
        Dictionary mapping `(lattice_size, chi_max, error_rate)` tuples to failure rates.
    error_bars : dict
        Dictionary mapping `(lattice_size, chi_max, error_rate)` tuples to error bars.
    lattice_sizes : list of int
        List of lattice sizes to consider.
    max_bond_dims : list of int
        List of bond dimensions to consider.
    mode : str, optional
        Plotting mode: "lattice_size" (varying bond dimensions for each lattice size)
        or "bond_dim" (varying lattice sizes for each bond dimension). Default is "lattice_size".
    xscale : str, optional
        Scale of the x-axis. Default is "linear".
    yscale : str, optional
        Scale of the y-axis. Default is "linear".

    Returns
    -------
    None
        Generates and shows the plots.
    """
    if mode not in ["lattice_size", "bond_dim"]:
        raise ValueError("Mode must be either 'lattice_size' or 'bond_dim'.")

    # Setup colormap
    cmap = matplotlib.colormaps["viridis_r"]

    # Mode 1: Fixed lattice size, vary bond dimensions
    if mode == "lattice_size":
        for lattice_size in lattice_sizes:
            plt.figure(figsize=(6, 5))
            norm = Normalize(vmin=0, vmax=len(max_bond_dims) - 1)

            for index, chi_max in enumerate(max_bond_dims):
                key = (lattice_size, chi_max)
                if key not in error_rates_dict:
                    print(
                        f"No data for lattice size {lattice_size}, bond dimension {chi_max}. Skipping."
                    )
                    continue

                error_rates = error_rates_dict[key]
                plt.errorbar(
                    error_rates,
                    [
                        failure_rates[lattice_size, chi_max, error_rate]
                        for error_rate in error_rates
                    ],
                    yerr=[
                        error_bars[lattice_size, chi_max, error_rate]
                        for error_rate in error_rates
                    ],
                    fmt="o--",
                    label=f"Bond dim: {chi_max}",
                    linewidth=3,
                    color=cmap(norm(index)),
                )

            plt.title(f"Failure Rate vs Error Rate (Lattice size = {lattice_size})")
            plt.xlabel("Error Rate")
            plt.ylabel("Failure Rate")
            plt.legend(fontsize=7)
            plt.grid()
            plt.xscale(xscale)
            plt.yscale(yscale)
            plt.show()

    # Mode 2: Fixed bond dimension, vary lattice sizes
    elif mode == "bond_dim":
        for chi_max in max_bond_dims:
            plt.figure(figsize=(6, 5))
            norm = Normalize(vmin=0, vmax=len(lattice_sizes) - 1)

            for index, lattice_size in enumerate(lattice_sizes):
                key = (lattice_size, chi_max)
                if key not in error_rates_dict:
                    print(
                        f"No data for lattice size {lattice_size}, bond dimension {chi_max}. Skipping."
                    )
                    continue

                error_rates = error_rates_dict[key]
                plt.errorbar(
                    error_rates,
                    [
                        failure_rates[lattice_size, chi_max, error_rate]
                        for error_rate in error_rates
                    ],
                    yerr=[
                        error_bars[lattice_size, chi_max, error_rate]
                        for error_rate in error_rates
                    ],
                    fmt="o--",
                    label=f"Lattice size: {lattice_size}",
                    linewidth=3,
                    color=cmap(norm(index)),
                )

            plt.title(f"Failure Rate vs Error Rate (Bond dim = {chi_max})")
            plt.xlabel("Error Rate")
            plt.ylabel("Failure Rate")
            plt.legend(fontsize=7)
            plt.grid()
            plt.xscale(xscale)
            plt.yscale(yscale)
            plt.show()


def plot_failure_statistics_fixed_rates(
    failure_rates: dict,
    error_bars: dict,
    lattice_sizes: list[int],
    max_bond_dims: list[int],
    error_rates: list[float],
    mode: str = "lattice_size",
    xscale: str = "linear",
    yscale: str = "linear",
):
    """
    Plot failure rates with error bars as a function of error rates, either varying bond dimensions
    at all lattice sizes or varying lattice sizes at all bond dimensions.

    Parameters
    ----------
    failure_rates : dict
        Dictionary mapping `(lattice_size, chi_max, error_rate)` tuples to failure rates.
    error_bars : dict
        Dictionary mapping `(lattice_size, chi_max, error_rate)` tuples to error bars.
    lattice_sizes : list of int
        List of lattice sizes to consider.
    max_bond_dims : list of int
        List of bond dimensions to consider.
    error_rates : list of float
        List of error rates to use for the x-axis.
    mode : str, optional
        Plotting mode: "lattice_size" (varying bond dimensions for each lattice size)
        or "bond_dim" (varying lattice sizes for each bond dimension). Default is "lattice_size".
    xscale : str, optional
        Scale of the x-axis. Default is "linear".
    yscale : str, optional
        Scale of the y-axis. Default is "linear".

    Returns
    -------
    None
        Generates and shows the plots.
    """
    # Validate mode
    if mode not in ["lattice_size", "bond_dim"]:
        raise ValueError("Mode must be either 'lattice_size' or 'bond_dim'.")

    # Colormap setup
    from matplotlib.colors import Normalize
    import matplotlib.pyplot as plt
    import matplotlib

    cmap = matplotlib.colormaps["viridis_r"]

    # Mode 1: Fixed lattice size, vary bond dimensions
    if mode == "lattice_size":
        for lattice_size in lattice_sizes:
            plt.figure(figsize=(6, 5))
            norm = Normalize(vmin=0, vmax=len(max_bond_dims) - 1)

            for index, chi_max in enumerate(max_bond_dims):
                failure_rate_values = []
                error_bar_values = []

                # Collect failure rates and error bars for the given error rates
                for error_rate in error_rates:
                    failure_rate_values.append(
                        failure_rates.get((lattice_size, chi_max, error_rate), None)
                    )
                    error_bar_values.append(
                        error_bars.get((lattice_size, chi_max, error_rate), 0)
                    )

                # Plot only if any failure rates exist
                if any(failure_rate_values):
                    plt.errorbar(
                        error_rates,
                        failure_rate_values,
                        yerr=error_bar_values,
                        fmt="o--",
                        label=f"Bond dim: {chi_max}",
                        linewidth=2,
                        color=cmap(norm(index)),
                    )

            plt.title(f"Failure Rate vs Error Rate (Lattice size = {lattice_size})")
            plt.xlabel("Error Rate")
            plt.ylabel("Failure Rate")
            plt.legend(fontsize=7)
            plt.grid()
            plt.xscale(xscale)
            plt.yscale(yscale)
            plt.show()

    # Mode 2: Fixed bond dimension, vary lattice sizes
    elif mode == "bond_dim":
        for chi_max in max_bond_dims:
            plt.figure(figsize=(6, 5))
            norm = Normalize(vmin=0, vmax=len(lattice_sizes) - 1)

            for index, lattice_size in enumerate(lattice_sizes):
                failure_rate_values = []
                error_bar_values = []

                # Collect failure rates and error bars for the given error rates
                for error_rate in error_rates:
                    failure_rate_values.append(
                        failure_rates.get((lattice_size, chi_max, error_rate), None)
                    )
                    error_bar_values.append(
                        error_bars.get((lattice_size, chi_max, error_rate), 0)
                    )

                # Plot only if any failure rates exist
                if any(failure_rate_values):
                    plt.errorbar(
                        error_rates,
                        failure_rate_values,
                        yerr=error_bar_values,
                        fmt="o--",
                        label=f"Lattice size: {lattice_size}",
                        linewidth=2,
                        color=cmap(norm(index)),
                    )

            plt.title(f"Failure Rate vs Error Rate (Bond dim = {chi_max})")
            plt.xlabel("Error Rate")
            plt.ylabel("Failure Rate")
            plt.legend(fontsize=7)
            plt.grid()
            plt.xscale(xscale)
            plt.yscale(yscale)
            plt.show()


def fit_failure_statistics(
    lattice_sizes: list[int],
    max_bond_dims: list[int],
    error_rates_dict: dict,
    failure_rates: dict,
    error_bars: dict,
    lower_cutoff: float,
    upper_cutoff: float,
    xscale: str = "linear",
    yscale: str = "linear",
):
    """
    Analyze and fit failure rates for different lattice sizes and bond dimensions
    using a scaling function, considering only error rates within two cutoff values.

    Parameters
    ----------
    lattice_sizes : list of int
        List of lattice sizes to analyze.
    max_bond_dims : list of int
        List of maximum bond dimensions to consider.
    error_rates_dict : dict
        Dictionary mapping (lattice_size, chi_max) to lists of error rates.
    failure_rates : dict
        Dictionary mapping (lattice_size, chi_max, error_rate) to failure rates.
    error_bars : dict
        Dictionary mapping (lattice_size, chi_max, error_rate) to error bars.
    lower_cutoff : float
        Lower cutoff value for the error rate. Only error rates above this value are considered.
    upper_cutoff : float
        Upper cutoff value for the error rate. Only error rates below this value are considered.
    xscale : str, optional
        Scale of the x-axis. Default is "linear".
    yscale : str, optional
        Scale of the y-axis. Default is "linear".

    Returns
    -------
    None
        Displays results and generates plots for each bond dimension.

    Notes
    -----
    The function fits failure rate data using a polynomial scaling function, ignoring
    error rates outside the specified cutoffs.
    Results and fit curves are visualized for each `chi_max`.
    """

    # Scaling function (polynomial approximation)
    def scaling_function(x, a0, a1, a2):
        return a0 + a1 * x + a2 * x**2

    # Fitting function
    def fit_function(p, p_th, nu, a0, a1, a2, d):
        x = (p - p_th) * d ** (1 / nu)
        return scaling_function(x, a0, a1, a2)

    # Loop over bond dimensions
    for chi_max in max_bond_dims:
        results = {}
        for lattice_size in lattice_sizes:
            error_rates = []
            failure_rates_flat = []
            weights = []

            # Filter data based on cutoff values
            for error_rate in error_rates_dict[(lattice_size, chi_max)]:
                if not (lower_cutoff <= error_rate <= upper_cutoff):
                    print(
                        f"Ignoring error rate {error_rate} for lattice size {lattice_size} and chi_max {chi_max} due to cutoffs."
                    )
                    continue

                failure_rate = failure_rates.get(
                    (lattice_size, chi_max, error_rate), None
                )
                error_bar = error_bars.get((lattice_size, chi_max, error_rate), None)

                if failure_rate is not None:
                    error_rates.append(error_rate)
                    failure_rates_flat.append(failure_rate)
                    weights.append(
                        1.0
                        if error_bar is None or error_bar == 0
                        else 1 / (error_bar**2)
                    )

            if not error_rates:
                print(
                    f"No valid data for lattice_size={lattice_size}, chi_max={chi_max}"
                )
                continue

            # Convert to numpy arrays
            error_rates = np.array(error_rates)
            failure_rates_flat = np.array(failure_rates_flat)
            weights = np.array(weights)

            # Objective function for fitting
            def objective_function(params):
                p_th, nu, a0, a1, a2 = params
                model = fit_function(error_rates, p_th, nu, a0, a1, a2, lattice_size)
                residuals = (failure_rates_flat - model) ** 2 * weights
                return np.sum(residuals)

            # Initial parameter guess
            initial_guess = [0.1, 1.0, 0.1, 0.1, 0.1]

            # Perform the fitting
            result = minimize(objective_function, initial_guess)

            # Extract parameters
            p_th, nu, a0, a1, a2 = result.x
            results[lattice_size] = {
                "p_th": p_th,
                "nu": nu,
                "a0": a0,
                "a1": a1,
                "a2": a2,
            }

            print(f"Lattice size: {lattice_size}, Bond dimension: {chi_max}")
            print(f"  Estimated threshold (p_th): {p_th*100:.5f}%")
            print(f"  Scaling exponent (nu): {nu:.5f}")

        # Plot the results
        plt.figure()
        for lattice_size in results:
            params = results[lattice_size]
            p_th, nu, a0, a1, a2 = (
                params["p_th"],
                params["nu"],
                params["a0"],
                params["a1"],
                params["a2"],
            )

            # Scatter plot with error bars
            valid_error_rates = [
                er for er in error_rates if lower_cutoff <= er <= upper_cutoff
            ]
            valid_failure_rates = [
                failure_rates_flat[i]
                for i, er in enumerate(error_rates)
                if lower_cutoff <= er <= upper_cutoff
            ]
            valid_error_bars = [
                error_bars.get((lattice_size, chi_max, er), 0)
                for er in valid_error_rates
            ]

            plt.errorbar(
                valid_error_rates,
                valid_failure_rates,
                yerr=valid_error_bars,
                fmt="o",
                label=f"Lattice size {lattice_size}",
            )

            # Fit line for visualization
            fit_p = np.linspace(min(valid_error_rates), max(valid_error_rates), 500)
            fit_y = fit_function(fit_p, p_th, nu, a0, a1, a2, lattice_size)
            plt.plot(
                fit_p,
                fit_y,
                label=f"Fit L={lattice_size}, p_th={p_th*100:.2f}%, nu={nu:.2f}",
            )

            plt.axvline(
                x=p_th,
                linestyle="--",
                color="gray",
                label=f"Threshold p_th={p_th*100:.2f}%",
            )

        plt.title(f"Failure Rate vs Error Rate for Bond Dimension {chi_max}")
        plt.xlabel("Physical Error Rate (p)")
        plt.ylabel("Logical Failure Rate (P_L)")
        plt.legend()
        plt.grid()
        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.show()
