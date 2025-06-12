"""Helper functions for handling the decoder data."""

import os
import re
import pickle
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

from scipy.stats import sem
from scipy.optimize import minimize

from numpy.polynomial.polynomial import polyfit, Polynomial


plt.rcParams["text.usetex"] = True  # Enable LaTeX in matplotlib
plt.rcParams["font.family"] = "serif"  # Optional: sets font family to serif


def load_data(file_key: str):
    """Load the experiment data from a pickle file."""
    with open(file_key, "rb") as pickle_file:
        data = pickle.load(pickle_file)
    return data


def process_failure_statistics(
    lattice_sizes: list[int],
    max_bond_dims: list[int],
    error_model: str,
    directory: str,
    precision: int = 5,
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
    precision : int, optional
        Precision to round the error rates to. Default is 5.

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
    logicals_distributions_dict : dict
        Dictionary mapping `(lattice_size, chi_max, error_rate)` tuples to logicals distributions.
    failures_statistics : dict
        Dictionary mapping `(lattice_size, chi_max, error_rate)` tuples to failure statistics.

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
    all_unique_error_rates = set()
    logicals_distributions_dict = {}
    failures_statistics_dict = {}

    for lattice_size in lattice_sizes:
        for chi_max in max_bond_dims:
            # Create a regex pattern to match the desired file format
            pattern = rf"^latticesize{lattice_size}_bonddim{chi_max}_errorrate[0-9\.]+(?:e[+-]?[0-9]+)?_errormodel{error_model}+_bias_prob[0-9\.]+(?:e[+-]?[0-9]+)?_numexperiments[0-9]+_tolerance[0-9\.]+(?:e[+-]?[0-9]+)?_cut[0-9\.]+(?:e[+-]?[0-9]+)?_seed\d+\.pkl$"

            all_logicals_distributions = (
                {}
            )  # Dictionary to store the logicals distributions
            all_failures_statistics = {}
            all_errors_statistics = {}  # Dictionary to store errors for each error rate
            error_rates = set()  # Use a set to avoid duplicates

            for file_name in os.listdir(directory):
                if re.match(pattern, file_name):
                    data = load_data(os.path.join(directory, file_name))

                    logicals_distributions = data["logicals_distributions"]
                    failures_statistics = data["failures"]
                    file_errors = data["errors"]
                    file_error_rate = round(data["error_rate"], precision)

                    if file_error_rate not in all_failures_statistics:
                        all_failures_statistics[file_error_rate] = []
                    all_failures_statistics[file_error_rate].extend(failures_statistics)

                    if file_error_rate not in all_errors_statistics:
                        all_errors_statistics[file_error_rate] = []
                    all_errors_statistics[file_error_rate].extend(file_errors)

                    if file_error_rate not in all_logicals_distributions:
                        all_logicals_distributions[file_error_rate] = []
                    all_logicals_distributions[file_error_rate].extend(
                        logicals_distributions
                    )

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
                    # Calculate mean failure rate skipping the nans
                    failures_statistics = np.array(failures_statistics, dtype=object)
                    mask = (
                        failures_statistics is None
                    )  # True wherever the element is None
                    failures_statistics[mask] = np.nan
                    failure_rates[(lattice_size, chi_max, error_rate)] = np.nanmean(
                        failures_statistics
                    )

                    # Calculate standard error of the mean (error bar)
                    try:
                        error_bars[(lattice_size, chi_max, error_rate)] = sem(
                            failures_statistics, nan_policy="omit"
                        )
                    except:
                        error_bars[(lattice_size, chi_max, error_rate)] = sem(
                            failures_statistics
                        )

                    # Store the errors
                    errors_dict[(lattice_size, chi_max, error_rate)] = errors_statistics

                    # Store the logicals distributions
                    logicals_distributions_dict[(lattice_size, chi_max, error_rate)] = (
                        all_logicals_distributions[error_rate]
                    )

                    failures_statistics_dict[(lattice_size, chi_max, error_rate)] = (
                        failures_statistics
                    )

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
        logicals_distributions_dict,
        failures_statistics_dict,
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

            for chi_max, errors in zip(max_bond_dims, errors_by_chi_max):
                current_errors = set(errors)  # Convert current list to set
                if current_errors == reference_errors:
                    continue
                else:
                    num_inconsistent = len(
                        reference_errors.symmetric_difference(current_errors)
                    )
                    total_inconsistencies += num_inconsistent
                    total_checked += num_total
                    print(
                        f"lattice_size={lattice_size}, chi_max={chi_max}: Inconsistent ({num_inconsistent}/{num_total}), {num_inconsistent/num_total*100:.2f}%"
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
    xlim: tuple = None,
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
    xlim : tuple, optional
        Limits for the x-axis. Default is automatic.
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
            plt.figure(figsize=(6, 4))
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
            if xlim:
                plt.xlim(xlim)
            plt.xscale(xscale)
            plt.yscale(yscale)
            plt.grid(True)
            plt.show()

    # Mode 2: Fixed bond dimension, vary lattice sizes
    elif mode == "bond_dim":
        for chi_max in max_bond_dims:
            plt.figure(figsize=(6, 4))
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
            if xlim:
                plt.xlim(xlim)
            plt.xscale(xscale)
            plt.yscale(yscale)
            plt.grid(True)
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
    precision: int = 5,
):
    """
    Analyze and fit failure rates for different lattice sizes and bond dimensions
    using the finite-size scaling ansatz from arXiv:2101.04125, considering only
    error rates within two cutoff values.
    """

    # Import colormap
    cmap = matplotlib.colormaps["viridis_r"]
    norm = Normalize(vmin=0, vmax=len(lattice_sizes) - 1)

    # Scaling function (quadratic approximation)
    def scaling_function(x, a0, a1, a2):
        return a0 + a1 * x + a2 * x**2

    # Fitting function based on arXiv:2101.04125
    def fit_function(p, p_th, nu, a0, a1, a2, L):
        x = (p - p_th) * L ** (1 / nu)
        return scaling_function(x, a0, a1, a2)

    # Loop over bond dimensions
    for chi_max in max_bond_dims:
        results = {}
        all_data_points = []  # Store all data points for collective plotting

        for lattice_size in lattice_sizes:
            filtered_error_rates = []
            filtered_failure_rates = []
            weights = []

            # Filter data based on cutoff values and rounding
            for error_rate in error_rates_dict.get((lattice_size, chi_max), []):
                rounded_error_rate = round(error_rate, precision)

                if not (lower_cutoff <= rounded_error_rate <= upper_cutoff):
                    continue
                failure_rate = failure_rates.get(
                    (lattice_size, chi_max, rounded_error_rate), None
                )
                error_bar = error_bars.get(
                    (lattice_size, chi_max, rounded_error_rate), None
                )

                if failure_rate is not None:
                    filtered_error_rates.append(rounded_error_rate)
                    filtered_failure_rates.append(failure_rate)

                    if error_bar is None or error_bar == 0:
                        weights.append(1.0)
                    else:
                        weights.append(1 / (error_bar**2))

            if not filtered_error_rates:
                print(
                    f"No valid data for lattice_size={lattice_size}, chi_max={chi_max}"
                )
                continue

            filtered_error_rates = np.array(filtered_error_rates)
            filtered_failure_rates = np.array(filtered_failure_rates)
            weights = np.array(weights)

            # Store data points for later plotting
            all_data_points.append(
                (filtered_error_rates, filtered_failure_rates, lattice_size)
            )

            # Objective function for fitting
            def objective_function(params):
                p_th, nu, a0, a1, a2 = params
                model = fit_function(
                    filtered_error_rates, p_th, nu, a0, a1, a2, lattice_size
                )

                if len(weights) != len(filtered_failure_rates):
                    print(
                        f"Warning: Weight length mismatch for L={lattice_size}, Chi={chi_max}"
                    )
                    return np.inf

                residuals = (filtered_failure_rates - model) ** 2 * weights
                return np.sum(residuals)

            # Initial parameter guess (based on reasonable values)
            initial_guess = [
                np.median(filtered_error_rates),
                1.0,
                np.mean(filtered_failure_rates),
                0.0,
                0.0,
            ]

            # Perform the fitting using a more robust optimizer
            result = minimize(objective_function, initial_guess, method="Powell")

            # Handle failed optimization
            if not result.success:
                print(
                    f"Warning: Fit did not converge for lattice_size={lattice_size}, chi_max={chi_max}"
                )
                continue

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

        # Skip plotting if no valid results exist
        if not results:
            print(f"Skipping plotting for chi_max={chi_max} due to no valid fits.")
            continue

        # Create the figure once, plot all data points and fits together
        plt.figure(figsize=(6, 4))
        has_valid_data = False

        # Plot all raw data points first
        for index, (error_rates, failurerates, lattice_size) in enumerate(
            all_data_points
        ):
            error_bars_plot = [
                error_bars.get((lattice_size, chi_max, error_rate), 0)
                for error_rate in error_rates
            ]
            plt.errorbar(
                error_rates,
                failurerates,
                yerr=error_bars_plot,
                fmt="o--",
                label=f"Lattice size: {lattice_size}",
                linewidth=3,
                color=cmap(norm(index)),
            )
            has_valid_data = True

        # Now plot the fitted curves
        for index, (lattice_size, params) in enumerate(results.items()):
            p_th, nu, a0, a1, a2 = (
                params["p_th"],
                params["nu"],
                params["a0"],
                params["a1"],
                params["a2"],
            )

            # Generate fit line for visualization
            fit_p = np.linspace(lower_cutoff, upper_cutoff, 500)
            fit_y = fit_function(fit_p, p_th, nu, a0, a1, a2, lattice_size)

            plt.plot(
                fit_p,
                fit_y,
                label=f"Fit L={lattice_size}, p_th={p_th*100:.2f}%, nu={nu:.2f}",
                linewidth=3,
                color=cmap(norm(index)),  # Ensure fit curve matches data color
            )

        if has_valid_data:
            plt.xlabel("Physical Error Rate (p)")
            plt.ylabel("Logical Failure Rate (P_L)")
            plt.legend(fontsize=7)
            plt.xscale(xscale)
            plt.yscale(yscale)
            plt.grid(True)
            plt.show()
        else:
            print(f"Skipping plot for chi_max={chi_max} due to no valid data.")


def plot_bond_dimension_scaling(
    failure_rates: dict,
    target_error_rate: float,
    threshold: float,
):
    """
    Plot the minimum bond dimension required to achieve a target error rate
    as a function of lattice size, with a linear fit.

    Parameters
    ----------
    failure_rates : dict
        Dictionary mapping `(lattice_size, bond_dimension, error_rate)` tuples
        to logical failure rates.
    target_error_rate : float
        The target error rate to consider for the plot.
    threshold : float
        The threshold for logical failure rates to consider in the plot.
    """
    min_bond_dim = {}

    for (lattice_size, bond_dimension, error_rate), fail_rate in failure_rates.items():
        if error_rate == target_error_rate and fail_rate <= threshold:
            if (
                lattice_size not in min_bond_dim
                or bond_dimension < min_bond_dim[lattice_size]
            ):
                min_bond_dim[lattice_size] = bond_dimension

    if not min_bond_dim:
        print("No data below threshold found.")
        return

    lattice_sizes = sorted(min_bond_dim.keys())
    bond_dims = [min_bond_dim[lattice_size] for lattice_size in lattice_sizes]

    coefs = polyfit(lattice_sizes, bond_dims, deg=2)
    poly = Polynomial(coefs)
    latticesize_fit = np.linspace(0, 50, 2000)
    bondim_fit = poly(latticesize_fit)

    plt.figure(figsize=(6, 4))
    plt.plot(lattice_sizes, bond_dims, "o", label="Minimum bond dim")
    plt.plot(
        latticesize_fit,
        bondim_fit,
        "--",
        label=f"Fit: $\chi \\sim {coefs[2]:.3f}L^2 + {coefs[1]:.3f}L + {coefs[0]:.3f} $",
        linewidth=3,
    )
    plt.xlabel("Lattice size $L$")
    plt.ylabel("Bond dimension $\chi$")
    plt.title(f"Bond dimension scaling at $p = {target_error_rate}$")
    plt.legend(fontsize=7)
    plt.grid(True)
    plt.xlim(min(lattice_sizes) - 1, max(lattice_sizes) + 1)
    plt.ylim(min(bond_dims) - 1, max(bond_dims) + 1)
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()
