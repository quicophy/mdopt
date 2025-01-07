"""This module allows visualising the MPS decoder and parity check MPOs for a CSS code."""

import numpy as np
from matrex import msro
from qecstruct import CssCode  # pylint: disable=E0611
import matplotlib.pyplot as plt
from examples.decoding.decoding import (
    css_code_logicals_sites,
    css_code_constraint_sites,
)


def plot_parity_check_mpo(
    code: CssCode, optimise_order=False, return_matrix=False, plot_type="both"
):
    """
    Plots the parity check MPOs for a CSS code with optional order optimisation.

    Parameters
    ----------
    code : CssCode
        The quantum CSS code object.
    optimise_order : bool, optional
        Whether to optimise the order of applying the constraints.
    return_matrix : bool, optional
        Whether to return the MPO location matrix.
    plot_type : str, optional
        Specifies which part of the code to plot.
        Options: "both", "X", "Z".
        - "both": Plots both X and Z parts of the code.
        - "X": Plots only the X-part of the code.
        - "Z": Plots only the Z-part of the code.

    Notes
    -----
    - The first sublist corresponds to specific tensor types:
        - XOR_LEFT, XOR_BULK, SWAP, XOR_RIGHT (for stabiliser parity check sites)
        - COPY_LEFT, XOR_BULK, SWAP, XOR_RIGHT (for logical parity check sites)
    """

    if not isinstance(code, CssCode):
        raise ValueError("Unsupported code type. Only CssCode is supported.")

    tensor_types = ["XOR_LEFT", "XOR_BULK", "XOR_RIGHT", "SWAP", "COPY_LEFT"]
    tensor_colors = {tensor: i + 1 for i, tensor in enumerate(tensor_types)}

    logicals_sites_x, logicals_sites_z = css_code_logicals_sites(code)
    logicals_sites = []
    if plot_type in ["both", "X"]:
        logicals_sites += logicals_sites_x
    if plot_type in ["both", "Z"]:
        logicals_sites += logicals_sites_z

    constraint_sites_x, constraint_sites_z = css_code_constraint_sites(code)
    constraint_sites = []
    if plot_type in ["both", "X"]:
        constraint_sites += constraint_sites_x
    if plot_type in ["both", "Z"]:
        constraint_sites += constraint_sites_z

    num_constraint_checks = sum(
        [
            code.num_x_stabs() if plot_type in ["both", "X"] else 0,
            code.num_z_stabs() if plot_type in ["both", "Z"] else 0,
        ]
    )
    num_logical_checks = sum(
        [
            code.num_x_logicals() if plot_type in ["both", "X"] else 0,
            code.num_z_logicals() if plot_type in ["both", "Z"] else 0,
        ]
    )
    num_sites = 2 * len(code) + code.num_x_logicals() + code.num_z_logicals()

    mpo_location_matrix = np.zeros(
        (num_constraint_checks + num_logical_checks, num_sites)
    )

    if optimise_order:
        mpo_matrix = np.zeros((num_constraint_checks + num_logical_checks, num_sites))
        for row_idx, sublist in enumerate(logicals_sites):
            for sublist_indices in sublist:
                for index in sublist_indices:
                    mpo_matrix[row_idx][index] = 1
        for row_idx, sublist in enumerate(constraint_sites):
            for sublist_indices in sublist:
                for index in sublist_indices:
                    mpo_matrix[row_idx + num_logical_checks][index] = 1

        optimised_order = msro(mpo_matrix)

        logical_indices = list(range(num_logical_checks))
        constraint_indices = list(
            range(num_logical_checks, num_logical_checks + num_constraint_checks)
        )

        logicals_order = [i for i in optimised_order if i in logical_indices]
        constraints_order = [i for i in optimised_order if i in constraint_indices]

        logicals_sites = [logicals_sites[i] for i in logicals_order]
        constraint_sites = [
            constraint_sites[i - num_logical_checks] for i in constraints_order
        ]

    # Fill matrix for logical sites
    for row_idx, site_group in enumerate(logicals_sites):
        for tensor_idx, tensor_type in enumerate(
            ["COPY_LEFT", "XOR_BULK", "SWAP", "XOR_RIGHT"]
        ):
            if tensor_idx < len(site_group):
                indices = site_group[tensor_idx]
                indices = (
                    [indices] if isinstance(indices, (int, np.integer)) else indices
                )
                for index in indices:
                    mpo_location_matrix[row_idx][index] = tensor_colors[tensor_type]

    # Fill matrix for constraint sites
    for row_idx, site_group in enumerate(constraint_sites):
        for tensor_idx, tensor_type in enumerate(
            ["XOR_LEFT", "XOR_BULK", "SWAP", "XOR_RIGHT"]
        ):
            if tensor_idx < len(site_group):
                indices = site_group[tensor_idx]
                indices = (
                    [indices] if isinstance(indices, (int, np.integer)) else indices
                )
                for index in indices:
                    mpo_location_matrix[num_logical_checks + row_idx][index] = (
                        tensor_colors[tensor_type]
                    )

    _, ax = plt.subplots(figsize=(10, 4))
    cmap = plt.colormaps["viridis"]

    # Define grid coordinates for pcolormesh
    x = np.arange(num_sites + 1)
    y = np.arange(num_constraint_checks + num_logical_checks + 1)

    # Use pcolormesh for plotting with gridlines
    im = ax.pcolormesh(
        x, y, mpo_location_matrix, cmap=cmap, edgecolors="k", linewidth=0.5
    )

    # Reverse the y-axis to mimic origin='upper'
    ax.invert_yaxis()

    # Set ticks and labels
    ax.set_xticks(np.arange(0, num_sites, 4) + 0.5)
    ax.set_yticks(np.arange(0, num_constraint_checks + num_logical_checks, 4) + 0.5)
    ax.set_xticklabels(np.arange(0, num_sites, 4))
    ax.set_yticklabels(np.arange(0, num_constraint_checks + num_logical_checks, 4))

    cbar = plt.colorbar(im, ax=ax, ticks=np.arange(0, len(tensor_types) + 1))
    cbar.ax.set_yticklabels(["None"] + tensor_types)

    ax.set_xlabel("MPS Site Index")
    ax.set_ylabel("Parity Check Index")
    plot_title_part = "X and Z parts" if plot_type == "both" else f"{plot_type}-part"
    ax.set_title(
        f"Parity Check MPO Structure ({plot_title_part}, "
        f"{'optimised' if optimise_order else 'Unoptimised'})"
    )

    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    plt.show()

    if return_matrix:
        return np.vectorize({v: k for k, v in tensor_colors.items()}.get)(
            mpo_location_matrix
        )
