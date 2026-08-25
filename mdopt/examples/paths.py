"""Locate the example assets that live outside the package.

The example *code* ships inside ``mdopt.examples`` so it can be imported from
anywhere (see issue #446). The things it reads and writes -- simulation data,
generated figures, notebooks -- deliberately stay in the repository's top-level
``examples/`` directory: the decoding datasets alone run to tens of gigabytes and
have no business inside an installed wheel.

That split means a script can no longer reach its data by walking up from
``__file__``. These helpers do the lookup instead, and raise a clear error rather
than silently writing to the wrong place when the assets are absent, which is the
normal situation for an installed (non-checkout) copy.

Set ``MDOPT_EXAMPLES_ASSETS`` to override the location, e.g. when the data lives
on scratch storage on a cluster.
"""

import os
from pathlib import Path
from typing import Optional

__all__ = [
    "assets_root",
    "decoding_assets",
    "data_dir",
    "figures_dir",
    "figure",
]

_ENV_VAR = "MDOPT_EXAMPLES_ASSETS"


def _find_repo_examples() -> Optional[Path]:
    """Walk up from this file for a checkout that still has ``examples/``.

    Anchored on ``pyproject.toml`` rather than on the directory name: this module
    already sits inside a directory called ``examples``, so matching on the name
    alone would find the package itself.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "examples").is_dir():
            return parent / "examples"
    return None


def assets_root() -> Path:
    """Directory holding the example assets, i.e. the repo's ``examples/``."""
    override = os.environ.get(_ENV_VAR)
    if override:
        root = Path(override).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                f"{_ENV_VAR} is set to {root!s}, which is not a directory."
            )
        return root

    found = _find_repo_examples()
    if found is None:
        raise FileNotFoundError(
            "Could not locate the example assets. They live in the repository's "
            "top-level 'examples/' directory and are not shipped with the "
            f"installed package. Set {_ENV_VAR} to point at them."
        )
    return found


def decoding_assets() -> Path:
    """The ``examples/decoding`` asset directory."""
    return assets_root() / "decoding"


def data_dir(name: str, required: bool = True) -> Path:
    """Path to a decoding dataset directory such as ``data-quantum-surface``."""
    path = decoding_assets() / name
    if required and not path.is_dir():
        raise FileNotFoundError(
            f"Dataset {name!r} not found at {path!s}. The datasets are large and "
            "are not tracked in git; copy or symlink them into place, or point "
            f"{_ENV_VAR} at a directory that holds them."
        )
    return path


def figures_dir() -> Path:
    """Directory for generated figures, created on demand."""
    path = decoding_assets() / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def figure(name: str) -> Path:
    """Absolute path for a figure file, independent of the current directory."""
    return figures_dir() / os.path.basename(name)
