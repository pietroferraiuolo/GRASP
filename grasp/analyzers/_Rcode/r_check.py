"""R package availability checks (lazy).

Authors
-------
- Pietro Ferraiuolo : written in 2024

Notes
-----
Historically this module called ``utils.chooseCRANmirror(ind=1)`` and
``utils.install_packages(...)`` at *import time*. That made simply
running ``python -c "import grasp"`` (or ``pytest``) fail inside any
offline / sandboxed environment because rpy2 tried to dial out to the
CRAN mirror list during module load.

Phase 3 of the cleanup makes this strictly lazy: importing this module
does nothing beyond setting the rpy2 log level. The first time the user
actually requests the R backend (``grasp.stats.fit_distribution(...)``
with ``backend='r'``), :func:`check_packages` is invoked; if a required
package is missing we now *raise* a :class:`RuntimeError` with a clear
install recipe instead of silently calling ``install.packages``.
"""

from __future__ import annotations

from logging import ERROR

REQUIRED_R_PACKAGES: tuple[str, ...] = ("minpack.lm", "mclust")
"""R packages the R backend depends on.

This is the **only** list that documents the runtime R dependencies; keep
it in sync with ``regression.R`` / ``gaussian_mixture.R``.
"""

_INSTALL_HINT = (
    "The R backend requires the package %s. Install it manually from an R "
    "session with\n"
    "    install.packages(\"%s\")\n"
    "or via apt / conda. Then set ``backend='python'`` (the default) to "
    "stay on the supported path."
)


def check_packages(packages: str | list[str] | tuple[str, ...]) -> None:
    """Verify the listed R packages are importable.

    Parameters
    ----------
    packages : str or sequence of str
        R package name(s) to check.

    Raises
    ------
    RuntimeError
        If any required package is missing. The exception message
        includes a copy-pasteable ``install.packages(...)`` recipe.
    ModuleNotFoundError
        If :mod:`rpy2` itself is not installed -- the R backend is now
        an optional extra (``pip install grasp[r]``).
    """
    try:
        from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
        from rpy2.robjects.packages import LibraryError, importr, isinstalled
    except ImportError as e:  # pragma: no cover - rpy2 missing.
        raise ModuleNotFoundError(
            "The R backend requires rpy2. Install the optional extra "
            "with `pip install grasp[r]` to enable it."
        ) from e

    rpy2_logger.setLevel(ERROR)

    if isinstance(packages, str):
        packages = [packages]
    for package in packages:
        if not isinstalled(package):
            raise RuntimeError(_INSTALL_HINT % (package, package))
        try:
            importr(package)
        except LibraryError as e:
            raise RuntimeError(
                f"R package `{package}` is installed but failed to import: {e}"
            ) from e
