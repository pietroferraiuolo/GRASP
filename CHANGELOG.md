# Changelog

All notable changes to **GRASP** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
loosely adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] -- 2026-05 cleanup

This release consolidates five stacked PRs (`cleanup/phase-{0..4}`) into
a single coherent modernisation pass. The R-backed numerical routines
are now formally deprecated; pure-Python equivalents based on
`scipy`, `lmfit`, `statsmodels` and `scikit-learn` run by default.

### Added (Python-native backends)
- `grasp.analyzers.backends._python` with `fit_distribution_python`,
  `linear_regression_python`, `gaussian_mixture_model_python` and
  `kfold_gmm_python`. (P3-1, P3-2, P3-3)
- `grasp.utils.rng.default_rng(seed=None)` for reproducible RNG;
  all stochastic public APIs accept a `seed=` kwarg. (P3-4)
- `PyFitResult`, `GaussianMixturePython` and `KFoldGMMResult`
  dataclasses with `.save()`/`.load()` via `joblib`. (P3-5)
- `grasp.stats.fit_poisson_mle` implementing the true Poisson MLE via
  `scipy.optimize.minimize` and `scipy.stats.poisson.logpmf`, replacing
  the previous NLS hack that called `factorial` on bin centres. (P1-8)
- `CartesianConversion(input_unit="deg"|"rad")` with proper unit handling
  and an `astropy`-aware wrapper. (P1-3, P1-4)
- New tests: ADQL DR2/DR3 qualifier regression (P1-1),
  `CartesianConversion` units (P1-3), Poisson MLE (P1-8), GMM port
  parity (P3-2), RNG reproducibility (P3-4), joblib round-trips (P3-5).
- `CITATION.cff` (P4-4) and this `CHANGELOG.md` (P4-5).
- `.pre-commit-config.yaml` (P4-7) wiring `ruff`, `ruff-format` and
  basic hygiene hooks.

### Changed
- Packaging: full PEP 621 `[project]` table in `pyproject.toml` with
  `requires-python = ">=3.10,<3.14"`, runtime deps copied from
  `requirements.txt`, optional extras `r`, `dev`, `docs`; `setup.py`
  reduced to a one-line shim; `MANIFEST.in` corrected to package
  `_Catalogue.xlsx`, `base.frm`, the R sources and the new `py.typed`
  marker. (P0-1, P2-1)
- Default backend for `fit_distribution`, `gaussian_mixture_model` and
  `kfold_gmm_estimator` is now Python; the R path is opt-in via
  `backend="r"` or the environment variable `GRASP_R_BACKEND=1`. Every
  R call emits a `DeprecationWarning`. (P3)
- `grasp/__init__.py` uses relative intra-package imports, explicit
  re-exports from `__version__`, and includes `BaseFormula`, the new
  backends, `default_rng`, etc., in `__all__`. (P0-2)
- `grasp.core.folder_paths` no longer creates data directories at
  import time. `initialize_data_layout()` and `ensure_data_dir(path)`
  expose the previous behaviour explicitly. (P2-2)
- `grasp.gaia.query` builds the ADQL `CONTAINS(POINT(...), CIRCLE(...))`
  predicate from `_split_gaia_table(...)` so DR2 tables are no longer
  silently joined with `gaiadr3.gaia_source.ra/dec`. (P1-1)
- `grasp.gaia.query.free_query` docstring example now uses the IVOA
  ADQL 2.0 `CIRCLE('ICRS', ra, dec, radius)` arity. (P1-2)
- `grasp/sysdata/base.frm` records `r_pc = 1000 / varpi_mas` instead of
  the unit-ambiguous `r_x = 1/omega`; the loader docstring documents
  the convention. (P1-5)
- `grasp/_utility/cluster.py` keeps `rt = rc * 10**logc` (Harris 1996,
  2010 edition: `c = log10(rt/rc)`) and now carries an inline citation
  in both code and class docstring. (P1-6)
- `grasp.plots`: fixed
  - inverted `teff_gspphot is not None` branch in `colorMagnitude`
    (the `None` branch now triggers the magnitude-only plot);
  - marginal-axis KDE in `doubleHistScatter` (was plotting `reg_y.x`
    against itself);
  - `histogram(scale="log")` xlabel formatting;
  - `regressionPlot` x-axis padding now uses `(xmax - xmin) * 0.02`
    on both ends. (P0-3)
- `grasp.stats` dead-code `DeprecationWarning(...)` replaced with a
  proper `warnings.warn(..., DeprecationWarning, stacklevel=2)`. (P0-4)
- `grasp.analyzers._Rcode.r_check` no longer mutates the user's R
  session at import time. Missing R packages raise a `RuntimeError`
  with install instructions only when the R backend is invoked. (P3-6)
- `grasp.analyzers._Rcode.r2py_models` imports `rpy2` lazily; `import
  grasp` works without the `[r]` extra.
- `print(...)` calls in `folder_paths.py`, `analyzers/calculus.py`,
  `_utility/cluster.py` and `gaia/query.py` migrated to a package-level
  `logging.getLogger("grasp")` logger (with a `NullHandler` so library
  consumers see nothing by default). (P2-3)
- Sphinx docs renamed from `ggcas.*.rst` to `grasp.*.rst`, the version
  is pulled from `importlib.metadata`, `numpydoc` and an intersphinx
  table are wired in, and `functions.rst` documents only
  `CartesianConversion`. (P4-2)
- GitHub Actions workflow upgraded to a Python 3.10/3.11/3.12/3.13
  matrix, `ruff check` replacing `flake8 --exit-zero`,
  `pytest -ra -q --cov=grasp --cov-report=xml` with coverage and
  `pip freeze` lockfile artefacts. R parity is an optional
  `continue-on-error` job. (P4-1)

### Fixed
- `grasp.functions.CartesianConversion.compute` no longer calls a
  non-existent `self.compute_error`; errors flow through the analytical
  `_errFormula`. (P1-3 follow-up)
- `grasp.core.osutils.load_data(tn=...)` no longer self-clobbers
  `tn` from `kwargs.get('tn')` (which always returned `None`).
- `test/test_query.py::test_free_query` previously passed an ADQL
  string as the `radius` argument to `free_gc_query`; it now calls
  `free_query` and asserts on the constructed query. (P0-5)

### Deprecated
- The `rpy2`-backed numerical routines (`fit_distribution(backend="r")`,
  `gaussian_mixture_model(backend="r")`, `kfold_gmm_estimator(backend="r")`)
  are deprecated and scheduled for removal in **GRASP 1.0**. Migrate to
  the Python backends (the default). Set `GRASP_R_BACKEND=1` only for
  parity testing.
- `grasp.functions` historically referenced `AngularSeparation`,
  `GcDistance` and `RadialDistance2D`; these were never re-implemented
  in the modern API and have been removed from the public surface.
  Use `astropy.coordinates` instead.

### Breaking
- **Exponential model sign**: `fit_distribution(method="exponential")`
  (both Python and R) now uses the *decay* form `A * exp(-b * x)`. The
  old Python code used `A * exp(l * x)`, so callers that stored or
  serialised the previous `(A, lambda)` parameters must flip the sign
  on the second coefficient when migrating. (P1-7)
- `fit_distribution` / `gaussian_mixture_model` /
  `kfold_gmm_estimator` now return `PyFitResult` /
  `GaussianMixturePython` / `KFoldGMMResult` by default, not the R-wrapped
  classes. Pass `backend="r"` to retain the legacy return types
  (subject to the deprecation warning above).
- `grasp.core.folder_paths` no longer side-effects directory creation
  at import time. Call `initialize_data_layout()` from your application
  entry point (or rely on the lazy `ensure_data_dir` helper) if you
  depended on the old behaviour.

### Removed
- `_Catalogue.xml` entry in `MANIFEST.in` (file does not exist).
- Erroneous `recursive-include my_package *` from `MANIFEST.in`.
- Import-time calls to `chooseCRANmirror` and `install.packages` in
  `r_check.py`.

[Unreleased]: https://github.com/pietroferraiuolo/GRASP/compare/v0.9.3...HEAD
