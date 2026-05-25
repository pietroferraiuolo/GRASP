# GRASP - Globular clusteR Astrometry and Photometry Software
 ![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/pietroferraiuolo/GRASP/python-test.yaml)

The GRASP package is a tool for astrophysical data analysis, mainly thought for Globular CLusters 
astrometric and photometric data retrievement and GCs dynamical evolution analysis.

## What's new (2026-05 cleanup)

The current development branch is the result of a multi-phase cleanup
(see [`CHANGELOG.md`](./CHANGELOG.md)). The most user-visible changes:

- **Pure-Python statistical backends** based on `scipy`, `lmfit`,
  `statsmodels` and `scikit-learn` are now the *default*. The R-backed
  paths still exist for parity testing but emit a `DeprecationWarning`
  on every call and are scheduled for removal in **GRASP 1.0**. Opt
  into the R backend with the `backend="r"` kwarg or the environment
  variable `GRASP_R_BACKEND=1`.
- **Supported Python versions**: 3.10, 3.11, 3.12 and 3.13.
- **Reproducible RNG**: every public stochastic API accepts a `seed=`
  keyword that is fed through `grasp.utils.rng.default_rng`.
- **No more side-effects at import time**: data directories are created
  lazily, R packages are installed on demand, and `rpy2` is imported
  only when the R backend is invoked.

## Table of Contents

- [Installation Guide](#installation-guide)
    - [Quick start](#quick-start-python-only)
    - [Optional: R backend for parity testing](#optional-r-backend-for-parity-testing)
    - [ANTLR4](#install-latex-parser-dependencies)
- [Examples](#retrieving-data)

## Installation Guide

GRASP requires Python **3.10-3.13**. A virtual environment (`venv` or
`conda`) is strongly recommended.

### Quick start (Python-only)

```bash
pip install git+https://github.com/pietroferraiuolo/GRASP.git
```

For an editable checkout with the developer tooling
(`pytest`, `ruff`, `mypy`):

```bash
git clone https://github.com/pietroferraiuolo/GRASP.git
cd GRASP
pip install -e ".[dev]"
```

Once the project is published on PyPI you will be able to:

```bash
pip install grasp          # Python-only (recommended)
pip install "grasp[r]"     # add the deprecated R backend for parity tests
pip install "grasp[docs]"  # Sphinx + numpydoc to rebuild the docs
```

### Optional: R backend for parity testing

The R routines historically lived in `grasp/analyzers/_Rcode`. They
remain available behind the optional `[r]` extra strictly so that
existing notebooks can be reproduced -- new code should use the
Python implementations.

1. Install R (e.g. on Debian/Ubuntu):

   ```bash
   sudo apt update
   sudo apt install r-base r-base-dev
   ```

2. Install GRASP with the `r` extra:

   ```bash
   pip install "grasp[r]"
   # or, from a checkout:
   pip install -e ".[r]"
   ```

3. The first R-backend invocation will raise a `RuntimeError` with
   install instructions for the required R packages (`minpack.lm`,
   `mclust`). Install them inside R:

   ```bash
   Rscript -e 'install.packages(c("minpack.lm", "mclust"), repos="https://cloud.r-project.org/")'
   ```

4. Either pass `backend="r"` explicitly or set
   `GRASP_R_BACKEND=1` in your environment.

### Install latex parser dependencies: ANTLR4

The `formulary` module mixes SymPy with a LaTeX parser that depends on
ANTLR4. Pin the runtime version:

```bash
pip install antlr4-python3-runtime==4.11
```

Equivalently:

```bash
conda install -c conda-forge antlr4-python3-runtime==4.11
```

<details>
<summary>Features Examples</summary>

### Retrieving data
Right now, the only implemented archive available for data retrievement is the GAIA archive.
It is comprehensive of various data tables, with the main table for data release `X`
being `gaiadrX.gaia_source`. To list all the available data tables:

```py
> import grasp

> grasp.available_tables() # or equivalentely grasp.gaia.query.available_tables()
"INFO: Retrieving tables... [astroquery.utils.tap.core]"
"INFO: Parsing tables... [astroquery.utils.tap.core]"
"INFO: Done. [astroquery.utils.tap.core]"
"external.apassdr9"
"external.catwise2020"
"external.gaiadr2_astrophysical_parameters"
.
. 
. # continuing with all available data tables
```

As for (gaia) data retrievement, there is the `grasp.gaia.query` module containing the `GaiaQuery`
class, which can be instanced with any of the availble tables, passed as a string. For example, if
one wants to work with GAIA DR2 data, simply:

```py
> dr2 = grasp.GaiaQuery('gaiadr2.gaia_source') # or grasp.gaia.query.GaiaQuery()
"Initialized with Gaia table: 'gaiadr2.gaia_source'"
```

Let's say we want to work with the latest (as of 2025) data release, DR3 (there is a fast alias for 
that):

```py
> dr3 = grasp.dr3()
"Initialized with Gaia table: 'gaiadr3.gaia_source'"
> dr3
"""
GAIADR3.GAIA_SOURCE
-------------------
This table has an entry for every Gaia observed source as published with this data release. 
It contains the basic source parameters, in their final state as processed by the Gaia Data 
Processing and Analysis Consortium from the raw data coming from the spacecraft. The table 
is complemented with others containing information specific to certain kinds of objects 
(e.g.~Solar--system objects, non--single stars, variables etc.) and value--added processing 
(e.g.~astrophysical parameters etc.). Further array data types (spectra, epoch measurements) 
are presented separately via Datalink resources.

<grasp.query.GaiaQuery class>"""
```

For an easy and fast astrometry (or photometry) data retrival, there are built-in functions.
Let's assume we want to retrieve astrometric data of all the sources falling within a circle on the 
sky, with radius $r=1.0\,\deg$ and center coordinates $(\alpha, \delta) = (6.02, -72.08) \deg$,
and we want to save the data obtained:

```py
> a_sample = dr3.get_astrometry(radius=1., ra=6.02, dec=-72.08, save=True)
"Not a Cluster: no model available"
"INFO: Query finished. [astroquery.utils.tap.core]"
"Sample number of sources: 229382"
"Path '.../graspdata/query/UNTRACKEDDATA' did not exist. Created."
".../graspdata/query/UNTRACKEDDATA/20250312_111553/query_data.txt"
".../graspdata/query/UNTRACKEDDATA/20250312_111553/query_info.ini"

> a_sample
"""
Gaia data retrieved at coordinates 
RA=6.02 DEC=-72.08

Data Columns:
source_id - ra - ra_error - dec - dec_error - 
parallax - parallax_error - pmra - pmra_error - pmdec -
"""
```

Every query will have a unique tracking numer identifier of the format `YYYYMMDD_hhmmss`. The query
returns a `grasp.Sample` object, which handles all data and cluster integration in one place. Common
`pandas` and `astropy.QTable methods are available`:

```py
> a_sample.head()
""" 
             SOURCE_ID        ra   ra_error        dec  dec_error  parallax  \
0  4689621262329503744  5.934553   0.537742 -72.252166   0.742929 -2.907118   
1  4689859169153623936  6.723713   0.234891 -71.538202   0.198667 -0.215250   
2  4688735017312170240  7.703532   0.189104 -72.891773   0.190288  0.002266   
3  4688735021593281792  7.674888   0.216690 -72.905171   0.215202 -0.011812   
4  4688735021595344896  7.688834  12.067017 -72.890149   5.270076       NaN   

   parallax_error      pmra  pmra_error     pmdec  pmdec_error  
0        0.691778  4.077426    0.623827 -0.533118     0.897007  
1        0.234868 -0.710963    0.298354 -0.390703     0.304819  
2        0.209810  0.565811    0.251735 -1.081986     0.262715  
3        0.232426  0.404157    0.278103 -1.143366     0.292574  
4             NaN       NaN         NaN       NaN          NaN  
"""

> a_sample.info()
"""
<Table length=229382>
     name       dtype    unit                              description                             n_bad
-------------- ------- -------- ------------------------------------------------------------------ -----
     SOURCE_ID   int64          Unique source identifier (unique within a particular Data Release)     0
            ra float64      deg                                                    Right ascension     0
      ra_error float32      mas                                  Standard error of right ascension     0
           dec float64      deg                                                        Declination     0
     dec_error float32      mas                                      Standard error of declination     0
      parallax float64      mas                                                           Parallax 38102
parallax_error float32      mas                                         Standard error of parallax 38102
          pmra float64 mas / yr                         Proper motion in right ascension direction 38102
    pmra_error float32 mas / yr       Standard error of proper motion in right ascension direction 38102
         pmdec float64 mas / yr                             Proper motion in declination direction 38102
   pmdec_error float32 mas / yr           Standard error of proper motion in declination direction 38102
"""
```

Since there is a big focus on globular clusters for the `grasp` package, the same result could be 
achieved by simply passing as arguments of the functions the radius and the GC name.

The center coordinates used in the previous example, are the coordinates for the center of the GC
*NGC 104* (as listed in the Harry's 2010 edition catalogue). So we can repeat the query like this:

```py
> a_sample = dr3.get_astrometry(radius=1., gc='ngc104', save=True)
"INFO: Query finished. [astroquery.utils.tap.core]"
"Sample number of sources: 229490"
"Path '.../graspdata/query/NGC104' did not exist. Created."
".../graspdata/query/NGC104/20250312_112905/query_data.txt"
".../graspdata/query/NGC104/20250312_112905/query_info.ini"

> a_sample
"""
Data sample for cluster NGC104

Data Columns:
source_id - ra - ra_error - dec - dec_error - 
parallax - parallax_error - pmra - pmra_error - pmdec - 
pmdec_error
"""
```

With the addition that we have now available useful data on the Cluster (<ins>NOTE</ins>: this is 
non other than an implementation of the `grasp.Cluster` class):

```py
> print(a_sample.gc)
"""
Harris Catalog 2010 edition Parameters

       Key                  Value
----------------------------------------
.id      Cluster Name       NGC104
.ra      Position in sky    RA  6.02 deg
.dec                        DEC -72.08 deg
.dist    Distance           4.50 kpc
.w0      W0 Parameter       8.82
.logc    Concentration      logc=2.07
.cflag                      Collapsed -> False
.rc      Core radius        0.006 deg
.rh      Half-Light radius  0.053 deg
.rt      Tidal Radius       0.705 deg
"""
```

All the "fixed" query functions (`.get_astrometry`, `.get_photometry`, `.get_rv`) support an additional
parameter the conditions to be applied on the query. As example

```py
> conditions = ['parallax IS NOT NULL', 'parallax > 0', 'pmra IS NOT NULL', 'pmdec IS NOT NULL'] # uses ADQL
> newsample = dr3.get_astrometry(radius=1., gc='ngc104', conds=conditions)
"INFO: Query finished. [astroquery.utils.tap.core]"
"Sample number of sources: 131482"
```

We can see how the conditions were applied, excluding quite the number of sources.

While for these functions the data retrieved is fixed, there is the `.free_query` functions which, as
the name suggests, accepts bot the `data` and `conditions` additional arguments to customize the query.

### Data visualization
Let us work with the `a_smple` and the `newsample` from before. All the possible builtin visualization
functions of the `grasp` package, are within the `plots` module.

```py
> from grasp import plots as gplt # gplt -> grasp.plots
```

Just as (cool) examples, let's visualize the `parallax` distributions of the samples, as well as the
`proper motion` plots.

```py
> gplt.histogram(a_sample.parallax, xlabel='parallax', xlim=(-2,2))
```
![px1](./docs/pxdis_1_1.png)

With so many sources, the visualization is not great, so that limits have been applied. One coul even
perform analysis at the fly, like _Kerlen Density Estimation_ (here instead of putting limits to the
visualization only, the sample itself has been restricted, to gain resolution on the histogram bins)

```py
> gplt.histogram(
    a_sample.parallax[(a_sample.parallax > -2) & (a_sample.parallax < 2)], 
    kde=True, 
    kde_kind='gaussian', 
    xlabel='parallax'
  )
"Correctly imported `minpack.lm`."
```
![px2](./docs/pxdist_fit_normal.png)

And one could see that the distribution is better fitted by a lorentian distribution function
rather than a normal:

```py
> gplt.histogram(
    a_sample.parallax[(a_sample.parallax > -2) & (a_sample.parallax < 2)], 
    kde=True, 
    kde_kind='lorentzian', 
    xlabel='parallax'
  )
"Correctly imported `minpack.lm`."
```
![px3](./docs/pxdist_fit_lorentz.png)

Another example of visualization are the builtin `proper motion` plot and the `doubleHistScatter` plot:

```py
> conditions = {
...     'pmra':'>-50',
...     'pmra':'<50',
...     'pmdec':'<50',
...     'pmdec':'>-50'
}
> restricted_sample = newsample.apply_conditions(conditions)
```

Here we are restricting the sample for visualization purposes and is equivalent to

```py
> restricted_sample = newsample[(newsample.pmra >-50) & (newsample.pmra < 50) & (newsample.pmdec >-50) & (newsample.pmdec < 50)]
```

with the only difference being that with the latter a `QTable` is returned, while for the `apply_conditions` function a `Sample` istance is returned. (<ins>note</ins> that the same result
could'have been achieved by directly applying these conditions to the initial query).

```py
> gplt.properMotion(restricted_sample)
```
![pm1](./docs/pmplot.png)

The `doubleHistScatter` plot is a scatter plot with projected distributions of the data in their axes
(with the `kde` option too)

```py
> gplt.doubleHistScatter(restricted_sample['pmra'], restricted_sample['pmdec'], ylabel='pmdec', xlabel='pmra', title='Proper Motion')
```
![dhs](./docs/dhs.png)


### Computing formulas
Formula computations are handled through the `formulary` module, which contains the `Formulary`
class, a collection of formulas either read by a `.frm` file or directly defined through the python
session. 

```py
> f = grasp.load_base_formulary()
> f
"_base_ formulary from file 'base.frm'"
"Type: latex"
```

The `load_base_fornumaly` function is a wrapper for the instance of the Formulary class reading
the `base.frm` file, which contains formulas included in the package. Let's see what we have:

```py
> f.display_all()
"""
Angular Separation
theta_{2*D} = 2*asin(sqrt((sin((alpha_{0} - alpha_{1})/2)**2*cos(delta_{1}))*cos(delta_{0}) + sin((delta_{0} - delta_{1})/2)**2))

Los Distance
r_{pc} = 1000/varpi_{mas}      # ϖ in mas -> r in pc (Gaia convention)

Radial Distance 2D
r_{2*d} = r_{c*g}*tan(theta_{2*D})

Gc Z Coordinate
d = -r_{c*g} + r_{x}

Radial Distance 3D
R = sqrt(d**2 + r_{2*d}**2)

In-Shell Dimentionless Poteff
Sigma = (-x + log(B, E)) - log(Delta_{N}/(sqrt(x)), E)

B Constant
B = 16*(A*(sqrt(2)*(pi**2*(beta**2*(r_{s}**2*(alpha*(m**3*(sigma**3*(dr*dx)))))))))

Dimentionless Poteff
Sigma = -log(1 - exp(-w + x), E)
"""
```

</details>
