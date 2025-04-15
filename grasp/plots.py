"""
Author(s)
---------
    - Pietro Ferraiuolo : Written in 2024

Description
-----------
Module which contains useful plots for visualizing globular cluster's kinetic data.
Refer to the individual functions documentation for more information about their use.

How to Use
----------
Just import the module

    >>> from grasp import plots as gplt
    >>> gplt.doubleHistScatter(...) # your data

"""

import numpy as _np
import seaborn as sns
import matplotlib.pyplot as _plt
from grasp.core import osutils as _osu
from grasp.stats import fit_distribution as _kde_estimator
from typing import Optional as _Optional, Union as _Union, Callable as _Callable
from grasp.analyzers._Rcode.r2py_models import (
    _kde_labels,
    RegressionModel as _RegressionModel,
    PyRegressionModel as _FakeRegModel,
)

label_font = {
    "family": "serif",
    "color": "black",
    "weight": "normal",
    "size": 15,
}
title_font = {
    "family": "serif",
    #"style": "italic",
    "color": "black",
    "weight": "semibold",
    "size": 20,
}

default_figure_size = (6.4, 5.2)


def doubleHistScatter(x, y, kde=False, kde_kind: str = "gaussian", **kwargs):
    """
    Make a 2D scatter plot of two data arrays, with the respective histogram distributions
    projected on each axis. The kde option allows for regression on the plotted data.

    Parameters
    ----------
    x : ArrayLike
        Data to be scattered on x-axis.
    y : ArrayLike
        Data to be scattered on y-axis.
    kde : bool, optional
        Option to show the regression on the data. Default is False.
    kde_kind : str, optional
        Kind of kernel density estimation to be computed. The default is 'gaussian'.
        Options:
        - 'gaussian'
        - 'boltzmann'
        - 'exponential'
        - 'king'
        - 'rayleigh'
        - 'maxwell'
        - 'lorentzian'
        - 'lognormal'
        - 'power'

    Other Parameters
    ----------------
    **kwargs : Additional parameters for customizing the plot.
        xlabel : str
            Label of the x-axis.
        ylabel : str
            Label of the y-axis.
        title : str
            Title of the figure.
        alpha : float
            Transparency of the data points of the scatter.
        colorx : str
            Color of the histogram on x-axis.
        colory : str
            Color of the histogram on y-axis.
        scatter_color : str
            Color of the scattered dots.
        size : int or float
            Size of the scattered data points.
        figsize : tuple
            Size of the figure.
        bins : int
            Number of bins for the histogram.
        xlim : tuple
            Limits for the x-axis.
        ylim : tuple
            Limits for the y-axis.

    """
    # xmin = _np.nanmin(x)
    # xmax = _np.nanmax(x)
    # ymin = _np.nanmin(y)
    # ymax = _np.nanmax(y)
    title = kwargs.get("title", "")
    xlabel = kwargs.get("xlabel", "")
    ylabel = kwargs.get("ylabel", "")
    xlim = kwargs.get("xlim", None)
    ylim = kwargs.get("ylim", None)
    alpha = kwargs.get("alpha", 0.7)
    colory = kwargs.get("colory", "blue")
    colorx = kwargs.get("colorx", "green")
    sc = kwargs.get("scatter_color", "black")
    s = _osu.get_kwargs(("size", "s"), 5, kwargs)
    fsize = kwargs.get("figsize", (5.6, 5.2))
    bins = kwargs.get("bins", int(1.5 * _np.sqrt(len(x))))
    fig = _plt.figure(figsize=fsize)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.015,
        hspace=0.015,
    )
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_histx.tick_params(axis="x", labelbottom=False, length=0.0)
    ax_histy.tick_params(axis="y", labelleft=False, length=0.0)
    # the scatter plot:
    ax.scatter(x, y, color=sc, alpha=alpha, s=s)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # ax_histx.set_ylabel("Counts")
    ax_histy.set_xlabel("Counts\n")
    ax.set_xlabel(xlabel, fontdict=label_font)
    ax.set_ylabel(ylabel, fontdict=label_font)
    hx = ax_histx.hist(x, bins=bins, color=colorx, alpha=0.6)
    hy = ax_histy.hist(y, bins=bins, orientation="horizontal", color=colory, alpha=0.6)
    _plt.suptitle(title, size=20, style="italic", family="cursive")
    ax_histx.set_xlim(xlim)
    ax_histy.set_ylim(ylim)
    ax_histy.xaxis.set_ticks_position("top")
    ax_histy.xaxis.set_label_position("top")
    ax_histx.yaxis.set_ticks_position("right")
    ax_histy.set_xticks(_np.arange(hy[0].max(), hy[0].max() + 1, 1))
    ax_histx.set_yticks(_np.arange(hx[0].max(), hx[0].max() + 1, 1))
    if kde:
        reg_x = _kde_estimator(x, kde_kind)
        reg_y = _kde_estimator(y, kde_kind)
        ax_histx.plot(
            reg_x.x,
            reg_x.y,
            color=colorx,
            label=f"$\mu$={reg_x.coeffs[1]:.3f}\n$\sigma^2$={reg_x.coeffs[2]:.3f}",
        )
        ax_histy.plot(
            reg_y.x,
            reg_y.x,
            color=colory,
            label=f"$\mu$={reg_y.coeffs[1]:.3f}\n$\sigma^2$={reg_y.coeffs[2]:.3f}",
        )
        ax_histx.legend(loc="best", fontsize="small")
        ax_histy.legend(loc="best", fontsize="small")
    _plt.show()


def colorMagnitude(sample=None, g=None, b_r=None, teff_gspphot=None, **kwargs):
    """
    Make a scatter plot to create a color-magnitude diagram of the sample, using
    BP and RP photometry and temperature information.

    Parameters
    ----------
    sample : _Sample or dict
        The sample data containing 'phot_g_mean_mag', 'bp_rp' and 'teff_gspphot'
        fields. If no sample is provided, the data fields must be provided.
    g : float | ArrayLike
        Gaia mean magnitude in the G band. For gaia samples it is the
        'phot_g_mean_mag' field.
    b_r : float | ArrayLike
        Gaia color, defined as the BP mean magnitude minus the RP mean magnitude.
        For gaia samples it is the 'bp_rp' field.
    teff_gspphot : float | ArrayLike
        Gaia computed effective surface temperature.

    Other Parameters
    ----------------
    **kwargs : dict
        bgc : tuple
            A tuple with three float values, indicating the RGB gradient which
            define a color (placeholder for the ax.set_facecolor function).
            Aliases: 'bgc', 'bgcolor', 'background_color'.
        alpha : float
            Transparency of the scatter points.
        cmap : str
            All accepted matplotlib colormap strings.
        figsize : tuple
            Size of the figure.

    """
    bgc = _osu.get_kwargs(
        ("bgc", "bgcolor", "background_color"), (0.9, 0.9, 0.9), kwargs
    )
    a = kwargs.get("alpha", 0.8)
    cmap = kwargs.get("cmap", "rainbow_r")
    fsize = kwargs.get("figsize", default_figure_size)
    fig, ax = _plt.subplots(nrows=1, ncols=1, figsize=fsize)
    ax.set_facecolor(bgc)
    if sample is not None:
        g = sample["phot_g_mean_mag"]
        b_r = sample["bp_rp"]
        teff_gspphot = sample["teff_gspphot"]
    elif g is None or b_r is None or teff_gspphot is not None:
        raise ValueError("You must provide a sample or the data fields.")
    _plt.scatter(b_r, g, c=teff_gspphot, alpha=a, cmap=cmap)
    _plt.colorbar(label=r"$T_{eff}$")
    _plt.ylim(max(g) + 0.51, min(g) - 0.51)
    _plt.xlabel(r"$G_{BP} - G_{RP}$", fontdict=label_font)
    _plt.ylabel(r"$G$", fontdict=label_font)
    _plt.title("Color-Magnitude Diagram", fontsize=17)
    _plt.show()


def properMotion(sample, **kwargs):
    """
    Make a scatter plot in the proper motion space of the sample.

    Parameters
    ----------
    sample : _Sample or dict
        The sample data containing 'pmra' and 'pmdec' fields.

    Other Parameters
    ----------------
    **kwargs : additional parameters for customizing the plot.
        color : str os ArrayLike
            Color of the scattered data points.
        s : int or float
            Size of the scattered data points.
        alpha : float
            Transparency of the scattered data points.
        figsize : tuple
            Size of the figure.

    """
    col = _osu.get_kwargs(("color", "c"), "black", kwargs)
    size = _osu.get_kwargs(("s", "size"), 3, kwargs)
    alpha = kwargs.get("alpha", 0.5)
    fsize = kwargs.get("figsize", default_figure_size)
    pmra = sample["pmra"]
    pmdec = sample["pmdec"]
    fig, ax = _plt.subplots(figsize=fsize)
    _plt.xlabel(r"$\mu_{\alpha*}$ [deg]", fontdict=label_font)
    _plt.ylabel(r"$\mu_\delta$ [deg]", fontdict=label_font)
    _plt.title("Proper Motion Distribution", fontdict=title_font)
    ax.axis("equal")
    _plt.scatter(pmra, pmdec, c=col, alpha=alpha, s=size)
    _plt.show()


def spatial(sample, **kwargs):
    """
    Make a scatter plot in the spatial plot, that is in the Ra-Dec plane.

    Parameters
    ----------
    sample : _Sample or dict
        The sample data containing 'ra' and 'dec' fields.

    Other Parameters
    ----------------
    **kwargs : additional parameters for customizing the plot.
    title : str
        Title of the plot.
    color : str os ArrayLike
        Color of the scattered data points.
    s : int or float
        Size of the scattered data points.
    alpha : float
        Transparency of the scattered data points.
    figsize : tuple
        Size of the figure.
    colorbar : bool
        If True, a colorbar will be shown.
    clabel : str
        Label for the colorbar. Aliases
        - 'colorbar_label'
        - 'cl'
    cmap : str
        Matplotlib colormap string for the color-coded points.

    """
    col = _osu.get_kwargs(("color", "c"), "black", kwargs)
    fsize = kwargs.get("figsize", default_figure_size)
    size = _osu.get_kwargs(("s", "size"), 5, kwargs)
    alpha = kwargs.get("alpha", 0.5)
    colorbar = kwargs.get("colorbar", False)
    clabel = _osu.get_kwargs(("colorbar_label", "clabel", "cl"), "", kwargs)
    cmap = kwargs.get("cmap", "viridis")
    title = kwargs.get("title", "Spatial Distribution")
    ra = sample["ra"]
    dec = sample["dec"]
    fig, ax = _plt.subplots(figsize=fsize)
    _plt.xlabel(r"$DEC$ [deg]", fontdict=label_font)
    _plt.ylabel(r"$RA$ [deg]", fontdict=label_font)
    _plt.title(title, fontdict=title_font)
    ax.axis("equal")
    _plt.scatter(ra, dec, c=col, alpha=alpha, s=size, cmap=cmap)
    if colorbar:
        _plt.colorbar(label=clabel)
    _plt.show()


def histogram(data, kde=False, kde_kind: str = "gaussian", out: bool = False, **kwargs):
    """
    Plots the data distribution with a histogram. The number of bins is defined
    as 1.5*sqrt(N). If kde is True, the kernel density estimation will be
    computed and plotted over the histogram.

    Parameters
    ----------
    data : ArrayLike
        Input dataset for the histogram.
    kde : bool, optional
        Option for the computation of the Gaussian Kernel density estimation
        of the histogram. The default is False.
    kde_kind : str, optional
        Kind of kernel density estimation to be computed. The default is
        'gaussian'.
        Options:
        - 'gaussian'
        - 'boltzmann'
        - 'exponential'
        - 'king'
        - 'rayleigh'
        - 'maxwell'
        - 'lorentzian'
        - 'lognormal'
        - 'power'

    Other Parameters
    ----------------
    **kwargs : Additional parameters for customizing the plot.
        bins : int
            Number of bins for the histogram.
        title : str
            Title of the plot.
        kde_verbose : bool
            If True, the kde iteration will be printed.
        xlabel : str
            Label of the plot's x-axis.
        alpha : float
            Transparency of the bins.
        hcolor : str
            Color of the histogram's bins.
        kcolor : str
            Color of the kde curve.
        figsize : tuple
            Size of the figure.
        xlim : tuple
            Limits for the x-axis.
        scale : str
            Scale of the y-axis. Options: 'linear', 'log'.
            Default is 'linear'.

    Returns
    -------
    hist_result : dict
        Dictionary containing the histogram results and the distribution
        mean and standard deviation. The keys are 'h' and 'kde'.

    Example
    -------
    The output can be used to make other plots and computations. For example

    >>> import numpy as np
    >>> from grasp import plots as gplt
    >>> x = np.random.randn(1000)
    >>> y = np.random.randn(1000)
    >>> b, n = gplt.histogram(x, y)

    Then, with the result, you can make for example a scatter plot

    >>> plt.scatter(x, y)

    and so on.
    """
    xlabel = kwargs.get("xlabel", "")
    alpha = kwargs.get("alpha", 1)
    hcolor = _osu.get_kwargs(("hist_color", "hcolor", "hc"), "gray", kwargs)
    kcolor = _osu.get_kwargs(("kde_color", "kcolor", "kc"), "red", kwargs)
    title = kwargs.get("title", xlabel + " Distribution")
    fsize = kwargs.get("figsize", default_figure_size)
    verbose = _osu.get_kwargs(("kde_verbose", "verbose", "v"), False, kwargs)
    scale = _osu.get_kwargs(("scale", "yscale"), "linear", kwargs)
    if scale == "log":
        xlabel = "log() " + xlabel + " )"
    bins = kwargs.get("bins", int(1.5 * _np.sqrt(len(data))))
    if "xlim" in kwargs:
        if isinstance(kwargs["xlim"], tuple):
            xlim = kwargs["xlim"]
        else:
            raise TypeError("'xlim' arg must be a tuple")
    else:
        xlim = None
    n_bin = int(1.5 * _np.sqrt(len(data)))
    _plt.figure(figsize=fsize)
    h = _plt.hist(data, bins=n_bin, color=hcolor, alpha=alpha)
    _plt.ylabel("counts")
    _plt.yscale(scale)
    _plt.xlabel(xlabel, fontdict=label_font)
    _plt.title(title, fontdict=label_font)
    bins = h[1][: len(h[0])]
    counts = h[0]
    res = {"h": {"counts": counts, "bins": bins}}
    if kde:
        regression = _kde_estimator(data, kde_kind, verbose=verbose)
        res["kde"] = regression.coeffs
        label = _kde_labels(kde_kind, regression.coeffs)
        _plt.plot(regression.x, regression.y, c=kcolor, label=label)
        _plt.legend(loc="best", fontsize="medium")
    if xlim is not None:
        _plt.xlim(xlim)
    _plt.show()
    if out:
        return res


def scatterXHist(x, y, xerr: _Optional[_Union[float, _np.ndarray]] = None, **kwargs):
    """
    Make a scatter plot of a quantity 'x', with its projected histogram, relative
    to a quantity 'y'.

    Parameters
    ----------
    x : ndarray
        Input data.
    y : ndarray
        Related data.
    xerr : int or ndarray, optional
        Error on the input data points. Can be either a single value or an array
        with the same size as the input data. The default is None.

    Other Parameters
    ----------------
    **kwargs : additional arguments for customizing the plot.
        xlabel : str
            Label on x axis. The default is 'x'.
        ylabel : str
            Label on y axis. The default is 'y'.
        title : str
            Title of the figure. Default is 'x distribution'
        color : str
            Color of the scattered data points.
        size : int or float
            Size of the scattered data points.
        figsize : tuple
            Size of the figure.

    Returns
    -------
    list
        List containing the mean and error of x.
    """
    # to do: add title, remove unit (if present) to mean text and automatic title
    # change the text in a legend, so that size is fixed
    xlabel = kwargs.get("xlabel", "x")
    ylabel = kwargs.get("ylabel", "y")
    color = _osu.get_kwargs(("c", "color"), "gray", kwargs)
    s = _osu.get_kwargs(("s", "size"), 7.5, kwargs)
    fsize = kwargs.get("figsize", default_figure_size)
    title = kwargs.get("title", xlabel + " distribution")
    nb2 = int(1.5 * _np.sqrt(len(x)))
    mean_x = _np.mean(x)
    fig, (ax0, ax1) = _plt.subplots(
        nrows=2, ncols=1, height_ratios=[1, 3.5], figsize=fsize, sharex=True
    )
    fig.subplots_adjust(hspace=0)
    if xerr is not None:
        if isinstance(xerr, float):
            xerr = _np.full(len(x), xerr)
        ax1.errorbar(
            x,
            y,
            xerr=xerr,
            fmt="x",
            color=color,
            linewidth=1.0,
            markersize=3,
            alpha=0.8,
        )
        err_xm = _np.sqrt(sum(i * i for i in xerr) / len(x))
    else:
        ax1.scatter(x, y, c=color, alpha=0.8, s=s)
        err_xm = _np.std(x) / _np.sqrt(len(x))
    ax1.set_ylim(y.min() * 0.8, y.max() * 1.2)
    # Media scritta
    ax1.text(
        x.max() * 0.1,
        y.max(),
        r"$<${}$>=(${:.2f}$\,\pm\,${:.2f}$)$".format(xlabel, mean_x, err_xm),
        color="black",
        fontsize=12,
    )
    vh = ax0.hist(x, bins=nb2, color=color, histtype="step", orientation="vertical")
    # linea
    ax1.plot(
        [mean_x, mean_x],
        [min(y) * (0.75), max(y) * (1.25)],
        linestyle="--",
        c="black",
        alpha=0.85,
    )
    ax0.plot([mean_x, mean_x], [0, vh[0].max()], linestyle="--", c="black", alpha=0.85)
    # labels
    ax1.set_ylabel(ylabel, fontdict=label_font)
    ax1.set_xlabel(xlabel, fontdict=label_font)
    ax0.set_ylabel("Counts")
    # minor ticks
    ax0.minorticks_on()
    ax1.minorticks_on()
    ax0.tick_params(axis="both", direction="in", size=6)
    ax0.tick_params(axis="y", which="minor", direction="in", size=0)
    ax0.tick_params(axis="x", which="minor", direction="in", size=3)
    ax1.tick_params(axis="both", direction="in", size=6)
    ax1.tick_params(which="minor", direction="in", size=3)
    title = xlabel + " distribution "
    fig = _plt.suptitle(title, size=20, style="italic", family="cursive")
    _plt.show()
    return [mean_x, err_xm]


def errorbar(data, dataerr, x=None, xerr=None, **kwargs):
    """
    Plot data with error bars.

    Both `x` and `y` data with errors are supported

    Parameters
    ----------
    data : ndarray
        Data to be plotted.
    dataerr : ndarray
        Errors associated with the data.
    x : ndarray, optional
        X-axis data. The default is None.
    xerr : ndarray, optional
        Errors associated with the x-axis data. The default is None.

    ### kwargs : Additional callbacks for matplotlib (see matplotlib.pyplot.errorbar documentation).

    fmt : str
        Scatter point shape.
    color : str
        Scatter point color.
        Aliases - 'sc' ; 'scolor' ; 'scatcol'.
    ecolor : str
        Error bar color.
        Aliases - 'ec' ; 'errc' ; 'errcolor' ; 'errorcolor'.
    markersize : float
        Scatter point size.
        Aliases - 's' ; 'ms'.
    capsize : float
        Error bar cap length.
    elinewidth : float
        Error bar thickness.
        Aliases - 'elw' ; 'errlinew'.
    barsabove : bool
        If True, the error bars will be plotted over the scatter points.
    title : str
        Title of the plot.
    xlabel : str
        Label of the x-axis.
    ylabel : str
        Label of the y-axis.
    figsize : tuple
        Size of the figure.

    """
    ec = _osu.get_kwargs(
        ("ecolor", "ec", "errc", "errcolor", "error_color"), "red", kwargs
    )
    sc = _osu.get_kwargs(("color", "scolor", "sc", "scatter_col"), "black", kwargs)
    elw = _osu.get_kwargs(("elinewidth", "elw", "errlinew"), 1, kwargs)
    ms = _osu.get_kwargs(("markersize", "ms", "s"), 2, kwargs)
    fsize = kwargs.get("figsize", default_figure_size)
    ba = kwargs.get("barsabove", False)
    cs = kwargs.get("capsize", 1.5)
    xlabel = kwargs.get("xlabel", "")
    ylabel = kwargs.get("ylabel", "")
    title = kwargs.get("title", "")
    fmt = kwargs.get("fmt", "x")
    x = _np.linspace(0, 1, len(data)) if x is None else x
    _plt.figure(figsize=fsize)
    _plt.errorbar(
        x,
        data,
        yerr=dataerr,
        xerr=xerr,
        fmt=fmt,
        capsize=cs,
        ecolor=ec,
        elinewidth=elw,
        markersize=ms,
        color=sc,
        barsabove=ba,
    )
    _plt.xlabel(xlabel, fontdict=label_font)
    _plt.ylabel(ylabel, fontdict=label_font)
    _plt.title(title, fontdict=title_font)
    _plt.show()


def regressionPlot(
    regression_model: _RegressionModel | _FakeRegModel | _Callable | str,
    x_data: _np.ndarray | list = None,
    y_data: _np.ndarray | list = None,
    f_type: str = "distribution",
    **kwargs,
):
    """
    Plot the regression model with the data and residuals.

    Parameters
    ----------
    regression_model : grasp.r2py_models.RegressionModel or str or callable
        The regression model to be plotted. You can either pass the already fitted
        model or a string indicating the kind of regression to be fitted. The
        available options are:
        - 'gaussian'
        - 'boltzmann'
        - 'exponential'
        - 'king'
        - 'rayleigh'
        - 'maxwell'
        - 'lorentzian'
        - 'lognormal'
        - 'power'
        If you pass a callable, it must be a function that takes the data as
        input and returns the fitted model. See `grasp.starts.fit_data` documentation
        for more information.
    x : ndarray or list, optional
        The indipendent variable data to be plotted. If not given, it will be
        simply be an array of the same size as the y data.
    y : ndarray or list, optional
        If `regression_model` is not passed as a RegressionModel class, and so already
        fitted, `y` must be provided as the data to be fitted. `regression_model`, then,
        becomes the kind of regression to be performed.
    f_type : str, optional
        The type of algorithm to use for regression. Options are
            'distribution': the distribution (histogram) of the data is fitted
            'datapoint': the data points are fitted
        Default is 'distribution'.

    Other Parameters
    ----------------
    **kwargs : Additional parameters for customizing the plot.
        figsize : tuple
            Size of the figure.
        xlim : tuple
            Limits for the x-axis.
        xlabel : str
            Label of the x-axis.
        title : str
            Title of the figure.
        fmt : str
            Main plot style. Only works when passing a linear regression
            model. Default is '-' (normal "solid" plot).
        size : int or float
            Size of the scattered data points in both plots.
            Alias: 's'.
        plot_color : str
            Color of the data plot. Aliases:
            - 'pcolor'
            - 'plotcolor'
            - 'pc'
        fit_color : str
            Color of the regression plot. Aliases:
            - 'fcolor'
            - 'fitcolor'
            - 'fc'
        residuals_color : str
            Color of the residuals. Aliases:
            - 'rcolor'
            - 'rescolor'
            - 'rc'

    """
    rm = _get_regression_model(regression_model, y_data, x_data, f_type)
    D = _np.linalg.norm([rm.x.min(), rm.x.max()]) * 0.02
    xlim = kwargs.get("xlim", (rm.x.min() - D, rm.x.max()))
    s = _osu.get_kwargs(("size", "s"), 2.5, kwargs)
    fsize = kwargs.get("figsize", default_figure_size)
    xlabel = kwargs.get("xlabel", "")
    title = kwargs.get("title", "")
    pc = _osu.get_kwargs(("plot_color", "pcolor", "plotcolor", "pc"), "black", kwargs)
    fc = _osu.get_kwargs(("fit_color", "fcolor", "fitcolor", "fc"), "red", kwargs)
    rc = _osu.get_kwargs(("residuals_color", "rcolor", "rescolor", "rc"), "red", kwargs)
    rfmt = kwargs.get("rfmt", "o-")
    fmt = kwargs.get("fmt", "--")
    fig, (fax, rax) = _plt.subplots(
        nrows=2, ncols=1, height_ratios=[2.75, 1], figsize=fsize, sharex=True
    )
    fig.subplots_adjust(hspace=0)
    # ---------------
    # data plot (with conditions)
    # ---------------
    if isinstance(rm, _RegressionModel) and rm.kind == "linear":
        x = rm.data["x"].to_numpy()
        y = rm.data["y"].to_numpy()
        fax.plot(x, y, c=pc, markersize=s, linewidth=1.0, alpha=0.8, label="Data")
    elif f_type == "datapoint":
        fax.scatter(
            rm.x,
            rm.data,
            c=pc,
            s=s,
            linewidth=1.0,
            alpha=0.8,
            label="Data",
        )
    else:
        fax.hist(
            rm.data, bins=len(rm.y), color=pc, histtype="step", alpha=0.85, label="Data"
        )
    # ---------------
    # fit plot in red color (default)
    # ---------------
    fax.plot(rm.x, rm.y, fmt, c=fc, label=_kde_labels(rm.kind, rm.coeffs))
    fax.set_ylabel("counts")
    fax.set_xlim(xlim)
    fax.legend(loc="best", fontsize="medium")
    # --------------
    # Residuals plot
    # --------------
    rax.set_ylabel("Residuals")
    rax.yaxis.set_label_position("left")
    rax.yaxis.tick_right()
    rax.set_xlabel(xlabel)
    rax.set_xlim(xlim)
    rax.plot(
        [rm.x.min() * 1.1, rm.x.max() * 1.1],
        [0, 0],
        c="gray",
        alpha=0.8,
        linestyle="--",
    )
    rax.plot(rm.x, rm.residuals, rfmt, c=rc, markersize=s, linewidth=1.0, alpha=0.8)
    fig.suptitle(title, size=20, style="italic", family="cursive")
    _plt.show()


def seaborn(which: str, *args, **kwargs):
    """
    Wrapper to make a seaborn plot.

    Check
    <a href="https://seaborn.pydata.org/index.html">seaborn documentation</a>
    for more info on plot types and parameters.

    Parameters
    ----------
    which : str
        The type of seaborn plot to be made.

    Other Parameters
    ----------------
    *args : Additional arguments for the seaborn plot.
    **kwargs : Additional parameters for customizing the plot.
    """
    plot_call = getattr(sns, which)
    plot_call(*args, **kwargs)


def _get_regression_model(regression_model, y_data, x_data, which):
    """
    Get the regression model to be used for the plot.

    This function checks if the regression model is already fitted or if
    it needs to be fitted with the data. If the regression model is a string,
    it will be fitted with the data. If the regression model is a callable,
    it will be used as the fitting function. If the regression model is a
    RegressionModel instance, it will be used as the regression model.
    """
    if isinstance(regression_model, (_RegressionModel, _FakeRegModel)):
        rm = regression_model
    elif y_data is not None:
        if not regression_model is None:
            if which == "distribution":
                model = _kde_estimator(y_data, method=regression_model, verbose=False)
                rm = model
            elif which == "datapoint":
                from grasp.stats import fit_data_points
                fit = fit_data_points(data=y_data, x_data=x_data, method=regression_model)
                fit = _FakeRegModel(fit, regression_model)
                rm = fit
        else:
            raise ValueError(
                "You must provide the `regression model` argument either as a string, a callable or a grasp.RegressionModel."
            )
    else:
        raise ValueError(
            "You must either provide a fitted RegressionModel of `y` data to be fitted with the `regression model` argument as a string or a callable."
        )
    return rm
