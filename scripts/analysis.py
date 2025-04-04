# %% [markdown]
# # Initialization

# %%
import os
import grasp
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.io import fits
from matplotlib import pyplot as plt

dr3 = grasp.dr3()
gc = grasp.Cluster("ngc6121")

try:
    device_name = os.getenv("COMPUTERNAME")
    if device_name == "DESKTOP-Work":
        tn1 = "20250402_204446"
        tn2 = "20250402_204448"
        acs = grasp.load_data(tn1)
        pcs = grasp.load_data(tn2)
    elif device_name == "LAPTOP-Work":
        tn1 = "20250401_164228"
        tn2 = "20250401_164231"
        pcs = grasp.load_data(tn1)
        acs = grasp.load_data(tn2)
    else:
        raise EnvironmentError("Unknown device name")
except Exception:
    astrometry_query = "SELECT source_id, ra, ra_error, dec, dec_error, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error, \
                        radial_velocity, radial_velocity_error, bp_rp, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, teff_gspphot, ra_dec_corr, pmra_pmdec_corr \
                        FROM gaiadr3.gaia_source \
                        WHERE CONTAINS(POINT('ICRS',gaiadr3.gaia_source.ra,gaiadr3.gaia_source.dec),CIRCLE('ICRS',245.897,-26.526,0.86))=1 \
                        AND parallax IS NOT NULL AND parallax>0.531632110579479 AND parallax<0.5491488193300\
                        AND abs(parallax_error/parallax)<0.50\
                        AND abs(pmra_error/pmra)<0.30 \
                        AND abs(pmdec_error/pmdec)<0.30 \
                        AND pmra IS NOT NULL AND abs(pmra)>0 \
                        AND pmdec IS NOT NULL AND abs(pmdec)>0 \
                        AND pmra BETWEEN -13.742720 AND -11.295338 \
                        AND pmdec BETWEEN -20.214805 AND -17.807517"

    photometry_query = "SELECT source_id, ra, ra_error, dec_error, dec, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error, radial_velocity, radial_velocity_error, \
                        bp_rp, phot_g_mean_mag, phot_bp_rp_excess_factor, teff_gspphot, ra_dec_corr, pmra_pmdec_corr \
                        FROM gaiadr3.gaia_source \
                        WHERE CONTAINS(POINT('ICRS',gaiadr3.gaia_source.ra,gaiadr3.gaia_source.dec),CIRCLE('ICRS',245.8958,-26.5256,0.86))=1 \
                        AND parallax IS NOT NULL AND parallax>0.531632110579479 AND parallax<0.5491488193300\
                        AND ruwe < 1.15 \
                        AND phot_g_mean_mag > 11 \
                        AND astrometric_excess_noise_sig < 2 \
                        AND pmra BETWEEN -13.742720 AND -11.295338 \
                        AND pmdec BETWEEN -20.2148 AND -17.807517"

    acs = dr3.free_query(astrometry_query, save=True)
    acs = grasp.Sample(acs, gc)
    pcs = dr3.free_query(photometry_query, save=True)
    pcs = grasp.Sample(pcs, gc)
    print("\nWARNING! Remember to updates tn after running the new query!!!")

# %% [markdown]
# # Data visualization

# %%
aps = acs.join(pcs)
aps.info()
grasp.plots.colorMagnitude(aps)

# %%
aps._merge_info.head()

# %% [markdown]
# # Angular Separation Analysis
#
# The `Great Circle` formula versus the `Vincenty Formula` for the computation of distances on a sphere

# %% [markdown]
# ## $\theta_V$   vs   $\theta_{GC}$

# %%
f = grasp.load_base_formulary()
f.substitute(
    "Angular separation", {"alpha_{0}": aps.gc.ra.value, "delta_{0}": aps.gc.dec.value}
)

from sympy import atan2

atan_arg_1 = "sqrt((cos(delta_1*sin((alpha_0 - alpha_1)/2)))**2 + (cos(delta_0)*sin(delta_1) - sin(delta_0)*cos(delta_1)*cos((alpha_0 - alpha_1)/2 ))**2)"
atan_arg_2 = (
    "(sin(delta_0)*sin(delta_1) + cos(delta_0)*cos(delta_1)*cos((alpha_0 - alpha_1)/2))"
)
atan = atan2(atan_arg_1, atan_arg_2)
f.add_formula("Vincenty angsep", atan)
f.substitute(
    "Vincenty angsep", {"alpha_0": aps.gc.ra.value, "delta_0": aps.gc.dec.value}
)


f.angular_separation

# %%
f["Vincenty angsep"]

# %%
f.var_order("Angular Separation")
print("")
f.var_order("Vincenty angsep")

# %%
ra, dec = (aps.ra.value, aps.dec.value)
print("Great Circle Distance computation\n")
theta_1 = f.compute(
    "Angular Separation", data={"alpha_{1}": ra, "delta_{1}": dec}, asarray=True
)
print("\nVincenty Distance computation\n")
theta_2 = f.compute(
    "Vincenty angsep", data={"alpha_1": ra, "delta_1": dec}, asarray=True
)

grasp.plots.doubleHistScatter(
    theta_2, theta_1, xlabel="Vincenty Formula", ylabel="Angular Separation"
)

# %%
t_ratio = theta_2 / theta_1
out = grasp.plots.histogram(
    t_ratio, kde=True, kde_kind="power", xlabel=r"$\theta_V\,/\,\theta_G$", out=True
)
fit = out["kde"]
print(f"A = {fit[0]:.2f}  ;  lambda = {fit[1]:.2f}")

# %%
fit = grasp.stats.fit_data(
    t_ratio[t_ratio < 400], fit="power", x_data=theta_1[t_ratio < 400]
)

coeffs = fit.coeffs

exp = grasp.stats._get_function("power")
exp = exp(theta_1, *coeffs)
print(f"Exponential fit: A = {coeffs[0]:.2f}  ;  lambda = {coeffs[1]:.2f}")

plt.figure()
plt.plot(theta_1, t_ratio, "o", markersize=1, label="Data")
plt.plot(theta_1, exp, c="r", label="fit")
plt.xlabel(r"$\log{\theta_G}$", fontdict=grasp.plots.label_font)
plt.ylabel(r"$\log{\theta_V\,/\,\theta_G}$", fontdict=grasp.plots.label_font)
plt.yscale("log")
plt.xscale("log")
plt.legend()

# %% [markdown]
# ## $r_{2D}(\theta_V)$ vs $r_{2D}(\theta_{GC})$

# %%
aps.gc.dist = 1851  # Baumgardt, Vasiliev: 2021 # pc
f.substitute("radial_distance_2d", {"r_{c}": aps.gc.dist})
f.radial_distance_2d

# %%
f.var_order("radial_distance_2d")

# %%
print(r"Computation using $\theta_{GC}$")
r2d_1 = f.compute("radial_distance_2d", data={"theta": theta_1}, asarray=True)
print("")
print(r"Computation using $\theta_{V}$")
r2d_2 = f.compute("radial_distance_2d", data={"theta": theta_2}, asarray=True)

r_ratio = r2d_2 / r2d_1

# %%
grasp.plots.doubleHistScatter(
    r2d_2, r2d_1, xlabel=r"$\theta_{V}$", ylabel=r"$\theta_{GC}$"
)

# %%
grasp.plots.histogram(r_ratio, kde=True, kde_kind="exponential", xlabel=r"$r_{2d}$")

# %%
reg_e = grasp.stats.regression(r_ratio[r_ratio < 400], "exponential", False)
reg_p = grasp.stats.regression(r_ratio[r_ratio < 400], "power", False)

# %%
fit_p = grasp.stats.fit_data(
    r_ratio[r_ratio < 400], fit="power", x_data=r2d_1[r_ratio < 400]
)
# fit_e = grasp.stats.fit_data(r_ratio[r_ratio<400], fit='exponential', x_data = r2d_1[r_ratio<400])

plt.figure()
plt.plot(r2d_1[r_ratio < 400], r_ratio[r_ratio < 400], "o", markersize=1, label="Data")
plt.plot(r2d_1[r_ratio < 400], fit_p.y, c="r", label="power fit")
# plt.plot(r2d_1[r_ratio<400], fit_e.y, c='g', label='exponential fit')
plt.xlabel(r"$\log{[r_{2d}(\theta_{GC})]}$", fontdict=grasp.plots.label_font)
plt.ylabel(
    r"$\log{[\frac{r_{2d}(\theta_{V})}{r_{2d}(\theta_{GC})}]}$",
    fontdict=grasp.plots.label_font,
)
plt.yscale("log")
plt.xscale("log")
plt.legend()

# %%
print(reg_e)
print('\n')
print(reg_p)

# %%
f.display_all()
