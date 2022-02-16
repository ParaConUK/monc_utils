"""
Created on Thu Jan 27 10:52:22 2022

@author: Peter Clark
"""
import numpy as np
import subfilter.thermodynamics.thermodynamics as th
from scipy.special import erf

def richardson(mod_S_sq, dbdz, minshear = 1E-4):
    """
    Richardson number.

    Parameters
    ----------
    mod_S_sq : xarray
        Contains modulus of shear tensor squared.
    dbdz : xarray
        Moist Richardson Number.
    minshear : float (Optional. Default=1E-4).
        Minimum modulus of shear allowed.

    Returns
    -------
    ri : xarray
        Richardson number clipped to 1 on positive side.

    """
    ri = np.clip(dbdz / np.clip(mod_S_sq, minshear, None), None, 1)

    ri.name = 'moist_Ri'
    return ri

def sigma_s(s_qt_qt, s_thL_qt, s_thL_thL, parms):
    """
    Compute sigma s.

    Standard deviation of saturation variable.

    Parameters
    ----------
    s_qt_qt : xarray DataArray
        Variance of qt.
    s_thL_qt : xarray DataArray
        Covariance of qt and theta_L.
    s_thL_thL : xarray DataArray
        Variance of theta_L.
    th_ref : xarray DataArray
        Reference theta profile.
    p_ref :xarray DataArray
        Reference pressure profile.

    Returns
    -------
    sigma_s : xarray
        Standard deviation of saturation variable.

    """
    ap = parms["alpha_L"] * parms["pi_ref"]

    sigma_sq = (s_qt_qt - 2 * s_thL_qt * ap + s_thL_thL * ap * ap)

    sigma_s = np.sqrt(sigma_sq) * parms["a_L"]

    sigma_s.name = "sigma_s"
    sigma_s.attrs["units"] = "kg/kg"

    return sigma_s

def gaussian_cloud(th_L, qt, th_ref, p_ref, s_qt_qt, s_thL_qt, s_thL_thL):
    """
    Compute cloud variables using Gaussian model.

    Deardorf/Mellor-Yamada scheme.

    Parameters
    ----------
    th_L : xarray DataArray
        Liquid Water Potential Temperature. (K)
    qt : xarray DataArray
        Specific Total Water Content. (kg/kg)
    th_ref: xarray DataArray
        Reference Potential Temperature.
    p_ref: xarray DataArray
        Reference Pressure (Pa).
    s_qt_qt : xarray DataArray
        Variance of qt.
    s_thL_qt : xarray DataArray
        Covariance of qt and theta_L.
    s_thL_thL : xarray DataArray
        Variance of theta_L.
    th_ref : xarray DataArray
        Reference theta profile.
    p_ref :xarray DataArray
        Reference pressure profile.

    Returns
    -------
    tuple of xarray DataArray
        (delta_q, qc, sig_s, cloud_fraction, qcl)

    """
    parms = th.cloud_params_monc(th_ref, p_ref)
    delta_q = (qt - parms["qs_ref"] - parms["alpha_L"] * parms["pi_ref"]
                                      * (th_L - th_ref))


    qc = delta_q * parms["a_L"]

    sig_s = sigma_s(s_qt_qt, s_thL_qt, s_thL_thL, parms)
    qn = qc / sig_s

    cloud_fraction = np.clip(0.5 * (1 - erf(-qn / np.sqrt(2))), 0, None)

    qcl = np.clip(sig_s * ( cloud_fraction * qn + (1/np.sqrt(2 * np.pi))
                   * np.exp(-qn * qn / 2)) , 0, None)

    cloud_fraction.name = "cloud_fraction"
    cloud_fraction.attrs["units"] = ""

    qcl.name = r"$q_cl^r$"
    qcl.attrs["units"] = "kg/kg"

    return (delta_q, qc, sig_s, cloud_fraction, qcl)
