"""
General Thermodynamics functions.

Peter Clark
"""

import numpy as np
import xarray as xr
import subfilter
import subfilter.utils.difference_ops as do
import subfilter.thermodynamics.thermodynamics_constants as tc


def esat(T):
    """
    Saturation Vapour Pressure over Water.

    Derived variable name: esat

    Parameters
    ----------
        T: numpy array or xarray DataArray
            Temperature (K)

    Returns
    -------
        res: numpy array or xarray DataArray
            Vapour pressure over water (Pa)
    """
    T_ref=tc.freeze_pt
    T_ref2=243.04-T_ref # Bolton uses 243.5
    es_Tref=610.94      # Bolton uses 611.2
    const=17.625        # Bolton uses 17.67
    res = es_Tref * np.exp(const * (T - T_ref)/(T+T_ref2))

    if type(res) is xr.core.dataarray.DataArray:
        res.name = 'esat'

    return res

def esat_ice(T):
    """
    Saturation Vapour Pressure over Ice.

    Derived variable name: esat_ice

    Magnus Teten,
    Murray (1967)

    Parameters
    ----------
        T: numpy array or xarray DataArray.
            Temperature (K)

    Returns
    -------
        res: numpy array or xarray DataArray
            Vapour pressure over ice(Pa)
    """
    T_ref=tc.triple_pt
    T_ref2= -7.66
    es_Tref=610.87
    const= 21.8745584
    res = es_Tref * np.exp(const * (T - T_ref)/(T+T_ref2))
    if type(res) is xr.core.dataarray.DataArray:
        res.name = 'esat_ice'
    return res

def inv_esat(es):
    """
    Temperature for given Saturation Vapour Pressure over Water.

    Parameters
    ----------
         es: numpy array or xarray DataArray
             Vapour pressure over water (Pa)

    Returns
    -------
        T: numpy array or xarray DataArray
            Temperature (K)
    """
    T_ref=tc.freeze_pt
    T_ref2=243.04-T_ref
#
# This is how constants are derived:
#    es_Tref=610.94
#    const=17.625
#    ln_es_Tref = np.log(es_Tref)
#    C1 = const * T_ref - ln_es_Tref * T_ref2
#    C2 = const + ln_es_Tref
#
    ln_es_Tref = 6.41499875468
    C1 = 5007.4243625
    C2 = 24.039998754
    ln_es =  np.log(es) / ln_es_Tref
    T = (T_ref2 * ln_es +  C1) / (C2 - ln_es)
    return T

def inv_esat_ice(es):
    """
    Temperature for given Saturation Vapour Pressure over Water.

    Magnus Teten,
    Murray (1967)

    Parameters
    ----------
         es: numpy array or xarray DataArray
             Vapour pressure over water (Pa)

    Returns
    -------
        T: numpy array or xarray DataArray
            Temperature (K)
    """
#    T_ref=tc.tc.triple_pt
    T_ref2=-7.66
#
# This is how constants are derived:
#    es_Tref = 610.87
#    const = 21.8745584
#    ln_es_Tref = np.log(es_Tref)
#    C1 = const * T_ref - ln_es_Tref * T_ref2
#    C2 = const + ln_es_Tref
#
    ln_es_Tref = 6.4148841705762614
    C1 = 6024.3923852906155
    C2 = 28.289442570576263

    ln_es =  np.log(es) / ln_es_Tref
    T = (T_ref2 * ln_es +  C1) / (C2 - ln_es)
    return T

def esat_over_Tkappa(T):
    """
    Computes :math:`e_s/T^{(1/kappa)}` (es in Pa).

    From Bolton 1980.

    Parameters
    ----------
        T: numpy array or xarray DataArray
            Temperature (K)

    Returns
    -------
        res: numpy array or xarray DataArray

    """
    T_ref = tc.freeze_pt
    T_ref2=217.8-T_ref
    es_over_Tkappa_Tref = 1.7743E-6
    const=12.992
    res = es_over_Tkappa_Tref * np.exp(const * (T - T_ref)/(T+T_ref2))
    return res

def exner(p):
    """
    Compute Exner Pressure.

    Derived variable name: exner


    Parameters
    ----------
        p: numpy array or xarray DataArray
            Pressure (Pa).

    Returns
    -------
        p_exner: numpy array or xarray DataArray

    """
    p_exner = (p / tc.p_ref_theta)**tc.kappa
    if type(p_exner) is xr.core.dataarray.DataArray:
        p_exner.name = 'exner'
    return p_exner

def inv_exner(p):
    """
    Compute 1/(Exner Pressure).

    Parameters
    ----------
        p: numpy array or xarray DataArray
            Pressure (Pa).

    Returns
    -------
        inv_p_exner: numpy array or xarray DataArray

    """
    inv_p_exner = (p / tc.p_ref_theta)**(-tc.kappa)
    return inv_p_exner


def temperature(theta, p):
    """
    Compute Potential Temperature.

    Derived variable name: T

    Parameters
    ----------
        theta: numpy array or xarray DataArray
            Potential temperature of dry air (K),
        p: numpy array or xarray DataArray
            Pressure (Pa).

    Returns
    -------
        T: numpy array or xarray DataArray
            Temperature (K).
    """
    T = theta * exner(p)
    if type(T) is xr.core.dataarray.DataArray:
        T.name = 'T'

    return T


def potential_temperature(T, p):
    """
    Compute Potential Temperature.

    Derived variable name: th

    Parameters
    ----------
        T: numpy array or xarray DataArray
            Temperature (K).
        p: numpy array or xarray DataArray
            Pressure (Pa).

    Returns
    -------
        theta: numpy array or xarray DataArray
            Potential temperature of dry air (K),
    """
    theta=T * inv_exner(p)
    if type(theta) is xr.core.dataarray.DataArray:
        theta.name = 'th'

    return theta

def moist_potential_temperature(T, p, m):
    """
    Compute Moist Potential Temperature.

    Parameters
    ----------
        T: numpy array or xarray DataArray
            Temperature (K).
        p: numpy array or xarray DataArray
            Pressure (Pa).
        m: numpy array or xarray DataArray
            Mixing ratio(kg/kg).

    Returns
    -------
        theta: numpy array or xarray DataArray
            Potential temperature of moist air (K) .

    """
    theta=T*(tc.p_ref_theta/p)**(tc.kappa*(1-tc.kappa_v * m))
    if type(theta) is xr.core.dataarray.DataArray:
        theta.name = 'th_m'

    return theta

def q_to_mix(q):
    """
    Convert specific humidity to mixing ratio.

    Derived variable name: m_x where input is q_x

    Parameters
    ----------
        q: numpy array or xarray DataArray
            Specific humidity (kg/kg).

    Returns
    -------
        m: numpy array or xarray DataArray
            Mixing Ratio(kg/kg)
    """
    qc=np.clip(q, 0, 0.999)
    m = qc / (1-qc)
    if type(m) is xr.core.dataarray.DataArray:
        m.name = 'm' + q.name[1:]

    return m

def mix_to_q(m):
    """
    Convert mixing ratio to specific humidity.

    Derived variable name: q_x where input is m_x

    Parameters
    ----------
        m: numpy array or xarray DataArray
            Mixing Ratio(kg/kg)

    Returns
    -------
        q: numpy array or xarray DataArray
            Specific humidity (kg/kg).
    """
    q = m / (1+m)
    if type(q) is xr.core.dataarray.DataArray:
        q.name = 'q' + m.name[1:]
    return q

def q_p_to_e(q, p):
    """
    Convert specific humidity and pressure to vapour pressure.

    Parameters
    ----------
        q: numpy array or xarray DataArray
            Specific humidity (kg/kg)
        p: numpy array or xarray DataArray
            Total Pressure (Pa)

    Returns
    -------
        e: numpy array or xarray DataArray
            Vapour pressure (Pa)
    """
    e = q * p / (q * (1-tc.epsilon) + tc.epsilon )
    return e

def e_p_to_q(e, p):
    """
    Convert vapour pressure and total pressure to specific humidity.

    Parameters
    ----------
        e: numpy array or xarray DataArray
            Vapour pressure (Pa)
        p: numpy array or xarray DataArray
            Pressure (Pa)

    Returns
    -------
        q: numpy array or xarray DataArray
            Specific humidity (kg/kg)
    """
    q = np.clip(tc.epsilon * e / (p - (1-tc.epsilon) * e), 0, 0.999)
    return q

def t_lcl_td(T, TD):
    """
    T at lifting condensation level from Dewpoint.

    Derived variable name: t_lcl_td

    From Bolton 1980

    Parameters
    ----------
        T: numpy array or xarray DataArray
            Temperature (K).
        TD: numpy array or xarray DataArray
            Dew point Temperature (K).

    Returns
    -------
        Tlcl : numpy array or xarray DataArray
            temperature at lifting condensation level (K)

    """
    T_ref = 56.0
    const = 800.0
    tlcl = 1 / (1 / (TD-T_ref)+np.log(T/TD) / const) + T_ref
    if type(tlcl) is xr.core.dataarray.DataArray:
        tlcl.name = 't_lcl_td'
    return tlcl

def t_lcl_e(T, e):
    """
    T at lifting condensation level from vapour presssure.

    Derived variable name: t_lcl_e

    From Bolton 1980

    Parameters
    ----------
        T: numpy array or xarray DataArray
            Temperature (K).
        e: numpy array or xarray DataArray
            Vapour pressure (Pa).

    Returns
    -------
        Tlcl : numpy array or xarray DataArray
            temperature at lifting condensation level (K)

    """
    T_ref = 55.0
    const = 2840.0
    C2 = -0.199829814012
    C3 = 3.5
    tlcl = const / (C3 * np.log(T) - np.log(e) + C2) + T_ref
    if type(tlcl) is xr.core.dataarray.DataArray:
        tlcl.name = 't_lcl_e'
    return tlcl

def t_lcl_rh(T, RH):
    """
    T at lifting condensation level from RH.

    Derived variable name: t_lcl_rh

    From Bolton 1980

    Parameters
    ----------
        T: numpy array or xarray DataArray
            Temperature (K).
        RH: numpy array or xarray DataArray
            Relative humidity (%)

    Returns
    -------
        Tlcl : numpy array or xarray DataArray
            temperature at lifting condensation level (K)

    """
    T_ref = 55.0
    const = 2840.0
    tlcl = 1 / (1 / (T-T_ref) - np.log(RH/100) / const) + T_ref
    if type(tlcl) is xr.core.dataarray.DataArray:
        tlcl.name = 't_lcl_rh'
    return tlcl

def latheat(T, sublim=0, model=0, focwil_T=None) :
    """
    Latent heat of condensation or sublimation.

    Parameters
    ----------
        T: numpy array or xarray DataArray
            Temperature (K).
        sublim: int (optional)
            = 1 return Latent heat of sublimation
        Model: int (optional)
            = 1 use UM fixed values
        focwil_T:  float (optional)
            use linear ramp in ice fraction from
            focwil_T to freezing.

    Returns
    -------
        latheat: numpy array or xarray DataArray
            Latent heat of condensation(K)
    """
    if model == 1 :
        el0 = tc.cvap_water
        elm = tc.cfus_water

        el=np.ones_like(T) * el0

    else :
# From Pruppacher and Klett
        el0 = 2.5e6
        p1 = 0.167e0
        pg = 3.67e-4
        el = el0*((tc.freeze_pt/T)**(p1+pg*T))
    if sublim or focwil_T is not None:
        TC=T-tc.freeze_pt
        lm0 = 333584.0
        lm1 = 2029.97
        lm2 = -10.4638
        elm=lm0+lm1*TC+lm2*TC*TC

    if sublim:
        focwii = np.zeros_like(el)
        focwii[T <= tc.freeze_p] = 1.0
        el +=  focwii * elm
    elif focwil_T is not None:
        focwii = np.clip((tc.freeze_pt - T) /(tc.freeze_pt - focwil_T), 0, 1)
        el += focwii * elm

    return el

def dewpoint(T, p, q) :
    """
    Dewpoint.

    Derived variable name: T_dew

    Parameters
    ----------
        T: numpy array or xarray DataArray
            Temperature.
        p: numpy array or xarray DataArray
            Pressure (Pa).
        q: numpy array or xarray DataArray
            specific humidity (kg/kg)

    Returns
    -------
        TD: Nnmpy array
            Dew-point temperature (K).
    """
    TD = T.copy(deep=True)

#  calculate vapour pressure, and from that the dewpoint in kelvins.
    v_pres = np.clip(q * p/( tc.epsilon + q), 1e-10, None)

    TD = np.clip(inv_esat(v_pres), None, T)
    if type(TD) is xr.core.dataarray.DataArray:
       TD.name = 'T_dew'

    return TD

def qsat(T, p) :
    """
    Saturation vapour pressure.

    Derived variable name: qsat

    Parameters
    ----------
    T : numpy array or xarray DataArray
        Temperature. (K)
    p : numpy array or xarray DataArray
        Pressure (Pa).

    Returns
    -------
    qs : numpy array or xarray DataArray
        Saturation specific humidity (kg/kg) over water.
    """
    es = esat(T)
    fsubw = 1.0 + 1.0E-8 * p * (4.5 + 6.0E-4 * (T - tc.freeze_pt) * (T - tc.freeze_pt) )
    es = es * fsubw
    qs = e_p_to_q(es, p)
    if type(qs) is xr.core.dataarray.DataArray:
        qs.name = 'qsat'

    return qs

def dqsatbydT(T, p) :
    """
    :math:`{alpha= dq_{s}}/{dT}`.

    Derived variable name: alpha

    Parameters
    ----------
    T : numpy array or xarray DataArray
        Temperature. (K)
    p : numpy array or xarray DataArray
        Pressure (Pa).

    Returns
    -------
    alpha : numpy array or xarray DataArray

    """
    alpha = tc.epsilon * tc.cvap_water * qsat(T, p) / \
            (tc.gas_const_air * T * T)
    if type(alpha) is xr.core.dataarray.DataArray:
       alpha.name = 'alpha'
    return alpha

def equiv_potential_temperature_approx(T, p, q):
    """
    Equivalent potential temperature.

    From Bolton 1980

    Parameters
    ----------
    T : numpy array or xarray DataArray
        Temperature. (K)
    p : numpy array or xarray DataArray
        Pressure (Pa).
    q: numpy array or xarray DataArray
        specific humidity (kg/kg)

    Returns
    -------
    theta_e: numpy array or xarray DataArray
        Fast estimate of equivalent potential temperature (K).

    """
    C1 = 3.376E3
    C2 = 2.54
    C3 = 0.81
    e = q_p_to_e(q, p)
    m = q_to_mix(q)
    T_LCL = t_lcl_e(T, e)
    theta= moist_potential_temperature(T , p, m)

    theta_e = theta * \
      np.exp((C1/T_LCL-C2) * m * (1 + C3 * m) )
    return theta_e

def equiv_potential_temperature(T, p, q) :
    """
    Equivalent potential temperature.

    Derived variable name: th_e

    From Bolton 1980

    Parameters
    ----------
    T : numpy array or xarray DataArray
        Temperature. (K)
    p : numpy array or xarray DataArray
        Pressure (Pa).
    q: numpy array or xarray DataArray
        specific humidity (kg/kg)

    Returns
    -------
    theta_e: numpy array or xarray DataArray
        Accurate estimate of equivalent potential temperature (K).

    """
    C1 = 3.036E3
    C2 = 1.78
    C3 = 0.448
    e = q_p_to_e(q, p)
    m = q_to_mix(q)
    T_LCL = t_lcl_e(T, e)
    theta_DL = T * ( tc.p_ref_theta/(p - e) )**tc.kappa \
               * (T / T_LCL)**(tc.kappa_v * m)

    theta_e = theta_DL * \
      np.exp((C1/T_LCL-C2) * m * (1 + C3 * m) )

    if type(theta_e) is xr.core.dataarray.DataArray:
        theta_e.name = 'th_e'

    return theta_e

def wet_bulb_potential_temperature(T, p, q):
    """
    Wet-bulb potential temperature.

    Derived variable name: th_w

    From Davies-Jones 2007

    Parameters
    ----------
    T : numpy array or xarray DataArray
        Temperature. (K)
    p : numpy array or xarray DataArray
        Pressure (Pa).
    q: numpy array or xarray DataArray
        specific humidity (kg/kg)

    Returns
    -------
    theta_w: numpy array or xarray DataArray
        Wet-bulb potential temperature (K) numpy array or xarray DataArray
    """
    A = 2675.0
    T_ref=tc.freeze_pt
    b = 243.04
    T_ref2 = b-T_ref      # Bolton uses 243.5
#    es_Tref = 610.94      # Bolton uses 611.2
                          # a
    const = 17.625        # Bolton uses 17.67
    Tref3 = 45.114 + T_ref
    Tref4 = 43.380 + T_ref
    C1 = -51.489
    C2 = 0.6069
    C3 = -0.01005

    th_E = equiv_potential_temperature_approx(T, p, q)
    th_W = th_E.copy(deep=True)
    if type(th_E) is xr.core.dataarray.DataArray:
        Ars = A * q_to_mix(qsat(th_E, p))
        dlnesdT = const * b / (th_E + T_ref2)**2
        th_W = xr.where(th_E < 257, th_E -  Ars/(1+Ars*dlnesdT),
                                    Tref3 + C1 * (T_ref/th_E)**tc.rk)
        th_W = xr.where(377 <= th_E, Tref4 + C1 * (T_ref/th_E)**tc.rk \
                                          + C2 * (th_E/T_ref)**tc.rk \
                                          + C3 * (th_E/T_ref)**(2*tc.rk), th_W)
        th_W = xr.where(674 <= th_E, np.nan, th_W)
    else:

        ir1 = th_E < 257
        ir2 = np.logical_and(257 <= th_E, th_E < 377)
        ir3 = np.logical_and(377 <= th_E, th_E < 674)
        ir4 = 674 <= th_E
        if np.size(ir1) !=  0:
            Ars = A * q_to_mix(qsat(th_E[ir1], p[ir1]))
            dlnesdT = const * b / (th_E[ir1] + T_ref2)**2
            th_W[ir1] = th_E[ir1] -  Ars/(1+Ars*dlnesdT)
        if np.size(ir2) !=  0:
            th_W[ir2] = Tref3 + C1 * (T_ref/th_E[ir2])**tc.rk
        if np.size(ir3) !=  0:
            th_W[ir3] = Tref4 + C1 * (T_ref/th_E[ir3])**tc.rk \
              + C2 * (th_E[ir3]/T_ref)**tc.rk \
              + C3 * (th_E[ir3]/T_ref)**(2*tc.rk)
        if np.size(ir4) !=  0:
            th_W[ir4] = np.nan
    if type(th_W) is xr.core.dataarray.DataArray:
        th_W.name = 'th_w'
    return th_W

def wet_bulb_temperature(T, p, q):
    """
    Wet-bulb temperature.

    Derived variable name: T_w

    From Davies-Jones 2007

    Parameters
    ----------
    T : numpy array or xarray DataArray
        Temperature. (K)
    p : numpy array or xarray DataArray
        Pressure (Pa).
    q: numpy array or xarray DataArray
        specific humidity (kg/kg)

    Returns
    -------
    theta_w: numpy array or xarray DataArray
        Wet-bulb temperature (K) numpy array or xarray DataArray
    """
    T_ref=tc.freeze_pt
    A = 2675.0
    const = 17.625        # Bolton uses 17.67
    b = 243.04
    T_ref2 = b-T_ref # Bolton uses 243.5
    pi = (p/tc.p_ref_theta)**tc.kappa

    pi2 = pi * pi

    D_pi=1/(0.1859 * (p/tc.p_ref_theta)+0.6512)

    k1_pi = 137.81 * pi -38.5 * pi2 - 53.737

    k2_pi = 56.831 * pi -4.392 * pi2 - 0.384

#    print D_pi, k1_pi, k2_pi

    th_E = equiv_potential_temperature(T, p, q)
    TE=th_E/pi
    inv_TE = (tc.freeze_pt/TE)**tc.rk
    TW = TE.copy(deep=True)
    if type(inv_TE) is xr.core.dataarray.DataArray:
        Ars = A * q_to_mix(qsat(TE, p))
        dlnesdT = const * b / (TE + T_ref2)**2
        TW = xr.where(D_pi < inv_TE , TE -  Ars/(1+Ars*dlnesdT),
                                      tc.freeze_pt + k1_pi
                                        - k2_pi * inv_TE)
        TW = xr.where( inv_TE <= 1.0, tc.freeze_pt + (k1_pi - 1.21)
                                        - (k2_pi - 1.21) * inv_TE, TW)
        TW = xr.where( inv_TE <= 0.4, tc.freeze_pt + (k1_pi - 2.66)
                                        - (k2_pi - 1.21) * inv_TE
                                        + 0.58 / inv_TE, TW)

    else:

        ir1 = D_pi < inv_TE
        ir2 = np.logical_and(1.0 < inv_TE, inv_TE <= D_pi)
        ir3 = np.logical_and(0.4 < inv_TE, inv_TE <=  1 )
        ir4 = inv_TE <= 0.4
        if np.size(ir1) !=  0:
            Ars = A * q_to_mix(qsat(TE[ir1], p))
            dlnesdT = const * b / (TE[ir1] + T_ref2)**2
    #        print '1:', TE, Ars, dlnesdT
            TW[ir1] = TE[ir1] -  Ars/(1+Ars*dlnesdT)
        if np.size(ir2) !=  0:
    #        print '2:', TE, inv_TE
            TW[ir2] = tc.freeze_pt + k1_pi[ir2] - k2_pi[ir2] * inv_TE[ir2]
        if np.size(ir3) !=  0:
    #        print '3:', TE, inv_TE
            TW[ir3] = tc.freeze_pt + (k1_pi[ir3] - 1.21) - (k2_pi[ir3] - 1.21) * inv_TE[ir3]
        if np.size(ir4) !=  0:
    #        print '4:', TE, inv_TE
            TW[ir4] = tc.freeze_pt + (k1_pi[ir4] - 2.66) - (k2_pi[ir4] - 1.21) * inv_TE[ir4] \
              +0.58 / inv_TE[ir4]
    if type(TW) is xr.core.dataarray.DataArray:
        TW.name = 'T_w'
    return TW

def liquid_water_potential_temperature(theta, q_cl, pi):
    """
    Liquid water potential temperature.

    Derived variable name: th_L

    Approximate form as in MONC

    Parameters
    ----------
    theta : numpy array or xarray DataArray
        Potential Temperature. (K)
    q_cl : numpy array or xarray DataArray
        q_cloud_liquid_mass.
    pi: numpy array or xarray DataArray
        Exner pressure.

    Returns
    -------
    theta_w: numpy array or xarray DataArray
        Wet-bulb potential temperature (K) numpy array or xarray DataArray
    """
    th_L = theta - tc.L_over_cp * q_cl / pi
    if type(th_L) is xr.core.dataarray.DataArray:
        th_L.name = 'th_L'
    return th_L

def virtual_potential_temperature(theta, q_v, q_cl):
    """
    Virtual potential temperature.

    Derived variable name: th_v

    Parameters
    ----------
    theta : numpy array or xarray DataArray
        Potential Temperature. (K)
    q_v : numpy array or xarray DataArray
        specific humidity
    q_cl : numpy array or xarray DataArray
        specific cloud liquid water content.

    Returns
    -------
    theta_v: numpy array or xarray DataArray
        Virtual potential temperature (K)
    """
    th_v = theta * (1 + tc.c_virtual * q_v - q_cl)
    if type(th_v) is xr.core.dataarray.DataArray:
        th_v.name = 'th_v'
    return th_v

def virtual_potential_temperature_monc(theta, thref, q_v, q_cl):
    """
    Virtual potential temperature.

    Derived variable name: th_v_monc

    Approximate form as in MONC

    Parameters
    ----------
    theta : numpy array or xarray DataArray
        Potential Temperature. (K)
    thref : numpy array or xarray DataArray
        Reference Potential Temperature (usually 1D). (K)
    q_v : numpy array or xarray DataArray
        specific humidity
    q_cl : numpy array or xarray DataArray
        specific cloud liquid water content.

    Returns
    -------
    theta_v: numpy array or xarray DataArray
        Virtual potential temperature (K)
    """
    th_v = theta + thref * (tc.c_virtual * q_v - q_cl)
    if type(th_v) is xr.core.dataarray.DataArray:
        th_v.name = 'th_v_monc'
    return th_v

def q_total(*args):
    """
    Specfic Total Water Content.

    Derived variable name: q_total

    Parameters
    ----------
    args : list of numpy array or xarray DataArrays
        specific water variables (e.g.[ q_v, q_cl]).

    Returns
    -------
    q_total : numpy array or xarray DataArray
        sum of args
    """
    qt = None
    for a in args:
        if a is not None:
            if qt is not None:
                qt += a
            else:
                qt = a
    if type(qt) is xr.core.dataarray.DataArray:
        qt.name = 'q_total'
    return qt

def buoyancy(th_v):
    """
    Buoyancy from theta_v.

    Derived variable name: buoyancy

    Parameters
    ----------
    th_v : numpy array or xarray DataArray
        Virtual potential temperature.

    Returns
    -------
    buoyancy : numpy array or xarray DataArray
    """
    xdname = [a for a in th_v.dims if a.startswith('x')][0]
    ydname = [a for a in th_v.dims if a.startswith('y')][0]
    mean_thv = th_v.mean(dim=(xdname, ydname))
    b = tc.grav * (th_v - mean_thv)/mean_thv
    if type(b) is xr.core.dataarray.DataArray:
        b.name = 'buoyancy'
    return b

def buoyancy_monc(th_v, thref):
    """
    Buoyancy from theta_v.

    Derived variable name: buoyancy_monc

    MONC approximation.

    Parameters
    ----------
    th_v : numpy array or xarray DataArray
        Virtual potential temperature.

    Returns
    -------
        buoyancy : numpy array or xarray DataArray
    """
    b = tc.grav * (th_v - thref)/thref
    if type(b) is xr.core.dataarray.DataArray:
        b.name = 'buoyancy_monc'
    return b

def dbdz(th, p, q_v, q_cl, z, zn):
    """
    Vertical Gradient of Buoyancy from theta_v.

    Derived variable name: dbdz

    Parameters
    ----------
    theta : numpy array or xarray DataArray
        Potential Temperature. (K)
    thref : numpy array or xarray DataArray
        Reference Potential Temperature (usually 1D). (K)
    p : numpy array or xarray DataArray
        Pressure (Pa).
    q_v : numpy array or xarray DataArray
        specific humidity
    q_cl : numpy array or xarray DataArray
        specific cloud liquid water content.
    th_v : numpy array or xarray DataArray
        Virtual potential temperature.

    Returns
    -------
        dbuoyancy/dz : numpy array or xarray DataArray
    """
    th_v = virtual_potential_temperature(th, q_v, q_cl)
    b = buoyancy(th_v)
    dbdz = do.d_by_dz_field(b, z, zn, grid= 'w')
    if type(dbdz) is xr.core.dataarray.DataArray:
        dbdz.name = 'dbdz'

    return dbdz

def dbdz_monc(th, thref, p, q_v, q_cl, z, zn):
    """
    Vertical Gradient of Buoyancy from theta_v.

    Derived variable name: db_moncdz

    MONC approximation

    Parameters
    ----------
    theta : numpy array or xarray DataArray
        Potential Temperature. (K)
    thref : numpy array or xarray DataArray
        Reference Potential Temperature (usually 1D). (K)
    p : numpy array or xarray DataArray
        Pressure (Pa).
    q_v : numpy array or xarray DataArray
        specific humidity
    q_cl : numpy array or xarray DataArray
        specific cloud liquid water content.
    th_v : numpy array or xarray DataArray
        Virtual potential temperature.

    Returns
    -------
        buoyancy : numpy array or xarray DataArray
    """
    th_v = virtual_potential_temperature_monc(th, thref, q_v, q_cl)
    b = buoyancy_monc(th_v, thref)
    dbdz = do.d_by_dz_field(b, z, zn, grid= 'w')

    if type(dbdz) is xr.core.dataarray.DataArray:
        dbdz.name = 'db_moncdz'

    return dbdz


def rh(T, p, q):
    """
    Relative Humidity.

    Derived variable name: rh

    Parameters
    ----------
        T: numpy array or xarray DataArray
            Temperature (K).
        p: numpy array or xarray DataArray
            Pressure (Pa).
        q: numpy array or xarray DataArray
            specific humidity (kg/kg)

    Returns
    -------
        rh: numpy array or xarray DataArray
            Relative Humidity.
    """
    rh=T.copy(deep=True)

#  calculate vapour pressure.
    e = np.clip( q * p/( tc.epsilon + q), 1e-10, None)

    es = esat(T)

    rh = np.clip(e / es, 0, 1)

    if type(rh) is xr.core.dataarray.DataArray:
        rh.name = 'rh'

    return rh

def rh_ice(T, p, q):
    """
    Relative Humidity wrt Ice.

    Derived variable name: rh_ice

    Parameters
    ----------
        T: numpy array or xarray DataArray
            Temperature.
        p: numpy array or xarray DataArray
            Pressure (Pa).
        q: numpy array or xarray DataArray
            specific humidity (kg/kg)

    Returns
    -------
        rh: numpy array or xarray DataArray
            Relative Humidity.
    """
    rh=T.copy(deep=True)

#  calculate vapour pressure.
    e = np.clip( q * p/( tc.epsilon + q), 1e-10, None)

    es = esat_ice(T)

    rh = np.clip(e / es, 0, 1)

    if type(rh) is xr.core.dataarray.DataArray:
        rh.name = 'rh_ice'

    return rh

def a_L_monc(T, p):
    """
    Cloud Factor.

    Derived variable name: a_L

    Parameters
    ----------
        T: numpy array or xarray DataArray
            Temperature.
        p: numpy array or xarray DataArray
            Pressure (Pa).

    Returns
    -------
        a_L: numpy array or xarray DataArray
            Factor used in calculating liqid water content.
    """
    alpha_L = dqsatbydT(T, p)
    a_L = 1.0 / (1.0 + tc.L_over_cp * alpha_L)
    if type(a_L) is xr.core.dataarray.DataArray:
        a_L.name = 'a_L'
    return a_L

def cloud_params_monc(th_ref, p_ref):
    """
    Cloud Parameters.

    MONC Approximation

    Parameters
    ----------
        th_ref: numpy array or xarray DataArray
            Reference Potential Temperature.
        p_ref: numpy array or xarray DataArray
            Reference Pressure (Pa).

    Returns
    -------
        dict
            Factors used in calculating liqid water content.
         "T_ref","pi_ref", "qs_ref", "a_L", "alpha_L".

    """
    pi_ref = exner(p_ref)
    T_ref = th_ref * pi_ref
    qs_ref = qsat(T_ref, p_ref)
    alpha_L = dqsatbydT(T_ref, p_ref)
    a_L = 1.0 / (1.0 + tc.L_over_cp * alpha_L)

    output_dir = {
        "T_ref": T_ref,
        "pi_ref": pi_ref,
        "qs_ref": qs_ref,
        "a_L": a_L,
        "alpha_L": alpha_L,
        }

    return output_dir

def betas_monc(th, p):
    """
    Beta factors in cloudy buoyancy calculation.

    Parameters
    ----------
    th : numpy array or xarray DataArray
        Potential Temperature. (K)
    p : numpy array or xarray DataArray
        Pressure (Pa).

    Returns
    -------
    tuple of numpy array or xarray DataArrays
        (bt, bq, bc, alpha_L, a_L).
    """
    betas = cloud_params_monc(th, p)
    betas["bt"] = 1/th
    betas["bq"] = 1/tc.epsilon -1
    betas["bc"] = betas["a_L"] * (latheat(betas["T_ref"], model=1)
                                  / (tc.cp_air * betas["T_ref"])
                                  - 1/tc.epsilon)

    return betas

def buoyancy_moist(th, th_ref, p, q_v, q_cl, thresh = 1.0e-5):
    """
    Buoyancy including cloud condensation.

    Derived variable name: buoyancy_moist

    MONC approximation

    Parameters
    ----------
    th : numpy array or xarray DataArray
        Potential Temperature. (K)
    thref : numpy array or xarray DataArray
        Reference Potential Temperature (usually 1D). (K)
    p : numpy array or xarray DataArray
        Pressure (Pa).
    q_v : numpy array or xarray DataArray
        specific humidity
    q_cl : numpy array or xarray DataArray
        specific cloud liquid water content.

    Returns
    -------
        buoyancy : numpy array or xarray DataArray
    """
    betas = betas_monc(th_ref, p)
    th_L = liquid_water_potential_temperature(th, q_cl, betas["pi_ref"])
    qt = q_total(q_v, q_cl)

    delta_q = (qt - betas["qs_ref"] - betas["alpha_L"] * betas["pi_ref"]
                                      * (th_L - th_ref))

    b_dry = th_L * betas["bt"] -1 + qt * betas["bq"]

    if type(th) is xr.core.dataarray.DataArray:
        bc_delta_q = xr.where(delta_q >= thresh, delta_q * betas["bc"], 0)
    else:
        bc_delta_q = np.zeros_like(delta_q)
        iwet = delta_q >= thresh
        bc_delta_q[iwet] = ( delta_q * betas["bc"])[iwet]

    b_wet = b_dry + bc_delta_q

    b = tc.g * b_wet

    if type(b) is xr.core.dataarray.DataArray:
        b.name = 'buoyancy_moist'

    return b

def dmoist_bdz(th, th_ref, p, q_v, q_cl, z, zn, thresh = 1.0e-5):
    """
    Vertical Gradient of (buoyancy including cloud condensation).

    Derived variable name: dmoist_bdz

    MONC approximation.
    This is db/dz with b = beta_t theta_l + beta_q q_t.
    Note - not to be used for vertical buoyancy flux.

    Parameters
    ----------
    th : numpy array or xarray DataArray
        Potential Temperature. (K)
    th_ref : numpy array or xarray DataArray
        Reference Potential Temperature (usually 1D). (K)
    p : numpy array or xarray DataArray
        Pressure (Pa).
    q_v : numpy array or xarray DataArray
        specific humidity
    q_cl : numpy array or xarray DataArray
        specific cloud liquid water content.
    z : xarray coord.
    zn : xarray coord.
    thresh : (Optional) float. Default is 1E-5.
        Threshold for cloud water.

    Returns
    -------
        buoyancy : numpy array or xarray DataArray
    """
    b = buoyancy_moist(th, th_ref, p, q_v, q_cl, thresh = thresh)

    dbdz = do.d_by_dz_field(b, z, zn, grid= 'w')

    if type(dbdz) is xr.core.dataarray.DataArray:
        dbdz.name = 'dmoist_bdz'

    return dbdz

def moist_dbdz(th, th_ref, p_ref, q_v, q_cl, z, zn, thresh = 1.0e-5):
    """
    Vertical Gradient of buoyancy (including cloud condensation).

    Derived variable name: moist_dbdz

    MONC approximation
    This is db/dz = beta_t dtheta_l/dz + beta_q dq_t/dz.
    Note - to be used for vertical buoyancy flux.

    Parameters
    ----------
    theta : numpy array or xarray DataArray
        Potential Temperature. (K)
    thref : numpy array or xarray DataArray
        Reference Potential Temperature (usually 1D). (K)
    p : numpy array or xarray DataArray
        Pressure (Pa).
    q_v : numpy array or xarray DataArray
        specific humidity
    q_cl : numpy array or xarray DataArray
        specific cloud liquid water content.
    z : xarray coord.
    zn : xarray coord.
    thresh : (Optional) float. Default is 1E-5.
        Threshold for cloud water.

    Returns
    -------
        buoyancy : numpy array or xarray DataArray
    """
    betas = betas_monc(th_ref, p_ref)
    th_L = liquid_water_potential_temperature(th, q_cl, betas["pi_ref"])
    qt = q_total(q_v, q_cl)

    delta_q = (qt - betas["qs_ref"] - betas["alpha_L"] * betas["pi_ref"]
                                      * (th_L - th_ref))
#    delta_q = qt - qs

    b_dry_t = betas["bt"]
    b_dry_q = betas["bq"]

    if type(th) is xr.core.dataarray.DataArray:
        bcc = xr.ones_like(th)
        bcc = xr.where(delta_q >= thresh, bcc * betas["bc"], 0)
    else:
        bcc = np.zeros_like(th)
        bct = np.ones_like(th)
        iwet = delta_q >= thresh
        bcc[iwet] = (bct * betas["bc"])[iwet]

    b_wet_t = b_dry_t - betas["alpha_L"] * betas["pi_ref"] * bcc
    b_wet_q = b_dry_q + bcc

    b_wet_t.name = 'beta_t_wet'
    b_wet_q.name = 'beta_q_wet'

    b_wet_t = do.grid_conform_z(b_wet_t, z, zn, 'z')
    b_wet_q = do.grid_conform_z(b_wet_q, z, zn, 'z')

    dbdz = tc.g * ( do.d_by_dz_field(th_L, z, zn, grid= 'w') * b_wet_t
                  + do.d_by_dz_field(qt,   z, zn, grid= 'w') * b_wet_q)

    if type(dbdz) is xr.core.dataarray.DataArray:
        dbdz.name = 'moist_dbdz'

    return dbdz

def saturation(th, th_ref, p_ref, q_v, q_cl):
    """
    Effective saturation.

    Derived variable name: saturation

    MONC approximation

    Parameters
    ----------
    th : numpy array or xarray DataArray
        Potential Temperature. (K)
    th_ref : numpy array or xarray DataArray
        Reference Potential Temperature (usually 1D). (K)
    p_ref : numpy array or xarray DataArray
        Reference Pressure (Pa).
    q_v : numpy array or xarray DataArray
        specific humidity
    q_cl : numpy array or xarray DataArray
        specific cloud liquid water content.

    Returns
    -------
    saturation : numpy array or xarray DataArray
    """
    betas = betas_monc(th_ref, p_ref)
    th_L = liquid_water_potential_temperature(th, q_cl, betas["pi_ref"])
#    T_L = th_L * pi
    qt = q_total(q_v, q_cl)
    s = (qt - betas["qs_ref"] - betas["alpha_L"] * betas["pi_ref"] * (th_L - th_ref))
    s = betas["a_L"] * s
    if type(s) is xr.core.dataarray.DataArray:
        s.name = 'saturation'

    return s

def cloud_fraction(q_cl, thresh = 1.0e-5):
    """
    Compute indicator function on cloud.

    Derived variable name: cloud_fraction

    Parameters
    ----------
    q_cl : numpy array or xarray DataArray
        specific cloud liquid water content.
    thresh : (Optional) float. Default is 1E-5.
        Threshold for cloud water.

    Returns
    -------
    numpy array or xarray DataArray

    """
    if type(q_cl) is xr.core.dataarray.DataArray:
        cf = xr.ones_like(q_cl)
        cf = xr.where(q_cl >= thresh, cf, 0)
        cf.name = "cloud_fraction"
        cf.attrs["units"] = ""
    else:
        cf = np.zeros_like(q_cl)
        iwet = q_cl >= thresh
        cf[iwet] = 1
    if type(cf) is xr.core.dataarray.DataArray:
        cf.name = 'cloud_fraction'
    return cf

derived_vars = {
    # '':
    #     {'vars': ,
    #      'func': ,
    #      'units': ''},
    'rh_ice':
        {'vars': ('T', 'p', 'q_vapour'),
         'func': rh_ice,
         'units': ''},
    'rh':
        {'vars': ('T', 'p', 'q_vapour'),
          'func': rh,
          'units': ''},
    'esat':
        {'vars': ('T',),
         'func': esat,
         'units': 'Pa'},
    'esat_ice':
        {'vars': ('T',),
         'func': esat_ice,
         'units': 'Pa'},
    'exner':
        {'vars': ('p',),
         'func': exner,
         'units': ''},
    'T':
        {'vars': ('th', 'p'),
         'func': temperature,
         'units': 'K'},
    'th':
        {'vars': ('T', 'p'),
         'func': potential_temperature,
         'units': 'K'},
    'th_m':
        {'vars': ('T', 'p', 'm_vapour'),
         'func': moist_potential_temperature,
         'units': 'K'},
    'm_vapour':
        {'vars': ('q_vapour',),
         'func': q_to_mix,
         'units': ''},
    'm_cloud_liquid_mass':
        {'vars': ('q_cloud_liquid_mass',),
         'func': q_to_mix,
         'units': ''},
    'm_total':
        {'vars': ('q_total',),
         'func': q_to_mix,
         'units': ''},
    'q_vapour':
        {'vars': ('m_vapour',),
         'func': mix_to_q,
         'units': 'kg/kg'},
    'q_cloud_liquid_mass':
        {'vars': ('m_cloud_liquid_mass',),
         'func': mix_to_q,
         'units': 'kg/kg'},
    't_lcl_td':
        {'vars': ('T', 'T_dew'),
         'func': t_lcl_td,
         'units': 'K'},
    't_lcl_e':
        {'vars': ('T', 'p_vapour'),
         'func': t_lcl_e,
         'units': 'K'},
    't_lcl_rh':
        {'vars': ('T', 'rh'),
         'func': t_lcl_rh,
         'units': ''},
    'T_dew':
        {'vars': ('T', 'p', 'q'),
         'func': dewpoint,
         'units': 'K'},
    'qsat':
        {'vars': ('T', 'p'),
         'func': qsat,
         'units': ''},
    'alpha':
        {'vars': ('T', 'p'),
         'func': dqsatbydT,
         'units': r'K$^{-1}$'},
    'T_w':
        {'vars': ('T', 'p', 'q_vapour'),
         'func': wet_bulb_temperature,
         'units': 'K'},
    'th_w':
        {'vars': ('T', 'p', 'q_vapour'),
         'func': wet_bulb_potential_temperature,
         'units': 'K'},
    'th_e':
        {'vars': ('T', 'p', 'q_vapour'),
         'func': equiv_potential_temperature,
         'units': 'K'},
    'th_L':
        {'vars': ('th','q_cloud_liquid_mass','piref'),
         'func': liquid_water_potential_temperature,
         'units': 'K'},
    'th_v':
        {'vars': ('th', 'q_vapour','q_cloud_liquid_mass'),
         'func': virtual_potential_temperature,
        'units': 'K'},
    'th_v_monc':
        {'vars': ('th','thref','q_vapour','q_cloud_liquid_mass'),
         'func': virtual_potential_temperature_monc,
        'units': 'K'},
    'q_total':
        {'vars': ('q_vapour','q_cloud_liquid_mass','q_ice_mass'),
         'func': q_total,
         'units': 'kg/kg'},
    'buoyancy':
        {'vars': ('th_v',),
         'func': buoyancy,
         'units': r'm s$^{-2}$'},
    'buoyancy_monc':
        {'vars': ('th_v', 'thref',),
         'func': buoyancy_monc,
         'units': r'm s$^{-2}$'},
    'buoyancy_moist':
        {'vars': ('th','thref','p','q_vapour','q_cloud_liquid_mass'),
         'func': buoyancy_moist,
         'units': r'm s$^{-2}$'},
    'moist_dbdz':
        {'vars':('th', 'thref', 'p', 'q_vapour', 'q_cloud_liquid_mass',
                 'z', 'zn') ,
          'func': moist_dbdz,
          'units': r's$^{-2}$'},
    'dmoist_bdz':
        {'vars':('th', 'thref', 'p', 'q_vapour', 'q_cloud_liquid_mass',
                 'z', 'zn') ,
          'func': dmoist_bdz,
          'units': r's$^{-2}$'},
    'dbdz':
        {'vars': ('th', 'p', 'q_vapour', 'q_cloud_liquid_mass',
                 'z', 'zn'),
         'func': dbdz,
         'units': r's$^{-2}$'},
    'dbdz_monc':
        {'vars': ('th', 'thref', 'p', 'q_vapour', 'q_cloud_liquid_mass',
                 'z', 'zn'),
         'func': dbdz_monc,
         'units': r's$^{-2}$'},
    'saturation':
        {'vars': ('th','thref','pref', 'q_vapour','q_cloud_liquid_mass'),
         'func': saturation,
        'units': 'kg/kg'},
    'cloud_fraction':
        {'vars': ('q_cloud_liquid_mass',),
          'func': cloud_fraction,
          'units': ''},
        }
