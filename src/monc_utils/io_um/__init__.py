"""
Created on Mon Aug  2 11:21:15 2021

@author: Peter Clark
"""
from loguru import logger

logger.disable("monc_utils.io_um")

um_datain_options = {'coord_type':'cartesian',
                     'grid_type':'LAM',
                     'ref_is_init_mean':True,
                     'offsets':{'x':0, 'y':0, 'z':0},
                     'out_prec':'float32',
                     }

PGRID = [False,False,False]
UGRID = [True,False,False]
VGRID = [False,True,False]
WGRID = [False,False,True]
TGRID = WGRID
BGRID = [True,True,False]

var_properties = {"u":{"grid":UGRID,
                       "units":'m.s-1'},
                  "v":{"grid":VGRID,
                       "units":'m.s-1'},
                  "w":{"grid":WGRID,
                       "units":'m.s-1'},
                  "th":{"grid":TGRID,
                        "units":'K'},
                  "theta":{"grid":TGRID,
                        "units":'K'},
                  "air_temperature":{"grid":TGRID,
                        "units":'K'},
                  "p":{"grid":PGRID,
                       "units":'Pa'},
                  "pressure":{"grid":PGRID,
                       "units":'Pa'},
                  "exner":{"grid":TGRID,
                       "units":''},
                  "q_vapour":{"grid":TGRID,
                              "units":'kg/kg'},
                  "q_cloud_liquid_mass":{"grid":TGRID,
                                         "units":'kg/kg'},
                  "q_ice_mass":{"grid":TGRID,
                                "units":'kg/kg'},
                  "u_b":{"grid":BGRID,
                         "units":'m.s-1'},
                  "v_b":{"grid":BGRID,
                         "units":'m.s-1'},
                  "traj_tracer_xr" : {"grid":TGRID,
                       "units":''},
                  "traj_tracer_xi" : {"grid":TGRID,
                       "units":''},
                  "traj_tracer_yr" : {"grid":TGRID,
                       "units":''},
                  "traj_tracer_yi" : {"grid":TGRID,
                       "units":''},
                  "traj_tracer_zr" : {"grid":TGRID,
                       "units":''},
                  }

stash_map = { 'u'                  : 'm01s00i002', 
              'v'                  : 'm01s00i003',
              'w'                  : 'm01s00i150',
              'upward_air_velocity': 'm01s00i150',
              'air_potential_temperature': 'm01s00i004',
              'theta'              : 'm01s00i004',
              'th'                 : 'm01s00i004',
              'surface_altitude'   : 'm01s00i033',
              'q_vapour'           : 'm01s00i010', 
              'specific_humidity'  : 'm01s00i010',
              'q_ice_mass'         : 'm01s00i012', 
              'q_cloud_liquid_mass': 'm01s00i254',
              'exner_rho'          : 'm01s00i255',
              'mr_liquid_cloud'    : 'm01s00i392',
              'mr_ice cloud'       : 'm01s00i393',
              'mr_rain'            : 'm01s00i394',
              'mr_graupel'         : 'm01s00i395',
              'mr_ice crystals'    : 'm01s00i396',
#              'dimensionless_exner_function' : 'm01s00i406',
              'exner'              : 'm01s00i406',
              'p_rho'              : 'm01s00i407',
              'p_th'               : 'm01s00i408',
              'p'                  : 'm01s00i408',
              'p_surf'             : 'm01s00i409',
              'cloud fraction'     : 'm01s00i266',
              'rainrate'           : 'm01s04i203',
              "traj_tracer_xr"     : 'm01s00i700',
              "traj_tracer_xi"     : 'm01s00i701',
              "traj_tracer_yr"     : 'm01s00i702',
              "traj_tracer_yi"     : 'm01s00i703',
              "traj_tracer_zr"     : 'm01s00i704',
              "upward_heat_flux"   : 'm01s03i216',
              "upward_water_vapour_flux" : 'm01s03i222',
              'rho'                : 'm01s15i271',
              'u_b'                : 'm01s15i002',
              'v_b'                : 'm01s15i003',
              'air_temperature'    : 'm01s16i004',
            }

def inverse_lookup(item, indict):
    name = item
    for lookup_name, lookup_item in indict.items():
        if lookup_item == item:
            name = lookup_name
            break
    return name