"""
Created on Mon Aug  2 11:21:15 2021

@author: Peter Clark
"""
from loguru import logger

logger.disable("monc_utils.io")

PGRID = [False,False,False]
UGRID = [True,False,False]
VGRID = [False,True,False]
WGRID = [False,False,True]
TGRID = PGRID
BGRID = [True,True,False]

#   Mapping of data locations on grid via logical triplet:
#   logical[u-point,v-point,w-point]
#          [False,  False,  False  ] --> (p,th,q)-point
var_properties = {"u":{"grid":UGRID,
                       "units":'m s-1'},
                  "v":{"grid":VGRID,
                       "units":'m s-1'},
                  "w":{"grid":WGRID,
                       "units":'m s-1'},
                  "th":{"grid":TGRID,
                        "units":'K'},
                  "theta":{"grid":TGRID,
                        "units":'K'},
                  "thref":{"grid":TGRID,
                        "units":'K'},
                  "p":{"grid":PGRID,
                       "units":'Pa'},
                  "pressure":{"grid":PGRID,
                       "units":'Pa'},
                  "pref":{"grid":PGRID,
                       "units":'Pa'},
                  "q_vapour":{"grid":TGRID,
                              "units":'kg/kg'},
                  "q_cloud_liquid_mass":{"grid":TGRID,
                                         "units":'kg/kg'},
                  "q_ice_mass":{"grid":TGRID,
                                "units":'kg/kg'},
                  "u_b":{"grid":BGRID,
                                         "units":'m s-1'},
                  "v_b":{"grid":BGRID,
                                         "units":'m s-1'},
                  }

