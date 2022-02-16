============
Introduction
============
 
The purpose of this set of modules is to implement a set of useful utilities for dealing with output from MONC (and similar data). 

==============      
Variable Names
==============
This package uses xarray - returned variables are xarray DataArrays. 
These have names; this package follows the following naming convention.

    * Variables or derived variables read from the source or reference file using :py:func:`~monc_utils.io.datain.get_data` have the name requested (possibly changed from the internal name in the source data by the aliases dictionary). For example, ``u``. They have grid specifications corrected to the appropriate dimensionnspecificatio  in MONC (for example, ``u`` will be on ``x_u``, ``y_p`` and ``zn``) and the time dimension renamed ``time``.    
    * Variables or derived variables read from the source or reference file using :py:func:`~monc_utils.io.datain.get_and_transform` behave as above but are then interpolated to the requested grid (``u``, ``v``, ``w`` or ``p,`` or a tuple made up of x, y and z dimension names) using :py:func:`~monc_utils.data_utils.difference_ops.grid_conform`. 
    * Variables or derived variables read from the source or reference file using :py:func:`~monc_utils.io.datain.get_data_on_grid` behave as above but have the grid name appended to the variable name, e.g. ``u_on_w``. If ``options['save_all']`` is set to ``'yes'``, the variable is retrieved from the derived data file if already there and stored to it if not.
    
================
Input Transforms
================

Basic variables (``'u'``, ``'v'``, ``'w'``, ``'th'``, ``'p'``, ``'q_vapour'``, ``'q_cloud_liquid_mass'``, ``'q_cloud_ice_mass'``) are expected to be available in the input file.
If MONC is used, the horizontal grid specification is corrected on input to ``'x_p'`` or ``'x_u'``, ``'y_p'`` or ``'y_v'`` as appropriate.
 
To facilitate use of other models, a list of aliases can be provided under the key 'aliases' to translate variable names. 

In order to facilitate comparisons and products, tools have been coded (efficiently but not very elegantly) to transform data from different points on the C-grid. Thus, second order terms can be computed correctly on required points just by specifying the ouput grid.

A number of derived variables have been implemented that are calculated provided the required inputs are available. These are provided in the :py:mod:`~monc_utils.thermodynamics.thermodynamics` module. 
Examples are:

+-----------+---------------------------------------------------------------+
|'th_L'     | Liquid water potential temperature  :math:`\theta_L`.         |
+-----------+---------------------------------------------------------------+
|'th_v'     | Virtual potential temperature  :math:`\theta_v`.              |
+-----------+---------------------------------------------------------------+
|'th_w'     | Wet bulb potential temperature  :math:`\theta_w`.             |
+-----------+---------------------------------------------------------------+
|'q_total'  | Total water  :math:`q_t`.                                     |
+-----------+---------------------------------------------------------------+
|'buoyancy' |:math:`(g/\overline{\theta_v})*(\theta_v-\overline{\theta_v})`,|
|           |where the mean is the domain mean.                             |
+-----------+---------------------------------------------------------------+


.. todo:: Code to calculate the deformation field and hence shear and vorticity has also been implemented but needs full integration.

.. todo:: The next step is to implement arbitrary derivatives, so one could specify in the variable list, e.g. ``'d_u_d_x_on_w'``. This has been implemented in the trajectory code and will be ported to here for compatibility.
    The :py:mod:`~monc_utils.data_utils.difference_ops` module now has general, grid-aware derivative and averaging functions. 
    These are used internally but the ability to use them in the input variable list has yet to be implemented, apart from some special variables like buoyancy gradient.

===============
Version History
===============

Latest version is 0.1.0

.. topic:: New at 0.1

    #. Complete re-structuring. Extracted from Subfilter repository.


