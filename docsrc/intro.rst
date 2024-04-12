============
Introduction
============
 
The purpose of this set of modules is to implement a set of useful utilities for dealing with output from MONC (and similar data). 

==============      
Variable Names
==============
This package uses xarray - returned variables are xarray DataArrays. 
These have names; this package follows the following naming convention.

    * Variables or derived variables read from the source or reference file using :py:func:`~monc_utils.io.datain.get_data` have the name requested (possibly changed from the internal name in the source data by the aliases dictionary). For example, ``u``. They have grid specifications corrected to the appropriate dimension specification in MONC (for example, ``u`` will be on ``x_u``, ``y_p`` and ``z_p``) and the time dimension renamed ``time``.    
    * Variables or derived variables read from the source or reference file using :py:func:`~monc_utils.io.datain.get_and_transform` behave as above but are then interpolated to the requested grid (``u``, ``v``, ``w`` or ``p,`` or a tuple made up of x, y and z dimension names) using :py:func:`~monc_utils.data_utils.difference_ops.grid_conform`. 
    * Variables or derived variables read from the source or reference file using :py:func:`~monc_utils.io.datain.get_data_on_grid` behave as above but have the grid name appended to the variable name, e.g. ``u_on_w``. If ``options['save_all']`` is set to ``'yes'``, the variable is retrieved from the derived data file if already there and stored to it if not.
    
	
Similar functionality has now been added to allow idealised UM data in NetCDF form to be read to look like MONC data, allowing similar derived variable. See :py:func:`~monc_utils.io_um.datain`.

================
Input Transforms
================

Basic variables (``'u'``, ``'v'``, ``'w'``, ``'th'``, ``'p'``, ``'q_vapour'``, ``'q_cloud_liquid_mass'``, ``'q_cloud_ice_mass'``) are expected to be available in the input file.
If MONC is used, the horizontal grid specification is corrected on input to ``'x_p'`` or ``'x_u'``, ``'y_p'`` or ``'y_v'`` as appropriate.
Similarly, ``'z'`` is changed to ``'z_w'`` and ``'zn'`` is to ``'z_p'``.
 
To facilitate use of other models, a list of aliases can be provided under the key 'aliases' to translate variable names. 

In order to facilitate comparisons and products, tools have been coded (efficiently but not very elegantly) to transform data from different points on the C-grid. Thus, second order terms can be computed correctly on required points just by specifying the ouput grid.

A number of derived variables have been implemented that are calculated provided the required inputs are available. These are provided in the :py:mod:`~monc_utils.thermodynamics.thermodynamics` module. 
Examples are:

+---------------+---------------------------------------------------------------+
|``'th_L'``     | Liquid water potential temperature  :math:`\theta_L`.         |
+---------------+---------------------------------------------------------------+
|``'th_v'``     | Virtual potential temperature  :math:`\theta_v`.              |
+---------------+---------------------------------------------------------------+
|``'th_w'``     | Wet bulb potential temperature  :math:`\theta_w`.             |
+---------------+---------------------------------------------------------------+
|``'q_total'``  | Total water  :math:`q_t`.                                     |
+---------------+---------------------------------------------------------------+
|``'buoyancy'`` |:math:`(g/\overline{\theta_v})*(\theta_v-\overline{\theta_v})`,|
|               |where the mean is the domain mean.                             |
+---------------+---------------------------------------------------------------+

Spatial derivatives can be specified e.g. ``'dbydx(th)'``.
Multiple (nested) derivatives are allowed, e.g. ``'dbydy(dbydx(th))'``.
These are read in using :py:func:`~monc_utils.io.datain.get_data` and computed on a native grid, but may be conformed to a requuired grid as per any other variable using :py:func:`~monc_utils.io.datain.get_data_on_grid`.
The derivatives are calculated using :py:mod:`~monc_utils.data_utils.difference_ops` module that now has general, grid-aware derivative and averaging functions.

.. topic:: New at 0.4.0

	The ``io_um`` package reads data from idealised UM runs NetCDF output in a form similar to MONC, so that all of the input transforms can be used. 
	It adds the ``'ref'`` suffix that reads in just the first time and returns a horizontal average. This is provided to approximate the MONC reference state used especially to compute buoyancy. 
	**This functionality is not available for MONC (and should not be needed)**. 

.. todo:: Code to calculate the deformation field and hence shear and vorticity has also been implemented but needs full integration.

===============
Version History
===============

Latest version is 0.4.0

.. topic:: New at 0.4.0

	#. Added ``io_um`` package.


.. topic:: New at 0.3.0

	#. Corrections to inv_esat and inv_esat_ice
	#. Addition of 
		- saturated_wet_bulb_potential_temperature
		- saturated_unsaturated_wet_bulb_potential_temperature

.. topic:: New at 0.2

    #. Specify spatial derivatives of variables at input.

.. topic:: New at 0.1

    #. Complete re-structuring. Extracted from Subfilter repository.


