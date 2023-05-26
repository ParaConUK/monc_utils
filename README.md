# monc_utils
Various utilities for handling data from the MONC large-eddy model.
See https://paraconuk.github.io/monc_utils/ for documentation.

See the [changelog](CHANGELOG.md) for a summary of changes.

Users should pip install to a suitable environment using

    pip install  git+https://github.com/ParaConUK/monc_utils.git

This will install into the standard library.

Developers should fork then clone the repository (please create a branch before making 
any changes!), open a terminal window and activate the python environment 
required, cd to the monc_utils directory and

    pip install -e .

This will install as if into the standard library but using the cloned code 
which can be edited. Please commit code improvements and discuss merging with 
the master branch with Peter Clark and other users.

New in version 0.3.0
#. Corrections to inv_esat and inv_esat_ice
#. Addition of 
	- saturated_wet_bulb_potential_temperature
	- saturated_unsaturated_wet_bulb_potential_temperature

New in version 0.2.0
#. The ability to read in spatial derivatives of variables.