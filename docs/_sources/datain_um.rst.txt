==================
The datain module.
==================
This module implements functions to various input tasks for :py:mod:`monc_utils`.

.. topic:: New at 0.4

    #. Provides input of UM idealised fields with functionality of :py:mod:`monc_utils.io.datain`

       - Using :py:func:`~monc_utils.io_um.datain.get_um_field` to read a UM field using its stash code.

       - Using :py:func:`~monc_utils.io_um.datain.get_um_data` to read a UM field using a name similar to MONC. 
	     Currently only a small  mapping from name to STASH code is provided.

       - Using :py:func:`~monc_utils.io_um.datain.get_um_data_on_grid` the result is interpolated to 
         the required grid and the name appended ``_on_grid``, with ``grid`` the requested grid.

       - Nesting is allowed, e.g. ``'dbydy(dbydx(th_L))'``.



Detailed Module Contents
------------------------
The entire module is documented below.

.. automodule:: monc_utils.io_um.datain
   :member-order: bysource
   :members:
   :undoc-members:

.. automodule:: monc_utils.io_um
   :member-order: bysource
   :members:
   :undoc-members:


    
