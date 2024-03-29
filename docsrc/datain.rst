==================
The datain module.
==================
This module implements functions to various input tasks for :py:mod:`monc_utils`.

.. topic:: New at 0.2

    #. The ability to read in spatial derivatives of variables. The syntax is e.g. ``'dbydx(th_L)'``. 
       This will read in ``th_L``, itself a derived variable, and produce an ``x`` derivative.

       - Using :py:func:`~monc_utils.io.datain.get_data` the derivative on it's native grid 
         (i.e. a centred difference) is returned, with the name requested.

       - Using :py:func:`~monc_utils.io.datain.get_data_on_grid` the result is interpolated to 
         the required grid and the name appended ``_on_grid``, with ``grid`` the requested grid.

       - Nesting is allowed, e.g. ``'dbydy(dbydx(th_L))'``.

.. topic:: New at 0.1

    #. Relocated to :py:mod:`monc_utils.io`.



Detailed Module Contents
------------------------
The entire module is documented below.

.. automodule:: monc_utils.io.datain
   :member-order: bysource
   :members:
   :undoc-members:

.. automodule:: monc_utils.io
   :member-order: bysource
   :members:
   :undoc-members:


    
