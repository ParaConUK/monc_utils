==========================
The difference_ops module.
==========================
This module implements averaging and differencing functions for data on a C-grid.

.. topic:: New at 0.2

    #. Specify spatial derivatives of variables at input. Derivatives are split into the :py:func:`~monc_utils.data_utils.difference_ops.d_by_dx_field` and :py:func:`~monc_utils.data_utils.difference_ops.d_by_dx_field_native` (and similarly for y, z) . The latter just perform a centred difference and update the grid specification for that dimension (so, for example, a derivative of a field on ``x_u`` points will be on ``x_p`` points. The former calls the latter then intterpolates to the required output grid.

.. topic:: New at 0.1

    #. Relocated to :py:mod:`monc_utils.data_utils`.


Detailed Module Contents
------------------------
The entire module is documented below.

.. automodule:: monc_utils.data_utils.difference_ops
   :member-order: bysource
   :members:
   :undoc-members:

.. automodule:: monc_utils.data_utils
   :member-order: bysource
   :members:
   :undoc-members:
