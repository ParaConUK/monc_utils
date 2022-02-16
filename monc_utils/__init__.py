import os
import yaml

# System-defined parameters
# Determine computation environment (currently a binary choice)
try:
    # Printing will fail where environmental variable is not defined.
    print(f'Job id: {os.environ["SLURM_JOB_ID"]}\n')
    executing_on_cluster = True
except:
    executing_on_cluster = False



# User configurable global options DEFAULT values
global_config = {
    'write_sleeptime': 3,                            # int, Seconds to wait after writing to flush caches.
    'use_concat': True,                              # [True, False] Whether to concatenate deformation field output.
    'chunk_size': 2**22,                             # int, Maximum horizontal chunk size for deformation calculation, bytes?.
    'no_dask': False,                                # [True, False] Whether to completely disable dask in all routines.
                                                     #   Smaller chunks generally take longer to compute.
    'dask_chunks': {'z':'auto', 'zn':'auto'},        # dict, coordinate-specific rules about how to chunk input data when
                                                     #   reading. For example:
                                                     #       if not subfilter.global_config['no_dask']:
                                                     #          dask.config.set({"array.slicing.split_large_chunks": True})
                                                     #          dask_chunks = subfilter.global_config['dask_chunks']
                                                     #       ...
                                                     #       dataset = xr.open_dataset(infile, chunks=dask_chunks)
                                                     #   -1 loads the dataset with dask using a single chunk for all arrays
                                                     #   {} loads the dataset with dask using engine preferred chunks if exposed
                                                     #      by the backend, otherwise as -1.
                                                     #      This is often faster, but uses more memory.
                                                     #   'auto' will use dask auto chunking.
    'use_map_overlap': True,                         # [True, False] When using dask, whether to use dask's map_overlap to
                                                     #   perform difference operator actions with periodic boundaries.
    'l_slurm_job_tag': False,                        # [True, False] When executing_on_cluster, whether to add the Slurm job
                                                     #   name to output file names.
    'output_precision': "float64",                   # str, ["float64", "float32"]  Presision of output data.
    'test_level': 0,                                 # int, [1, 2] Case ID to select variable lists for testing.
    }


# Configure changes to global_config
def set_global_config(in_opts):
    """
    Configure changes to global_config with user-supplied values

    Parameters
    ----------
    in_opts : str or dict
        If str, path to configuration .yaml file containing a global_config entry set.
        If dict, set of items to update.

    Returns
    -------

    """
    if type(in_opts) is str:
        with open(in_opts) as c:
            update_config = yaml.load(c, Loader = yaml.SafeLoader)
        if 'global_config' in update_config:
            global_config.update(update_config['global_config'])
        else:
            print('[WARN] Attempted set_global_config via yaml, but global_config entry not found.')
    elif type(in_opts) is dict:
        global_config.update(in_opts)
    else:
        raise TypeError(f"The input parameter type {type(in_opts)} is not valid.  \
                          Use either str path to yaml file or dict to change subfilter global_config.")
    _set_consistent_global_config()


# Rules for consistent global_config options
def _set_consistent_global_config():
    if global_config['no_dask']:
        global_config['dask_chunks'] = None          # Input data will be xarray only, not dask.

# Verify consistent configuration options
_set_consistent_global_config()
