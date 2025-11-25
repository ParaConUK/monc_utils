"""
monc_utils module.

Created on Wed Feb 16 17:58:34 2022

@author: Todd Jones and Peter Clark
"""
import yaml

def monc_utils_options(config_file:str=None):
    """
    Set default options.

    Parameters
    ----------
    config_file : str
        Path to configuration .yaml file. The default is None.

    Returns
    -------
    options : dict
        Options including optional updates
    update_config : dict
        Updates from config_file

    """
    update_config = None
    options = {
                'override': True,      # Overwrite output file if it exists.
                'input_file': None,    # For user convenience, not required
                'ref_file': None,      # For user convenience, not required
                'outpath': None,       # For user convenience, not required
              }
    if config_file is not None:
        with open(config_file) as c:
            update_config = yaml.load(c, Loader = yaml.SafeLoader)

        options.update(update_config['options'])

    return options, update_config
