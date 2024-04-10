from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='monc_utils',
    url='https://github.com/ReadingClouds/monc_utils',
    author='Peter Clark',
    author_email='p.clark@reading.ac.uk',
    contributors='Todd Jones',
    # Needed to actually package something
    packages=['monc_utils', 
              'monc_utils/io',
              'monc_utils/io_um',
              'monc_utils/thermodynamics',
              'monc_utils/data_utils',
              ],
    # Needed for dependencies
    install_requires=['numpy', 'scipy', 'dask', 'xarray'],
    # *strongly* suggested for sharing
    version='0.4.0',
    # The license can be anything you like
    license='MIT',
    description='python code to improve io and compute useful quantities from MONC output.',
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README.md').read(),
)