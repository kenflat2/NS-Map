from setuptools import setup, find_packages

VERSION = '1.3'
DESCRIPTION = 'Implementation of NS-Map'
LONG_DESCRIPTION = 'Implementation of NS-Map, a nonstationary extension of the ecological forecasting method S-Map.'

setup(
    name = 'NSMap',
    version = VERSION,
    author = 'Kenneth Gee',
    author_email = 'kenflat2@gmail.com',
    description = DESCRIPTION,
    long_description = LONG_DESCRIPTION,
    packages = find_packages(),
    install_requires = ['numpy', 'scipy'],
    keywords = ['Ecology', 'Forecasting', 'EDM', 'Empirical Dynamical Modeling', 'Nonstationarity'],
    classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)