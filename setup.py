import setuptools
import re

# Version is maintained in the __init__.py file
with open("nmrtoolslab/__init__.py") as f:
    try:
        VERSION = re.findall(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)[0]
    except IndexError:
        raise RuntimeError('Unable to determine version.')

setuptools.setup(
    name="nmrtoolslab",
    version=VERSION,
    author="Cyril Charlier",
    author_email="charlier@insa-toulouse.fr",
    description="Tools for NMR work within the lab",
    long_description="add long description here",
    long_description_content_type="text/markdown",
    url="https://github.com/NMRTeamTBI/nmrToolsLab",
    packages=setuptools.find_packages(),
    python_requires='>=3.5',
    install_requires=['pandas>=0.17.1', 'scipy>=0.12.1', 'nmrglue>=0.6', 'numpy>=1.14.0', 'plotly', 'pybaselines>=1.0.0', 'biopandas'],
    package_data={'': ['data/*.png', ], },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        ]
)
