from setuptools import setup, find_packages

from pyfcm import __version__

setup(
    name='pyfcm',
    version=__version__,
    description="Flexible, extensible fuzzy c-means clustering in python.",
    url="https://bm424.github.io/pyfcm/",
    author="Ben Martineau",
    author_email="bm424@cam.ac.uk",
    license="MIT",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    keywords='clustering cluster fuzzy classifier classifiers',
    packages=find_packages(exclude=['docs', 'tests']),
    install_requires=['numpy', 'scipy', 'scikit-learn']

)