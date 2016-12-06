from setuptools import setup, find_packages

from skcmeans import __version__

setup(
    name='scikit-cmeans',
    version=__version__,
    description="Flexible, extensible fuzzy c-means clustering in python.",
    url="https://bm424.github.io/scikit-cmeans/",
    author="Ben Martineau",
    author_email="bm424@cam.ac.uk",
    license="MIT",
    classifiers=[
        'Development Status :: 4 - Beta',
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