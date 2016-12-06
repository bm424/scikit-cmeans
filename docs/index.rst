.. scikit-cmeans documentation master file, created by
   sphinx-quickstart on Wed Oct 26 13:53:02 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to scikit-cmeans's documentation!
=========================================

scikit-cmeans is a (currently rather small) package designed to facilitate
exploration of fuzzy clustering algorithms in a way that is both readily
usable and easily extended. The API is loosely based on that of
`scikit-learn <http://scikit-learn.org/>`_.

.. image:: _images/demo.png
    :align: center
    :alt: Example of a fuzzy cluster result


Many packages already distribute versions of the C-means algorithm, but fuzzy
clustering is a rich field. It is often necessary or desirable to explore
different parameters or methods rapidly, or develop small tweaks to
algorithms without having to rewrite the entire codebase. This project aims
to make that as easy as possible, by providing a simple API promoting
small, decoupled, connectable tools.


Contents
--------

.. toctree::
   :maxdepth: 2

   examples
   skcmeans.algorithms
   skcmeans.initialization


Installation
------------

Installing from GitHub
......................

To install the latest version of scikit-cmeans, you can use ``pip`` to install
from the main GitHub repository.

You must ensure that you have ``pip`` installed. See
`here <https://docs.python.org/3/installing/>`_ for more information.

.. code-block:: bash

    pip install git+git://github.com/bm424/scikit-cmeans.git@master


.. warning::

    Installation has not yet thoroughly tested.


What's in the Box
-----------------

Out-of-the-box, scikit-cmeans provides algorithms for hard clustering,
probabilistic and possibilistic c-means clustering, and a plugin for the
Gustafson-Kessel variant for ellipsoidal clusters. Any of the basic
algorithms can be used with any distance metric available from scipy, or use
a custom distance function. Data of any dimensionality is supported


What's Coming Up
----------------

- Plotting tools
- A couple of new algorithms (for example, the Rousseeuw-Trauwaert-Kaufman
  variant)
- More examples and demonstrations
- New initialization methods and improvements on those existing


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

