from unittest import TestCase

import numpy as np

from pyfcm.algorithms import GustafsonKesselMixin


class TestGustafsonKesselMixin(TestCase):

    def setUp(self):
        self.x = np.arange(64).reshape(16, 4)
        self.clusterer = GustafsonKesselMixin()

    def test_metric(self):
        c = self.x[:2, :]
        self.clusterer.centers = c
        distance = self.clusterer.distances(self.x)
        self.assertEqual(distance.shape, (2, 16))
