from unittest import TestCase, mock

import numpy as np

from skcmeans.initialization import initialize_random


@mock.patch('scikit-cmeans.initialization.check_random_state')
class TestInitializeRandom(TestCase):

    def setUp(self):
        self.x = np.arange(64).reshape(16, 4)

    def test_number_of_samples(self, check_random_state):
        check_random_state.return_value.permutation.return_value = np.arange(3)
        initialize_random(self.x, 2, None)
        check_random_state.return_value.permutation.assert_called_with(16)

    def test_return_shape(self, check_random_state):
        check_random_state.return_value.permutation.return_value = np.arange(4)
        selected = initialize_random(self.x, 2, None)
        self.assertEqual(selected[1].shape, (2, 4))
        self.assertEqual(selected[1].shape, (2, 4))



