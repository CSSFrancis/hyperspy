import os

from unittest import TestCase
import matplotlib.pyplot as plt
import pytest
import numpy as np

from hyperspy.io import load
from hyperspy.misc.test_utils import assert_warns

MY_PATH = os.path.dirname(__file__)


class TestSEQReader2(TestCase):

    def setUp(self):
        self.seq = os.path.join(MY_PATH, 'seq_files')
        self.file0 = os.path.join(self.seq, 'test.seq')
        print(self.file0)

    def test_load_seq_lazy(self):
        s = load(self.file0, lazy=True)
        print(s.data)

    def test_load_seq(self):
        s = load(self.file0, lazy=False)
        s.plot()
        print(np.shape(s.data))
        print(s.inav[1].data)
        plt.show(block=True)
