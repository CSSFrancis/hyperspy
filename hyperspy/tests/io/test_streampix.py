import os

from unittest import TestCase
import matplotlib.pyplot as plt
plt.show()
import pytest
import numpy as np

from hyperspy.io import load
from hyperspy.misc.test_utils import assert_warns

MY_PATH = os.path.dirname(__file__)


class TestSEQReader(TestCase):

    def setUp(self):
        self.seq = os.path.join(MY_PATH, 'seq_files')
        self.file0 = os.path.join(self.seq, 'test.seq')
        print(self.file0)

    def test_load_seq_lazy(self):
        s = load(self.file0, lazy=True)
        print("the size is :", np.shape(s.inav[1].data.compute()))
        print(s.axes_manager)
        s.plot()
        plt.ion()
        plt.show(block=False)
        plt.draw()
        plt.show()

    def test_load_seq(self):
        s = load(self.file0, lazy=False)
        print(np.shape(s.data))
        print(np.shape(s.inav[1].data))
        print(s.axes_manager)
        s.plot()
        plt.show()
