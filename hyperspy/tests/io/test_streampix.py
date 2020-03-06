import os

from unittest import TestCase
import matplotlib.pyplot as plt
plt.show()
import pytest
import numpy as np
import numpy.testing as ntest
from hyperspy.io import load
from hyperspy.io_plugins.seq import SeqReader

MY_PATH = os.path.dirname(__file__)


class TestSEQReader(TestCase):

    def setUp(self):
        self.seq = os.path.join(MY_PATH, 'seq_files')
        self.file0 = os.path.join(self.seq, 'test.seq')
        #self.file0 = '/media/hdd/home/data/TEM/MgNiY-2020-01-17/13-50-27.443.seq'
        self.file1 = os.path.join(self.seq, 'test1.seq')
        print(self.file0)
        self.reader0 = SeqReader(file=self.file0)
        self.reader1 = SeqReader(file=self.file1)

    def test_get_dark_ref(self):
        # Testing with a dark and a gain reference
        self.reader0._get_dark_ref()
        assert(self.reader0.dark_ref is not None)
        assert(type(self.reader0.dark_ref[0, 0]) is np.float32)
        # Testing without a gain and a dark reference
        self.reader1._get_dark_ref()
        assert(self.reader1.dark_ref is None)

    def test_get_gain_ref(self):
        # Testing with a gain and a dark reference
        self.reader0._get_gain_ref()
        assert(self.reader0.gain_ref is not None)
        assert (type(self.reader0.gain_ref[0, 0]) is np.float32)
        # Testing without a gain or dark reference
        self.reader1._get_gain_ref()
        assert(self.reader1.gain_ref is None)

    def test_apply_gain_dark(self):
        # With dark and gain reference
        self.reader0.parse_header()
        uncorrected_data = self.reader0.get_image_data()
        self.reader0._get_gain_ref()
        self.reader0._get_dark_ref()
        corrected_data = self.reader0.get_image_data()
        uncorrected_data = np.array(uncorrected_data, dtype=np.float32)
        true_correction = (uncorrected_data - self.reader0.dark_ref)*self.reader0.gain_ref
        true_correction[true_correction < 0] = 0
        true_correction = np.round(true_correction)

        ntest.assert_array_almost_equal(true_correction, corrected_data)

    def test_apply_gain_dark_lazy(self):
        # With dark and gain reference
        self.reader0.parse_header()
        uncorrected_data = self.reader0.get_image_data(lazy=True).compute()
        self.reader0._get_gain_ref()
        self.reader0._get_dark_ref()
        corrected_data = self.reader0.get_image_data(lazy=True).compute()
        uncorrected_data = np.array(uncorrected_data, dtype=np.float32)
        #print(uncorrected_data)
        true_correction = (uncorrected_data - self.reader0.dark_ref)*self.reader0.gain_ref
        true_correction[true_correction < 0] = 0
        true_correction = np.round(true_correction).astype(np.int16)
        for c,cd in zip(true_correction, corrected_data):
            print(sum(cd == c))
        print(corrected_data[-1])
        print(true_correction[-1])
        ntest.assert_array_almost_equal(true_correction, corrected_data)


    def test_parse_metadata(self):
        self.reader0.parse_metadata_file()

    def test_load_seq_lazy(self):
        s = load(self.file0, lazy=True)
        signal_shape = np.shape(s.inav[1].data.compute())
        assert(signal_shape[0] == 128)
        assert (signal_shape[1] == 128)
        s.plot()
        plt.show()

    def test_load_seq(self):
        s = load(self.file0, lazy=False)
        signal_shape = np.shape(s.inav[1].data)
        assert(signal_shape[0]==128)
        assert (signal_shape[1] == 128)
        s.plot()
        plt.show()

