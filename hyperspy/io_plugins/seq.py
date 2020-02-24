# Plugin characteristics
# ----------------------
format_name = 'seq seqential file'
description = """The file format used by StreamPix
and an output for Direct Electron detectors"""
full_support = False
# Recognised file extension
file_extensions = ('seq')
default_extension = 0
# Reading capabilities
reads_images = True
reads_spectrum = False
reads_spectrum_image = True
# Writing capabilities
writes = False

import os
import logging
import dateutil.parser

import numpy as np
import traits.api as t

import hyperspy.misc.io.utils_readfile as iou
from hyperspy.exceptions import DM3TagIDError, DM3DataTypeError, DM3TagTypeError
import hyperspy.misc.io.tools
from hyperspy.misc.utils import DictionaryTreeBrowser
from hyperspy.docstrings.signal import OPTIMIZE_ARG
import struct


_logger = logging.getLogger(__name__)


class SeqReader(object):
    """ Class to read .seq files. File format from StreamPix and Output for Direct Electron Cameras
    """

    def __init__(self, f):
        self.f = f
        self.metadata_dict = None
        self.frame_width = None
        self.frame_height = None
        self.frame_length =None
        self.num_frames = None
        self.img_bytes = None
        self.dark_ref = None
        self.gain_ref = None
        self.fps = None

    def _get_dark_ref(self):
        try:
            with open (self.f+".dark.mrc") as dark_ref:
                bytes = dark_ref.read(self.frame_width * self.frame_height * 4)
                self.dark_ref = np.frombuffer(bytes, dtype=np.float32)
        except FileNotFoundError:
            print("No Dark Reference image found.  The Dark reference should be in the same directory "
                  "as the image and have the form testfile.seq.dark.mrc")

    def _get_gain_ref(self):
        try:
            with open (self.f+".gain.mrc") as dark_ref:
                bytes = dark_ref.read(self.frame_width * self.frame_height * 4)
                self.dark_ref = np.frombuffer(bytes, dtype=np.float32)  # 32 bit float image gain and dark
                # Images kept in linear form.
        except FileNotFoundError:
            print("No gain Reference image found.  The Dark reference should be in the same directory "
                  "as the image and have the form testfile.seq.dark.mrc")

    def parse_header(self):
        print(self.f)
        with open(self.f, mode='rb') as file:  # b is important -> binary
            file.seek(548)
            read_bytes = file.read(20)
            self.frame_width = struct.unpack('<L', read_bytes[0:4])[0]
            self.frame_height = struct.unpack('<L', read_bytes[4:8])[0]
            self.frame_length=self.frame_height*self.frame_width
            _logger.info('Each frame is %i x %i pixels', (self.frame_width, self.frame_width))
            file.seek(572)

            read_bytes = file.read(4)
            self.num_frames = struct.unpack('<i', read_bytes)[0]
            _logger.info('%i number of frames found', self.num_frames)

            file.seek(580)
            read_bytes = file.read(4)
            self.img_bytes = struct.unpack('<L', read_bytes[0:4])[0]

            file.seek(584)
            read_bytes = file.read(8)
            self.fps = struct.unpack('<d', read_bytes)[0]
            _logger.info('Image acquired at %i frames per second', self.fps)
        return

    def parse_metadata(self):
        metadata = {'General': {'original_filename': os.path.split(self.f)[1],},
                    "Signal": {'signal_type': "Signal2D"}, }
        try:
            with open(self.f+".metadata") as meta:
                pass
        except FileNotFoundError:
            print("No gain Reference image found.  The Dark reference should be in the same directory "
                  "as the image and have the form testfile.seq.dark.mrc")
        return metadata

    def parse_axes(self):
        axes = []
        axes.append({
            'name':'kx',
            'offset': 0,
            'scale': 1,
            'size': self.frame_width,
            'navigate': False,
            'index_in_array': 0})
        axes.append({
            'name': 'ky',
            'offset': 0,
            'scale': 1,
            'size': self.frame_height,
            'navigate': False,
            'index_in_array': 1})
        axes.append({
            'name': 'x',
            'offset': 0,
            'scale': 1,
            'size': self.num_frames,
            'navigate': True,
            'index_in_array': 2})
        try:
            with open(self.f+".metadata") as meta:
                pass
        except FileNotFoundError:
            print("No Gain Reference image found.  The Gain reference should be in the same directory "
                  "as the image and have the form testfile.seq.gain.mrc")

        return axes

    def _get_image(self):
        with open(self.f, mode='rb') as file:
            data = np.empty(self.num_frames*self.frame_length, dtype=np.uint16)  # creating an empty array
            file.seek(8192)
            for i in range(self.num_frames):
                file.seek(8192+i*self.img_bytes)
                if self.dark_ref is not None and self.gain_ref is not None:
                    data[i] = (np.fromfile(file, np.uint16, count=1) - self.dark_ref) * self.gain_ref
                else:
                    data[i*self.frame_length:(i+1)*self.frame_length] = np.fromfile(file,
                                                                                    np.uint16,
                                                                                    count=self.frame_length)
            data = data.reshape((self.frame_width,self.frame_height, self.num_frames))
        return data

    def read_data(self, lazy=False):
        if lazy:
            from dask import delayed
            from dask.array import from_delayed
            val = delayed(self._get_image, pure=True)
            data = from_delayed(val, shape=(self.frame_width, self.frame_height,self.num_frames), dtype=np.uint16)
        else:
            data = self._get_image()
        return data


def file_reader(filename, lazy=False):
    """Reads a .seq file.

    Parameters
    ----------
    filename: str
        The filename to be loaded
    lazy : bool, default False
        Load the signal lazily.
    """
    seq = SeqReader(filename)
    seq.parse_header()
    seq._get_dark_ref()
    seq._get_gain_ref()
    metadata = seq.parse_metadata()
    axes = seq.parse_axes()
    data = seq.read_data(lazy=lazy)
    dictionary = {
        'data': data,
        'metadata': metadata,
        'axes': axes,
        'original_metadata': metadata,}

    return [dictionary, ]
    file_reader.__doc__ %= (OPTIMIZE_ARG.replace('False', 'True'))