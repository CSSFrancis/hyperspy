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

data_types = {8: np.uint8, 16: np.uint16, 32: np.uint32}  # Stream Pix data types
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
        self.bit_depth = None
        self.image_dtype_list = None

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
            image_info_dtype= [(("ImageWidth"),("<u4")),
                               (("ImageHeight"), ("<u4")),
                               (("ImageBitDepth"), ("<u4")),
                               (("ImageBitDepthReal"), ("<u4"))]
            image_info = np.fromfile(file, image_info_dtype, count=1)[0]
            self.frame_width = image_info[0]
            self.frame_height = image_info[1]
            self.bit_depth = data_types[image_info[2]]  # image bit depth
            self.bit_depth_real = image_info[3]  # actual recorded bit depth
            self.frame_length = self.frame_height*self.frame_width
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
            dtype_list = [(("Array"), self.bit_depth, (self.frame_width, self.frame_height)),
                          (("t_value"),("<u4")),
                          (("Miliseconds"), ("<u2")),
                          (("Microseconds"), ("<u2"))
                          ]
            data = np.empty(self.num_frames, dtype=dtype_list)  # creating an empty array
            file.seek(8192)
            for i in range(self.num_frames):
                file.seek(8192+i*self.img_bytes)
                if self.dark_ref is not None and self.gain_ref is not None:
                    data[i] = (np.fromfile(file, dtype_list, count=1) - self.dark_ref) * self.gain_ref
                else:
                    data[i] = np.fromfile(file, dtype_list, count=1)
        return data["Array"]

    def read_data(self, lazy=False):
        if lazy:
            from dask import delayed
            from dask.array import from_delayed
            val = delayed(self._get_image, pure=True)
            dt =(("Array"), self.bit_depth, (self.frame_width, self.frame_height))
            data = from_delayed(val, shape=self.num_frames, dtype=)
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