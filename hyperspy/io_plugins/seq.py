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

    _complex_type = (15, 18, 20)
    simple_type = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

    def __init__(self, f):
        self.f = f
        self.metadata_dict = None
        self.frame_width = None
        self.frame_height = None
        self.num_frames = None
        self.img_bytes = None
        self.dark_ref = None
        self.gain_ref = None
        self.fps = None

    def _get_dark_ref(self):
        try:
            with open (self.f+".dark.mrc") as dark_ref:
                bytes = dark_ref.read(self.frame_width * self.frame_height * 4)
                self.dark_ref = np.reshape(np.frombuffer(bytes, dtype=np.float32),
                                           (self.frame_width,self.frame_height))
        except FileNotFoundError:
            print("No Dark Reference image found.  The Dark reference should be in the same directory "
                  "as the image and have the form testfile.seq.dark.mrc")

    def _get_gain_ref(self):
        try:
            with open (self.f+".gain.mrc") as dark_ref:
                bytes = dark_ref.read(self.frame_width * self.frame_height * 4)
                self.dark_ref = np.reshape(np.frombuffer(bytes, dtype=np.float32),
                                           (self.frame_width, self.frame_height))  # 32 bit float image gain and dark
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
            'index_in_array': -1})
        try:
            with open(self.f+".metadata") as meta:
                pass
        except FileNotFoundError:
            print("No gain Reference image found.  The Dark reference should be in the same directory "
                  "as the image and have the form testfile.seq.dark.mrc")

        return axes

    def _get_image(self, start):
        with open(self.f, mode='rb') as file:
            file.seek(start)
            read_bytes = file.read(self.frame_width*self.frame_height*2)
            # loading from buffer
            frame = np.reshape(np.frombuffer(read_bytes, dtype=np.uint16), (self.frame_width, self.frame_height))
            if self.dark_ref is not None and self.gain_ref is not None:
                frame = (frame - self.dark_ref) * self.gain_ref
            frame.astype(dtype=np.int32)  # This should probably happen before gain and darkref are applied
            return frame

    def read_data(self, lazy=False):
        if lazy:
            from dask import delayed
            from dask.array import from_delayed, stack
            img_list = [from_delayed(delayed(self._get_image)(start=i*self.img_bytes + 8192), # Need to determine proper start
                                     shape=(self.frame_width, self.frame_height),
                                     dtype=np.uint32)
                        for i in range(self.num_frames)]
            d = stack(img_list, axis=-1)  # adding navigation axis
            return d


def file_reader(filename, record_by=None, order=None, lazy=False,
                optimize=True):
    """Reads a DM3 file and loads the data into the appropriate class.
    data_id can be specified to load a given image within a DM3 file that
    contains more than one dataset.

    Parameters
    ----------
    record_by: Str
        One of: SI, Signal2D
    order : Str
        One of 'C' or 'F'
    lazy : bool, default False
        Load the signal lazily.
    %s
    """
    seq = SeqReader(filename)
    seq.parse_header()
    seq._get_dark_ref()
    seq._get_gain_ref()
    metadata = seq.parse_metadata()
    axes = seq.parse_axes()
    data = seq.read_data(lazy=lazy)
    print(data)
    dictionary = {
        'data': data,
        'metadata': metadata,
        'axes': axes,
        'original_metadata': metadata,}

    return [dictionary, ]
    return
    file_reader.__doc__ %= (OPTIMIZE_ARG.replace('False', 'True'))