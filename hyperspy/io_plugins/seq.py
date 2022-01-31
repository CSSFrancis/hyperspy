# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <http://www.gnu.org/licenses/>.

# Plugin characteristics
# ----------------------
format_name = 'seq sequential file'
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
import numpy as np
import struct
import dask.array as da
import xmltodict


_logger = logging.getLogger(__name__)

data_types = {8: np.uint8, 16: np.uint16, 32: np.uint32}  # Stream Pix data types


class SeqReader(object):
    """ Class to read .seq files. File format from StreamPix and Output for Direct Electron Cameras
    """
    def __init__(self,
                 seq=None,
                 dark=None,
                 gain=None,
                 metadata=None,
                 xml_file=None,
                 celeritas=False):
        self.dark_file = dark
        self.gain_file = gain
        self.xml_file = xml_file
        self.metadata_file = metadata
        self.metadata_dict = {}
        self.axes = []
        self.image_dict = {}
        self.dark_ref = None
        self.gain_ref = None
        self.image_dtype = None
        self.celeritas = celeritas
        self.seq = seq

    def _get_xml_file(self):
        if self.xml_file is None:
            _logger.warning("No xml file is given. While the program might still run it is important to have "
                            "and xml file.  At faster FPS the XML file will contain information necessary to"
                             "properly load the images")
            return
        with open(self.xml_file, "r") as file:
            dict = xmltodict.parse(file.read())
            file_info = dict["Configuration"]["FileInfo"]
            for key in file_info:
                file_info[key] = file_info[key]["@Value"]
            self.segment_prebuffer = int(file_info["SegmentPreBuffer"])  # divide frames by this
            self.image_dict["NumFrames"] = int(file_info["TotalFrames"])
            self.xml_metadata = file_info
            return

    def parse_header(self, celeritas=False):
        with open(self.seq, mode='rb') as file:  # b is important -> binary
            file.seek(548)
            image_info_dtype = [("ImageWidth", "<u4"),
                                ("ImageHeight", "<u4"),
                                ("ImageBitDepth", "<u4"),
                                ("ImageBitDepthReal", "<u4")]
            image_info = np.fromfile(file, image_info_dtype, count=1)[0]
            self.image_dict['ImageWidth'] = image_info[0]
            self.image_dict['ImageHeight'] = image_info[1]
            self.image_dict['ImageBitDepth'] = data_types[image_info[2]]  # image bit depth
            self.image_dict["ImageBitDepthReal"] = image_info[3]  # actual recorded bit depth
            self.image_dict["FrameLength"] = image_info[0] * image_info[1]
            _logger.info('Each frame is %i x %i pixels', (image_info[0], image_info[1]))
            file.seek(572)

            read_bytes = file.read(4)
            self.image_dict["NumFrames"] = struct.unpack('<i', read_bytes)[0]
            _logger.info('%i number of frames found', self.image_dict["NumFrames"])
            _logger.info(self.image_dict["NumFrames"])

            file.seek(580)
            read_bytes = file.read(4)
            self.image_dict["ImgBytes"] = struct.unpack('<L', read_bytes[0:4])[0]

            file.seek(584)
            read_bytes = file.read(8)
            self.image_dict["FPS"] = struct.unpack('<d', read_bytes)[0]
            image_size = int((image_info[2]/8 *
                          self.image_dict["ImageWidth"] *
                          self.image_dict["ImageHeight"]))
            if celeritas:
                if self.segment_prebuffer is None:
                    _logger.warning("Trying to guess the segment pre-factor... Please load .xml File to help")
                    # try to guess it?
                    if self.image_dict["ImageWidth"] == 512:
                        self.segment_prebuffer = 16
                    elif self.image_dict["ImageWidth"] == 256:
                        self.segment_prebuffer = 64
                    else:
                        self.segment_prebuffer = 4
                buffer = (image_info[2] - image_size)/self.segment_prebuffer
                self.image_dtype = [("Array", self.image_dict["ImageBitDepth"],
                                     (self.image_dict["ImageWidth"],
                                      self.image_dict["ImageHeight"]/self.segment_prebuffer)),
                                    ("seconds", "<u4"),
                                    ("Milliseconds", "<u2"),
                                    ("Microseconds", "<u2"),
                                    ("Buffer", bytes, buffer - 8)]
            else:
                self.image_dtype = [("Array", self.image_dict["ImageBitDepth"],
                                     (self.image_dict["ImageWidth"], self.image_dict["ImageHeight"])),
                                    ("seconds", "<u4"),
                                    ("Milliseconds", "<u2"),
                                    ("Microseconds", "<u2"),
                                    ("Buffer", bytes, self.image_dict["ImgBytes"]-image_size-8)]
        return

    def parse_metadata_file(self):
        """ This reads the metadata from the .metadata file """
        try:
            with open(self.seq + ".metadata", 'rb') as meta:
                meta.seek(320)
                image_info_dtype = [("SensorGain", np.float64),
                                    ("Magnification", np.float64),
                                    ("PixelSize", np.float64),
                                    ("CameraLength", np.float64),
                                    ("DiffPixelSize", np.float64)]
                m = np.fromfile(meta, image_info_dtype, count=1)[0]
                self.metadata_dict["SensorGain"] = m[0]
                self.metadata_dict["Magnification"] = m[1]
                self.metadata_dict["PixelSize"] = m[2]
                self.metadata_dict["CameraLength"] = m[3]
                self.metadata_dict["DiffPixelSize"] = m[4]

        except FileNotFoundError:
            _logger.info("No metadata file.  The metadata should be in the same directory "
                         "as the image and have the form xxx.seq.metadata")
        return

    def create_axes(self):
        axes = []
        axes.append({'name': 'time', 'offset': 0, 'scale': 1, 'size': self.image_dict["NumFrames"],
                     'navigate': True, 'index_in_array': 0})
        axes[0]['scale'] = 1 / self.image_dict["FPS"]

        axes.append({'name': 'y', 'offset': 0, 'scale': 1, 'size': self.image_dict["ImageHeight"],
                     'navigate': False, 'index_in_array': 1})
        axes.append({'name': 'x', 'offset': 0, 'scale': 1, 'size': self.image_dict["ImageWidth"],
                     'navigate': False, 'index_in_array': 2})

        if self.metadata_dict is not {} and self.metadata_dict["PixelSize"] != 0:
            # need to still determine a way to properly set units and scale
            axes[-2]['scale'] = self.metadata_dict["PixelSize"]
            axes[-1]['scale'] = self.metadata_dict["PixelSize"]
        return axes

    def create_metadata(self):
        metadata = {'General': {'original_filename': os.path.split(self.seq)[1]},
                    'Signal': {'signal_type': 'Signal2D'}}
        if self.metadata_dict is not {}:
            metadata['Acquisition_instrument'] = {'TEM':
                                                  {'camera_length': self.metadata_dict["CameraLength"],
                                                   'magnification': self.metadata_dict["Magnification"]}}
        return metadata

    def get_image_data(self, lazy=False, celeritas=False):
        if celeritas:
            mm = [np.memmap(s, offset=8192, dtype=self.image_dtype, ) for s in self.seq]
            if lazy:
                data = [da.from_array(m["Array"]) for m in mm]
            else:
                data = [m["Array"] for m in mm]
            seconds = mm[0]["seconds"]
            milliseconds = mm[0]["Milliseconds"]
            microseconds = mm[0]["Microseconds"]
            data = np.stack(data, axis=0)
        else:
            mm = np.memmap(self.seq, offset=8192, dtype=self.image_dtype, )
            data = mm["Array"]
            seconds = mm["seconds"]
            milliseconds = mm["Milliseconds"]
            microseconds = mm["Microseconds"]
        time = [seconds, milliseconds, microseconds]
        max_pix = 2 ** self.image_dict["ImageBitDepthReal"]
        if lazy:
            data = da.from_array(data)
        if self.dark_ref is not None:
            data = data - self.dark_ref
            data[data > max_pix] = 0
        if self.gain_ref is not None:
            data = data * self.gain_ref
        return data, time


def file_reader(filename,
                dark=None,
                gain=None,
                metadata=None,
                xml_file=None,
                lazy=False,
                celeritas=False
                ):
    """Reads a .seq file.
    Parameters
    ----------
    filename: str
        The filename to be loaded
    lazy : bool, default False
        Load the signal lazily.
    nav_shape: tuple
        The shape of the navigation axis to be loaded
    chunk_shape: tuple
        The shape for each chunk to be loaded. This has some performance implications when dealing with
        the data later...
    """
    seq = SeqReader(filename,
                    dark,
                    gain,
                    metadata,
                    xml_file)
    if celeritas:
        seq._get_xml_file()
    seq.parse_header()
    seq.dark_ref = read_ref(dark,
                            height=seq.image_dict["ImageHeight"],
                            width=seq.image_dict["ImageWidth"])
    seq.gain_ref = read_ref(gain,
                            height=seq.image_dict["ImageHeight"],
                            width=seq.image_dict["ImageWidth"])
    seq.parse_metadata_file()
    axes = seq.create_axes()
    metadata = seq.create_metadata()
    data, time = seq.get_image_data(lazy=lazy,)
    metadata["General"]["Timestamps"] = time
    dictionary = {
        'data': data,
        'metadata': metadata,
        'axes': axes,
        'original_metadata': metadata,
    }
    return [dictionary, ]


def read_ref(file_name,
             height,
             width,
             ):
    """Reads a reference image from the file using the file name as well as the width and height of the image. """
    if file_name is None:
        return
    try:
        with open(file_name, mode='rb') as file:
            file.seek(0)
            read_bytes = file.read(8)
            frame_width = struct.unpack('<i', read_bytes[0:4])[0]
            frame_height = struct.unpack('<i', read_bytes[4:8])[0]
            file.seek(256 * 4)
            bytes = file.read(frame_width * frame_height * 4)
            dark_ref = np.reshape(np.frombuffer(bytes, dtype=np.float32), (height, width))
        return dark_ref
    except FileNotFoundError:
        _logger.warning("No Dark Reference image found.  The Dark reference should be in the same directory "
                        "as the image and have the form xxx.seq.dark.mrc")
        return