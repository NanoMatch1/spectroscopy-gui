'''Module for loading dat files. Returns a dataclass placeholder object.
dat files are text documents containing the full collected data, with all scans unaveraged. The headers are indicated by leading % symbols, data begins after the last header line.'''

import numpy as np
from dataset_core.data_structures.thz import THzData, BaseTHzData
from os import path
from dataset_core.io.loaders.registry import BaseLoader, register_loader

@register_loader
class DATLoader(BaseLoader):

    extension = '.dat'
    errors = []

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.filename = path.basename(filepath)

    def _simple_load(self):
        '''Reads the entire file as plain text.'''
        with open(self.filepath, 'r') as file:
            data = file.read()
        return data
    
    def _parse_data(self, spectrum: list):
        '''Parses a list of strings into numeric data.'''
        numeric_data = []
        for row in spectrum:
            try:
                string_row = row.split(' ')
                numeric_row = [float(value) for value in string_row]
                numeric_data.append(numeric_row)
            except ValueError:
                self.errors.append(f"Could not parse row: {row}")
                continue
        try:
            numeric_data = np.array(numeric_data)
        except Exception as e:
            self.errors.append(f"Could not convert data to numpy array: {e}")
        return numeric_data

    def _simple_split(self, raw_data: str):
        '''Parses the data from simple_load method. Returns a dict with single scan to mimic the acc data structure for THz analysis methods.'''

        header = []
        spectrum = []

        rows = raw_data.split("\n")
        for row in rows:
            row = row.strip()
            if row.startswith('%'):
                header.append(row.strip('%').strip())
            else:
                if row == '': # skip empty lines
                    continue
                spectrum.append(row)

        scan_dict = {'scan_0': {'header': header, 'spectrum': spectrum}}
        return scan_dict

    def load(self):
        '''Loads the data using simple load and parse methods.'''
        new_data = []
        raw_data = self._simple_load()
        parsed_data = self._simple_split(raw_data)
        
        for key, value in parsed_data.items():
            data = self._parse_data(value['spectrum'])
            new_data.append(BaseTHzData(data=data, headers=value['header'])) # parse each scan into BaseTHzData object
        
        return THzData(data=new_data, header=None, filename=self.filename, data_type='dat')
