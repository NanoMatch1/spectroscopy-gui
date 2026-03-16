'''Module for loading acc files. Returns a dataclass placeholder object.
acc files are text documents containing the full collected data, with all scans unaveraged. The headers are indicated by leading % symbols, and each spectrum/collection is demarcated by %%.'''

import numpy as np
from dataset_core.data_structures.thz import THzData, BaseTHzData
from os import path
from dataset_core.io.loaders.registry import BaseLoader, register_loader

@register_loader
class ACCLoader(BaseLoader):

    extension = '.acc'
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
        '''Parses the data from simple_load method.'''
        scans = raw_data.split("%%") # split by scans
        scan_dict = {}

        for index, item in enumerate(scans):
            header = []
            spectrum = []
            rows = item.split("\n")
            for row in rows:
                row = row.strip()
                if row.startswith('%'):
                    header.append(row.strip('%').strip())
                else:
                    if row == '': # skip empty lines
                        continue
                    spectrum.append(row)

            scan_dict[f'scan_{index}'] = {'header': header, 'spectrum': spectrum}
        return scan_dict

    def load(self):
        '''Loads the data using simple load and parse methods.'''
        new_data = []
        raw_data = self._simple_load()
        parsed_data = self._simple_split(raw_data)
        for key, value in parsed_data.items():
            data = self._parse_data(value['spectrum'])
            new_data.append(BaseTHzData(data=data, headers=value['header'])) # parse each scan into BaseTHzData object
        
        return THzData(data=new_data, header=None, filename=self.filename, data_type='acc')
