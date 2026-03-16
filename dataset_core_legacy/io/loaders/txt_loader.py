'''Module for loading txt files. Currently returns a THz data class. #TODO: make it return Spectrum class, and let DataSet handle THzData conversion.
#TODO: Take intelligent sniffing registry from analysis-spectroscopy'''

import numpy as np
from dataset_core.data_structures.thz import THzData, BaseTHzData
from os import path
from dataset_core.io.loaders.registry import BaseLoader, register_loader

@register_loader
class TXTLoader(BaseLoader):

    extension = '.txt'
    errors = []

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.filename = path.basename(filepath)

    def _simple_load(self):
        '''Reads the entire file as plain text.'''
        with open(self.filepath, 'r') as file:
            data = file.read()
        return data
    
    def _parse_data(self, filestring: list):
        '''Parses a list of strings into numeric data.'''
        numeric_data = []
        headers = []

        rows = filestring.split("\n")

        for row in rows:
            try:
                string_row = row.split(' ')
                numeric_row = [float(value) for value in string_row]
                numeric_data.append(numeric_row)
            except ValueError:
                self.errors.append(f"Could not parse row: {row}")
                headers.append(row)
                continue

        try:
            numeric_data = np.array(numeric_data)
        except Exception as e:
            self.errors.append(f"Could not convert data to numpy array: {e}")
        return numeric_data, headers

    def load(self):
        '''Loads the data using simple load and parse methods.'''
        new_data = []
        raw_data = self._simple_load()
        data, headers = self._parse_data(raw_data)
        new_data.append(BaseTHzData(data=data, headers=headers)) # parse each scan into BaseTHzData object
        return THzData(data=new_data, header=None, filename=self.filename, data_type='txt')
