import numpy as np

class DataObject:

    '''A generic data object to hold parsed data and headers.
    Attributes:
        data (np.ndarray): The numeric data array.
        header (list): The list of header strings.
    '''

    def __init__(self, data: np.ndarray, header: list) -> None:
        self.data = data
        self.header = header
