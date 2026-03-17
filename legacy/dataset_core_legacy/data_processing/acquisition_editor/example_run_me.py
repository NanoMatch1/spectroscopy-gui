import os.path as osp

from acquisition_editor import process_directory
    
this_dir = osp.dirname(__file__)
data_dir = osp.join(this_dir, "examples")
process_directory(data_dir)
