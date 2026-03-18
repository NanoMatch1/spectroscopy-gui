from acquisition_editor import process_directory
import os.path as osp

this_dir = osp.dirname(__file__)
process_directory(osp.join(this_dir, "acquisition_editor", "examples"))
