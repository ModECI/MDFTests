import mdf_to_onnx as m2o

import os
import argparse

def file_path(path):
    if os.path.isfile(path) is False:
        raise argparse.ArgumentTypeError(path+" is an invalid file.")
    return path

def dir_path(path):
    if os.path.isdir(path) is False:
        raise argparse.ArgumentTypeError(path + " is an invalid directory.")
    return path

def get_args():
    parser = argparse.ArgumentParser(description='Create a new dataset of <#bins> windows of <#windowlen> each from a raw datset')
    parser.add_argument('--input', '-i', help='Full path of MDF file', type=file_path, required=True)
    parser.add_argument('--opdir', '-od', help='Full path of destination directory', type=dir_path, required=False)

    cmd_args = parser.parse_args()
    if not cmd_args.opdir:
        cmd_args.opdir = os.path.dirname(cmd_args.input) + os.sep

    return cmd_args.input, cmd_args.opdir


if __name__ == '__main__':
    print('Invoking MDF to ONNX translator')
    input, opdir = get_args()
    m2o.convert_mdf_to_onnx(input, opdir)
