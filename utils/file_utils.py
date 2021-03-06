import os, json
import numpy as np

def write_data_file(filename, filepath, data):
    create_sub_directories(filepath)
    if not os.listdir(filepath):
        with open(f'{filepath}/{filename}.csv', 'w') as f:
            np.savetxt(f, data, fmt="%s", delimiter=',')
    else:
        with open(f'{filepath}/{filename}.csv', 'a') as f:
            np.savetxt(f, data, fmt="%s", delimiter=',')

def write_meta_data_file(filepath, meta_data):
    """writes a meta data file as json to the chosen file path

    Args:
        filepath (str): relative file path from the working directory
        meta_data (dict): meta data
    """
    create_sub_directories(filepath)
    with open(f'{filepath}/meta.json', 'w') as f:
        f.write(json.dumps(meta_data, indent=4))

def create_sub_directories(path):
    """checks and creates all non-existing sub directories of a given path

    Args:
        name (str): the path to create directories for (should be entered as linux '/' notation)
    """
    directories = path.split('/')
    current_path = ''
    for directory in directories:
        current_path = current_path.__add__(directory).__add__('/')
        if directory == '..':
            continue
        if not os.path.isdir(current_path):
            os.mkdir(current_path)
