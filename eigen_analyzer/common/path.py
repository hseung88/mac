import os
import glob
from pathlib import Path
from datetime import datetime
from hydra.utils import to_absolute_path
from common.logging import logger


# --------------------------- #
#        Path util            #
# --------------------------- #


def file_exists(path, fname):
    """ check if there exists a file with the specified name in the given path """
    fpath = Path(path_join(path, fname))
    return fpath.is_file()


def file_name(path):
    # split the path into a pair (head, tail) and returns the tail only
    fname = os.path.basename(path)

    base, ext = fname.rsplit('.', 1)
    return base, ext


def dir_name(path):
    path = dir_path(path)

    return os.path.basename(path)


def dir_path(path):
    path = os.path.dirname(path)

    return path


def path_join(path, *path2):
    if path is None or len(path) == 0:
        logger.error('Invalid path string (1st arg)')

    return os.path.join(path, *path2)


def abs_path(root, rel_path):
    if root is None:
        root = os.getcwd()
    return os.path.normpath(os.path.join(root, rel_path))


def makedirs(dirname, warn_if_exists=False):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    else:
        if warn_if_exists:
            warn_if_not_empty(dirname)


def list_files(path, filter_string='*'):
    filter_string = path_join(path, filter_string)
    files = list(file for file in glob.glob(filter_string))
    return files


def symlink(path_origin, *paths, use_relative_path=True):
    for item in paths:
        if os.path.exists(item):
            os.remove(item)

        if use_relative_path:
            src_path = os.path.relpath(path_origin,
                                       start=os.path.dirname(item))
        else:
            src_path = path_origin
        try:
            os.symlink(src_path, item)
        except FileExistsError:
            os.unlink(item)
            os.symlink(src_path, item)


def hyd_normpath(relpath):
    abs_path = to_absolute_path(relpath)
    return os.path.normpath(abs_path)


def create_working_dir(prefix, exp_name=None, chdir=True):
    if exp_name is None:
        exp_name = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    wdir = path_join(prefix, exp_name)
    makedirs(wdir)
    working_dir = abs_path(None, wdir)

    if chdir:
        os.chdir(wdir)

    return working_dir


def warn_if_not_empty(dirpath):
    if (os.path.exists(dirpath)
        and os.path.isdir(dirpath)
        and len(os.listdir(dirpath)) > 0):
        logger.warning(f"The provided dir is not empty: {dirpath}")
