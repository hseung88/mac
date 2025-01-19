import math
import os
from datetime import datetime
from hydra.utils import to_absolute_path
#from common.logging import logger


# --------------------------- #
#        Path util            #
# --------------------------- #
unit_list = list(zip(["bytes", "kB", "MB", "GB", "TB", "PB"],
                     [0, 0, 1, 2, 2, 2]))


def file_name(path):
    # split the path into a pair (head, tail) and returns the tail only
    fname = os.path.basename(path)

    base, ext = fname.rsplit('.', 1)
    return base, ext


def dir_name(path):
    path = dir_path(path)

    return os.path.basename(path)


def dir_path(path):
    if os.path.isfile(path):
        path = os.path.dirname(path)
    else:
        path = os.path.normpath(path)

    return path


#def path_join(path, *path2):
#    if path is None or len(path) == 0:
#        logger.error('Invalid path string (1st arg)')
#
#    return os.path.join(path, *path2)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


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


def fsize_format(num):
    """
    Human readable file size.
    copied from
    http://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
    """
    if num == 0:
        return "0 bytes"
    if num == 1:
        return "1 byte"

    exponent = min(int(math.log(num, 1024)), len(unit_list) - 1)
    quotient = float(num) / 1024 ** exponent
    unit, num_decimals = unit_list[exponent]
    format_string = "{:.%sf} {}" % num_decimals
    return format_string.format(quotient, unit)


def create_working_dir(prefix, chdir=True):
    now = datetime.now()
    str_datetime = now.strftime('%Y-%m-%d-%H-%M-%S')

    wdir = path_join(prefix, str_datetime)
    makedirs(wdir)

    if chdir:
        os.chdir(wdir)

    return wdir
