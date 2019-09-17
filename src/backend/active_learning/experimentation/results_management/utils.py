import shutil
from typing import Set


def do_nothing(*args, **kwargs):
    return


def sets_differ(a: Set, b):
    return len(a.symmetric_difference(b)) > 0


def delete_path(path):
    shutil.rmtree(path)
    return
