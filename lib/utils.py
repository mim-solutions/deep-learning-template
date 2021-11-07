"""Technical utils and helpers not specific to data science: types, iterables, timers, profilers etc."""
import os

from pathlib import Path
from typing import Union, Sequence, Tuple

# https://github.com/joerick/pyinstrument/
import pyinstrument


PathLike = Union[str, os.PathLike]


def get_root_dir_of_repo() -> Path:
    """Return absolute path to the root directory of the code repository."""
    # parent ot this file is lib/, parent.parent is the root.
    return Path(__file__).parent.parent.resolve()


# for hydra; yaml doesn't support tuples
def to_tuple(seq: Sequence) -> Tuple:
    return tuple(seq)


class TreeProfiler(object):
    """Used as a context manager to profile a block of code:

    with TreeProfiler(show_all=True):
        # your code goes here

    Note that it does not work well with multiprocessing.
    """

    def __init__(self, show_all=False):
        self.profiler = pyinstrument.Profiler()
        self.show_all = show_all  # verbose output of pyinstrument profiler

    def __enter__(self):
        print("WITH TREE_PROFILER:")
        self.profiler.start()

    def __exit__(self, *args):
        self.profiler.stop()
        print(self.profiler.output_text(unicode=True, color=True, show_all=self.show_all))
