# TODO break into file now that we have the utils submodule (also update __init__)

import contextlib
import copy
import importlib
import logging

# import lz4.frame
import lzma  # lzma is slow but reallllly gets that file size down...
import os
import pickle
import threading

import numpy as np
import pandas as pd
import tqdm


# https://stackoverflow.com/questions/38543506/change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo=None):
        return dotdict({key: copy.deepcopy(value) for key, value in self.items()})


def map_np_array(a, d):
    n = np.ndarray(a.shape)
    for k in d:
        n[a == k] = d[k]
    return n


def estimate_IFR(age):
    # from best fit in https://www.medrxiv.org/content/10.1101/2020.07.23.20160895v4.full.pdf
    # std err is 0.17 on the const and 0.003 on the linear term
    return np.exp(-7.56 + 0.121 * age) / 100.0


def bin_age_csv(filename, out_filename):
    df = pd.read_csv(filename, header=None, names=["fips", "age", "N"])
    pop_weighted_IFR = df.N.to_numpy() * estimate_IFR(df.age.to_numpy())
    df = df.assign(IFR=pop_weighted_IFR)
    df["age_group"] = pd.cut(
        df["age"], np.append(np.arange(0, 76, 5), 120), right=False
    )
    df = df.groupby(["fips", "age_group"]).sum()[["N", "IFR"]].unstack("age_group")

    df = df.assign(IFR=df.IFR / df.N)

    df.to_csv(out_filename)


def date_to_t_int(dates, start_date):
    return np.array([(date - start_date).days for date in dates], dtype=int)


def _cache_files(fname_list, cache_name):
    tmp = {f: open(f, "rb").read() for f in fname_list}
    with lzma.open("run_cache/" + cache_name + ".p.xz", "wb") as f:
        pickle.dump(tmp, f)


def cache_files(*argv):
    thread = threading.Thread(target=_cache_files, args=argv)
    thread.start()


def unpack_cache(cache_file):
    with lzma.open(cache_file, "rb") as f:
        tmp = pickle.load(f)

    # os.mkdir(cache_file[:-5])
    for fname in tmp:
        new_file = cache_file[:-5] + "/" + fname
        os.makedirs(os.path.dirname(new_file), exist_ok=True)
        with open(new_file, "wb") as f:
            f.write(tmp[fname])


def _banner():
    print(r" ____             _          ")
    print(r"| __ ) _   _  ___| | ___   _ ")
    print(r"|  _ \| | | |/ __| |/ / | | |")
    print(r"| |_) | |_| | (__|   <| |_| |")
    print(r"|____/ \__,_|\___|_|\_\\__, |")
    print(r"                       |___/ ")
    print(r"                             ")
