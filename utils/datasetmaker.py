
from collections import namedtuple
import os
import time
import pickle
import random

import natsort
import multiprocessing
import numpy as np

import m8r

num_cores = multiprocessing.cpu_count()


def print_ascii(txt):
    print(txt)
    
    
def run_process(p):
    os.system(p)
    
    
def from_rsf(file, return_dict=False):
    rsf = sf.Input(file)
    ndim = len(rsf.shape())
    if return_dict:
        d = dict()
        for i in range(ndim):
            for j in ['n', 'd', 'o']:
                key = f'{j}{i+1}'
                val = rsf.float(key)
                print(f"\tdict['{key}'] <-- {key} <-- {val}")
                d[key] = val
    else:
        d = [rsf.float(f"d{i+1}") for i in range(ndim)]
    n = [rsf.int(f"n{i+1}") for i in range(ndim)]
    a = np.zeros(n[::-1], dtype=np.float32)
    rsf.read(a)
    # a = from_binary(file, dtype=np.float32)[-np.prod(n):].reshape(n).transpose()
    data = np.swapaxes(a, 0, -1)
    return data, d
    
    
def print_fn_signature(_what, _from, _to):
    """ Prints function routing info """
    print(f'###\n{_what}:\n\tfrom:\t{_from}\n\tto:\t\t{_to}')


def print_list_signature(lst, header='List'):
    """ Prints first and last elemenst of a list """
    if lst:
        print(f'{header}, {len(lst)} items:\t{lst[0]}\t...\t{lst[-1]}')

        
def cmd(command, disable=False):
    # print(f'\t{command}')
    if not disable:
        process = subprocess.call(command, shell=True)
    return 0


def run_process(p, verbose=0):
    if verbose:
        print(p)
    os.system(p)


def get_filenames(root_data, keys='.', filenames_only=False, exclude_keys='@', verbose=1):
    """ Fetches filenames from a directory which contain one of the keys in them

    Args:
        root_data (str): path to folder with target files
        keys (str, list): string keys to pick a file from directory
        filenames_only (bool): return short filenames if True, full path otherwise
        exclude_keys (str, list): string keys to exclude a file

    Returns:
        List of strings with file names
    """
    def to_list(a):
        return a if isinstance(a, (list, tuple)) else [a]
    filenames = []
    keys = to_list(keys)
    exclude_keys = to_list(exclude_keys)

    for f in os.listdir(root_data):
        for k in keys:
            if k in f:
                for k_ in exclude_keys:
                    if k_ not in f:
                        if filenames_only:
                            filenames.append(f)
                        else:
                            filenames.append(os.path.join(root_data, f))
    filenames = natsorted(filenames)
    if verbose:
        print_list_signature(filenames, 'Files')
    return filenames

    
def split_rsf(root_data='./', root_destination='./', delete_old=False, key='.rsf', return_cmd=False, verbose=1):
    """ Splits all .rsf files in a folder along 3rd dimension into
    individual .rsf files. Runs command line under the hood.

    Args:
        root_data (str): path to folder with target files
        root_destination (str): path to folder where to store ouputs
        delete_old (bool): delete source files if True, keep them otherwise
        key (str): file extention to apply the transform
    """
    if '.rsf' in root_data:
        key = '.rsf'
    elif '.hh' in root_data:
        key = '.hh'

    os.makedirs(root_destination, exist_ok=True)
    cwd = os.getcwd()
    if not key in root_data:
        filenames = get_filenames(root_data, keys=key, filenames_only=True, verbose=verbose)
    else:
        root_data, filenames = os.path.split(root_data)
        filenames = [filenames]
    if verbose:
        print_fn_signature(f'Split {key}', root_data, root_destination)
    commands = []
    to_be_deleted = []
    os.chdir(root_data)
    for file in filenames:
        source = file
        dat = rdr.from_rsf(source)[0]
        if len(dat.shape) == 3:
            h, w, c = dat.shape
            if c > 1:
                to_be_deleted.append(file)
                for ic in range(c):
                    new_fname = file.replace(key, '') + f'_{ic}{key}'
                    target = new_fname.replace('.rsf', '.hh')
                    command = f'< {source} sfwindow f3={ic} n3=1 --out=stdout > {target}'

                    _from = os.path.join(root_data, target)
                    _to = root_destination
                    if _to != os.path.split(_from)[0]:
                        command += f' && cd {cwd} && ' \
                                   f'mv {_from} {_to} && ' \
                                   f'cd {root_data}'

                    commands.append(command)

    if len(commands) >= int(3 * num_cores / 4):
        print('Start parallel pool...')
        with multiprocessing.Pool(processes=num_cores) as pool:
            pool.map(run_process, commands)
    else:
        for command in commands:
            run_process(command)

    if delete_old:
        for file in to_be_deleted:
            cmd(f'sfrm {file}')

    os.chdir(cwd)

def split_hh(root_data='./', root_destination='./', delete_old=False, key='.hh', verbose=1):
    split_rsf(root_data=root_data, root_destination=root_destination, delete_old=delete_old, key=key, verbose=verbose)
    
    
class DatasetMaker(object):
    def __init__(self, par, verbose=0):

        if isinstance(par, dict):
            MyTuple = namedtuple('MyTuple', sorted(par))
            dct = MyTuple(**par)
        else:
            dct = par.syn

        self.format = dct.format
        self.root_to = dct.dir_dataset
        self.root_from = dct.dir_syn
        self.jump_files = dct.jump_files
        self.jump_folders = dct.jump_folders
        self.pre = dct.pre
        self.post = dct.post
        self.split_from_files = dct.from_files
        self.delete_old = dct.delete_old
        self.verbose = verbose
        self.partitions = dct.partitions

        self.split_hh = dct.split_hh
        self.file_keys = dct.file_keys if isinstance(dct.file_keys, list) else [dct.file_keys]
        self.rules = dct.rules
        self._rule_template = '< #INP #PRE #CMD #POST out=stdout > #OUT'
        self.all_commands = []
        self.move_commands = []
        self.created_files = {m: [] for m in self.rules.keys()}
        self.processed_files = []
        self.delete_commands = []

    def reset(self):
        self.processed_files = []
        self.delete_commands = []
        self.all_commands = []
        self.move_commands = []
        self.created_files = {m: [] for m in self.rules.keys()}

    def print_created(self):
        if self.verbose:
            for k, v in self.created_files.items():
                print(k)
                for f in v:
                    print(f'\t{f}')

    @property
    def root_train(self):
        return os.path.join(self.root_to, 'train/')

    def from_files(self, path_to_files, tag=''):
        """ Apply selected operations to all files in the directory """
        if '.hh' in path_to_files or '.rsf' in path_to_files or '.su' in path_to_files:
            # If file specified then unpack it directly to destination
            path_to_dest = self.root_to
            filemode = True
            # delete_old = False
        else:
            # If folder, then unpack to the same folder
            path_to_dest = path_to_files
            filemode = False
            # delete_old = True

        delete_old = self.delete_old

        if self.split_hh:
            split_hh(path_to_files, path_to_dest, delete_old=delete_old, verbose=self.verbose)
        try:
            subfiles = [f for f in os.listdir(path_to_dest) if all(k in f for k in self.file_keys)]
            if self.verbose:
                print(f'There are {len(subfiles)} files with {self.file_keys} in {path_to_dest}. Delete old={delete_old}!')

            subfiles = subfiles[::self.jump_files]
            for f in subfiles:
                file = os.path.join(path_to_dest, f)
                self.process_file(file, self.root_train, tag=tag)
                # if filemode:
                #     self.delete_commands.append(f'rm {file}')
                self.processed_files.append(file)
        except NotADirectoryError:
            print(f'Not found {path_to_dest}! Ignore.')

    def from_folders(self, path_to_folders, extra_levels=['']):
        folders = natsort.natsorted(os.listdir(path_to_folders))
        folders = [f for f in folders if 'fig' not in f]
        if self.verbose:
            print(f'From {len(folders)} folders in {path_to_folders}')
        for m in folders[::self.jump_folders]:
            root_su = os.path.join(path_to_folders, m)
            for extra in extra_levels:
                root_su = os.path.join(root_su, extra)
            self.from_files(root_su, tag=m)

    def process_file(self, file, destination, tag=''):
        filename = tag + '_' + file.split('/')[-1].replace('.su', '')
        pre = self.pre
        post = self.post

        if '.hh' not in filename:
            filename += '.hh'

        for mode, rule in self.rules.items():
            save_folder = os.path.join(destination, mode)
            new_file = os.path.join(save_folder, filename)
            os.makedirs(save_folder, exist_ok=True)

            command = self._rule_template.replace('#INP', file).replace('#PRE', pre).replace('#CMD', rule).replace(
                '#POST', post).replace('#OUT', new_file)
            self.all_commands.append(command)
            self.created_files[mode].append(new_file)

    def save_list_of_created_files(self, where):
        with open(os.path.join(where, 'created_files.pickle'), 'wb') as handle:
            pickle.dump(self.created_files, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def read_list_of_created_files(self, from_where):
        with open(os.path.join(from_where, 'created_files.pickle'), 'rb') as handle:
            self.created_files = pickle.load(handle)

    def move_list_to(self, list_of_what, where):
        os.makedirs(where, exist_ok=True)
        for what in list_of_what:
            command = f'mv {what} {where}'
            self.move_commands.append(command)

    def split_created_files(self):
        n = len(self.created_files[next(iter(self.rules.keys()))])
        partitions = [int(np.ceil(n * p / 100)) for p in self.partitions]
        partitions[-1] = n - np.sum(partitions[:-1])
        if self.verbose:
            print(partitions)
        assert np.sum(partitions) == n, f'Sums dont match, got {np.sum(partitions)}, need {n}'
   
        random.seed(0)
        shuffled_indices = list(np.arange(n))
        random.shuffle(shuffled_indices)
        for mode in self.rules.keys():
            shuffled = [self.created_files[mode][i] for i in shuffled_indices]
            self.move_list_to(shuffled[-partitions[-1]:], os.path.join(self.root_to, 'test', mode))
            shuffled = shuffled[:len(shuffled) - partitions[-1]]
            self.move_list_to(shuffled[-partitions[-2]:], os.path.join(self.root_to, 'val', mode))


    def run(self, **kwargs):
        print_ascii('SynSplit')
        if len(kwargs) > 0:
            for k, v in kwargs.items():
                print(f'Override self.{k} = {v}')
                setattr(self, k, v)
        os.makedirs(self.root_to, exist_ok=True)

        t1 = time.time()
        if self.split_from_files:
            self.from_files(self.root_from)
        else:
            self.from_folders(self.root_from, extra_levels=['su'])

        print('Start parallel pool...')
        pool = multiprocessing.Pool(processes=num_cores)
        pool.map(run_process, self.all_commands)

        self.save_list_of_created_files(self.root_to)
        self.read_list_of_created_files(self.root_to)
        self.split_created_files()

        pool.map(run_process, self.move_commands)
        pool.map(run_process, self.delete_commands)

        pool.close()
        pool.join()


        t2 = time.time()
        print(f'Done. {t2 - t1} sec')