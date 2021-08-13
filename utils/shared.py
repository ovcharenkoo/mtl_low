import os
import glob
import numpy as np
import pickle
import torch
from natsort import natsorted

import segyio

import itertools
import multiprocessing
import multiprocessing.pool
from skimage.transform import resize

import utils.loaders as ld
import matplotlib.pyplot as plt

from IPython.display import clear_output


def is_empty(p): return False if (os.path.exists(p) and os.listdir(p)) else True


class Survey:
    """A container to pass seismic survey design between scripts"""
    def __init__(self, **kwargs):
        self.src = None # object with sources
        self.rec = None # object with receivers
        self.vp = None # placeholder for velocity model
        self.dx = None # grid spacing
        self.wb_taper = None # Waterbottom placeholder
        self.log_idx = None # location of log (grid nodes)
        self.log_loc = None # location of log, meters
        self.bpw = None # Source wavelet
        self.dDT = None # Time sampling for forward modeling
        self.dNT = None # Number of time steps for forward modeling
   
    def __str__(self):
        return str(self.__dict__)
        
        
def cmd(c):
    """Run command line script
    Args:
        c(str): shell command
    """
    print(c)
    os.system(c)


def get_fnames_pattern(pattern, exclude_str='.it'): 
    """ Get list of files matching pattern
    Args:
        pattern(str): e.g. './outputs/su/*.su'
    """
    fnames = natsorted(glob.glob(pattern))
    fnames = [f for f in fnames if exclude_str not in f]
    print(f'{len(fnames)} files found in {pattern}')
    return fnames


def make_cube_and_loader(fnames):
    """Make Pytorch loader from list of filenames
    Args:
        fnames(list): list of filenames
    Returns:
        loader, np.ndarray(len(fnames), noffsets, ntimes), list(dict(scalars))
    """
    cube = []
    for i, f in enumerate(fnames):
        clear_output()
        print(f'{i+1}/{len(fnames)}\n{f}')
        with segyio.su.open(f, "r+", endian='little', ignore_geometry=True) as dst:
            raw = dst.trace.raw[:]
            print(raw.shape)
            cube.append(np.expand_dims(raw, 0))

    cube = np.concatenate(cube, 0)
    (cube_h, cube_l, cube_u), scalers =  split_hlm(cube, par_default)
    loader = TriLoader(cube_h, cube_l, cube_u, np.zeros_like(cube_u), par_default)
    return loader, cube, scalers


def match_amp(have, want):
    """Make data have amplitude of another data from the 'robustly found' patch.
    Selection of this patch makes no much difference as long it is far form boundaries.
    Args:
        have(np.ndarray): [noffsets, ntimes] - where to apply the amplitude
        want(np.ndarray): [noffsets, ntimes] - from where take the amplitude
    """
    max_have = np.max(have[20:50,:-50])
    max_want = np.max(want[20:50,:-50])
    return have / max_have * max_want


def save_object(obj, filename):
    """Save object as pickle file"""
    print(f'Save obj to {filename}')
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

        
def load_object(filename):
    """Load object from pickle file"""
    print(f'Load obj from {filename}')
    try:
        with open(filename, 'rb') as handle:
            obj = pickle.load(handle)
    except Exception as e:
        obj = None
        print(f'Failed! {e}')
    return obj


def save_obj(fname, data):
    """Same as above (need to check their identity before deleting this func)"""
    print(f'Save obj to {fname}')
    with open(fname, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_obj(fname):
    """Same as above (need to check their identity before deleting this func)"""
    print(f'Load obj from {fname}')
    with open(fname, 'rb') as handle:
        data = pickle.load(handle)
    return data


def split_tr_te(f, frac):
    """Split list of filenames into training and testing partitions
    Args:
        f(list): list of filenames
        frac(float): fraction of test partition
    """
    n = int(frac * len(f))
    return f[n:], f[:n]


class TriLoader(torch.utils.data.Dataset):
    """Combine data cubes [nsamp, noffsets, ntimes] into the data loader"""
    def __init__(self, cube1, cube2, cube3, cube4, par=None):
        self.cube1, self.cube2, self.cube3, self.cube4 = cube1, cube2, cube3, cube4
        self.par = par
        print(cube1.shape, cube2.shape, cube3.shape, cube4.shape)
    
    def __len__(self):
        return self.cube1.shape[0]
    
    def __getitem__(self, item):
        def p(v, i):
            out = v[i:i+1, ...].astype(np.float32)
            return out
        
        return (p(self.cube1, item),
                p(self.cube2, item),
                p(self.cube3, item),
                p(self.cube4, item))

    
def get_median_max(x):
    """Get max of absolute value (the function name is misleading from the legacy code)"""
    return np.max(np.abs(x))
    
    
def crop_norm(x, norm=True, ncrop=376 * 8):
    """ Crop the data along last dimension and fit into [-1, 1] range 
    Args:
        x(np.ndarray): data to crop [..., ntimes]
        ncrop(int): how many samples to keep from the last dim 
    Returns:
        cropped data, scaler used for normalization
    """
    x = np.pad(x, ((0, 0), (12, 0)))
    x = x[..., :ncrop]
    if norm:
        s = get_median_max(x)
        x /= s
    else:
        s = 1.
    return x, s


def split_hlm(data, par):
    """ Apply a set of band-pass filters and pre-processing to input dataset
    Args:
        data(np.ndarray): [nsamp, noffset, ntimes] - seismic dataset of common-shot-gathers
        par(dict): a set of parameters
    Returns:
        a list of 9 items: 
            pre-processed data: high-freq, low-freq, mid-freq, ultra-low, freq
            raw data: same as above but without normalization 
            scaler: dict 
    """
    print(f'Start split for {data.shape}')
    if len(data.shape) == 2:
        data = np.expand_dims(data, 0)
        
    ns, no, nt = data.shape
    rules = par['rules']
    modes = natsorted(rules.keys())
    outs = {'high': [], 'low': [], 'ulow': []}
    scalers = []
    
    def process_one_shot(_d, par):
        """Split single shot into high(input), low(target) and ultra-low (visual evaluation) partions"""
        
        # Divide data max(abs) so it is in [-1, 1]
        _d, s = crop_norm(_d.copy(), norm=True)
        
        # Remember the single scalar you divided by (NOT USED)
        local_scalers = {}
        local_scalers['max_shot'] = s
        dat = {}

        for mode in modes:
            # For every key in rules apply bandpass filtering
            rule = rules[mode]
            dat[mode] = ld.bandpass(_d.copy(), dt=par['dt'], pad=(0,8), **rule)
            
            # If mid or high, delete low frequencies
            if not 'low' in mode:
                dat[mode] = ld.zero_below_freq(dat[mode], fhi=par['fedge'], dt=par['dt'])
            
            # Sparsify the data
            dat[mode] = dat[mode][...,::8]
        
        out = []
        symbols = ['high', 'low', 'ulow']
        for k in symbols:
            val = np.expand_dims(dat[k], 0)
            if 'low' in k:
                s = get_median_max(val)
                val = val / s
                local_scalers[f'max_{k}'] = s
            out.append(val) 
            
        scalers.append(local_scalers)
        return out

    # Split in parallel
    with multiprocessing.pool.ThreadPool() as pool:
        list_of_lists = pool.starmap(process_one_shot, zip(data, itertools.repeat(par)))
   
    outs = []
    print('Start making cubes from lists for list {}, where each sublist {}'.format(len(list_of_lists), len(list_of_lists[0])))
    for i in range(len(list_of_lists[0])):
        print(f'{i+1}/{len(list_of_lists[0])}', end='\r')
        cube = np.concatenate([o[i].copy() for o in list_of_lists])
        print(cube.shape)
        outs.append(cube)
    
    # 0,1,2,3 and 4,5,6,7, and [...]
    return outs[:3], scalers
            

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!
# GLOBAL RULES FOR DATA PARTITIONING
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!
order=8 # order of band-pass filters
par_default = {'rules': {'high': {'flo': 4, 'btype': 'high', 'order': order}, 
                     'low':  {'fhi': 5, 'btype': 'low', 'order': order},
                     'ulow':  {'fhi': 2.5, 'btype': 'low', 'order': order},
                    },
               'dt': 0.002, # time sampling of raw data
               'fedge': 4, # hard-code zeros below this frequency
              }

# The overlapping frequency range (To recover amplitudes for FWI)
par_ref = {'fs': 1/0.002, 'flo': 4, 'fhi': 5, 'order': 8, 'btype': 'band'}


def get_model_under_shot(model_cube, shot_data, shot_filename):
    """ Crop the subsurface model covered by geometric location of streamer"""
    noffset, ntime = shot_data.shape[-2:]
    imodel = int(shot_filename.split('/')[-1].split('_')[1])
    ishot = int(shot_filename.split('/')[-1].split('.')[-2].replace('shot', ''))

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # HARD CODED SYN MODEL DATASET SPACINGS
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!
    xsrc_start = 500
    xsrc_step = 1000
    xsrc_dx = 25
    idx_start = int((xsrc_start + ishot * xsrc_step) / xsrc_dx)
    idx_end = idx_start + noffset
    
    # [0..this_vmax/max(all_models)]
    full_model = model_cube[imodel, ...]
    full_model = np.flip(full_model, 0)
    
    # DEPTH x OFFSET
    this_model = full_model[:, idx_start:idx_end]
    # OFFSET x DEPTH
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # HARD CODED TIME DIM
    this_model = resize(this_model.T, (noffset, 376))
    assert this_model.shape[0] == noffset, f'Failed, {this_model.shape}! {shot_filename}, model={imodel}, shot={ishot}, noffset={noffset}'
    return np.expand_dims(this_model, 0)


def get_cubes_from_loader(loader, model_cube):
    """ Take a loader of shot gathers and the dataset of random subsurface models and extract the subsurface for each shot 
    Args:
        loader(torch loader): the loader of ordered shots from the dataset
        model_cube(np.ndarray): [nmodels, nz, nx] all generated random subsurface models used for data generation
    Returns:
        data array: np.ndarray(nshots, noffsets, ntime), local subsurface array: np.ndarray(nshots, noffsets, nz)
    """
    cube_dat = []
    cube_mod = []
    nsamp = len(loader)
    for i in range(nsamp):
        print(f'{i+1} / {nsamp}', end='\r')
        this_data, this_file = loader.__getitem__(i)
        cube_dat.append(np.expand_dims(this_data, 0))
        cube_mod.append(get_model_under_shot(model_cube, this_data, this_file))
    cube_dat = np.concatenate(cube_dat, 0)
    cube_mod = np.concatenate(cube_mod, 0)
    print(cube_dat.shape, cube_mod.shape)
    return cube_dat, cube_mod


def init_loaders(root_src, root_dst, par=None, frac=0.1, limit=None):
    """Main function that makes training dataset from shot gathers and a set of generated subsurface models.
    
    When run for the first time it reads multiple files from the data directory, splits these data into high, low and ultra-low frequency bands and stores them in .npy cubes. The motivation is to accelerate training since such a pre-processing is static and there is no need to do it on the fly. However, this significantly increases RAM consumption.
    """
    par = par if par is not None else par_default
    
    if not os.path.exists(os.path.join(root_src, 'cube_src_tr_h.npy')):
        print(f'Not found! {os.path.join(root_src, "cube_src_tr_h.npy")}')
        
        #=============================================
        # TRAIN
        #=============================================
        # Field data train
        par_ld={'crop': (None, None), 'skip': (1,1), 'norm': False}
        cube_dst_tr = np.load(os.path.join(root_dst, 'data_cgg_ext.npy'))
        print(cube_dst_tr.shape)

        # Syn data train
        fnames = ld.parse_files(os.path.join(root_src, 'train/raw/'), '*.hh')
        loader_src_tr = ld.Loader(fnames, par_ld)
        
        # Load cube of velocity models and fit it into [0, 1], dividing by max
        path_model_cube = os.path.join(root_src, 'rand_models.npy')
        model_cube = np.load(path_model_cube)
        model_cube -= np.min(model_cube) # [0, max - min]
        model_cube /= np.max(model_cube) # [0, 1]
        model_cube *= 2. # [0, 2]
        model_cube -= 1. # [-1, 1]
        # The reverse opertaions are 1. +1; 2. /2; 3. * (box_max - box_min); 4. +box_min
        print(f'Load model cube from {path_model_cube}, {model_cube.shape}\n\tmax:\t{model_cube.max()}\n\tmin:\t{model_cube.min()}')
        
        # Get cubes from loader
        cube_src_tr, cube_src_tr_models = get_cubes_from_loader(loader_src_tr, model_cube)

        #=============================================
        # TEST
        #=============================================
        # Syn data test
        fnames = ld.parse_files(os.path.join(root_src, 'val/raw/'), '*.hh')
        loader_src_te = ld.Loader(fnames, par_ld)

        # Get cubes from loader
        cube_src_te, cube_src_te_models = get_cubes_from_loader(loader_src_te, model_cube)

        #=============================================
        # SPLIT
        #=============================================
        # DST data
        # Processed / Raw
        (cube_dst_tr_h, cube_dst_tr_l, cube_dst_tr_u), _ = split_hlm(cube_dst_tr, par)
        
        # Testing = Training
        cube_dst_te_h, cube_dst_te_l, cube_dst_te_u = cube_dst_tr_h.copy(), cube_dst_tr_l.copy(), cube_dst_tr_u.copy()

        # SRC data
        (cube_src_tr_h, cube_src_tr_l, cube_src_tr_u), _ =  split_hlm(cube_src_tr, par)
        
        (cube_src_te_h, cube_src_te_l, cube_src_te_u), _ =  split_hlm(cube_src_te, par)
        print(cube_src_tr_h.shape)

        print(f'Save cubes to {root_src}...')
        np.save(os.path.join(root_src, 'cube_src_tr_h.npy'), cube_src_tr_h)
        np.save(os.path.join(root_src, 'cube_src_te_h.npy'), cube_src_te_h)
        np.save(os.path.join(root_src, 'cube_src_tr_l.npy'), cube_src_tr_l)
        np.save(os.path.join(root_src, 'cube_src_te_l.npy'), cube_src_te_l)
        np.save(os.path.join(root_src, 'cube_src_tr_u.npy'), cube_src_tr_u)
        np.save(os.path.join(root_src, 'cube_src_te_u.npy'), cube_src_te_u)

        np.save(os.path.join(root_src, 'cube_dst_tr_h.npy'), cube_dst_tr_h)
        np.save(os.path.join(root_src, 'cube_dst_te_h.npy'), cube_dst_te_h)
        np.save(os.path.join(root_src, 'cube_dst_tr_l.npy'), cube_dst_tr_l)
        np.save(os.path.join(root_src, 'cube_dst_te_l.npy'), cube_dst_te_l)
        np.save(os.path.join(root_src, 'cube_dst_tr_u.npy'), cube_dst_tr_u)
        np.save(os.path.join(root_src, 'cube_dst_te_u.npy'), cube_dst_te_u)
        
        np.save(os.path.join(root_src, 'cube_src_tr_models.npy'), cube_src_tr_models)
        np.save(os.path.join(root_src, 'cube_src_te_models.npy'), cube_src_te_models)
        
    else:
        print(f'Load cubes from {root_src}...')
        cube_src_tr_h = np.load(os.path.join(root_src, 'cube_src_tr_h.npy'))
        cube_src_te_h = np.load(os.path.join(root_src, 'cube_src_te_h.npy'))
        cube_src_tr_l = np.load(os.path.join(root_src, 'cube_src_tr_l.npy'))
        cube_src_te_l = np.load(os.path.join(root_src, 'cube_src_te_l.npy'))
        cube_src_tr_u = np.load(os.path.join(root_src, 'cube_src_tr_u.npy'))
        cube_src_te_u = np.load(os.path.join(root_src, 'cube_src_te_u.npy'))

        cube_dst_tr_h = np.load(os.path.join(root_src, 'cube_dst_tr_h.npy'))
        cube_dst_te_h = np.load(os.path.join(root_src, 'cube_dst_te_h.npy'))
        cube_dst_tr_l = np.load(os.path.join(root_src, 'cube_dst_tr_l.npy'))
        cube_dst_te_l = np.load(os.path.join(root_src, 'cube_dst_te_l.npy'))
        cube_dst_tr_u = np.load(os.path.join(root_src, 'cube_dst_tr_u.npy'))
        cube_dst_te_u = np.load(os.path.join(root_src, 'cube_dst_te_u.npy'))

        cube_src_tr_models = np.load(os.path.join(root_src, 'cube_src_tr_models.npy'))
        cube_src_te_models = np.load(os.path.join(root_src, 'cube_src_te_models.npy'))
        
    loader_src_tr = TriLoader(cube_src_tr_h, cube_src_tr_l, cube_src_tr_u, cube_src_tr_models, par)
    loader_src_te = TriLoader(cube_src_te_h, cube_src_te_l, cube_src_te_u, cube_src_te_models, par)
    loader_dst_tr = TriLoader(cube_dst_tr_h, cube_dst_tr_l, cube_dst_tr_u, np.zeros_like(cube_dst_tr_u), par)
    loader_dst_te = TriLoader(cube_dst_te_h, cube_dst_te_l, cube_dst_te_u, np.zeros_like(cube_dst_te_u), par)
    
    jloader_tr = ld.JointLoader(loader_src_tr, loader_dst_tr)
    jloader_te = ld.JointLoader(loader_src_te, loader_dst_te)
    
    print('Processed data loaders: {} {}'.format(len(jloader_tr), len(jloader_te)))
    
    return jloader_tr, jloader_te


def dcn(t):
    """Detach and convert to numpy a torch tensor"""
    return t.detach().cpu().numpy()

def save_image(p, img, title='', clip=1):
    """Save image at given path"""
    vlim = clip * np.max(np.abs(img))
    plt.figure(); plt.imshow(img, vmin=-vlim, vmax=vlim); 
    fname = os.path.join(p, title)
    print(f'Save {fname}...')
    plt.savefig(fname)
    plt.pause(1e-4); 
    plt.close();