import os
import glob
import numpy as np
import torch
import scipy
import scipy.signal
from scipy import fftpack, signal
import segyio
from natsort import natsorted

try:
    import m8r as sf
except:
    print('Madagascar not found! Install m8r from ahay.org')


def is_empty(p): return False if (os.path.exists(p) and [f for f in os.listdir(p) if f != '.gitignore']) else True


def divmax(x): return x / np.max(np.abs(x))


def zero_below_freq(dat, fhi, dt, disable=False, reverse=False):
    """ Input zeros into frequency spectrum of data below or above specified frequency.
        by Oleg Ovcharenko, KAUST, 2021
        
    Args:
        dat(np.ndarray): 2D array [noffsets, ntimes]
        fhi(float): threshold frequency, Hz
        dt(float): temporal sampling, sec
        disable(bool): do nothing, return input data
        reverse(bool): when True, set zeros above fhi, otherwise below
    """
    if disable:
        return dat

    h, w = dat.shape[-2:]
    dat_fx = np.fft.rfft(dat, w)
    ff = np.fft.rfftfreq(dat.shape[-1], d=dt)
    if not reverse:
        where_to_zero = np.where(ff < fhi)[0]
    else:
        where_to_zero = np.where(ff >= fhi)[0]
    dat_fx[..., where_to_zero] = 0. + 0. * 1j
    out = np.fft.irfft(dat_fx, w)
    return out


def convolve(star, psf):
    star_fft = fftpack.fftshift(fftpack.fftn(star))
    psf_fft = fftpack.fftshift(fftpack.fftn(psf))
    return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(star_fft*psf_fft)))


def deconvolve(star, psf, ax=-1):
    star_fft = fftpack.fftshift(fftpack.fftn(star, axes=ax), axes=ax)
    psf_fft = fftpack.fftshift(fftpack.fftn(psf, axes=ax), axes=ax)
    return np.real(fftpack.ifftn(fftpack.ifftshift(star_fft/psf_fft, axes=ax), axes=ax))


def load_su(fname):
    """ Read .su into a np.ndarrays """
    with segyio.su.open(fname, endian='little', ignore_geometry=True) as f:
        d = np.array([np.copy(tr) for tr in f.trace])
        print(f'< {fname} > np.array({d.shape})')
        return d
    
def write_su(fname):
    """ Write a np.ndarray in to a .su file """
    pass

def from_rsf(file, verbose=False):
    rsf = sf.Input(file)
    ndim = len(rsf.shape())
    if verbose: print(f'Load {file}\n\tdim: {ndim}')
    d = dict()
    for i in range(ndim):
        for j in ['n', 'd', 'o']:
            key = f'{j}{i+1}'
            val = rsf.float(key)
            d[key] = int(val) if (val).is_integer() else val
            if verbose: print(f"\tdict['{key}'] <-- {key} <-- {d[key]}")
    n = [rsf.int(f"n{i+1}") for i in range(ndim)]
    a = np.zeros(n[::-1], dtype=np.float32)
    rsf.read(a)
    # a = from_binary(file, dtype=np.float32)[-np.prod(n):].reshape(n).transpose()
    data = np.swapaxes(a, 0, -1)
    if verbose: print(f'\tdata <-- {data.shape}')
    return data, d


def load_bin(p, dims): 
    f = open(p); vp = np.fromfile (f, dtype=np.dtype('float32').newbyteorder ('<')); f.close();
    vp = vp.reshape(*dims); vp = np.transpose(vp); vp = np.flipud(vp); print(f"{vp.shape}"); return vp


def write_bin(d, f):
    print(f'Save {d.shape} as binary to {f}')
    with open(f, "wb") as file:
        binary_format = bytearray(d)
        file.write(binary_format)


def load_hh(f, verbose=False):
    data, opts = from_rsf(f, verbose)
    return data.swapaxes(-1, -2), opts


def write_hh(data, opts, filename=None, verbose=False):
    if verbose: print(f'Save {filename}')
    data = data.astype(np.float32); yy = sf.Output(filename); yy.filename = '/dev/null'; yy.pipe = True
    for k, v in opts.items(): 
        if verbose: print(f'\t{k} <-- {v}')
        yy.put(k, v)
    yy.write(data); yy.close()
    

def parse_files(root, pattern, verbose=1):
    files = natsorted(glob.glob(os.path.join(root, pattern)))
    if verbose:
        if len(files) > 0:
            print(f'Found {len(files)} files in {root}:\n\t{files[0]}\n\t{files[-1]}')
        else:
            print(f'No files matching {pattern} in {root}. Try {natsorted(os.listdir(root))[0]}...')
    return files


def prep(dat, par):
    nx, nt = par['crop']
    sx, st = par['skip']
    dat = dat[:nx:sx, :nt:st]
    if par['norm']:
        si = np.max(np.abs(dat)[:])
        dat /= si
    return dat


class CubeLoader(torch.utils.data.Dataset):
    def __init__(self, dat, par):
        self.dat = dat
        self.par = par
    
    def __len__(self):
        return self.dat.shape[0]
    
    def __getitem__(self, item):
        dat = self.dat[item, ...]
        dat = prep(dat, self.par)
        return (dat.astype(np.float32), item)


class Loader(torch.utils.data.Dataset):
    def __init__(self, f_inp, par):
        self.f_inp = f_inp
        self.par = par

    def __len__(self):
        return len(self.f_inp)

    def __getitem__(self, item):
        fname = self.f_inp[item]
        dat, opts = load_hh(fname)
        dat = prep(dat, self.par)
        return (dat.astype(np.float32), fname)


class RawLoader(torch.utils.data.Dataset):
    """Reads raw .hh data into (h, w) np.array and returns it without any pre-processing"""
    def __init__(self, f_inp):
        self.f_inp = f_inp

    def __len__(self):
        return len(self.f_inp)

    def __getitem__(self, item):
        fname = self.f_inp[item]
        data, opts = load_hh(fname)
        return (data.astype(np.float32), fname)


class LimitLoader(torch.utils.data.Dataset):
    """Given loader with self.f_inp leaves only first n items"""
    def __init__(self, l1, n):
        self.f_inp = l1.f_inp[:n]
        self.main_loader = RawLoader(self.f_inp)
     
    def __len__(self):
        return len(self.f_inp)
    
    def __getitem__(self, item):
        return self.main_loader.__getitem__(item)
    

class CatLoader(torch.utils.data.Dataset):
    def __init__(self, l1, l2):
        super().__init__()
        self.f_inp = self.l1.f_inp + self.l2.f_inp
        self.main_loader = Loader(f_inp)
    
    def __len__(self):
        return len(self.f_inp)
    
    def __getitem__(self, item):
        return self.main_loader.__getitem__(item)
    

class JointLoader(torch.utils.data.Dataset):
    def __init__(self, l1, l2):
        super().__init__()
        self.l1 = l1
        self.l2 = l2
        self.len_max = max(len(l1), len(l2))
    
    def __len__(self):
        return self.len_max
    
    def __getitem__(self, item):
        item1 = item if item < len(self.l1) else item % len(self.l1)
        o1 = self.l1.__getitem__(item1)
        item2 = item if item < len(self.l2) else item % len(self.l2)
        o2 = self.l2.__getitem__(item2)
        return (*o1, *o2)
        
        
        
#============================================================================   
# 
#============================================================================   
        
        
        
import scipy.signal

def const_bandpass_below_freq(dat_tx, fhi, dt, disable=False, reverse=False):
    h, w = dat_tx.shape[-2:]
    dat_fx = np.fft.rfft(dat_tx, w)
    ff = np.fft.rfftfreq(dat_tx.shape[-1], d=dt)
    if not disable:
        if not reverse:
            where_to_zero = np.where(ff < fhi)[0]
            edge = max(where_to_zero)
        else:
            where_to_zero = np.where(ff >= fhi)[0]
            edge = min(where_to_zero)
#         dat_fx[..., where_to_zero] = dat_fx[..., [edge for _ in range(len(where_to_zero))]]
        dat_fx[..., where_to_zero] = 0.
        # print(f'Edge {edge}')
    out = np.fft.irfft(dat_fx, w)
    return out

def butter_bandpass(flo=None, fhi=None, fs=None, order=8, btype='band'):
#def butter_bandpass(flo=None, fhi=None, fs=None, order=5, btype='band'):
    nyq = 0.5 * fs
    if btype == 'band':
        low = flo / nyq
        high = fhi / nyq
        lims = [low, high]
    elif btype == 'low':
        high = fhi / nyq
        lims = high
    elif btype == 'high':
        low = flo / nyq
        lims = low

    #b, a = scipy.signal.butter(order, lims, btype=btype)
    #return b, a
    sos = scipy.signal.butter(order, lims, btype=btype, output='sos')
    return sos


def bandpass(data, flo=None, fhi=None, dt=None, fs=None, order=4, btype='band', verbose=0, pad=(0, 8), upscale=0):
    """ Filter frequency content of 2D data in format [offset, time]

    Args:
        data (ndarray): [offset, time]
        flo (float): low coner frequency
        fhi (float): high corner frequency
        dt (float): sampling interval (introduced for back-compatibility). You can enter either one dt or fs
        fs (float): 1/dt, sampling frequency, Hz
        order:
        btype (str): band, high or low
            * band: limit from both left and right
            * high: limit from right only
            * low: limit from left only
        verbose (bool): print details

    Returns:
        ndarray
    """
    
    if not fs:
        fs = 1/dt
    
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    
    if upscale:
        no, nt = data.shape
        data = signal.resample(data, nt * upscale, axis=-1)
        fs = fs * upscale
        
    if pad:
        no, nt = data.shape
        tmp = np.zeros((no, nt + pad[0] + pad[1]))
        tmp[:, pad[0]:nt+pad[0]] = data
        data = tmp.copy()

    if verbose:
        print(f'Bandpass:\n\t{data.shape}\tflo={flo}\tfhi={fhi}\tfs={fs}')
    #b, a = butter_bandpass(flo, fhi, fs, order=order, btype=btype)
    #y = scipy.signal.filtfilt(b, a, data)

    sos = butter_bandpass(flo, fhi, fs, order=order, btype=btype)
    y = scipy.signal.sosfiltfilt(sos, data)
    
    if pad:
        y = y[:, pad[0]:-pad[1] if pad[1] else None]
    
    if upscale:
        y = y[:, ::upscale]
    return y



def bandpass2(data, 
              flo=None, fhi=None, fbtype='band', dt=None, forder=4,
              klo=None, khi=None, kbtype='band', dx=None, korder=4,
              verbose=0, pad=(0, 0, 0, 0)):
    """ Filter frequency content of 2D data in format [offset, time]

    Args:
        data (ndarray): [offset, time]
        flo (float): low coner frequency
        fhi (float): high corner frequency
        fs (float): 1/dt, sampling frequency, Hz
        order:
        btype (str): band, high or low
            * band: limit from both left and right
            * high: limit from right only
            * low: limit from left only
        verbose (bool): print details

    Returns:
        ndarray
    """
    fs = 1/dt
    
    if dx:
        ks = 1/dx
        
    if pad:
        no, nt = data.shape
        tmp = np.zeros((no + pad[0] + pad[1], nt + pad[2] + pad[3]))
        tmp[pad[0]:no+pad[0], pad[2]:nt+pad[2]] = data
        data = tmp.copy()

    if verbose:
        print(f'Bandpass:\n\t{data.shape}\tflo={flo}\tfhi={fhi}\tfs={fs}')
    
    if flo or fhi:
        sos = butter_bandpass(flo, fhi, fs, order=forder, btype=fbtype)
        data = scipy.signal.sosfiltfilt(sos, data)
    
    if klo or khi:
        sos = butter_bandpass(klo, khi, ks, order=korder, btype=kbtype)
        data = scipy.signal.sosfiltfilt(sos, data.T).T
    
    if pad:
        data = data[pad[0]:-pad[1] if pad[1] else None, 
              pad[2]:-pad[3] if pad[3] else None]
    return data


class BandpassLoader(torch.utils.data.Dataset):
    """Case-specific loader. Bandpasses data according to rules and returns all of them as a tuple"""
    def __init__(self, l1, par, unroll=True, peel=False):
        self.l1 = l1
        self.par = par
        self.unroll = unroll
        self.peel = peel
        self.modes = sorted(par['rules'].keys())
        print(self.modes)

    def __len__(self):
        return len(self.l1)

    def __getitem__(self, item, disable=False):
        d = self.l1.__getitem__(item)
        outs = []
        for _d in d:
            if isinstance(_d, np.ndarray):
                dat = {}
                for mode in self.modes:
                    rule = self.par['rules'][mode]
                    dat[mode] = bandpass(_d, fs=1/self.par['dt'], **rule)
                    dat[mode] = const_bandpass_below_freq(dat[mode], self.par['fedge'], self.par['dt'],
                                                disable=True if mode == 'low' or mode == 'raw' else False)
                    dat[mode] = bandpass(dat[mode], fs=1/self.par['dt'], **rule)
                    dat[mode] = dat[mode][...,:-100]
                    dat[mode] = dat[mode][...,::8]
                    if not self.peel:
                        dat[mode] = np.expand_dims(dat[mode].astype(np.float32), 0)
                    if self.unroll:
                        outs.append(dat[mode].copy())
                if not self.unroll:
                    outs.append(dat.copy())
            else:
                outs.append(_d)
        
        return outs
    
    
def mutter(d, k, b, r=0, flip=False):
    """ Linear muter with Gaussian blur
    Args:
        d(np.ndarray): data, [noffset, ntime]
        k(float): slope of the line
        b(float): intercept of the line
        r(float): smoothening radius for Gaussian blur
        flip(bool): change masking polarity 
    """
    dat = d.copy()
    no, nt = d.shape[-2:]
    mz = np.repeat(np.arange(nt)[np.newaxis, ...], no, 0)
    mx = np.repeat(np.arange(no)[..., np.newaxis], nt, 1)
    mask = np.ones((no, nt))
    if flip:
        mask[mz > k * mx + b] = 0.
    else:
        mask[mz < k * mx + b] = 0.
    if r > 0:
        mask = scipy.ndimage.gaussian_filter(mask, r)
    if len(d.shape) == 3:
        mask = np.expand_dims(mask, 0)
    return dat * mask

    
def make_noise_cube(p):
    print(f'Load {p}')
    c = np.load(p)
    mask = mutter(np.ones(c.shape[-2:]), 3/4, 0)
    c = c * np.expand_dims(1 - mask, 0)
    print(c.shape)

    c = c[: ,:, :int(mask.shape[0] * 3/4)]
    c = np.flip(c[:, ::-1, :], -1) + c
    c = c[...,10:-10]
    c = np.concatenate([c, c], -1)
    print(c.shape)
    return c


class NoiseAdder(torch.utils.data.Dataset):
    def __init__(self, l1, cube_hf):
        super().__init__()
        self.l1 = l1
        self.cube_hf = cube_hf
        self.cn0, self.cn1, self.cn2 = cube_hf.shape
            
    def __len__(self):
        return len(self.l1)
    
    def get_noise_sample(self, shp):
        idx_h = np.random.randint(self.cn2 - shp[-1] - 1)
        idx_b = np.random.randint(self.cn0)
        
        nhf = np.expand_dims(self.cube_hf[idx_b, :, idx_h:idx_h+shp[-1]], 0)
        
        if np.random.rand() > 0.5:
            nhf = np.flip(nhf, -2)
            
        if np.random.rand() > 0.5:
            nhf = np.flip(nhf, -1)
        return nhf

    def __getitem__(self, item):
        # Get [hs, ls, ms, hf, lf, mf]
        v = self.l1.__getitem__(item)
        hs, ls, us, mods, hf, lf, uf, modf = v
        
        nhf = self.get_noise_sample(v[0].shape)
        hs += nhf
        
        nhf = self.get_noise_sample(v[0].shape)
        hf += nhf
        
        return hs, ls, us, mods, hf, lf, uf, modf
    
    
class FlipLoader(torch.utils.data.Dataset):
    def __init__(self, l1, p=0.5):
        super().__init__()
        self.l1 = l1
        self.p = p
            
    def __len__(self):
        return len(self.l1)
    
    def __getitem__(self, item):
        datas = self.l1.__getitem__(item)
        if np.random.rand() > self.p:
            datas = [np.flip(data, -2).copy() for data in datas]
        return datas
