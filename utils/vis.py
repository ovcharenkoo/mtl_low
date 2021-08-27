import os
import copy
import numpy as np
import scipy.signal
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import matplotlib.gridspec as gridspec

from utils.loaders import bandpass, butter_bandpass


fontsize = 10
params = {
    # 'text.latex.preamble': ['\\usepackage{gensymb}'],
    # 'image.origin': 'lower',
    'image.interpolation': 'nearest',
    # 'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 300,  # to adjust notebook inline plot size
    'axes.labelsize':fontsize,  # fontsize for x and y labels (was 10)
    'axes.titlesize':fontsize,
    'font.size':fontsize,  # was 10
    'legend.fontsize': fontsize,  # was 10
    'xtick.labelsize':fontsize,
    'ytick.labelsize':fontsize,
    # 'text.usetex': True,
    # 'figure.figsize': [3.39, 2.10],
    'font.family': 'serif',
}
matplotlib.rcParams.update(params)


def savefig(img_name, disable=False):
    if not disable:
        root_home = os.path.expanduser("~")
        root_pic = os.path.join(root_home, 'Dropbox/WORK_HARD/low_elastic/tex/GEOPHYSICS_21/pic')
        os.makedirs(root_pic, exist_ok=True)
        par_pic = {'dpi': 150}
        if not '/' in img_name:
            print(f'Save {os.path.join(root_pic, img_name)}')
            plt.savefig(os.path.join(root_pic, img_name), **par_pic)
        else:
            print(f'Save {img_name}')
            plt.savefig(img_name, **par_pic)
    else:
        print('Saving figures disabled!')
    

def plot_model(v, title='', axis='on', colorbar=True, cax_label='km/s', figsize=None, cmap='RdBu_r', ax=None, dpi=None, **kwargs): 
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if dpi:
            fig.set_dpi(dpi)
            
    im = ax.imshow(v, cmap=cmap, **kwargs); 
    plt.axis(axis)
    ax.set_title(title); 
    ax.invert_yaxis();
    if colorbar:
        divider = make_axes_locatable(ax); cax = divider.append_axes("right", size="5%", pad=0.05);
        plt.colorbar(im, cax=cax, label=cax_label);


def plot_acquisition(v, dx, src=None, rec=None, title='', cax_label='', log=None, **kwargs):
    nz, nx = v.shape; par = {'extent': [0, nx*dx/1000, 0, nz*dx/1000], 'cmap': 'RdBu_r'}; par.update(kwargs);
    plt.figure(); ax = plt.gca(); im = ax.imshow(v/1000, **par); plt.title(title); ax.invert_yaxis(); plt.xlabel('km'); plt.ylabel('km');
    if rec is not None: 
        map_rec = rec.x / dx < nx
        plt.scatter(rec.x[map_rec]/1000, rec.y[map_rec]/1000, 1, color='m'); 
    if src is not None:
        map_src = src.x / dx < nx
        plt.scatter(src.x[map_src]/1000, src.y[map_src]/1000, 1, color='w'); 
    
    if log is not None:
        ax = plt.gca()
        _log = log['data']
        vh = log['loc'] * np.ones_like(_log) / 1000 
        ax.plot(vh, np.arange(len(_log))*dx/1000, 'k--')
        ax.plot(vh + (_log[::-1] - min(_log)) / 1000, np.arange(len(_log))*dx/1000, 'k')
    
    divider = make_axes_locatable(ax); cax = divider.append_axes("right", size="5%", pad=0.05); 
    plt.colorbar(im, cax=cax, label=cax_label);
    

def plot_signal(signals, dt=None, title='', xlabel='Time, sec', ylabel='', **kwargs):
    fig, ax = plt.subplots(1, 1)
    if not isinstance(signals, list):
        signals = [signals]
    
    for signal in signals:
        if len(signal.shape) < 2:
            signal = np.expand_dims(signal, 0)

        if dt:
            ax_ticks = np.arange(signal.shape[-1]) * dt
            plt.plot(ax_ticks, signal.T, **kwargs)
        else:
            plt.plot(signal.T, **kwargs)
    
    plt.title(title)
    if dt:
        plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    
def plot_shot(shot, pclip=1.0, title='', colorbar=True, dt=None, dx=None, 
              figsize=None, axis=True, ax=None, cax_label='', dpi=None, **kwargs):
    """ Plot a seismic shot gather given as 2d array [offset, time] """
    if isinstance(shot, list):
        shot = np.concatenate(shot, 0)
        
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize if figsize else None);
        if dpi:
            fig.set_dpi(dpi)
        
    vlim = pclip * np.max(np.abs(shot)); 

    if dt or dx:
        nx, nt = shot.shape                                                 nz0 * dx/1000] if nz0 else nz * dx / 1000, 

        kwargs['extent'] = [0, nx*dx/1000 if dx else nx, 0, nt*dt if dt else nt]
        if dx: 
            ax.set_xlabel('km');
        else: 
            ax.set_visible(False)
        if dt: 
            ax.set_ylabel('sec');
        else: 
            ax.set_visible(False)
            
    if not 'cmap' in kwargs.keys():
        kwargs['cmap'] = 'Greys'
    kwargs['vmin'] = -vlim if not 'vmin' in kwargs.keys() else kwargs['vmin']
    kwargs['vmax'] = vlim if not 'vmax' in kwargs.keys() else kwargs['vmax']
    
    im = ax.imshow(np.flip(shot, -1).T, **kwargs)
    
    if colorbar:
        divider = make_axes_locatable(ax); cax = divider.append_axes("right", size="5%", pad=0.05);
        plt.colorbar(im, cax=cax, label=cax_label);
        
    ax.set_title(title); 
    ax.axis('auto' if axis else 'off'); ax.invert_yaxis();
    

def plot_spectra_dictlist(dictlist, dt, fmax=10, phase=False, title='', norm=False, ampmax=None, ax=None):
    """ Plot multiple spectra of seismic data on the same plot. 
    Input data format:
    [{'data': arr1, 'line': 'r--', 'label': 'data1'},
     {'data': arr2, 'line': 'b', 'label': 'data2'},] """
    if ax is None:
        fig, ax = plt.subplots()
    for d in dictlist:
        ff, ss = get_spectrum(d['data'], dt, phase=phase)
        if ampmax:
            ss[ss > ampmax] = ampmax
        if norm:
            ss /= np.max(np.abs(ss))
        nfmax = int(fmax / (ff[1]-ff[0]))
        ax.plot(ff[:nfmax], ss[:nfmax],
                 d['line'], label=d['label'])
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Gain' if not phase else 'Angle, rad'); 
    ax.set_title(title)
    ax.grid(True); ax.legend(loc='best');
    
    
def plot_nxm(ds, value_shift=0, transpose=False, **kwargs):
    """ Given a nested list [[im1, im2, ...], [im3, im4, ...]] plot it as a table """
    img = np.concatenate([np.concatenate(dd, 0) for dd in ds], 1)
    img += value_shift
    plot_shot(img if not transpose else img.T, axis=False, **kwargs)
    

def compare_shots(a, b, pclip=0.95, title=''):
    img = np.concatenate((a, b, a-b), 0); 
    plot_shot(img, pclip, title=title)

    
def plot_spectrum(shot, dt, title='', fmax=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ps = np.sum(np.abs(np.fft.fft(shot)) ** 2, axis=-2); freqs = np.fft.fftfreq(len(ps), dt); idx = np.argsort(freqs)
    causal = int(len(ps) // 2); freqs, ps = freqs[idx], ps[idx]; freqs = freqs[-causal:]; ps = ps[-causal:]; freqs = freqs[freqs < (fmax if fmax else np.max(freqs))]; 
    n = len(freqs); 
    ax.plot(freqs[:n], ps[:n], label=title); 
    ax.set_xlabel('Frequency (Hz)'); 
    ax.set_ylabel('Gain'); ax.grid(True);


def plot_logs(m1, m2, idx=2, title=''):
    plt.figure(); plt.plot(m1.vp[::-1,idx], 'k--'); plt.plot(m1.vs[::-1,idx], 'k--'); plt.plot(m1.rho[::-1,idx], 'k--');
    plt.plot(m2.vp[::-1,idx], label='vp'); plt.plot(m2.vs[::-1,idx], label='vs'); plt.plot(m2.rho[::-1,idx], label='rho'); plt.legend() 


def imshow_dict(d, title='', sym=True, lims=None, return_lims=False, disable=False, **kwargs):
    imgs = []
    keys = []
    clip = 0.9 if not 'pclip' in kwargs.keys() else kwargs['pclip']
    for k, v in sorted(d.items()):
        if len(v.shape) == 2:
            v = np.expand_dims(v, 0)
        # print(v.shape)
        v = v[0, ...].transpose(-1, -2)
        imgs.append(v)
        keys.append(k)
    title += ', '.join(keys)
    img = np.concatenate(imgs, axis=1)
    _lims = [np.min(img), np.max(img)]
    if not disable:
        plt.figure()
        if lims is not None:
            vlim = clip * max(np.abs(lims[0]), np.abs(lims[1]))
            plt.imshow(img, vmin=-vlim, vmax=vlim)
        elif sym:
            vlim = clip * np.max(np.max(np.abs(img)))
            plt.imshow(img, vmin=-vlim, vmax=vlim)
        if title:
            plt.title(title)
        plt.colorbar(orientation="horizontal")
    if return_lims:
        return _lims
    else:
        return None

def imshow_shot(ds, title='', pclip=0.1, **kwargs):
    img = np.concatenate([d.squeeze().T for d in ds], -1)
    vlim = pclip * np.max(np.abs(img))
    
    kwargs['vmin'] = -vlim if not 'vmin' in kwargs.keys() else kwargs['vmin']
    kwargs['vmax'] = vlim if not 'vmax' in kwargs.keys() else kwargs['vmax']
 
    plt.figure()
    if len(img.shape) > 2:
        img = img[0, ...]
    plt.imshow(img, **kwargs);
    plt.title(title)
    plt.pause(1e-4)
    

def imshow_diff(d1, d2, key, pclip, title, lims):
    diff = dict_sub(d1, d2)
    imshow_dict({key: np.concatenate((d1[key], d2[key], diff[key]), -2)}, 
                pclip=pclip, title=title, lims=lims)


def get_spectrum(t, dt, phase=False):
    t_fft = np.fft.fft(t)
    if not phase:
        ps = np.sum(np.abs(t_fft) ** 2, axis=-2)
    else:
        ps = np.sum(np.arctan2(np.imag(t_fft), np.real(t_fft)), axis=-2)
    freqs = np.fft.fftfreq(len(ps), dt)
    idx = np.argsort(freqs)
    causal = int(len(ps) // 2)
    freqs, ps = freqs[idx], ps[idx]
    return freqs[-causal:], ps[-causal:] / t.shape[-2]


def plot_spectra_dict(ds, dt, fmax, labels=None, norm=False, fig_ax=None, phase=False, **kwargs):
    colors = ['b', 'r', 'g', 'm', 'c', 'y']
    lines = 2 * ['', '--', '-.', ':']

    if fig_ax is None:
        fig1, ax1 = plt.subplots()
    else:
        fig1, ax1 = fig_ax
        
    if not isinstance(ds, list):
        ds = [ds]

    for il, d in enumerate(ds):
        for ik, (k, v) in enumerate(d.items()):
            ff, ss = get_spectrum(v, dt, phase=phase)
            if norm:
                ss /= np.max(ss)
            nfmax = int(fmax / (ff[1]-ff[0]))
            ax1.plot(ff[:nfmax], ss[:nfmax], 
                     f'{colors[il]}{lines[ik]}', 
                     label=f'{labels[il] if labels is not None else ""}{k}', 
                     **kwargs)
            
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')


def del_key(d, key):
    if not isinstance(key, list):
        key = [key]
    r = copy.deepcopy(dict(d))
    for k in key:
        if k in r.keys():
            del r[k]
    return r

def plot_filters(rules, dt, fmax=None, title='', exclude=None, fedge=None):
    print(rules)
    fs = 1 / dt
    fig1, ax1 = plt.subplots()
    if exclude:
        for key in exclude:
            rules = del_key(rules, key)

    for mode, rule in rules.items():
        #b, a = butter_bandpass(fs=fs, **rule)
        #w, h = scipy.signal.freqz(b, a)
        sos = butter_bandpass(fs=fs, **rule)
        w, h = scipy.signal.sosfreqz(sos)
        w_ = (fs * 0.5 / np.pi) * w
        nfmax = len(w_[w_ < fmax])
        ax1.plot(w_[:nfmax], abs(h[:nfmax]), label="%s" % mode)
    if fedge is not None:
        fedge = w_[w_ < fedge]
        ax1.vlines(fedge[-1], 0, 1, colors='r', linestyles='--')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')
    if title:
        plt.title(title)
    # plt.pause(0.001)


def insert_zeros(trace, tt=None):
    """Insert zero locations in data trace and tt vector based on linear fit"""

    if tt is None:
        tt = np.arange(len(trace))

    # Find zeros
    zc_idx = np.where(np.diff(np.signbit(trace)))[0]
    x1 = tt[zc_idx]
    x2 = tt[zc_idx + 1]
    y1 = trace[zc_idx]
    y2 = trace[zc_idx + 1]
    a = (y2 - y1) / (x2 - x1)
    tt_zero = x1 - y1 / a

    # split tt and trace
    tt_split = np.split(tt, zc_idx + 1)
    trace_split = np.split(trace, zc_idx + 1)
    tt_zi = tt_split[0]
    trace_zi = trace_split[0]

    # insert zeros in tt and trace
    for i in range(len(tt_zero)):
        tt_zi = np.hstack(
            (tt_zi, np.array([tt_zero[i]]), tt_split[i + 1]))
        trace_zi = np.hstack(
            (trace_zi, np.zeros(1), trace_split[i + 1]))

    return trace_zi, tt_zi


def wiggle_input_check(data, tt, xx, sf, verbose):
    ''' Helper function for wiggle() and traces() to check input
    '''

    # Input check for verbose
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a bool")

    # Input check for data
    if type(data).__module__ != np.__name__:
        raise TypeError("data must be a numpy array")

    if len(data.shape) != 2:
        raise ValueError("data must be a 2D array")

    # Input check for tt
    if tt is None:
        tt = np.arange(data.shape[0])
        if verbose:
            print("tt is automatically generated.")
            print(tt)
    else:
        if type(tt).__module__ != np.__name__:
            raise TypeError("tt must be a numpy array")
        if len(tt.shape) != 1:
            raise ValueError("tt must be a 1D array")
        if tt.shape[0] != data.shape[0]:
            raise ValueError("tt must have same as data's rows")

    # Input check for xx
    if xx is None:
        xx = np.arange(data.shape[1])
        if verbose:
            print("xx is automatically generated.")
            print(xx)
    else:
        if type(xx).__module__ != np.__name__:
            raise TypeError("tt must be a numpy array")
        if len(xx.shape) != 1:
            raise ValueError("tt must be a 1D array")
        if tt.shape[0] != data.shape[0]:
            raise ValueError("tt must have same as data's rows")
        if verbose:
            print(xx)

    # Input check for streth factor (sf)
    if not isinstance(sf, (int, float)):
        raise TypeError("Strech factor(sf) must be a number")

    # Compute trace horizontal spacing
    ts = np.min(np.diff(xx))

    # Rescale data by trace_spacing and strech_factor
    data_max_std = np.max(np.std(data, axis=0))
    data = data / data_max_std * ts * sf

    return data, tt, xx, ts


def wiggle(data, tt=None, xx=None, color='k', alpha=1., sf=0.15, verbose=False, ax=None, fill=False):
    '''Wiggle plot of a sesimic data section
    Syntax examples:
        wiggle(data)
        wiggle(data, tt)
        wiggle(data, tt, xx)
        wiggle(data, tt, xx, color)
        fi = wiggle(data, tt, xx, color, sf, verbose)
    Use the column major order for array as in Fortran to optimal performance.
    The following color abbreviations are supported:
    ==========  ========
    character   color
    ==========  ========
    'b'         blue
    'g'         green
    'r'         red
    'c'         cyan
    'm'         magenta
    'y'         yellow
    'k'         black
    'w'         white
    ==========  ========
    '''

    # (Time, Offset) --> (Offset, Time)
    data = data.T

    # Input check
    data, tt, xx, ts = wiggle_input_check(data, tt, xx, sf, verbose)

    # Plot data using matplotlib.pyplot
    Ntr = data.shape[1]

    if not ax:
        ax = plt.gca()

    for ntr in range(Ntr):
        trace = data[:, ntr]
        offset = xx[ntr]

        if verbose:
            print(offset)

        trace_zi, tt_zi = insert_zeros(trace, tt)
        if fill:
            ax.fill_betweenx(tt_zi, offset, trace_zi + offset,
                             where=trace_zi >= 0,
                             facecolor=color)

        ax.plot(trace_zi + offset, tt_zi, color, alpha=alpha)

    ax.set_xlim(xx[0] - ts, xx[-1] + ts)
    ax.set_ylim(tt[0], tt[-1])
    ax.invert_yaxis()
    
    
def color_to_color_linestyle(compound_color):
    this_color = ''.join(filter(str.isalpha, compound_color))
    this_linestyle = ''.join([s for s in compound_color if not s.isalpha()])
    this_linestyle = this_linestyle if this_linestyle else 'solid'
    return this_color, this_linestyle

from matplotlib.lines import Line2D
def plot_wiggles(ds, n=9, colors=['k--', 'b', 'r'], alphas=1., legend=None, figsize=None, return_data=False):
    """ds (list) of numpy arrays """
    if not isinstance(ds, list):
        ds = [ds]
    if not isinstance(alphas, list):
        alphas = [alphas for _ in range(len(ds))]
    
    to_return = []

    fig, ax = plt.subplots(1, 1, figsize=None)
    idxs = [int(np.floor(v)) for v in np.linspace(0, ds[0].shape[-2] - 1, n)]
    print(idxs)
    for i, d in enumerate(ds):
        if len(d.shape) == 3:
            d = d[0, ...]
        elif len(d.shape) == 4:
            d = d[0, 0, ...]
        to_plot = d[idxs, :]
        to_return.append(to_plot)
        wiggle(to_plot, color=colors[i], alpha=alphas[i])
        
    if legend:
        lines = []
        for i in range(len(ds)):
            this_color, entered_style = color_to_color_linestyle(colors[i])
            this_linestyle = entered_style if entered_style else 'solid'
            lines.append(Line2D([0, 1], [0, 1], color=this_color, linestyle=this_linestyle))
        plt.legend(lines, legend)

    if return_data:
        return to_return
        
        
def plot_metrics(m, title=''):
    plt.figure()
    for phase, loss_dict in m.items():
        color_phase = '--' if phase == 'val' else ''
        for ln, val in loss_dict.items():
            plt.plot(val, color_phase, label=f'{ln}_{phase}')
    plt.legend()
    plt.title(title)
#     plt.pause(0.001)
    plt.show()

def _handle_np_zeros(t, eps=1e-6):
    t[np.abs(t) < eps] = 1
    return t

def tracewise_rms(x):
    return np.sqrt(np.sum(np.square(x), axis=-1) / x.shape[-1])


def rms(x):
    return np.sqrt(np.sum(np.square(x)) / np.prod(x.shape))


def relative_rms(ref, x, digits=2):
    return np.round(rms(x) / rms(ref), 2)


def metric_pearson2(x, y):
    """ Pearson's coefficient, rho = \frac{cov(x, y)}{std_x * std_y}
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    Args:
        x: [b, c, h, w] np.array
        y: [b, c, h, w] np.array

    Returns:
        [b, c, h] np.array
    """
    ex = x - np.expand_dims(np.mean(x, axis=-1), -1)
    ey = y - np.expand_dims(np.mean(y, axis=-1), -1)
    m = np.mean(ex * ey, axis=-1) \
        / _handle_np_zeros(np.std(x, axis=-1)) \
        / _handle_np_zeros(np.std(y, axis=-1))
    return m

def _check_range(vv, vmin=0, vmax=1):
    vv[vv > vmax] = vmax
    vv[vv < vmin] = vmin
    return vv

def ax_imshow(ax, im, title='', **kwargs):
    ax.imshow(im.T, cmap='Greys', **kwargs); 
    ax.axis('off');
    if title:
        ax.set_title('Input'); 

def imgrid(inp, lbl, preds, pclip=1.0, titles='', 
           diff_of_diff=True, analysis=True, figsize=None, 
           scatter_size=1, scatter_color_rms='b', scatter_color_pear='yellow',
           labels=None, par_text=None, **kwargs):
    """
    inp (np.ndarray): [noffset x ntime]
    lbl (np.ndarray): [noffset x ntime]
    preds (list): list of np.ndarrays, [noffset x ntime]
    """
    nrows = 2
    ndnns = len(preds)
    
    if not titles:
        titles = 4 * ['']
    
    nx, nt = inp.shape
    par_scatter_range = {'vmin': 2, 'vmax': nt-2}
    
    if diff_of_diff:
        move = 2
    else:
        move = 1
        
#     fig = plt.figure(figsize=(2 * (ndnns + move), 2 * nrows) if not figsize else figsize)
    fig = plt.figure(figsize=(2 * (ndnns + move), (ndnns + move)/1.5) if not figsize else figsize)
    gs1 = gridspec.GridSpec(nrows, ndnns + move)
    gs1.update(wspace = 0.025, hspace = 0.025)
    
    ax = plt.subplot(gs1[0]); 
    ax_imshow(ax, inp, **kwargs);

    diffs = []
    coeffs_rms = []
    coeffs_pearson = []
    for ie in range(ndnns):
        pred = preds[ie]
        diff = lbl - pred
        diffs.append(diff)

        ax = plt.subplot(gs1[1 + ie]); 
        ax_imshow(ax, pred, **kwargs);
        if labels is not None:
            par_text = {'x': 0., 
                        'y': 0.,
                        'color': 'white',
                       'fontsize': 10,
                       'verticalalignment': 'top',
                       'horizontalalignment': 'left',
                       'fontfamily':'sans-serif',
#                        'name': 'Helvetica', 
#                        'transform': ax.transAxes
                       } if par_text is None else par_text
            ax.text(s=labels[1+ie], **par_text)
        
        ax = plt.subplot(gs1[ndnns + move + ie + 1]); 
        ax_imshow(ax, diff, **kwargs);
    #         diff.imshow(im_id, fig_ax=(fig, ax), **pargs, title='{:.3f}'.format(diff.rms[im_id] / lbl.rms[im_id]))

        if analysis:
            loc_y = lbl.shape[-1] // 2
            _xx = np.arange(lbl.shape[-2])
            _yy = loc_y * np.ones_like(_xx)
            ax.plot(_xx, _yy, 'k--', lw=0.5)

            # Pearson similarity measure, best when 1, worst -1
            coeff_pearson = metric_pearson2(lbl, pred)
            _yy = loc_y * (1 - coeff_pearson)
            _yy = _check_range(_yy, **par_scatter_range)
            coeffs_pearson.append(_yy)
            ax.scatter(_xx, _yy, s=scatter_size, 
#                        c=cm.hot((_yy / loc_y + 1) / 2), 
                       c=scatter_color_pear,
                       edgecolor='none')

            # RMS similarity = 1 - rms(x - y) / rms(x). Best when 1, worst 0
            coeff_rms = 1 - tracewise_rms(lbl - pred) / _handle_np_zeros(tracewise_rms(lbl))
            coeff_rms[coeff_rms < -1] = -1
            _yy = loc_y * (1 - coeff_rms)
            _yy = _check_range(_yy, **par_scatter_range)
            coeffs_rms.append(_yy)
            ax.scatter(_xx, _yy, s=scatter_size, 
#                        c=cm.winter_r(_yy / loc_y), 
                       c=scatter_color_rms,
                       edgecolor='none')

    if diff_of_diff:
        ax = plt.subplot(gs1[1 + ie + 1])
        ax_imshow(ax, preds[1] - preds[0], **kwargs)
        
        ax = plt.subplot(gs1[move + ie + ndnns + 2])
        ax_imshow(ax, diffs[1] - diffs[0], **kwargs)
        
        if analysis:
            _yy = loc_y * np.ones_like(_xx)
            _yy = _check_range(_yy, **par_scatter_range)
            ax.plot(_xx, _yy, 'k--', lw=0.5)

            _yy = coeffs_pearson[1] - coeffs_pearson[0] + loc_y
            _yy = _check_range(_yy, **par_scatter_range)
            ax.scatter(_xx, _yy, s=scatter_size, 
#                        c=cm.hot((_yy / loc_y + 1) / 2), 
                       c=scatter_color_pear,
                       edgecolor='none')

            _yy = coeffs_rms[1] - coeffs_rms[0] + loc_y
            _yy = _check_range(_yy, **par_scatter_range)
            ax.scatter(_xx, _yy, s=scatter_size, 
#                        c=cm.winter_r(_yy / loc_y), 
                       c=scatter_color_rms,
                       edgecolor='none')

    ax = plt.subplot(gs1[ndnns + move])
    ax_imshow(ax, lbl, **kwargs)
#     plt.margins(0, 0)

def get_spectrum2(shot, dt=None, dx=None, fmax=None, kmax=None):
    shot = abs(np.fft.fftshift(np.fft.fft2(shot)))
    fax = None
    kax = None
    nx, nt = shot.shape       
    if dt:
        fax = np.fft.fftfreq(shot.shape[-1], d=dt)
        idx = np.argsort(fax)
        causal = int(nt // 2);
        fax, shot = fax[idx], shot[:, idx];
        fax = fax[-causal:]; shot = shot[:, -causal:]
        fax = fax[fax < (fmax if fmax else np.max(fax))]; 
        n = len(fax);
        fax = fax[:n]; shot = shot[:, -n:]
    if dx:
        kax = np.fft.fftfreq(shot.shape[0], d=dx)
        idx = np.argsort(kax)
        kax, shot = kax[idx], shot[idx, :]
        shot = np.fft.fftshift(shot, 0)
        idx = np.abs(kax) < (kmax if kmax else np.max(kax))
        kax = kax[idx];
        shot = shot[idx, :]
    return shot, fax, kax


def plot_spectrum2(shot, pclip=1.0, title='', colorbar=False, dt=None, dx=None, fmax=30, kmax=None,
                   figsize=None, axis=True, plot_only=False, **kwargs):
    """
    dt : temporal sampling [sec]
    dx : spatial sampling [m]
    """
    
    if isinstance(shot, list):
        shot = np.concatenate(shot, 0)
        
    if dx:
        dx = dx / 1000
        
    nx, nt = shot.shape
    

    fft2_shot, fax, kax = get_spectrum2(shot, dt, dx, fmax, kmax)
    
    if not plot_only:
        shot = fft2_shot.copy()
        
    vlim = pclip * np.max(np.abs(shot)); plt.figure(figsize=figsize if figsize else None); 
    
    if dt or dx:
        kwargs['extent'] = [np.min(kax) if dx else 0, np.max(kax) if dx else nx, 
                            0, np.max(fax) if dt else nt]
        if dx: 
            plt.xlabel('Wavenumber, cycles / km') 
        else: 
            plt.gca().get_xaxis().set_visible(False) 
        if dt: 
            plt.ylabel('Frequency, Hz') 
        else: 
            plt.gca().get_yaxis().set_visible(False)
    kwargs['vmin'] = 0 if not 'vmin' in kwargs.keys() else kwargs['vmin']
    kwargs['vmax'] = vlim if not 'vmax' in kwargs.keys() else kwargs['vmax']
    plt.imshow(shot.T, cmap='RdBu_r', **kwargs); 
    if colorbar: plt.colorbar()
    plt.title(title); plt.axis('auto' if axis else 'off');

    
def make_stripe_mask(_shape, width, step):
    h, w = _shape
    nlines = int(w / (width + step)) + 1

    segment = [0 for i in range(width)] + [1 for i in range(step)]
    mute_vec = nlines * segment
    mute_vec = mute_vec[:h]
    mute_vec = [i for i, val in enumerate(mute_vec) if val > 0]

    mask = np.zeros((h, w))
    mask[mute_vec, :] = 1
    return mask.astype(np.bool8)

def plot_compare_stripes(shot_a, shot_b, width=20, step=20, **kwargs):
    """
    Args:
        shot_a(np.ndarray): 2D array, [noffset, ntime]
        shot_b(np.ndarray): 2D array, [noffset, ntime]
        width(int): width of the stripes in pixels
        step(int): step of the stripes in pixels
    """
    stripes = make_stripe_mask(shot_a.shape, width, step)
    shot_plot = copy.copy(shot_a)
    shot_plot[stripes] = shot_b[stripes]
    plot_shot(shot_plot, **kwargs)
    
    
def plot_log_model(mm, dx, nx0=None, nz0=None, _src=None, title='', log=None, log_location=None, cmap='RdBu_r', axis=True, cax_label='km/s', **kwargs):
    v = mm.copy() / 1000
    plt.figure(); ax = plt.gca();
    nz, nx = mm.shape[-2:]
    if _src is not None:
        map_src = _src.x / dx < nx0
        plt.scatter(_src.x[map_src]/1000, _src.y[map_src]/1000, 1, color='w'); 
    im = ax.imshow(v[:,:nx0], cmap=cmap, extent=[0, 
                                                 nx0 * dx / 1000 if nx0 else nx * dx / 1000, 
                                                 0, 
                                                 nz0 * dx/1000 if nz0 else nz * dx / 1000], 
                   origin='upper', **kwargs); 
    divider = make_axes_locatable(ax); 
    cax = divider.append_axes("right", size="5%", pad=0.05); cbar = plt.colorbar(im, cax=cax); cbar.set_label(cax_label);
    if axis:
        ax.set_xlabel('km'); ax.set_ylabel('km'); ax.set_title(title); 
    else:
        ax.axis('off')
    ax.invert_yaxis();

    if log is not None:
        vh = log_location * np.ones_like(log) / 1000 
        ax.plot(vh, np.arange(len(log))*dx/1000, 'k--')
        ax.plot(vh + (log[::-1] - min(log)) / 1000, np.arange(len(log))*dx/1000, 'k')
      
    
def get_spectrum_fx(shot, dt, fmax=None):
    shot = np.fft.rfft(shot, n=(shot.shape[-1]*2-1), axis=-1)
    nx, nt = shot.shape       
    fax = np.fft.fftfreq(shot.shape[-1], d=2*dt)
    idx = np.argsort(fax)
    fax, shot = fax[idx], shot[:, idx];
    causal = int(nt // 2);
    fax = fax[-causal:]; shot = shot[:, -causal:]
    fax = fax[fax < (fmax if fmax else np.max(fax))]; 
    n = len(fax);
    fax = fax[:n]; shot = shot[:, :n]
    return shot, fax


def plot_fx(shot, dt, component='real', pclip=1.0, title='', colorbar=False, dx=None, fmax=30, kmax=None,
                   figsize=None, axis=True, plot_only=False, **kwargs):
    """
    dt : temporal sampling [sec]
    dx : spatial sampling [m]
    """
    if isinstance(shot, list):
        shot = np.concatenate(shot, 0)
        
    if dx:
        dx = dx / 1000
    nx, nt = shot.shape
    
    fft2_shot, fax = get_spectrum_fx(shot, dt, fmax)
    print(fft2_shot.shape)
    
    if component == 'imag':
        fft2_shot = np.imag(fft2_shot)
    elif component == 'abs':
        fft2_shot = np.abs(fft2_shot)
    else:
        fft2_shot = np.real(fft2_shot)
        
    if not plot_only:
        shot = fft2_shot.copy()
        
    vlim = pclip * np.max(np.abs(shot)); plt.figure(figsize=figsize if figsize else None); 
    
    if dt or dx:
        kwargs['extent'] = [0, dx * nx if dx else nx, 
                            0, np.max(fax) if dt else nt]
        if dx: 
            plt.xlabel('Offset, km') 
        else: 
            plt.gca().get_xaxis().set_visible(False) 
        if dt: 
            plt.ylabel('Frequency, Hz') 
        else: 
            plt.gca().get_yaxis().set_visible(False)
    kwargs['vmin'] = -vlim if not 'vmin' in kwargs.keys() else kwargs['vmin']
    kwargs['vmax'] = vlim if not 'vmax' in kwargs.keys() else kwargs['vmax']
    plt.imshow(np.flip(shot.T, 0), cmap='RdBu_r', **kwargs); 
    if colorbar: plt.colorbar()
    plt.title(title); plt.axis('auto' if axis else 'off');
    
    
def plot_data3d_slices(data3d, ncols = 15, verbose=0, **kwargs):
    nimgs = data3d.shape[0]
    nrows = int(np.ceil(nimgs / ncols))
    nempty = nrows * ncols - nimgs
    if verbose:
        print(f'Grid of {nrows} rows and {ncols} cols: {nrows} x {ncols}')
        print(f'Append {nempty} empty images')

    _np, _no, _nt = data3d.shape
    extended_pred = np.concatenate([data3d, np.zeros((nempty, _no, _nt))])
    table = [[extended_pred[ncols*irow + icol, ...] for icol in range(ncols)] for irow in range(nrows)]
    plot_nxm(table, figsize=(ncols, nrows), colorbar=False, **kwargs)
    
    
# from https://github.com/audeering/audtorch/blob/master/audtorch/metrics/functional.py
def pearsonr(x, y, batch_first=True):
    assert x.shape == y.shape

    if batch_first:
        dim = -1
    else:
        dim = 0

    centered_x = x - x.mean(axis=dim, keepdims=True)
    centered_y = y - y.mean(axis=dim, keepdims=True)

    covariance = (centered_x * centered_y).sum(axis=dim, keepdims=True)

    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = x.std(axis=dim, keepdims=True)
    y_std = y.std(axis=dim, keepdims=True)

    corr = bessel_corrected_covariance / (x_std * y_std)

    return corr


def concordance_cc(x, y, batch_first=True):
    assert x.shape == y.shape

    if batch_first:
        dim = -1
    else:
        dim = 0

    bessel_correction_term = (x.shape[dim] - 1) / x.shape[dim]

    r = pearsonr(x, y, batch_first)
    x_mean = x.mean(axis=dim, keepdims=True)
    y_mean = y.mean(axis=dim, keepdims=True)
    x_std = x.std(axis=dim, keepdims=True)
    y_std = y.std(axis=dim, keepdims=True)
    ccc = 2 * r * x_std * y_std / (x_std * x_std
                                   + y_std * y_std
                                   + (x_mean - y_mean)
                                   * (x_mean - y_mean)
                                   / bessel_correction_term)
    return ccc

def rms_tracewise(lbl, pred, eps=1e-6):
    diff = lbl - pred
    diff_rms = np.sqrt(np.sum(diff ** 2, axis=-1, keepdims=True))
    lbl_rms = np.sqrt(np.sum(lbl ** 2, axis=-1, keepdims=True))
    return diff_rms / handle_zeros(lbl_rms, eps)


def handle_zeros(t, eps):
    """ Avoid division by zero """
    t[t < eps] = 1.
    return t

def get_rms(t):
    """ Compute Root-Mean-Squared value for a 3D tensor """
    return np.sqrt(np.sum(t ** 2, axis=-1) / t.shape[1])

def get_local_rms(inp, win2, eps_noise):
    """ Compute RMS for entire input image """
    h, w = inp.shape[-2:]
    k = np.zeros_like(inp)

    assert k.shape[-2:] == inp.shape[-2:], 'Shapes of coefficients and data do not match!'

    for irow in range(win2, w - win2):
        k[..., irow] = get_rms(inp[..., irow - win2:irow + win2])

    k[..., :win2] = np.concatenate(win2 * [np.expand_dims(k[..., win2 + 1], -1)], -1)
    k[..., -win2:] = np.concatenate(win2 * [np.expand_dims(k[..., -win2 - 1], -1)], -1)
    k = handle_zeros(k, eps_noise)
    return k

def agc(dat, win=150, amp=1, eps=1e-6):
    """ Automatic Gain Control 
    Args:
        win(int): window size where to balance amplitudes
        target_rms(float): desired amplitude in window
        eps(float): tolerance to noise
    Returns:
        data after AGC, mask used to apply the agc
    """
    win2 = int(win / 2)
    k = get_local_rms(dat, win2, eps)
    return dat / k * amp, k
