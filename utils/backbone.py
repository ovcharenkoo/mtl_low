import os
import torch
import torch.nn as nn
import pickle
from natsort import natsorted

def get_latest(p):
    files = glob.glob(p)
    return natsorted(files)[-1]


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'use_bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func)


# a complex model consisted of several nets, and each net will be explicitly defined in other py class files
class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def init(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.save_dir = opt.save_folder
        self.model_names = []
        self.running_metrics = {}
        
    def set_input(self, inputData):
        self.input = inputData

    def forward(self):
        pass

    def optimize_parameters(self, train_dataset, val_dataset):
        pass

    def get_current_losses(self):
        pass

    def update_learning_rate(self):
        pass

    def test(self):
        with torch.no_grad():
            self.forward()

    # save models to the disk
    def save_networks(self, which_epoch, subfolder=''):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net%s.pth' % (which_epoch, name)
                save_dir = os.path.join(self.save_dir, subfolder)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if isinstance(net, torch.nn.DataParallel):
                    torch.save(net.module.state_dict(), save_path)
                    print(f'Save DataParallel model to {save_path}')
                else:
                    torch.save(net.state_dict(), save_path)
                    print(f"Save model to {save_path}")

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)


    # load models from the disk
    def load_networks(self, load_path, which_epoch=0):
        for name in self.model_names:
            try:
                print(f'Load weights for {name}')
                if isinstance(name, str):
                    net_name = 'net' + name
                    net = getattr(self, net_name)
                    model_dict = net.state_dict()
                    if isinstance(net, torch.nn.DataParallel):
                        print('Load DataParallel state dict!')
                        net = net.module
                    complete_path = os.path.join(load_path, f'{which_epoch}_{net_name}.pth')
                    print('Loading model from %s' % complete_path)
                    # # if you are using PyTorch newer than 0.4 (e.g., built from
                    # # GitHub source), you can remove str() on self.device
                    state_dict = torch.load(complete_path,
                                            map_location=torch.device('cpu') if not torch.cuda.is_available() else torch.device(0))
                    # # patch InstanceNorm checkpoints prior to 0.4
                    for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                        self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))

                    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}


                    # Copy only weights which match shape. E.g. if a pre-trained model was trained for a 3-channel image,
                    # the loading will fail when applied to a one-channel image. We want to use a pre-trained model on
                    # face-dataset, whereas seismic has only one channel.
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        try:
                            if model_dict[k].shape == v.shape:
                                # print(f'{k}: {model_dict[k].shape}')
                                # print(f'{k}: {v.shape}')
                                # continue
                                new_state_dict[k] = v
                            else:
                                new_state_dict[k] = model_dict[k]
                                print(f'\tEXCLUDE {k}! Loaded {v.shape}, but model expected {model_dict[k].shape}')
                        except:
                            print(f'\tEXCEPTION for key {k} while loading state_dict!')

                    # net.load_state_dict(state_dict)
                    net.load_state_dict(new_state_dict)

                    # try:
                    #     net.module.state_dict(torch.load(load_path))
                    # except AttributeError:
                    #     net.state_dict(torch.load(load_path))
            except FileNotFoundError:
                print(f'Not found {complete_path}!')
                
                
#     def load_history(self, load_dir):
#         hname = os.path.join(load_dir, 'history.pkl')
#         print(f'Loading history from {hname}...')
#         try:
#             with open(hname, 'rb') as handle:
#                 self.running_metrics = pickle.load(handle)
#         except Exception as e:
#             print(f'Failed to load history, {e}')
            
    def load_history(self, load_dir):
        for name in self.model_names:
            hname = os.path.join(load_dir, f'history{name}.pkl')
            print(f'Loading history from {hname}...')
            try:
                metric_name = 'running_metrics' + name
                with open(hname, 'rb') as handle:
                    setattr(self, metric_name, pickle.load(handle))
            except Exception as e:
                print(f'Failed to load history form {hname}, {e}')
                
    def load_lr_history(self, load_dir):
        name = '_lr'
        hname = os.path.join(load_dir, f'history{name}.pkl')
        print(f'Loading history from {hname}...')
        try:
            metric_name = 'running_metrics' + name
            with open(hname, 'rb') as handle:
                setattr(self, metric_name, pickle.load(handle))
        except Exception as e:
            print(f'Failed to load history form {hname}, {e}')

    def save_history(self, save_dir):
        for name in self.model_names:
            hname = os.path.join(save_dir, f'history{name}.pkl')
            print(f'Saving history to {hname}...')
            try:
                metric_name = 'running_metrics' + name
                metric = getattr(self, metric_name)
                with open(hname, 'wb') as handle:
                    pickle.dump(metric, handle)
            except Exception as e:
                print(f'Failed to save history to {hname}, {e}')
    
    def save_lr_history(self, save_dir):
        name = '_lr'
        hname = os.path.join(save_dir, f'history{name}.pkl')
        print(f'Saving history to {hname}...')
        try:
            metric_name = 'running_metrics' + name
            metric = getattr(self, metric_name)
            with open(hname, 'wb') as handle:
                pickle.dump(metric, handle)
        except Exception as e:
            print(f'Failed to save history to {hname}, {e}')

    # print network information
    def print_networks(self, verbose=True):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    # set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

    def init(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.save_dir = opt.checkpoint_dir
        # self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

    def forward(self, *input):
        return super(BaseNet, self).forward(*input)

    def test(self, *input):
        with torch.no_grad():
            self.forward(*input)

    def save_network(self, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(self.cpu().state_dict(), save_path)