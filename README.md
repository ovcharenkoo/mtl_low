
# Multi-task learning for low-frequency extrapolation and elastic model building from seismic data


by [Ovcharenko Oleg](https://ovcharenkoo.com/), [Vladimir Kazei](https://vkazei.com/), [Tariq Alkhalifah](https://sites.google.com/a/kaust.edu.sa/tariq/home) and [Daniel Peter](https://github.com/danielpeter). This repository contains the general workflow and experiments for our [paper](https://google.com/).


Note that examples below include **experiments on synthetic data only** since posting field data requires special permissions.

![workflow](./assets/arch.png)

## How to start
Download the data and run notebooks. All notebooks are set for inference / view by default. Meaning that these will not run any heavy calculations unless reset otherwise. Instead, these will use the pre-trained weights and data to partially reproduce results from the paper. The `Need extra` column indicates whether you need to compile or install third-party software to properly run these notebooks (check dependencies section for details).


| Filename | Need extra | Description |
| -------- | ----- | ---- |
| ex0_create_training_dataset.ipynb | Yes | Generate training dataset of synthetic waveforms | 
| ex1_unet_l.ipynb | | Train UNet to predict low-frequency data | 
| ex2_multi_l.ipynb | | Train Multi-column network to predict low-frequency data |
| ex3_multi_lm.ipynb |  | Train Multi-column network to predict low-frequency data and local subsurface model|
| ex4_multi_lc.ipynb | | Train Multi-column network to predict low-frequency data using extra trace-wise correlation loss term | 
| ex5_multi_lcm.ipynb | | Train Multi-column network to predict low-frequency data and local subsurface model. Also use the trace-wise correlation loss to fit the data |
| ex6_fwi_marmousi_with_predicted.ipynb | Yes | Run / view full-waveform inversion from predicted low-frequency data and initial subsurface model| 
| ex7_fwi_marmousi_without_predicted.ipynb | Yes | Run / view full-waveform inversion from bandlimited data and poor initial model |
| ex8_make_pictures.ipynb | | Compare all trained networks and make key figures | 
| shared_data_loading.ipynb | | This notebook is called by ex1-5|
| assets | | Folder with images for README| 
| pretrained_files | | Download and place pre-trained data here | 
| utils |  | Code components | 

### Prerequsites and dependencies
* Python 3.8
* PyTorch 1.8
* CUDA 11.0

For the rest of Python dependencies check `requirements.txt`.

To run FWI and data generation notebooks on your machine (examples 0, 6 and 7) you would need to download and compile [DENISE-Black-Edition](https://github.com/daniel-koehn/DENISE-Black-Edition) software for numerical wave propagation, followed by changing relevant paths in aforementioned notebooks. Moreover, for generation of the training dataset from scratch, you would need to install [Madagascar](https://github.com/ahay/src) software for seismic data processing.

### Installation
```
git clone https://github.com/ovcharenkoo/mtl_low.git
cd mtl_low/
python -m venv env
source env/bin/activate
pip install -r requirements.txt

jupyter notebook .
```

### Downloads
Unzip files by running `tar -xvf arhive.tar.gz` and place complete folders according to the table

| Link | Size | Destination | Description
| ---- | -----| ------------| ----------- |
| [data.tar.gz](https://www.dropbox.com/s/58zckalcm6wlp06/data.tar.gz?dl=1) | ~ 13 Gb | `./pretrained_files/data/*` | training and validation datasets
| [trained_nets.tar.gz](https://www.dropbox.com/s/a8wvncp86iiob0d/trained_nets.tar.gz?dl=1) | ~ 300 Mb| `./pretrained_files/trained_nets/*` | Pre-trained network weights
| [fwi_outputs.tar.gz](https://www.dropbox.com/s/jpnb18j62jqrs22/fwi_outputs.tar.gz?dl=1) | ~ 10 Mb | `./pretrained_files/fwi_outputs/*` | Inverted subsurface models etc.

## Acknowledgments
Our implementation is heavily influenced and contains code blocks from [Inpainting GMCNN](https://github.com/shepnerd/inpainting_gmcnn).

## Contact
oleg.ovcharenko@kaust.edu.sa