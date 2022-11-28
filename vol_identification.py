from pathlib import Path
import numpy as np
import zarr
from numcodecs import Blosc
import mat73
import logging
from dat_file_functs import Struct, LoadConfig, calc_total_planes
from data_stream_io import map_dats_to_volume, load_dats_for_vol

# enter dat dir
dat_dir=""
dat_label=None #int

dat_dir=Path(dat_dir)
info_path = Path(f"{dat_dir.parent}/{dat_dir.stem}_info.mat")
config_path = Path(f"{dat_dir}/acquisitionmetadata.ini")

daq = Struct(mat73.loadmat(info_path)).info.daq
# load config file
config=LoadConfig(config_path)
# generate dat file loader
dat_loader = config.get_dat_dimensions()
# load and sort dat spools files
total_planes = calc_total_planes(daq)
file_names = dat_loader.sort_spool_filenames(total_planes)
# prepare slices and dat vols for each camera vol
dat_vol_arrays, dat_slice_array = map_dats_to_volume(daq, dat_loader.n_planes)



filenames = [x[:-10] for x in file_names]
filenames = np.array(filenames)
corrupt_dat_ix = np.where(filenames == 520000000)

for vol_ix, cam_vol in enumerate(dat_vol_arrays):
    for dat_vol in cam_vol:
        if dat_vol == corrupt_dat_ix:
            print(f"Corrupt_volume {vol_ix}")