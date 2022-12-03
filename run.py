import logging
import argparse
from pathlib import Path
from dat_conversion import make_dark_plane, convert_dat_to_arr
from dat_file_functs import create_directory
import numpy as np
import zarr
from numcodecs import Blosc
import mat73
import logging
from dat_file_functs import Struct, LoadConfig, calc_total_planes
from data_stream_io import map_dats_to_volume, load_dats_for_vol
from tqdm import trange




def main(args):
    # 1: define paths
    # ---------------------------------
    root_dir = Path(args.PathData)
    exp_dir = Path(args.PathExp)
    
    create_directory(parent_path=exp_dir.parent, dir_name=f"{exp_dir.stem}")

    # print out the args
    logging.warning(f"\nROOT DATA PATH\t{root_dir}\nEXPORT PATH\t{exp_dir}")

    # 2: Make dark volume
    # ---------------------------------
    dark_vol_dir = list(sorted(name for name in root_dir.glob("**/*dark*") if name.is_dir()))[-1]
    dark_vol_path = Path(f"{exp_dir}/dark_plane.npy")
    dark_plane = make_dark_plane(dat_dir=dark_vol_dir, export_path=dark_vol_path)
    logging.info(f"Dark volume exported")
    # HR dir path
    # list(sorted((name for name in root_dir.glob("**/") if 
    #              "HR" in str(name) and 
    #              "dark" not in str(name)), 
    #             reverse=True))

    # 3: Split on UV behaviour TODO
    # ---------------------------------
    # infer the behaviour mapping and UV stimulation
    
    # Export each section with volume:frameID
    
    
    # 4: Load the dat files and iterate through each section
    # ---------------------------------
    flyback = args.flyback
    
    dat_dir = list(sorted((name for name in root_dir.glob("**/") if 
                           "HR" not in str(name) and 
                           "dark" not in str(name))))[-2]
    info_path = Path(f"{dat_dir.parent}/{dat_dir.stem}_info.mat")
    assert info_path.exists(), f"invalid info path: {info_path}"
    config_path = Path(f"{dat_dir}/acquisitionmetadata.ini")
    assert config_path.exists(), f"invalid config path: {config_path}"

    
    # load daq info
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
    # instantiate experiment array
    compressor = Blosc(cname='zstd', clevel=args.cLevel, shuffle=Blosc.BITSHUFFLE)
    zarr_filepath = Path(f"{exp_dir}/{root_dir.stem}.zarr")
    z_arr = zarr.open(f'{zarr_filepath}', 
                      mode='w', 
                      shape=(int(daq.numberOfScans),
                             int(daq.pixelsPerLine-flyback),
                             dat_loader.x_crop,
                             dat_loader.y_crop),
                      chunks=(1, None),
                      compressor=compressor,
                      dtype=np.uint16,)
    # add attributes i.e. relative frame ids and vol ids
    # ...
    
    # create a dark vol is background subtraction applied
    dark_vol = np.tile(dark_plane, (int(daq.pixelsPerLine-flyback),1,1)).astype(z_arr.dtype)
    # interate through each camera volume and fill with sliced dat vol
    for i in trange(int(daq.numberOfScans)):
        volume = load_dats_for_vol(dat_loader,
                                     dat_dir, 
                                     file_names, 
                                     dat_vol_arrays[i], 
                                     dat_slice_array[i])[...,:-flyback]
        # reorder indices - t, z, y, x
        volume = volume.transpose(2,0,1)
        # subtract dark vol
        if dark_plane is not None:
            volume+=110
            volume-=dark_vol
            
        z_arr.oindex[i] = volume
    
    vol_rate = daq.pixelrate / daq.pixelsPerLine


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Define the parameters used for converting dat files')
    parser.add_argument('-pD', '--PathData', help="path to imaging directories data", default='.', type=str)
    parser.add_argument('-pE', '--PathExport', help="path to export aggregated data", default='.', type=str)
    parser.add_argument('-c', '--cLevel', help="Level of compression of zarr file." / 
                        "Uses zstd Blosc.BITSHUFFLE compression. Default n = 5.",
                        default=5, type=int)
    parser.add_argument('-f', '--flyback', help="flyback frames used. Default n = 2.",
                        default=2, type=int)
    args = parser.parse_args()
    main(args)
