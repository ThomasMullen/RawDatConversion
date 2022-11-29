from pathlib import Path
import numpy as np
import zarr
from numcodecs import Blosc
import mat73
import logging
from dat_file_functs import Struct, LoadConfig, calc_total_planes
from data_stream_io import map_dats_to_volume, load_dats_for_vol
from tqdm import trange

# function make dark vol
def make_dark_plane(dat_dir:str, export_path:str=None)->None:
    if export_path is None:
        export_path=Path(f"{dat_dir.parent}/{dat_dir.stem}_dark_plane")
    logging.info("Make dark plane")
    z_arr = convert_dat_to_arr(dat_dir)
    dark_plane = np.mean(z_arr, axis=(0,1))
    np.save(export_path, dark_plane)
    return

# function build stack
def convert_dat_to_arr(dat_dir:str, zarr_filepath:str, flyback:int=2, dark_plane_path:str=None, compressor=None)->np.ndarray:
    # parse filepaths
    dat_dir=Path(dat_dir)
    info_path = Path(f"{dat_dir.parent}/{dat_dir.stem}_info.mat")
    config_path = Path(f"{dat_dir}/acquisitionmetadata.ini")
    
    # check if dark plane can be loaded
    if dark_plane_path is not None:
        dark_plane_path = Path(dark_plane_path)
    if dark_plane_path is None or not dark_plane_path.exists():
        dark_plane=None
        logging.warning("No dark plane loaded - background subtraction not applied")
    else:
        dark_plane=np.load(dark_plane_path)
        logging.info("Loaded dark plane loaded")
    
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
    z_arr = zarr.open(f'{zarr_filepath}', 
                      mode='w', 
                      shape=(int(daq.numberOfScans),
                             int(daq.pixelsPerLine-flyback),
                             dat_loader.x_crop,
                             dat_loader.y_crop),
                      chunks=(10, None),
                      compressor=compressor,
                      dtype=np.uint16,)
    # create a dark vol is background subtraction applied
    if dark_plane is not None:
        dark_vol = np.tile(dark_plane, (int(daq.pixelsPerLine-flyback),1,1)).astype(z_arr.dtype)
    # interate through each camera volume and fill with sliced dat vol
    for i in trange(int(daq.numberOfScans)):
        if i > 20:
            break        
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
        # z_arr.set_basic_selection((i,slice(None)), volume)

    return z_arr
    

def main():
    # dark volume to subtract
    # dark_dat_dir=Path("/Volumes/TomMullen/10dpf20221119Fish01/test3/dark_offset_run1_HR")
    export_path=Path(r"H:\10dpf20221119Fish01\test3\dark_offset_run1_HR_dark_plane")
    # make_dark_plane(dat_dir=dark_dat_dir, 
    #                 export_path=export_path)
    
    dat_dir=Path(r"H:\10dpf20221119Fish01\test3\fullrun_run1")
    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
    output_z_arr = convert_dat_to_arr(dat_dir=f"{dat_dir}",
                                      flyback=2, 
                                      dark_plane_path=Path(f"{export_path}.npy"), 
                                      zarr_filepath=r"E:\2022\SCAPETestFullRun.zarr", 
                                      compressor=None)

if __name__ == '__main__':
    

    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
    stats.dump_stats(r'C:\Users\orger\VScodeProjects\RawDatConversion')
    
    
    
