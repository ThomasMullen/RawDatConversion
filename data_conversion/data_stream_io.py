from pathlib import Path
import numpy as np

from data_conversion.dat_file_functs import Struct, find_dat_slice_ix, DatLoader


def map_dats_to_volume(daq: Struct, planes_per_dat:int):
    """
    produces an iterative list which contains the dat indices for each volume and there corresponding slices
    :param planes_per_vol:
    :param planes_per_dat:
    :param n_scans:
    :return:
    """
    # list plane indices for the defined volume
    planes_per_vol = int(daq.pixelsPerLine)
    vol_plane_ix = [np.arange(planes_per_vol) + (vol_ix * planes_per_vol) for vol_ix in range(int(daq.numberOfScans))]
    # assign a dat file number to each plane
    spool_file_needed_for_plane = [np.floor(plane_ix / planes_per_dat).astype(int) for plane_ix in vol_plane_ix]
    get_file_and_planes_ixs = (np.unique(x, return_index=True) for x in spool_file_needed_for_plane)
    get_file_and_planes = ((y[0], find_dat_slice_ix(np.diff(np.concatenate([y[1], [planes_per_vol]])), planes_per_dat))
                           for y in get_file_and_planes_ixs)
    dat_vol, dat_slice = zip(*list(get_file_and_planes))
    return dat_vol, dat_slice


def load_dats_for_vol(dat_loader:DatLoader, dat_dir:str, dat_filenames, dat_ixs, dat_slices)->np.ndarray:
    """load set of dat files to compose a volume.
    loads a image volume from all of the dat files and dat slices required to be loaded

    Args:
        dat_loader (DatLoader): dat loader object
        dat_dir (str): directory pointing to dat files
        dat_filenames (list): list of dat file names
        dat_ixs (list): dat indixes used for volume
        dat_slices (tuple): slice of reshaped dat file used for the volume

    Returns:
        np.ndarray: volume of array
    """
    return np.concatenate([dat_loader.load_dat_file(Path(f"{dat_dir}",f"{dat_filenames[datix]}"))[..., plane_slices]
                           for datix, plane_slices in zip(dat_ixs, dat_slices)], axis=2)
    
    

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # save image files directory
    dump_dir = Path(f"/Users/thomasmullen/Desktop/dump")
    data_dir = Path(f"/Volumes/TomsWork/SCAPE/Data/2022-05-17/Exp1/2022-05-17-huc-7dpf-f1_run1")
