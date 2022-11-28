import unittest
from dat_file_functs import LoadConfig
from pathlib import Path
import numpy as np
import mat73
from dat_file_functs import Struct
from data_stream_io import map_dats_to_volume, load_dats_for_vol

mat_file=Path(r"/Users/thomasmullen/RawDatConversion/dummy_data/exp_info.mat")
dat_path = Path("/Users/thomasmullen/RawDatConversion/dummy_data/raw_dats")
config_path = Path("/Users/thomasmullen/RawDatConversion/dummy_data/acquisitionmetadata.ini")
info_path = Path("/Users/thomasmullen/RawDatConversion/dummy_data/exp_info.mat")

dark_img = Struct(mat73.loadmat("/Users/thomasmullen/RawDatConversion/dummy_data/processed/dataVolMean.mat"))
dark_vol = Struct(mat73.loadmat("/Users/thomasmullen/RawDatConversion/dummy_data/processed/datanoSC.mat"))
single_plane_mat = dark_vol.imagedata[0]
a = Struct(mat73.loadmat("/Users/thomasmullen/RawDatConversion/dummy_data/processed/a_output.mat"))
datfile_list = np.apply_along_axis(lambda row: row.astype('|S1').tostring().decode('utf-8'),
                    axis=1,
                    arr=a.a)


def test_spooled_filenames():
    config=LoadConfig(config_path)
    dat_loader = config.get_dat_dimensions()
    total_planes=1001
    file_names = dat_loader.sort_spool_filenames(total_planes)
    file_names = np.array([x[:-9] for x in file_names])
    np.testing.assert_array_equal(file_names, np.insert(datfile_list, 0, '0000000000'))

def test_load_dat():
    config=LoadConfig(config_path)
    dat_loader = config.get_dat_dimensions()
    
    true_dat0_path=Path('/Users/thomasmullen/RawDatConversion/dummy_data/processed/datCrop0.mat')
    true_dat0 = Struct(mat73.loadmat(true_dat0_path)).datCrop0.astype(int)
    true_dat1_path=Path('/Users/thomasmullen/RawDatConversion/dummy_data/processed/datCrop1.mat')
    true_dat1 = Struct(mat73.loadmat(true_dat1_path)).datCrop1.astype(int)
    
    dat_arr0=dat_loader.load_dat_file(f"{dat_path}/{'0000000000spool.dat'}")
    dat_arr1=dat_loader.load_dat_file(f"{dat_path}/{'1000000000spool.dat'}")
    
    np.testing.assert_array_equal(dat_arr0, true_dat0)
    np.testing.assert_array_equal(dat_arr1, true_dat1)
 
 
def test_map_dats_to_volume():
    config=LoadConfig(config_path)
    dat_loader = config.get_dat_dimensions()
    
    info = Struct(mat73.loadmat(info_path))
    daq = info.info.daq
    
    dat_vol_arrays, dat_slice_array = map_dats_to_volume(daq, dat_loader.n_planes) 
    
    images_needed = Path('/Users/thomasmullen/RawDatConversion/dummy_data/processed/imagesNeeded.mat')
    images_needed = Struct(mat73.loadmat(images_needed)).imagesNeeded.astype(int)
    first_img, last_img = images_needed[0]-1, images_needed[-1]
    assert last_img == list(dat_slice_array[0])[-1].stop
    assert first_img == list(dat_slice_array[0])[0].start
    
    
def test_load_volume():
    true_vol0_path=Path('/Users/thomasmullen/RawDatConversion/dummy_data/processed/datanoSC.mat')
    true_vol0 = Struct(mat73.loadmat(true_vol0_path)).imagedata[0].astype(int)
    
    info = Struct(mat73.loadmat(info_path))
    daq = info.info.daq
    
    flyback=2
    config=LoadConfig(config_path)    
    dat_loader = config.get_dat_dimensions()    
    # n_dats_per_vol = dat_loader.calc_number_dats_per_vol(daq)
    total_planes = 1001
    # spool_file_needed = dat_loader.calc_spool_files_needed(daq)
    file_names = dat_loader.sort_spool_filenames(total_planes)
    
    dat_vol_arrays, dat_slice_array = map_dats_to_volume(daq, 
                                                         dat_loader.n_planes)

    
    vol = load_dats_for_vol(dat_loader,
                                     dat_path, 
                                     file_names, 
                                     dat_vol_arrays[0], 
                                     dat_slice_array[0])[...,:-flyback]
    vol = vol.transpose(2,0,1)
    np.testing.assert_array_equal(vol, true_vol0)