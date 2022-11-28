import unittest
from dat_file_functs import LoadConfig
from pathlib import Path

test_file=Path(r"/Users/thomasmullen/RawDatConversion/dummy_data/acquisitionmetadata.ini")

def test_constructed_members():
    config = LoadConfig(test_file)
    assert config.aoi_height == 350
    assert config.aoi_width == 500
    assert config.aoi_stride == 1040, "Check calculate_expected_stride() funct"
    assert config.image_size_bytes == 366080
    assert config.pixel_encoding == 'Mono16'
    assert config.is_16_bit == True
    assert config.planes_per_datfile == 22
    
def test_extra_rows():
    config = LoadConfig(test_file)
    assert config.calculate_extra_rows() == 2
    assert isinstance(config.calculate_extra_rows(), int)

def test_calc_padded_rows():
    config = LoadConfig(test_file)
    assert config.calculate_padded_rows() == 40
    assert isinstance(config.calculate_extra_rows(), int)

    
def test_image_byte_dimensions():
    config = LoadConfig(test_file)
    extra_rows = config.calculate_extra_rows()
    assert extra_rows == 2
    n_rows, n_cols = config.calculate_image_byte_dimension(extra_rows)
    assert n_cols == 520
    assert n_rows == 352
    assert isinstance(n_cols, int)
    assert isinstance(n_rows, int)
    
def test_image_crop():
    config = LoadConfig(test_file)
    x_crop, y_crop = config.calculate_image_crop()
    assert x_crop == 500
    assert y_crop == 349
    assert isinstance(x_crop, int)
    assert isinstance(y_crop, int)
    
def test_get_dat_dims():
    config = LoadConfig(test_file)
    dat_dims = config.get_dat_dimensions()
    assert dat_dims.n_planes==22 
    assert dat_dims.n_rows==352
    assert dat_dims.n_cols==520
    assert dat_dims.x_crop==500
    assert dat_dims.y_crop==349
    