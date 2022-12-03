import configparser
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import logging


def create_directory(parent_path, dir_name=''):
    parent_path = Path(f"{parent_path}")
    # make export directory
    dir_path = parent_path.joinpath(dir_name)
    # if already exists return directory path file
    if dir_path.exists():
        logging.warning(f"Directory '{dir_path}' already exists.")
        return dir_path
    # otherwise make directory
    dir_path.mkdir()
    logging.warning(f"Directory '{dir_path}' created.")
    return dir_path

class Struct(object):
    """
    Converts nested dictionary (e.g. .mat file) into an object-like structure. Pass in dictionary
    """

    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b , (list, tuple)):
                setattr(self, a, [Struct(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, Struct(b) if isinstance(b, dict) else b)


@dataclass
class DatDimension:
    """
    Contain the volume and crop dimensions of a dat file
    """
    n_planes: int
    n_rows: int
    n_cols: int
    x_crop: int
    y_crop: int


class DatLoader:
    
    def __init__(self, planes_per_datfile, n_rows, n_cols, x_crop, y_crop) -> None:
        self.n_planes = int(planes_per_datfile)
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.x_crop = int(x_crop)
        self.y_crop = int(y_crop)
        
    def read_dat_file(self, filepath:str):
        """
        read in data file into a numpy array.
        :param filepath: of the .dat
        :return: 1d raw numpy array of the dat file
        """
        with open(f"{filepath}", "r") as f:
            return np.fromfile(f, dtype=np.uint16)
        
    def reshape_dat(self, arr:np.ndarray)->np.ndarray:
        """reshape dat file to fit that of scape volume

        Args:
            arr (np.ndarray): 1D dat array

        Returns:
            np.ndarray: reshape dat array
            
        Example:
        >>>> ini_filepath = f"{data_dir}/acquisitionmetadata.ini"
        >>>> acquisition_config = LoadConfig(ini_filepath)
        >>>> planes_per_datfile, n_rows, n_cols = acquisition_config.planes_per_datfile,
        >>>>                                   acquisition_config.calculate_image_byte_dimension(acquisition_config.extra_rows())
        >>>> reshape_dat(timeseries, n_cols, n_rows, planes_per_datfile)
        """
        # note: reshape with Fortran order to meet that of matlab code
        return arr.reshape((self.n_cols, self.n_rows, self.n_planes), order='F')

    def crop_dat_vol(self, arr:np.ndarray):
        """
        Crops the data array defined by x_crop and y_crop, in order to remove padding from the image files.
        :return: cropped 3D data array (x, y, z)

        **Note:** data array dimension is by default defined as (x, y, z). It will only follow the pipeline standard array
        order of (z, y, x) after the .zarr file has been exported.

        tempvol(:,:,find(spoolsNeeded==file2load))=SCAPE_rawdata(Xcrop(1):Xcrop(2),Ycrop(1):Ycrop(2) ,imagesNeeded(find(spoolsNeeded==file2load)));
        """
        return arr[:self.x_crop, :self.y_crop]
    
    def export_dot_vol(self, filepath:str)->None:
        pass
    
    def load_dat_file(self, dat_filepath:str):
        """
        read in .dat file and crop it's dimensions using config file
        :param dat_filepath: filepath of .dat file
        :param acqusition_config_struct: the read in acquisition ini struct
        :return: np array of .dat file
        """
        dat_arr = self.read_dat_file(dat_filepath)
        dat_arr = self.reshape_dat(dat_arr)
        dat_arr = self.crop_dat_vol(dat_arr)
        return dat_arr
    
    def calc_number_dats_per_vol(self, daq:Struct)->int:
        """calculate the number of dat files required for each volume

        Args:
            dat_dim (DatDimension): dimensions of dat file based on the Config file
            daq (Struct): daq struct from matlab info file

        Returns:
            int: number of dats per volume
        """
        planes_per_vol=daq.pixelsPerLine.astype(int)
        return int(np.ceil(planes_per_vol/self.n_planes))
    
    def calc_spool_files_needed(self, daq:Struct)->int:
        """calculate the number of dat files need for each volume

        Args:
            daq (Struct): daq struct from matlab info shee
            dat_dim (DatDimension): dat dimenions from load ini config file

        Returns:
            int: number of spool files
        """
        total_planes = calc_total_planes(daq)
        return round(total_planes / self.n_planes)
    
    def sort_spool_filenames(self, total_planes:int):
        r"""
        The format of the filenames are numerically labelled in order but the id is reversed. This function returns the
        labels in a sorted array
        Note: need to generalise this for more than 10 digits, for now this will work.
        max_digits is set to 10 `{i:010}`.

        :maths: \mathbf{A} = \left[\begin{array}{ccc} 1 & 3 & 5\\ 2 & 5 & 1\\ 2 & 3 & 8\end{array}\right].

        [1, 0, 0, 0, 0, 0, 0, 0, 0]
        [2, 0, 0, 0, 0, 0, 0, 0, 0]
        [3, 0, 0, 0, 0, 0, 0, 0, 0]
        [4, 0, 0, 0, 0, 0, 0, 0, 0]
        [5, 0, 0, 0, 0, 0, 0, 0, 0]
        [6, 0, 0, 0, 0, 0, 0, 0, 0]
        [7, 0, 0, 0, 0, 0, 0, 0, 0]
        [8, 0, 0, 0, 0, 0, 0, 0, 0]
        [9, 0, 0, 0, 0, 0, 0, 0, 0]
        [0, 1, 0, 0, 0, 0, 0, 0, 0]
        [1, 1, 0, 0, 0, 0, 0, 0, 0]
        [2, 1, 0, 0, 0, 0, 0, 0, 0]
        [3, 1, 0, 0, 0, 0, 0, 0, 0]

        :return:
        """
        n_spool_files = int(np.ceil(total_planes / self.n_planes))
        file_names = [str(f'{i:010}')[::-1] + "spool.dat" for i in range(n_spool_files)]
        return file_names

def calc_total_planes(daq:Struct)->int:
        """calculate the total number of planes used for entire experiment

        Args:
            daq (Struct): daq struct from matlab info file

        Returns:
            int: number of planes
        """
        n_scans=daq.numberOfScans.astype(int)
        planes_per_vol=daq.pixelsPerLine.astype(int)
        return int(n_scans * planes_per_vol)


class LoadConfig:
    """
    load parameters to structure from the data acquisition `.ini` file.
    """

    def __init__(self, filepath):
        read_config = configparser.ConfigParser()
        read_config.read(filepath, encoding='utf-8-sig')
        self.path = filepath
        self.print_config_sections()
        self.aoi_height = read_config.getint("data", 'aoiheight')
        self.aoi_width = read_config.getint("data", 'aoiwidth')
        self.aoi_stride = read_config.getint("data", 'aoistride')
        self.image_size_bytes = read_config.getint("data", 'imagesizebytes')
        self.pixel_encoding = read_config.get("data", 'pixelencoding')
        self.is_16_bit = (self.pixel_encoding.lower() == 'mono16')
        self.planes_per_datfile = read_config.getint("multiimage", 'imagesperfile')
        self.calculate_expected_stride()

    def print_config_sections(self):
        """Print out the sections in the config file - should have 'data'"""
        read_config = configparser.ConfigParser()
        read_config.read(self.path, encoding='utf-8-sig')
        print(read_config.sections())

    def print_data(self):
        """Print out the list of feilds in the .ini file"""
        read_config = configparser.ConfigParser()
        read_config.read(self.path, encoding='utf-8-sig')
        [print(i) for i in read_config.items('data')]

    def calculate_expected_stride(self):
        """Calculate the expected stride depending on the bit encoding"""
        if self.is_16_bit:
            self.expected_stride = (self.aoi_width * 2)
        else:
            self.expected_stride = (self.aoi_width * 3 / 2)

    def calculate_image_byte_dimension(self, extra_rows):
        """
        finds the actual number of rows and columns in the data saved by the cameera.
        :param extra_rows:
        :param image_size_bytes: parameter found in the image acqDataStruct.data.imagesizebytes
        :return: (number of rows, number of columns)
        """
        # self.planes_per_datfile
        n_rows = self.aoi_height + extra_rows
        n_cols = int(self.image_size_bytes / (2 * n_rows))
        if n_cols != self.aoi_stride / 2:
            RuntimeError("Something wrong in numberColumns calculation")
        return n_rows, n_cols

    def calculate_padded_rows(self)->int:
        """2 rows always padded at the end."""
        return self.aoi_stride - self.expected_stride

    def calculate_extra_rows(self)->int:
        extra_rows = int(self.image_size_bytes / self.aoi_stride) - self.aoi_height
        return extra_rows

    def calculate_image_crop(self):
        """
        As the camera pads the images, this gives you the region to crop for the real data.
        :return: x_crop, y_crop
        """
        row_pad_bytes = self.calculate_padded_rows()
        extra_rows = self.calculate_extra_rows()
        n_row, n_col = self.calculate_image_byte_dimension(extra_rows)

        x_crop = n_col - row_pad_bytes / 2
        y_crop = n_row - extra_rows - 1
        return int(x_crop), int(y_crop)

    def get_dat_dimensions(self) -> DatLoader:
        x_crop, y_crop = self.calculate_image_crop()
        n_rows, n_cols = self.calculate_image_byte_dimension(self.calculate_extra_rows())
        return DatLoader(self.planes_per_datfile, n_rows, n_cols, x_crop, y_crop)





def read_dat_file(filepath):
    """
    read in data file into a numpy array.
    :param filepath: of the .dat
    :return: 1d raw numpy array of the dat file
    """
    with open(f"{filepath}", "r") as f:
        return np.fromfile(f, dtype=np.uint16)


def reshape_dat(dat_arr, n_cols, n_rows, images_per_file):
    """

    :param dat_arr:
    :param n_cols:
    :param n_rows:
    :param images_per_file:
    :return:

    Example:
    --------

    >>>> ini_filepath = f"{data_dir}/acquisitionmetadata.ini"
    >>>> acquisition_config = LoadConfig(ini_filepath)
    >>>> planes_per_datfile, n_rows, n_cols = acquisition_config.planes_per_datfile,
    >>>>                                   acquisition_config.calculate_image_byte_dimension(acquisition_config.extra_rows())
    >>>> reshape_dat(timeseries, n_cols, n_rows, planes_per_datfile)
    """

    # note: change index order, permute i.e. x.transpose(3,1,2)

    return dat_arr.reshape((n_cols, n_rows, images_per_file))


def crop_dat_vol(dat_arr, x_crop, y_crop):
    """
    Crops the data array defined by x_crop and y_crop, in order to remove padding from the image files.
    :return: cropped 3D data array (x, y, z)

    **Note:** data array dimension is by default defined as (x, y, z). It will only follow the pipeline standard array
    order of (z, y, x) after the .zarr file has been exported.

    tempvol(:,:,find(spoolsNeeded==file2load))=SCAPE_rawdata(Xcrop(1):Xcrop(2),Ycrop(1):Ycrop(2) ,imagesNeeded(find(spoolsNeeded==file2load)));
    """
    return dat_arr[:x_crop, :y_crop]


def load_dat_file(dat_filepath, dat_dimensions: DatDimension):
    """
    read in .dat file and crop it's dimensions using config file
    :param dat_filepath: filepath of .dat file
    :param acqusition_config_struct: the read in acquisition ini struct
    :return: np array of .dat file
    """
    dat_arr = read_dat_file(dat_filepath)
    dat_arr = reshape_dat(dat_arr, dat_dimensions.n_cols, dat_dimensions.n_rows, dat_dimensions.n_planes)
    dat_arr = crop_dat_vol(dat_arr, dat_dimensions.x_crop, dat_dimensions.y_crop)
    return dat_arr


def skew_correct_conversion_factors(experiment_info: Struct, planes_per_vol: int) -> list:
    r"""
    returns the dimensions of the array for a skewedv volume depending on the angle of the light-sheet plane.

    :param: experiment_info: struct object of the experiment info file
    :param: planes_per_vol: number of planes per a volume
    :return: dimenions of the z, y, x axes.

    Example:
    --------
    >>> info_file = f"{parent_dir}/20210805_ECLAV3_6dpf_Fish2_run1_info.mat"
    >>> info = Struct(mat73.loadmat(info_file))
    >>> zyx_shape = skew_correct_conversion_factors(info, 120)
    """

    scan_angle = experiment_info.info.daq.scanAngle
    x_width = experiment_info.info.GUIcalFactors.xK_umPerVolt * scan_angle / planes_per_vol
    conversion_factors = [experiment_info.info.GUIcalFactors.y_umPerPix, experiment_info.info.GUIcalFactors.z_umPerPix,
                          x_width]
    return conversion_factors


def find_dat_slice_ix(dat_len_arr, plane_p_dat):
    """
    the number of planes in the dat file that will contribute to the imaging volume
    [7, 34, 34, 34, 34, 20]
    to find which slice of the dat file depends on where the dat file is pulled from i.e., first dat file will be
    the last i planes whereas the last dat file will be the first dat planes.

    :param dat_len_arr: number of planes in the dat files corresponding to the image volume
    :param plane_p_dat: planes per a dat file
    :return: array of slices corresponding to the start and end of the dat file.
    """
    # all slices start from the first dat plane
    start_ix = np.zeros_like(np.array(dat_len_arr))
    # first dat volume will start from the remaining left over
    start_ix[0] = plane_p_dat - dat_len_arr[0]
    # all slices end depending on it's length
    end_ix = np.array(dat_len_arr)
    # first dat file end at the maximum
    end_ix[0] = plane_p_dat
    return np.array([slice(start, end) for start, end in zip(start_ix, end_ix)])
