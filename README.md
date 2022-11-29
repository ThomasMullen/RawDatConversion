Background
----------
Image data is acquired from the camera as a continuous stream of bytes. These bytes as stored into .dat files. To reconstruct the 1D array into a formatted structure requires a `.ini` configure file to describe the original image shape and the bit-type
and also an `info.mat` file.

To construct the raw `.dat` files into the 4D dataset (t, z, y, x) we need to know the configuration if the DAQ board and the experiment info file. This is wrapped in two separate classes: `Struct` and `LoadConfig` which is found in [`dat_file_functs.py`](./dat_file_functs.py).

## Load experiment info file
```python
import mat73
from pathlib import Path
dat_dir=Path(r"home/10dpf20221119Fish01/quickrun_run1")
info_path = Path(f"{dat_dir.parent}/{dat_dir.stem}_info.mat")
daq = Struct(mat73.loadmat(info_path)).info.daq
```
>Note: Parameteras about the camera config can also be attained by loading the `info_path` e.g. 
    ```
    camera = Struct(mat73.loadmat(info_path)).info.camera
    ```

## Load `.ini` config file
This has attributes describing the `aoi_height`, `aoiwidth`, `aoistride`, padding, `imagesizebytes` and pixel encoding in order to reshape and separate bit stream into relevant `.dat` shape.
```python
import mat73
from pathlib import Path
dat_dir=Path(r"home/10dpf20221119Fish01/quickrun_run1")
config_path = Path(f"{dat_dir}/acquisitionmetadata.ini")
config=LoadConfig(config_path)
```

You can print out sections of the `.ini` file:
```python
config.print_config_sections()
```
or `data` variables:
```python
config.print_data()
```

### Create `DatLoader` object
To load `.dat` file and write them to arrays you need to create a `DatLoader` class (see [`dat_file_functs.py`](./dat_file_functs.py)) which is constructed by the dimensions of the `.dat` file and crop dimensions.

This can automatically be build by calling `get_dat_dimensions()` from `LoadConfig` class:
```python
import mat73
from pathlib import Path
dat_dir=Path(r"home/10dpf20221119Fish01/quickrun_run1")
config_path = Path(f"{dat_dir}/acquisitionmetadata.ini")
config=LoadConfig(config_path)
dat_loader = config.get_dat_dimensions()
```

## `DatLoader`
Each .dat file will encode M number of planes, and there will be N number of planes that completes 1 volume (N > M). You
will need multiple .dat files to recover a full volume, and these .dat files may rollover to the next volume.

>Get sorted list of `.dat` filenames`info_path` e.g. 
    ```
    dat_loader.sort_spool_filenames(total_planes)
    ```

Each camera volume is composed of a set of `.dat` files, but the number of dat array per a volume may not factorise. Therefore, we need to specify which dat file and corresponding slices compose a camera volume. This is calculated in function `map_dats_to_volume()`:

```python
dat_vol_arrays, dat_slice_array = map_dats_to_volume(daq, dat_loader.n_planes)
```

This returns a list with each element corresponding to a camera volume which has an array of `.dat` file indexes and slices.