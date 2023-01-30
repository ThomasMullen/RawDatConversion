import sys
import logging
import argparse
from pathlib import Path
import numpy as np
import zarr
from numcodecs import Blosc
import mat73
from tqdm import trange
from skimage import io as skio
import cProfile, pstats
import io
import itertools
from bg_space import AnatomicalSpace
#  check image layouts
from skimage.measure import block_reduce
from skimage.registration import phase_cross_correlation
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

from data_conversion.dat_conversion import make_dark_plane, convert_dat_to_arr
from data_conversion.dat_file_functs import Struct, LoadConfig, DatLoader, calc_total_planes, create_directory, get_pixel_space_calibration
from data_conversion.data_stream_io import map_dats_to_volume, load_dats_for_vol
from tracking.merge_stim import merge_tracking


def load_single_vol(vol_id:int, dat_loader:DatLoader, dat_dir:str, file_names:list[str], dat_vol_arrays: tuple, dat_slice_array:tuple, daq:Struct, flyback:int, dark_plane:np.ndarray=None)->np.ndarray:
    """load a single volume

    Args:
        vol_id (int): volume index wantto pull out from the experiment
        dat_loader (DatLoader): dat lodaer class
        dat_dir (str): directory with dat files
        file_names (list[str]): file names of the dat files
        dat_vol_arrays (tuple): list with the dat volumes that compose a volumes
        dat_slice_array (tuple): slices of each dat volume used to make the image volume
        daq (Struct): daq structure
        flyback (int): flyback frames used
        dark_plane (np.ndarray, optional): dark plane imaged. Defaults to None.

    Returns:
        np.ndarry: volume
    """
    volume = load_dats_for_vol(dat_loader,
                                        dat_dir, 
                                        file_names, 
                                        dat_vol_arrays[vol_id], 
                                        dat_slice_array[vol_id])[...,:-flyback]
    volume = volume.transpose(2,0,1)
    if dark_plane is None:
        return volume
    
    # subtract dark vol
    dark_vol = np.tile(dark_plane, (int(daq.pixelsPerLine-flyback),1,1)).astype(volume.dtype)
    volume+=110
    volume-=dark_vol
    return volume


def downsample_hr_vol(hr_info_path:str, lr_info_path:str, lr_vol:np.ndarray, hr_vol:np.ndarray)->np.ndarray:
    """downsample high resolution vol to match low resoluion vol

    Args:
        hr_info_path (str): info mat file for high res
        lr_info_path (str): info mat file for low res
        lr_vol (np.ndarray): low res volume
        hr_vol (np.ndarray): high res volum

    Returns:
        np.ndarray: down sampled high res volume to match low res volume
    """
    print(lr_vol.shape, hr_vol.shape)
    # change to low space
    hr_converter = get_pixel_space_calibration(hr_info_path)
    lr_converter = get_pixel_space_calibration(lr_info_path)
    hr_space = AnatomicalSpace("pli", resolution=(hr_converter.x_per_pix, hr_converter.y_per_pix, hr_converter.z_per_pix))
    lr_space = AnatomicalSpace("pli", resolution=(hr_vol.shape[0]/lr_vol.shape[0], lr_converter.y_per_pix, lr_converter.z_per_pix))
    downsampled_vol = hr_space.map_stack_to(lr_space, hr_vol)
    print(lr_vol.shape, downsampled_vol.shape)
    return downsampled_vol


if __name__ == "__main__":
    # Establish dirs
    root_dir = Path(r"G:\20221204\h2bhuc_f01_4dpf")
    tracking_dir = Path(r"G:\20221204\h2bhuc_f01_4dpf\tracking\h2bhuc_f01_4dpf")
    exp_dir = Path(r"D:\dump\h2bhuc_f01_4dpf")
    
    # 1: define paths
    # ---------------------------------
    args=argparse.Namespace(PathData=r"G:\20221204\h2bhuc_f01_4dpf",
                        PathTracking=r"G:\20221204\h2bhuc_f01_4dpf\tracking\h2bhuc_f01_4dpf",
                        PathExport=r"D:\dump\h2bhuc_f01_4dpf",
                        cLevel=5,
                        flyback=2,
                        UVPad=2,
                        preStim=[300, 300, 300],
                        postStim=[300, 300, 600],
                        SubtractDarkVol=1
                        )
      
    # define log file path
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fhandler = logging.FileHandler(filename=Path(f"{exp_dir.parent}",f"{exp_dir.stem}.log"), mode='a')
    logger.addHandler(fhandler)
    logging.info('Initiate log file')
    formatter = logging.Formatter('%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s')
    fhandler.setFormatter(formatter)
    # Arguments used
    logging.info(args)
    
    # Argument Checks
    assert len(args.preStim)==len(args.postStim), f"mismatch with {len(args.preStim)} pre-stimulus and {len(args.postStim)} post-stimulus periods"
    pre_ts = args.preStim
    post_ts = args.postStim
    
    create_directory(parent_path=exp_dir.parent, dir_name=f"{exp_dir.stem}")

    # print out the args
    logging.info(f"\nROOT DATA PATH\t{root_dir}\nEXPORT PATH\t{exp_dir}")


    # 3: Split on UV behaviour
    # ---------------------------------
    # infer the behaviour mapping and UV stimulation
    tail_df, stim_onset, stim_offset = merge_tracking(tracking_dir, exp_dir)
    assert stim_offset.shape[0] == len(post_ts), "Number of detected stimuli need to match the number of periods"
    
    # Export each section with volume:frameID
    
    
    # 4: Load the dat files and iterate through each section
    # ---------------------------------
    flyback = args.flyback
    
    
    # 2: Make dark volume
    # ---------------------------------
    dark_vol_dir = list(sorted(name for name in root_dir.glob("*dark*") if name.is_dir()))[-1]
    dark_vol_path = Path(f"{exp_dir}","dark_plane.npy")
    logging.info(f"\nDARK PLANE PATH\t{dark_vol_path}")
    dark_plane = make_dark_plane(dat_dir=dark_vol_dir, export_path=dark_vol_path)
    logging.info(f"Dark volume exported")
    
    # 2: High Res Volume
    # ---------------------------------
    # HR dir path
    hr_dat_dir = list(sorted((name for name in root_dir.glob("**/") if 
                 "HR" in str(name) and 
                 "dark" not in str(name)), 
                reverse=True))[-1]
    hr_info_path = Path(f"{hr_dat_dir}_info.mat")
    hr_config_path = Path(f"{hr_dat_dir}",f"acquisitionmetadata.ini")
    
    # load high res stack
    # load daq info
    daq = Struct(mat73.loadmat(hr_info_path)).info.daq
    # load config file
    config=LoadConfig(hr_config_path)
    # generate dat file loader
    dat_loader = config.get_dat_dimensions()
    # load and sort dat spools files
    total_planes = calc_total_planes(daq)
    file_names = dat_loader.sort_spool_filenames(total_planes)
    # prepare slices and dat vols for each camera vol
    dat_vol_arrays, dat_slice_array = map_dats_to_volume(daq, dat_loader.n_planes)
    hr_shape=(int(daq.numberOfScans),
            int(daq.pixelsPerLine-flyback),
            dat_loader.x_crop,
            dat_loader.y_crop)
    
    # load high resolution volume
    hr_volume = load_single_vol(0, 
                        dat_loader, 
                        hr_dat_dir, 
                        file_names, 
                        dat_vol_arrays, 
                        dat_slice_array, 
                        daq, 
                        flyback, 
                        dark_plane)
    
    # save the high resolution volume as a tiff
    skio.imsave(Path(f"{exp_dir}","high_resolution.tif"), hr_volume.astype(np.uint16))
    
    
    # 2: Load Low Res Volume
    # ---------------------------------
    dat_dir = list(sorted((name for name in root_dir.glob(f"*{root_dir.stem}*") if 
                           "HR" not in str(name) and 
                           "dark" not in str(name) and 
                           name.is_dir())))[-1]
    info_path = Path(f"{dat_dir.parent}",f"{dat_dir.stem}_info.mat")
    assert info_path.exists(), f"invalid info path: {info_path}"
    config_path = Path(f"{dat_dir}",f"acquisitionmetadata.ini")
    assert config_path.exists(), f"invalid config path: {config_path}"

    # config Dat files
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
    
    
    # load sample low-res vol
    i = 276
    lr_volume = load_single_vol(i, 
                    dat_loader, 
                    dat_dir, 
                    file_names, 
                    dat_vol_arrays, 
                    dat_slice_array, 
                    daq, 
                    flyback, 
                    dark_plane)
    # save the high resolution volume as a tiff
    skio.imsave(Path(f"{exp_dir}","low_resolution.tif"), np.clip(lr_volume,0,220).astype(np.uint16))
    
    
    # # change to low space
    mapped_hr = downsample_hr_vol(hr_info_path, info_path, lr_volume, hr_volume)
  
    # 2: Test motion correction
    # ---------------------------------
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,5))
    ax1.imshow(block_reduce(np.clip(lr_volume,0,220), block_size=(1,1,1), func=np.median)[60])
    ax2.imshow(block_reduce(mapped_hr, block_size=(1,1,1), func=np.median)[60])
    fig.savefig(f"{exp_dir}/vol_subsample.png")
    
    shifts = phase_cross_correlation(mapped_hr, np.clip(lr_volume,0,200), upsample_factor=10, space='real', return_error=False)
    # apply test shift correction
    aligned_frame = ndi.shift(lr_volume, tuple(shifts), cval=0., mode='constant', prefilter=True, order=3)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(8,5))
    ax1.imshow(block_reduce(np.clip(aligned_frame,0,150), block_size=(1,1,1), func=np.mean)[60])
    ax2.imshow(block_reduce(mapped_hr, block_size=(1,1,1), func=np.mean)[60], alpha=.6, cmap='Oranges')
    ax2.imshow(block_reduce(np.clip(aligned_frame,0,150), block_size=(1,1,1), func=np.mean)[60], alpha=.8, cmap='Blues')
    ax3.imshow(block_reduce(mapped_hr, block_size=(1,1,1), func=np.mean)[60])  
    fig.savefig(f"{exp_dir}/motion_correction_comparison.png")
  
        
    # 2: Load all volumes
    # ---------------------------------
    # calculate volume rate
    vol_rate = daq.pixelrate / daq.pixelsPerLine
    logging.info(f"volume rate {vol_rate}")
    
    frame_pad = args.UVPad
    
    # iterate through each trial
    for trial_ix, (stim_on, stim_off, t_init, t_fin) in enumerate(zip(stim_onset.itertuples(), stim_offset.itertuples(), pre_ts, post_ts)):
        logging.info(f"Trial {trial_ix} Started")
        logging.info(f"iterables\n{trial_ix}, {stim_on}, {stim_off}, {t_init}, {t_fin}")
        # calculate initial vol_ix
        pre_v = np.ceil(vol_rate*t_init).astype(int)
        post_v = np.ceil(vol_rate*t_fin).astype(int)
        logging.info(f"pre_v {pre_v}")
        logging.info(f"post_v {post_v}")
    
        logging.info(f"initial {stim_on.vol_id-pre_v}, {stim_on.vol_id-frame_pad}")
        logging.info(f"final {stim_off.vol_id+1+frame_pad}, {stim_off.vol_id+post_v}")
        # slice on the period and stimuli activity
        trial_dat_slice_array = list(itertools.chain(*[dat_slice_array[stim_on.vol_id-pre_v:stim_on.vol_id-frame_pad], 
                                                dat_slice_array[stim_off.vol_id+1+frame_pad:stim_off.vol_id+post_v]]))
        trial_dat_vol_arrays = list(itertools.chain(*[dat_vol_arrays[stim_on.vol_id-pre_v:stim_on.vol_id-frame_pad], 
                                                dat_vol_arrays[stim_off.vol_id+1+frame_pad:stim_off.vol_id+post_v]]))
        
        timepoints = len(trial_dat_slice_array)
            
        # instantiate experiment array
        compressor = Blosc(cname='zstd', clevel=args.cLevel, shuffle=Blosc.BITSHUFFLE)
        zarr_filepath = Path(f"{exp_dir}",f"{root_dir.stem}_{trial_ix:02}.zarr")
        z_arr = zarr.open(f'{zarr_filepath}', 
                        mode='w', 
                        shape=(timepoints,
                                int(daq.pixelsPerLine-flyback),
                                dat_loader.x_crop,
                                dat_loader.y_crop),
                        chunks=(1, None),
                        compressor=compressor,
                        dtype=np.uint16,)
        
        # add attributes i.e. relative frame ids and vol ids
        fid_vals = tail_df.loc[tail_df.vol_id == stim_off.vol_id-pre_v, 'FrameID'].values
        z_arr.attrs['FID_init'] = (fid_vals[0], fid_vals[-1])
        z_arr.attrs['VID_init'] = stim_on.vol_id - pre_v
        
        fid_vals = tail_df.loc[tail_df.vol_id == stim_on.vol_id+post_v, 'FrameID'].values
        z_arr.attrs['FID_fin'] = (fid_vals[0], fid_vals[-1])
        z_arr.attrs['VID_fin'] = stim_off.vol_id + post_v
        
        z_arr.attrs['FID_on'] = tail_df.loc[tail_df.vol_id == stim_on.vol_id-frame_pad, 'FrameID'].values[0]
        z_arr.attrs['VID_on'] = stim_on.vol_id
        
        z_arr.attrs['FID_off'] = tail_df.loc[tail_df.vol_id == stim_off.vol_id+frame_pad, 'FrameID'].values[-1]
        z_arr.attrs['VID_off'] = stim_off.vol_id
        
        z_arr.attrs['vol_rate'] = vol_rate
        z_arr.attrs['pre_stim_time'] = t_init
        z_arr.attrs['post_stim_time'] = t_fin
        z_arr.attrs['VID_stim_ix'] = (stim_on.vol_id-frame_pad) - (stim_on.vol_id-pre_v)
        z_arr.attrs['VID_stim_ix'] = pre_v-frame_pad
        z_arr.attrs['FID_stim_ix'] = tail_df.loc[tail_df.vol_id == pre_v-frame_pad, 'FrameID'].values[0]
        z_arr.attrs['exp_name'] = Struct(mat73.loadmat(info_path)).info.scanName
        
        # create a dark vol is background subtraction applied
        dark_vol = np.tile(dark_plane, (int(daq.pixelsPerLine-flyback),1,1)).astype(z_arr.dtype)
        # interate through each camera volume and fill with sliced dat vol
        
        # PARELLISE
        # from joblib import Parallel, delayed        
        # def _align_vol_hr(i, 
        #         dat_loader, 
        #         dat_dir, 
        #         file_names, 
        #         dat_vol_arrays, 
        #         dat_slice_array, 
        #         daq, 
        #         flyback, 
        #         dark_plane):
        #     lr_volume = load_single_vol(i, 
        #         dat_loader, 
        #         dat_dir, 
        #         file_names, 
        #         dat_vol_arrays, 
        #         dat_slice_array, 
        #         daq, 
        #         flyback, 
        #         dark_plane)
        #     shifts = phase_cross_correlation(mapped_hr, np.clip(lr_volume,0,200), upsample_factor=10, space='real', return_error=False)
        #     # apply test shift correction
        #     aligned_frame = ndi.shift(lr_volume, tuple(shifts), cval=0., mode='constant', prefilter=True, order=3)
        #     return aligned_frame
        
        # def align_vol_hr():
            
        #     Parallel(n_jobs=10)(delayed(_align_vol_hr)() for i in range(timepoints))
            
        timeseries_shifts=[]
        for i in trange(timepoints):
            volume = load_dats_for_vol(dat_loader,
                                        dat_dir, 
                                        file_names, 
                                        trial_dat_vol_arrays[i], 
                                        trial_dat_slice_array[i])[...,:-flyback]
            # reorder indices - t, z, y, x
            volume = volume.transpose(2,0,1)
            # subtract dark vol
            if args.SubtractDarkVol == 1:
                volume+=110
                volume-=dark_vol
            
            # rough motion correction
            volume = volume[volume > 300] = np.median(volume)
            shifts = phase_cross_correlation(mapped_hr, np.clip(volume,0,200), upsample_factor=10, space='real', return_error=False)
            aligned_frame = ndi.shift(volume, tuple(shifts), cval=0., mode='constant', prefilter=True, order=3)
            
            z_arr.oindex[i] = volume
        np.save(f"{exp_dir}/shifts{trial_ix}.npy", np.array(timeseries_shifts))
        logging.info("Trial Exported")