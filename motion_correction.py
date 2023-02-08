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
import tifffile

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


def downsample_hr_vol(hr_info_path:str, lr_info_path:str, lr_shape:tuple, hr_vol:np.ndarray)->np.ndarray:
    """downsample high resolution vol to match low resoluion vol

    Args:
        hr_info_path (str): info mat file for high res
        lr_info_path (str): info mat file for low res
        lr_vol (tuple): shape of low res dims
        hr_vol (np.ndarray): high res volume

    Returns:
        np.ndarray: down sampled high res volume to match low res volume
    """
    print(lr_shape, hr_vol.shape)
    # change to low space
    hr_converter = get_pixel_space_calibration(hr_info_path)
    lr_converter = get_pixel_space_calibration(lr_info_path)
    hr_space = AnatomicalSpace("pli", resolution=(hr_converter.x_per_pix, hr_converter.y_per_pix, hr_converter.z_per_pix))
    lr_space = AnatomicalSpace("pli", resolution=(hr_vol.shape[0]/(2*lr_shape[0]), lr_converter.y_per_pix, lr_converter.z_per_pix))
    downsampled_vol = hr_space.map_stack_to(lr_space, hr_vol)
    print(lr_shape, downsampled_vol.shape)
    return downsampled_vol


def main(args):
    # 1: Define logger and paths
    # ---------------------------------
    root_dir = Path(args.PathData)
    tracking_dir = Path(args.PathTracking)
    exp_dir = Path(args.PathExport)
    
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
    # print out the args
    logging.info(f"\nROOT DATA PATH\t{root_dir}\nEXPORT PATH\t{exp_dir}")
    
    # Argument Checks
    assert len(args.preStim)==len(args.postStim), f"mismatch with {len(args.preStim)} pre-stimulus and {len(args.postStim)} post-stimulus periods"
    pre_ts = args.preStim
    post_ts = args.postStim
    mip = bool(args.MIP)
    create_directory(parent_path=exp_dir.parent, dir_name=f"{exp_dir.stem}")

    # 2: Make dark plane
    # ---------------------------------
    dark_vol_dir = list(sorted(name for name in root_dir.glob("*dark*") if name.is_dir()))[-1]
    dark_vol_path = Path(f"{exp_dir}","dark_plane.npy")
    logging.info(f"\nDARK PLANE PATH\t{dark_vol_path}")
    dark_plane = make_dark_plane(dat_dir=dark_vol_dir, export_path=dark_vol_path)
    logging.info(f"Dark volume exported")
        
    # 3: Pull behaviour
    # ---------------------------------
    # infer the behaviour mapping and UV stimulation
    tail_df, stim_onset, stim_offset = merge_tracking(tracking_dir, exp_dir)
    assert stim_offset.shape[0] == len(post_ts), "Number of detected stimuli need to match the number of periods"
    
    # 4: Define low-res paths and config dat files
    # ---------------------------------
    flyback = args.flyback
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
    # low res volume shame
    lr_shape = (int(daq.pixelsPerLine) - flyback, dat_loader.x_crop, dat_loader.y_crop)
    # load and sort dat spools files
    total_planes = calc_total_planes(daq)
    file_names = dat_loader.sort_spool_filenames(total_planes)
    # prepare slices and dat vols for each camera vol
    dat_vol_arrays, dat_slice_array = map_dats_to_volume(daq, dat_loader.n_planes)
    # calculate volume rate
    vol_rate = daq.pixelrate / daq.pixelsPerLine
    logging.info(f"volume rate {vol_rate}")
    # mark padding before and aftrer UV stim
    pre_frame_pad, post_frame_pad = args.UVPad
    
    # 5: Make high-res anatomy stack
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
    hr_daq = Struct(mat73.loadmat(hr_info_path)).info.daq
    # load config file
    hr_config=LoadConfig(hr_config_path)
    # generate dat file loader
    hr_dat_loader = hr_config.get_dat_dimensions()
    # load and sort dat spools files
    total_hr_planes = calc_total_planes(hr_daq)
    hr_file_names = hr_dat_loader.sort_spool_filenames(total_hr_planes)
    # prepare slices and dat vols for each camera vol
    hr_dat_vol_arrays, hr_dat_slice_array = map_dats_to_volume(hr_daq, hr_dat_loader.n_planes)
    hr_shape=(int(hr_daq.numberOfScans),
            int(hr_daq.pixelsPerLine-flyback),
            hr_dat_loader.x_crop,
            hr_dat_loader.y_crop)
    
    # load high resolution volume
    hr_volume = load_single_vol(0, 
                        hr_dat_loader, 
                        hr_dat_dir, 
                        hr_file_names, 
                        hr_dat_vol_arrays, 
                        hr_dat_slice_array, 
                        hr_daq, 
                        flyback, 
                        dark_plane)
    
    # save the high resolution volume as a tiff
    skio.imsave(Path(f"{exp_dir}","high_resolution.tif"), hr_volume.astype(np.uint16))
    
    # downsample high-res to low-res space
    mapped_hr = downsample_hr_vol(hr_info_path, info_path, lr_shape, hr_volume)
    skio.imsave(Path(f"{exp_dir}","high_resolution_downsampled.tif"), mapped_hr.astype(np.uint16))
    
    # 6: Load dat files & motion correction
    # ---------------------------------
    # iterate through each trial
    for trial_ix, (stim_on, stim_off, t_init, t_fin) in enumerate(zip(stim_onset.itertuples(), stim_offset.itertuples(), pre_ts, post_ts)):
        logging.info(f"Trial {trial_ix} Started")
        logging.info(f"iterables\n{trial_ix}, {stim_on}, {stim_off}, {t_init}, {t_fin}")
        # calculate initial vol_ix
        pre_v = np.ceil(vol_rate*t_init).astype(int)
        post_v = np.ceil(vol_rate*t_fin).astype(int)
        logging.info(f"pre_v {pre_v}")
        logging.info(f"post_v {post_v}")
    
        logging.info(f"\n\ninitial exp-vol: {stim_on.vol_id-pre_v}\t initial stim-vol: {stim_on.vol_id-pre_frame_pad}")
        logging.info(f"final stim-vol: {stim_off.vol_id+1+post_frame_pad}\t final exp-vol: {stim_off.vol_id+post_v}\n\n")
        # slice on the period and stimuli activity
        trial_dat_slice_array = list(itertools.chain(*[dat_slice_array[stim_on.vol_id-pre_v:stim_on.vol_id-pre_frame_pad], 
                                                dat_slice_array[stim_off.vol_id+1+post_frame_pad:stim_off.vol_id+post_v]]))
        trial_dat_vol_arrays = list(itertools.chain(*[dat_vol_arrays[stim_on.vol_id-pre_v:stim_on.vol_id-pre_frame_pad], 
                                                dat_vol_arrays[stim_off.vol_id+1+post_frame_pad:stim_off.vol_id+post_v]]))
        
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
        # start of trial landmarks
        fid_vals = tail_df.loc[tail_df.vol_id == stim_on.vol_id-pre_v, 'FrameID'].values
        z_arr.attrs['FID_init'] = (fid_vals[0], fid_vals[-1]) # upper and lower frame id at start of the trial
        z_arr.attrs['VID_init'] = stim_on.vol_id - pre_v # volume id at start of the trial
        
        # end of trial landmarks
        fid_vals = tail_df.loc[tail_df.vol_id == stim_off.vol_id+post_v, 'FrameID'].values
        z_arr.attrs['FID_fin'] = (fid_vals[0], fid_vals[-1]) # upper and lower frame id at end of the trial
        z_arr.attrs['VID_fin'] = stim_off.vol_id + post_v # volume id at end of the trial
        
        # stimulus onset landmarks - accounts for padding
        fid_vals = tail_df.loc[tail_df.vol_id == stim_on.vol_id-pre_frame_pad, 'FrameID'].values
        z_arr.attrs['FID_on'] = (fid_vals[0], fid_vals[-1])
        z_arr.attrs['VID_on'] = stim_on.vol_id-pre_frame_pad
        
        # stimulus onset landmarks - accounts for padding
        fid_vals = tail_df.loc[tail_df.vol_id == stim_off.vol_id+post_frame_pad, 'FrameID'].values
        z_arr.attrs['FID_off'] = (fid_vals[0], fid_vals[-1])
        z_arr.attrs['VID_off'] = stim_off.vol_id+post_frame_pad
        
        z_arr.attrs['vol_rate'] = vol_rate
        z_arr.attrs['pre_stim_time'] = t_init
        z_arr.attrs['post_stim_time'] = t_fin
        z_arr.attrs['VID_stim_ix'] = (stim_on.vol_id-pre_frame_pad) - (stim_on.vol_id-pre_v)
        z_arr.attrs['VID_stim_ix'] = pre_v-pre_frame_pad
        z_arr.attrs['FID_stim_ix'] = tail_df.loc[tail_df.vol_id == pre_v-pre_frame_pad, 'FrameID'].values[0]
        z_arr.attrs['exp_name'] = Struct(mat73.loadmat(info_path)).info.scanName
        
        # create a dark vol is background subtraction applied
        dark_vol = np.tile(dark_plane, (int(daq.pixelsPerLine-flyback),1,1)).astype(z_arr.dtype)

        # Max intensiry projection
        if mip:
            mip_arr = []
            aligned_mip_arr = []
        
        timeseries_shifts=[]
        # interate through each camera volume and fill with sliced dat vol
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
            volume[volume > 300] = np.median(volume)
            shifts = phase_cross_correlation(mapped_hr, np.clip(volume,0,200), upsample_factor=10, space='real', return_error=False)
            aligned_frame = ndi.shift(volume, tuple(shifts), cval=0., mode='constant', prefilter=True, order=3)
            timeseries_shifts.append(shifts)
                
            z_arr.oindex[i] = aligned_frame
            if mip:
                down_samp_arr = block_reduce(volume, block_size=(10,3,3), func=np.median)
                aligned_down_samp_arr = block_reduce(aligned_frame, block_size=(10,3,3), func=np.median)
                mip_arr.append(np.sum(down_samp_arr, axis=0))
                aligned_mip_arr.append(np.sum(aligned_down_samp_arr, axis=0))
        logging.info("Trial Exported")
        if mip:
            mip_arr = np.array(mip_arr, dtype=np.uint16)
            tifffile.imsave(f"{exp_dir}/raw_mip{trial_ix:02}.tiff", list(mip_arr))
            aligned_mip_arr = np.array(aligned_mip_arr, dtype=np.uint16)
            tifffile.imsave(f"{exp_dir}/mc_mip{trial_ix:02}.tiff", list(aligned_mip_arr))
            
    # 7: Form binary mask & downsample voxels
    # ---------------------------------
    
    pass


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Define the parameters used for converting dat files')
    # parser.add_argument('-pD', '--PathData', help="path to imaging directories data", default='.', type=str)
    # parser.add_argument('-pT', '--PathTracking', help="path to tracking directories data", default='.', type=str)
    # parser.add_argument('-pE', '--PathExport', help="path to export aggregated data", default='.', type=str)
    # parser.add_argument('-c', '--cLevel', help="Level of compression of zarr file. Uses zstd Blosc.BITSHUFFLE compression. Default n = 5.",
    #                     default=5, type=int)
    # parser.add_argument('-f', '--flyback', help="flyback frames used. Default n = 2.",
    #                     default=2, type=int)
    # # parser.add_argument('-uvP', '--UVPad', help="Extra frames excluded from UV stimulation. Default n = 0.",
    # #                     default=0, type=int)
    # parser.add_argument('-uvP', '--UVPad', help="Extra frames excluded from UV stimulation, first arg is pre-frame padding. Default n = [0,1].",
    #                     nargs=2, default=[0,1], type=int)
    # parser.add_argument('-pre', '--preStim', help="Seconds acquired before stimulus.",
    #                     nargs='+', type=int)
    # parser.add_argument('-pos', '--postStim', help="Seconds acquired after stimulus.",
    #                     nargs='+', type=int)
    # parser.add_argument('-m', '--MIP', help="Save a max intensity projection of cleaned volumes. Default n = 1.",
    #                     default=1, type=int)
    # parser.add_argument('-dV', '--SubtractDarkVol', help="bool subtract darkvolume from volumes. Default true", default='1', type=int)
    # args = parser.parse_args()
    
    # debug
    # ---
    args=argparse.Namespace(PathData="/Volumes/SCAPE 1/Tom/20230203/20230203_huc_h2b_gcamp6f_7dpf_f02troubleshoot",
                            PathTracking="/Volumes/SCAPE 1/Tom/20230203/20230203_huc_h2b_gcamp6f_7dpf_f02troubleshoot/tracking/20230203_huc_h2b_gcamp6f_7dpf_f02troubleshoot",
                            PathExport="/Users/thomasmullen/Desktop/dump/RawDatConversionTest",
                            cLevel=5,
                            flyback=2,
                            UVPad=[0,1],
                            preStim=[10, 10, 10, 10],
                            postStim=[10, 10, 10, 10],
                            MIP=1,
                            SubtractDarkVol=1
                            )
    
    main(args)
