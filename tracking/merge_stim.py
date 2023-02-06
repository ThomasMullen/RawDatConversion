from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tracking.smoothing_functs import one_euro_filter

def add_uv_column(cam_df:pd.DataFrame, stim_df:pd.DataFrame)->pd.DataFrame:
    
    cam_df['uv_onset'] = 0
    
    for onset in stim_df.Beg:
        # pull out the closest frame ID
        frame_id = cam_df.iloc[(cam_df.PhotodiodeValue - onset).abs().argsort()[0],0]
        print(onset, frame_id)
        # assign one to new value
        cam_df.loc[cam_df.FrameID == frame_id, 'uv_onset'] = 1
    
    return cam_df


def find_onset_offset(binary_serie):
    """
    Finds the index of a series that switches from 1->0 (onset), from 0->1 (offset), and the number of rows for the
    duration of the onset (duration).
    :param binary_serie: pandas series
    :return: list of indexes each col refers to onset, offset, duration respectively.
    """
    # PAD WITH ZEROS:
    zer = pd.Series([0], index=[binary_serie.index[-1]])
    binary_serie = pd.concat([binary_serie,zer])
    zer = pd.Series([0], index=[binary_serie.index[0]])
    binary_serie = pd.concat([zer,binary_serie])
    cd_diff = binary_serie.diff().abs()
    cd_diff = cd_diff.iloc[1:-1]  # remove start and end values
    index_ = cd_diff[cd_diff == 1].index
    # Not sure if it shoud be added to the beginning or end
    if len(index_) % 2 == 1:
        last_index = binary_serie.index[-1]
        index_ = pd.concat([index_, pd.Index([last_index])])
    index_range = index_.to_numpy().reshape(-1, 2)
    onset = index_range[:, 0]
    offset = index_range[:, 1]
    duration = offset - onset
    return onset, offset, duration

def merge_tracking(tracking_dir, exp_dir, verbose=True):
    # create file path
    tracking_dir = Path(tracking_dir)
    cam_path = Path(f"{tracking_dir}",f"{tracking_dir.stem}scape sync reader.txt")
    tail_path = Path(f"{tracking_dir}",f"{tracking_dir.stem}mp tail tracking.txt")
    stim_path = Path(f"{tracking_dir}",f"{tracking_dir.stem}stim control.txt")
    
    # make hdf5 file
    store = pd.HDFStore(f'{exp_dir}/store.h5')
    
    stim_df = pd.read_csv(stim_path, delimiter=' ') # catch if broken
    cam_df = pd.read_csv(cam_path, delimiter=' ')
    tail_df = pd.read_csv(tail_path, delimiter=' ')
    tail_df=cam_df.merge(tail_df,how='right',on='FrameID').ffill()
    # drop any nans
    tail_df.dropna(axis=0, inplace=True)
    
    # mapstim and tail via AbsoluteTime
    
    df = tail_df[['angle0', 'angle1', 'angle2', 'angle3', 'angle4', 'angle5',
        'angle6', 'angle7', 'angle8', 'angle9', 'angle10', 'angle11', 'angle12',
        'angle13', 'angle14', 'angle15']]
    df = df.cumsum(axis=1).rename(
        columns = lambda x: 'cum_' + x)
    # smooth signal
    df = df.apply(one_euro_filter, axis=0, raw=True, fc_min=.2, beta=3, rate=700)
    if verbose:
        # save tail trace
        fig, ax = plt.subplots()
        df.cum_angle10.plot(ax=ax, title="tail trace")
        fig.savefig(f"{exp_dir}/tail_trace.png")
        df.cum_angle10[5000:6000].plot(ax=ax, title="tail trace")
        fig.savefig(f"{exp_dir}/short_tail_trace.png")
    
    tail_df = pd.concat([tail_df, df], axis=1)
    
    
    ''' Create a vol_id to merge with imaging volumes '''
    # find where galvo changes
    cam_df['galvo_diff'] = cam_df.GalvoValue.diff().abs()
    # save galvo diff
    if verbose:
        fig, ax = plt.subplots()
        cam_df.galvo_diff.plot(ax=ax, title="galvo diff")
        fig.savefig(f"{exp_dir}/galvo_diff.png")
        cam_df.GalvoValue.plot(ax=ax, title="galvo vals")
        fig.savefig(f"{exp_dir}/galvo_vals.png")
    # returns rows of galvo changes
    df_filtered_galvo = cam_df[cam_df.galvo_diff>.5]
    vol_id = pd.DataFrame(np.arange(df_filtered_galvo.shape[0]+1), 
                          columns=['vol_id'], 
                          index= np.insert(df_filtered_galvo.index.values,0,0))
    vol_id['frame_id'] = np.append(cam_df.iloc[0, 0], df_filtered_galvo.FrameID.values)
    # merge vol and frame id onto original behaviour tracking df
    tail_df = tail_df.merge(vol_id, how='left', left_on='FrameID', right_on='frame_id').ffill()
    # format vol_id
    tail_df[['frame_id', 'vol_id']] = tail_df[['frame_id', 'vol_id']].astype(int)
    
    
    # identify uv flash
    tail_df['uv']=0
    # define UV threshold based on mean photodioade value
    uv_thresh = tail_df.PhotodiodeValue.mean() + .05
    tail_df.loc[tail_df.PhotodiodeValue>uv_thresh, 'uv']=1
    
    # merge pulse cycle stimulus
    onset, offset, duration = find_onset_offset(tail_df.uv)
    # onset
    onset_diff = np.diff(onset)
    onset_ix = np.where(onset_diff>2000)[0] + 1
    merged_onset = onset[np.insert(onset_ix, 0, 0)]
    # offset
    roffset=offset[::-1]
    roffset_diff = abs(np.diff(roffset))
    roffset_ix = np.where(roffset_diff>2000)[0]
    roffset_ix = np.hstack([roffset_ix, roffset_ix[-1]+1])
    merged_offset = roffset[roffset_ix][::-1]
    # assign values
    tail_df['uv_stim']=0
    for on, off in zip(merged_onset, merged_offset):
        tail_df.loc[on:off, 'uv_stim']=1
    
    
    stim_onset = tail_df.loc[merged_onset, ['FrameID','vol_id']]
    stim_offset = tail_df.loc[merged_offset, ['FrameID','vol_id']]
    
    # save to file
    store[f'{tracking_dir.stem}/tail_df']=tail_df
    store[f'{tracking_dir.stem}/stim_onset']=stim_onset
    store[f'{tracking_dir.stem}/stim_offset']=stim_offset
    store.close()
    
    return tail_df, stim_onset, stim_offset