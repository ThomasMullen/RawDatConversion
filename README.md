Explanation
----------
Image data is acquired from the camera as a continuous stream of bytes. These bytes as stored into .dat files. To 
reconstruct the 1D array into a formatted structure requires a .ini configure file to describe the original image shape
and also an info.mat file.

Each .dat file will encode M number of planes, and there will be N number of planes that completes 1 volume (N > M). You
will need multiple .dat files to recover a full volume, and these .dat files may rollover to the next volume.

Key parameters:
----------
:acquisition_config.planes_per_datfile:         planes per a .dat file
:info.info.daq.pixelsPerLine:                   planes per a volume
:info.info.daq.numberOfScans:                   number of volumes
:n_scans * planes_per_vol:                      total number of planes saved
:round(total_planes / planes_per_datfile):      total number of .dat file
:planes_per_vol/planes_per_datfile:             number of .dat files for a volume
