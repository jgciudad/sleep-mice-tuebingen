#!/usr/bin/env python

import argparse
import sys
import time
from glob import glob
from os.path import join, basename, isfile, realpath, dirname

import h5py
import numpy as np
import tables
import pandas as pd
import mne
import scipy

sys.path.insert(0, realpath(join(dirname(__file__), '..')))

from base.data.downsample import downsample
from base.config_loader import ConfigLoader
from base.data.data_table import create_table_description, COLUMN_LABEL, COLUMN_MOUSE_ID


def parse():
    parser = argparse.ArgumentParser(description='data transformation script')
    parser.add_argument('--experiment', '-e', required=False, default='standard_config',
                        help='name of experiment to transform data to')

    return parser.parse_args()

def load_kornum_labels(label_path):
    # files = [fn for fn in glob(join(config.DATA_DIR, '**/*.tsv'), recursive=True)]
    # labels_path = [f for f in files if rec_name in f]
    # if len(labels_path) == 1:
    #     labels_path = labels_path[0]
    # else:
    #     raise Exception("More than one file with recording name " + rec_name)
    df = pd.read_csv(join(config.KORNUM_DATA_DIR, label_path), skiprows=9, engine='python', sep='\t', index_col=False).iloc[:, 4]
    # df.loc[(df != 1) & (df != 2) & (df != 3)] = 4

    # transform integer labels to category name
    def int_class_to_string(row):
        if row == 1:
            return 'WAKE'
        if row == 2:
            return 'NREM'
        if row == 3:
            return 'REM'
        if (row != 1) & (row != 2) & (row != 3):
            return 'ARTIFACT' 
    stages = df.apply(int_class_to_string)

    return stages.to_list(), stages.index.values.tolist()

def load_raw_kornum_recording(label_path):
    """
    :param file_path: path to the .edf recording
    :param resample_rate: new sampling rate in Hertz
    :return: numpy array with the signal
    """
    signal_path = join(config.KORNUM_DATA_DIR, label_path.split("/")[0], 'EDF', basename(label_path)[:-4] + '.edf')
    
    data = mne.io.read_raw_edf(signal_path)
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names

    features = {}

    for c, channel in enumerate(channels):
        # if resample_rate:
        # new_num_samples = raw_data.shape[1]/info['sfreq']*resample_rate
        # if new_num_samples.is_integer() is False:
        #     raise Exception("New number of samples is not integer")
                    # raw_data = scipy.signal.resample(x=raw_data, num=int(new_num_samples), axis=1)
        
        if channel == 'EEG EEG1A-B':
            c_name = 'EEG1'
        elif channel == 'EEG EEG2A-B':
            c_name = 'EEG2'
        elif channel == 'EMG EMG':
            c_name = 'EMG'

        features[c_name] = downsample(raw_data[c, :], sr_old=info['sfreq'], sr_new=config.SAMPLING_RATE,
                                            fmax=0.4 * config.SAMPLING_RATE, outtype='sos', method='pad')

    sample_start_times = np.arange(0, features[c_name].shape[0], config.SAMPLING_RATE*config.SAMPLE_DURATION, dtype=int)

    return features, sample_start_times.tolist()

def read_kornum_recording(label_path: str):
    """reads data from rec_name.edf and rec_name.tsv

    Returns:
        (dict, list, list): features in the form of a dict with entries for each CHANNEL, labels as a list, list of
        start times of samples as indexes in features
    """

    labels, epoch_number = load_kornum_labels(label_path)
    features_dict, sample_start_times = load_raw_kornum_recording(label_path)

    return features_dict, labels, sample_start_times, epoch_number


def load_spindle_labels(labels_path,
                scorer):

    df = pd.read_csv(join(config.SPINDLE_DATA_DIR, labels_path), header=None)

    # column names: {1, 2, 3, n, r, w}
    # 1=wake artifact, 2=NREM artifact, 3=REM artifact

    if scorer==1:
        labels = df[1]
    elif scorer==2:
        labels = df[2]

    # rename classes and convert class artifacts to unique artifact class
    def rename_class(row):
        if row == 'w':
            return 'WAKE'
        if row == 'n':
            return 'NREM'
        if row == 'r':
            return 'REM'
        if (row != 'w') & (row != 'n') & (row != 'r'):
            return 'ARTIFACT' 
    stages = labels.apply(rename_class)

    return stages.to_list(), stages.index.values.tolist()

def load_raw_spindle_recording(label_path):
    """
    :param file_path: path to the .edf recording
    :param resample_rate: new sampling rate in Hertz
    :return: numpy array with the signal
    """
    signal_path = join(config.SPINDLE_DATA_DIR, label_path.split("/")[0], 'recordings', basename(label_path)[:-4] + '.edf')
    
    data = mne.io.read_raw_edf(signal_path)
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names

    features = {}

    for c, channel in enumerate(channels):
        # if resample_rate:
        # new_num_samples = raw_data.shape[1]/info['sfreq']*resample_rate
        # if new_num_samples.is_integer() is False:
        #     raise Exception("New number of samples is not integer")
                    # raw_data = scipy.signal.resample(x=raw_data, num=int(new_num_samples), axis=1)

        features[channel] = downsample(raw_data[c, :], sr_old=info['sfreq'], sr_new=config.SAMPLING_RATE,
                                            fmax=0.4 * config.SAMPLING_RATE, outtype='sos', method='pad')

    sample_start_times = np.arange(0, features[channel].shape[0], config.SAMPLING_RATE*config.SAMPLE_DURATION, dtype=int)

    return features, sample_start_times.tolist()

def read_spindle_recording(label_path: str,  scorer: int):
    """reads data from rec_name.edf and rec_name.tsv

    Returns:
        (dict, list, list): features in the form of a dict with entries for each CHANNEL, labels as a list, list of
        start times of samples as indexes in features
    """

    labels, epoch_number = load_spindle_labels(labels_path=label_path,
                                       scorer=scorer)
    features_dict, sample_start_times = load_raw_spindle_recording(label_path)

    return features_dict, labels, sample_start_times, epoch_number



# def read_h5(h5_f_name: str):
#     """reads data from `h5_f_name`

#     Returns:
#         (dict, list, list): features in the form of a dict with entries for each CHANNEL, labels as a list, list of
#         start times of samples as indexes in features
#     """
#     features = {}  # save features per CHANNEL
#     # open h5 file in DATA_DIR
#     with h5py.File(join(config.DATA_DIR, h5_f_name), 'r') as h5_f:
#         # check if the sampling rates for all CHANNELS are the same, if not, throw an exception
#         sampling_rates = [h5_f[c].attrs['samprate'][0] for c in config.CHANNELS]
#         if np.all(sampling_rates == sampling_rates[0]):
#             sr = sampling_rates[0]
#         else:
#             raise ValueError('sampling rates in original h5 file are not equal:', sampling_rates)

#         # iterate over CHANNELS, load and downsample data
#         for channel in config.CHANNELS:
#             data_c = h5_f[channel][:]
#             # convert the integer data (2 bytes (int16)) into doubles
#             scale_c = h5_f[channel].attrs['scale']
#             offset_c = h5_f[channel].attrs['offset']
#             features[channel] = np.double(data_c) * scale_c / 6553.6 + offset_c

#             # downsample data to the SAMPLING_RATE specified in config
#             features[channel] = downsample(features[channel], sr_old=sr, sr_new=config.SAMPLING_RATE,
#                                            fmax=0.4 * config.SAMPLING_RATE, outtype='sos', method='pad')

#         # load scores from h5 file and map them to stages using SCORING_MAP
#         scores = h5_f['Scoring/scores'][:]
#         t_scoring_map = {v: k for k in config.SCORING_MAP for v in config.SCORING_MAP[k]}
#         labels = [t_scoring_map[s] if s in t_scoring_map else 'ignore' for s in scores]
#         # load start times of samples and transform them to indexes in the feature map
#         sample_start_times = h5_f['Scoring/times'][:]
#         sample_start_times = (sample_start_times / sr * config.SAMPLING_RATE).astype('int')

#     return features, labels, sample_start_times


def write_data_to_table(table: tables.Table, features: dict, labels: list, start_times: list, mouse_id: int, epoch_number: list):
    """writes given data to the passed table, each sample is written in a new row"""
    sample = table.row

    # iterate over samples and create rows
    for sample_start, label in zip(start_times, labels):
        # determine idxs of data to load from features
        sample_end = int(sample_start + config.SAMPLE_DURATION * config.SAMPLING_RATE)

        # try to load data from sample_start to sample_end, if there is not enough data, ignore the sample
        try:
            sample[COLUMN_MOUSE_ID] = mouse_id
            for c in config.CHANNELS:
                sample[c] = features[c][sample_start:sample_end]
            # map stage from h5 file to stage used for classification
            # if the mapping in STAGE_MAP is None, the sample is ignored
            if label not in config.STAGE_MAP or config.STAGE_MAP[label] is None:
                continue
            sample[COLUMN_LABEL] = config.STAGE_MAP[label]
            sample.append()
        except ValueError:
            print(f"""
            While processing epoch [{sample_start}, {sample_end}] with label {label}:
            not enough datapoints in file (n = {len(list(features.values())[0])})
            This epoch is ignored.
            """)
    # write data to table
    table.flush()


# def get_files_for_dataset(dataset: str):
#     """determines which files to transform for a given dataset, see DATA_SPLIT"""
#     # load all h5 files in DATA_DIR and sort them
#     files = [basename(fn) for fn in glob(join(config.DATA_DIR, '**/*.tsv'), recursive=True)]
#     files.sort()

#     # search for files corresponding to mice name in DATA_SPLIT in files
#     file_list = [f for mouse_name in config.DATA_SPLIT[dataset] for f in files if mouse_name in f]

#     return file_list


def transform():
    """transform files in DATA_DIR to pytables table"""
    # load description of table columns
    table_desc = create_table_description(config)

    # if the transformed data file already exists, ask the user if he wants to overwrite it
    # if isfile(config.DATA_FILE):
    #     question = f"{realpath(config.DATA_FILE)} already exists, do you want to override? (y/N): "
    #     response = input(question)
    #     if response.lower() != 'y':
    #         exit()

    # print(f'data is loaded from {realpath(config.DATA_DIR)}')

    # open pytables DATA_FILE
    with tables.open_file(config.DATA_FILE, mode='w', title='data from Kornum') as f:

        # datasets to create, if there is a dataset without any data, an empty table is created inside DATA_FILE
        datasets = ['train', 'valid', 'test', 'spA_scorer1', 'spA_scorer2', 'spD_scorer1', 'spD_scorer2']

        for dataset in datasets:
            print('writing', dataset, 'data...')

            # create tables for every dataset
            table = f.create_table(f.root, dataset, table_desc, dataset + ' data')

            # determine which files to transform for each dataset based on DATA_SPLIT
            # file_list = get_files_for_dataset(dataset)
            # iterate over files, load them and write them to the created table
            for i, label_path in enumerate(config.DATA_SPLIT[dataset]):
                print('mouse [{:d}/{:d}]: {:s}'.format(i + 1, len(config.DATA_SPLIT[dataset]), label_path))
                start = time.time()

                # load only numbers in mice name
                # mouse_id = int(''.join(filter(str.isdigit, f_name)))
                # read h5 file, load and downsample data

                if dataset == 'train' or dataset == 'valid' or dataset == 'test': 
                    features, labels, times, epoch_number = read_kornum_recording(label_path)
                elif dataset == 'spA_scorer1' or dataset == 'spD_scorer1':
                    scorer = 1
                    features, labels, times, epoch_number = read_spindle_recording(label_path, scorer)
                elif dataset == 'spA_scorer2' or dataset == 'spD_scorer2':
                    scorer = 2
                    features, labels, times, epoch_number = read_spindle_recording(label_path, scorer)

                write_data_to_table(table, features, labels, times, label_path, epoch_number)

                print('execution time: {:.2f}'.format(time.time() - start))
                print()

        print(f)


if __name__ == '__main__':
    args = parse()
    # load config, dirs are not needed here, because we do not write a log
    config = ConfigLoader(experiment='kornum_config', create_dirs=False)

    # transform files
    transform()
