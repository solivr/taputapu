#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

"""
This file contains code to process IAM dataset (http://www.fki.inf.unibe.ch/databases/iam-handwriting-database).
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Mapping
from tqdm import tqdm
import re
from imageio import imread, imsave
from taputapu.transform.crop import crop_image


class Info_IAM:
    train_samples = 6161
    test_samples = 1861
    validation1_samples = 900
    validation2_samples = 940

    filename_trainset = 'trainset.txt'
    filename_testset = 'testset.txt'
    filename_validationset1 = 'validationset1.txt'
    filename_validationset2 = 'validationset2.txt'

    strikeout_char = '#'


class Linefile_IAM:
    column_names = ["id", "seg_state", "gray_level", "n_components", "x", "y", "w", "h", "transcription"]
    column_types = {'id': str, 'seg_state': str, 'gray_level': np.uint8, 'n_components': np.int32,
                    'x': np.int32, 'y': np.int32, 'w': np.int32, 'h': np.int32, 'transcription': str}
    skiprows = 23
    n_total_rows = 13353


class Wordfile_IAM:
    column_names = ["id", "seg_state", "gray_level", "x", "y", "w", "h", "gram_tag", "transcription"]
    column_types = {'id': str, 'seg_state': str, 'gray_level': np.uint8, 'x': np.int32,
                    'y': np.int32, 'w': np.int32, 'h': np.int32, 'gram_tag': str, 'transcription': str}
    skiprows = 18
    n_total_rows = 115320


def load_ascii_txt_file(filename: str, skiprows=True, asserting_n_rows=True) -> pd.DataFrame:
    """
    Load ascii/*.txt files
    :param filename: filename of txt file to load
    :param skiprows: if True will first rows containing text, if False assumes data start from the first row
    :param asserting_n_rows: if True will check that the number of lines read correspond to the exact number of samples
    :return: a DataFrame with the content of .txt
    """
    if 'lines' in filename:
        filetype = Linefile_IAM

    elif 'words' in filename:
        filetype = Wordfile_IAM
    else:
        raise NotImplementedError

    data = pd.read_csv(filename, delim_whitespace=True, skiprows=filetype.skiprows if skiprows else None,
                       encoding='utf8', error_bad_lines=False, header=None,
                       names=filetype.column_names, dtype=filetype.column_types, quoting=3)

    if asserting_n_rows:
        assert len(data) == filetype.n_total_rows, "Parsing of file didn't get all the rows. " \
                                                   "Parsed {} instead of {}".format(len(data), filetype.n_total_rows)

    return data


def generate_segments(filename: str, images_dir: str, export_dir: str,
                      margin_wh: Tuple[int, int]=(0, 0)) -> None:
    """
    Crop the lines/words in `filename` and export them as .png files to `export_dir`
    :param filename: filename containing the lines/words, coordinates and transcriptions info
    :param images_dir: directory where the original full sized image are
    :param export_dir: export directory to save the cropped segments
    :param margin_wh: margin to add to the original box's coordinates
    :return:
    """
    # Load filename containing the desired segments info (lines/words)
    data_segments = load_ascii_txt_file(filename)

    for index, row in tqdm(data_segments.iterrows(), total=len(data_segments)):

        # Do not load image everytime if the line is in the same one
        current_basename = None
        basename_image = '-'.join(row.id.split('-')[:2]) + '.png'
        if current_basename != basename_image:
            filename_image = os.path.join(images_dir, basename_image)
            image = imread(filename_image, pilmode='L')

        # Crop segment
        segment_img = crop_image(image, (row.x, row.y, row.w, row.h), margin_wh=margin_wh)

        # Export it
        filename_segment = os.path.join(export_dir, row.id + '.png')
        imsave(filename_segment, segment_img)


def create_experiment_csv(filename: str, image_directory: str, output_filename,
                          original_iam_file=True, map_strikeouts=False) -> None:
    """
    Generates the csv file needed to train tf_crnn with format : path_to_img_segment;transcription_non_formatted
    :param filename: filename of line / words in IAM/ascii
    :param image_directory: directory where the image segments are located
    :param output_filename: filename of the output .csv file
    :param original_iam_file : if True considers it is the original file provided by IAM db maintainers,
    if False supposes a generated file
    :param map_strikeouts: If True will map '#' to '[-]'
    :return:
    """
    # Load filename containing the desired segments info (lines/words)
    if original_iam_file:
        data_segments = load_ascii_txt_file(filename)
    else:
        data_segments = load_ascii_txt_file(filename, skiprows=False, asserting_n_rows=False)

    list_filenames = []
    list_transcriptions = []
    for index, row in tqdm(data_segments.iterrows(), total=len(data_segments)):
        filename = os.path.join(image_directory, row.id + '.png')

        # Verify image exists
        if not os.path.isfile(filename):
            print('Image does not exist in {}'.format(filename))
            continue

        try:
            transcription = row.transcription.replace('|', ' ')
            # Remove space after ( char)
            transcription = re.sub(r'([?(])+\s', r'\1', transcription)
            # Remove space before .,?!) chars
            transcription = re.sub(r'\s+([?.,;!)])', r'\1', transcription).strip()
            if map_strikeouts:
                transcription = _map_strikeouts_to_char(transcription)
        except AttributeError:
            print('Transcription does not exists in {}'.format(index))
            continue

        list_filenames.append(filename)
        list_transcriptions.append(transcription)

    # Create dataframe with filenames and transcriptions
    df = pd.DataFrame({'path': list_filenames, 'transcription': list_transcriptions})
    df.to_csv(output_filename, sep=';', encoding='utf-8', header=False, index=False, escapechar="\\", quoting=3)


def _make_lookup_id_set(root_dir_set_list: str) -> Mapping[str, str]:
    """
    From the files listing the id for each set (train, test, validation1, validation2), generate a dictionary that will
    have as keys the id of lines/words and as values the name of set it belongs to ('train', 'test',
    'validation1', 'validation2')
    :param root_dir_set_list: path to the folder containing the split files .txt
    :return: a dictionary with {id: set}
    """

    _tuples = [(Info_IAM.filename_trainset, 'train'), (Info_IAM.filename_testset, 'test'),
               (Info_IAM.filename_validationset1, 'val1'), (Info_IAM.filename_validationset2, 'val2')]

    # Get all the id for each set
    dataframes = list()
    for filename_set, set_id in _tuples:
        tmp_df = pd.read_csv(os.path.join(root_dir_set_list, filename_set), encoding='utf8', header=None, names=['id'])
        tmp_df = tmp_df.assign(set=len(tmp_df) * [set_id])

        dataframes.append(tmp_df)

    # Concatenate all the dataframes
    df_all = pd.concat(dataframes)

    # Create a dictionary {id : set}
    dic_set_belonging = dict()
    for index, row in df_all.iterrows():
        dic_set_belonging[row.id] = row.set

    return dic_set_belonging


def generate_splits_txt(filename: str, rootdir_set_files: str, exportdir_set_files: str) -> None:
    """
    This function generates the train / test / validation1 / validation2 txt file
    with the same pattern as the ascii/{lines, words}.txt file.
    :param filename: full data txt file (e.g lines.txt)
    :param rootdir_set_files: path to directory where the set split .txt files are
    :param exportdir_set_files: directory to save the new generated files
    :return:
    """
    data = load_ascii_txt_file(filename)
    lookup_id_set = _make_lookup_id_set(rootdir_set_files)

    basename_file = os.path.basename(filename).split('.')[0]

    train_list, test_list, val1_list, val2_list = list(), list(), list(), list()

    # Because word id and line id are formatted a bit differently we need to separate these cases.
    # The id in the lookup table corresponds to the id of line. Words have an additional 'id of word' appended to it

    def _get_id(raw_id):
        if 'line' in basename_file:
            return raw_id
        elif 'word' in basename_file:
            return raw_id[:-3]
        else:
            raise NotImplementedError

    for index, row in tqdm(data.iterrows()):
        key_lookup = _get_id(row.id)
        try:
            if lookup_id_set[key_lookup] == 'train':
                train_list.append(row)
            elif lookup_id_set[key_lookup] == 'test':
                test_list.append(row)
            elif lookup_id_set[key_lookup] == 'val1':
                val1_list.append(row)
            elif lookup_id_set[key_lookup] == 'val2':
                val2_list.append(row)
        except KeyError:
            print('{} does not exist'.format(key_lookup))

    # Check that we have the same number of elements being exported
    assert Info_IAM.train_samples == len(train_list), \
        "Training : {} != {}".format(Info_IAM.train_samples, len(train_list))
    assert Info_IAM.test_samples == len(test_list), \
        "Testing : {} != {}".format(Info_IAM.test_samples, len(test_list))
    assert Info_IAM.validation1_samples == len(val1_list), \
        "Validation1 : {} != {}".format(Info_IAM.validation1_samples, len(val1_list))
    assert Info_IAM.validation2_samples == len(val2_list), \
        "Validation2 : {} != {}".format(Info_IAM.validation2_samples, len(val2_list))

    # Export dataframes to txt files
    tuples_export = [(train_list, os.path.join(exportdir_set_files, '{}_train.txt'.format(basename_file))),
                     (test_list, os.path.join(exportdir_set_files, '{}_test.txt'.format(basename_file))),
                     (val1_list, os.path.join(exportdir_set_files, '{}_validation1.txt'.format(basename_file))),
                     (val2_list, os.path.join(exportdir_set_files, '{}_validation2.txt'.format(basename_file)))]
    for set_list, export_filename in tuples_export:
        pd.DataFrame(set_list).to_csv(export_filename, sep=' ', header=False, index=False, quoting=3)


def _map_strikeouts_to_char(string: str, input_char: str=Info_IAM.strikeout_char, output_char: str='[-]') -> str:
    """
    Replaces the input character for strikeout by a new character/set of chars
    :param string: string to process
    :param input_char: char representing strike-outs in original string
    :param output_char: new char or string representing strike-outs in output string
    :return:
    """

    return re.sub(input_char, output_char, string)
