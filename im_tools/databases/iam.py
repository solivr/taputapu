#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

"""
This file contains code to process IAM dataset (http://www.fki.inf.unibe.ch/databases/iam-handwriting-database).
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple
from tqdm import tqdm
import re
from imageio import imread, imsave
from im_tools.transform.crop import crop_image


def load_ascii_txt_file(filename: str) -> pd.DataFrame:
    """
    Load ascii/*.txt files
    :param filename: filename of txt file to load
    :return: a DataFrame with the content of .txt
    """
    columns_names = None
    columns_type = None
    if 'lines' in filename:
        columns_names = ["id", "seg_state", "gray_level", "n_components", "x", "y", "w", "h", "transcription"]
        columns_type = {'id': str, 'seg_state': str, 'gray_level': np.uint8, 'n_components': np.int32,
                        'x': np.int32, 'y': np.int32, 'w': np.int32, 'h': np.int32, 'transcription': str}
    elif 'words' in filename:
        columns_names = ["id", "seg_state", "gray_level", "x", "y", "w", "h",
                         "gram_tag", "transcription"]
        columns_type = {'id': str, 'seg_state': str, 'gray_level': np.uint8, 'x': np.int32,
                        'y': np.int32, 'w': np.int32, 'h': np.int32, 'gram_tag': str, 'transcription': str}

    data_lines = pd.read_csv(filename, delim_whitespace=True, comment='#',
                             encoding='utf8', error_bad_lines=False, header=None,
                             names=columns_names, dtype=columns_type, quoting=3)

    return data_lines


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


def create_experiment_csv(filename: str, image_directory: str, output_filename) -> None:
    """
    Generates the csv file needed to train tf_crnn with format : path_to_img_segment;transcription_non_formatted
    :param filename: filename of line / words in IAM/ascii
    :param image_directory: directory where the image segments are located
    :param output_filename: filename of the output .csv file
    :return:
    """
    # Load filename containing the desired segments info (lines/words)
    data_segments = load_ascii_txt_file(filename)

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
            # Remove space befor .,?!) chars
            transcription = re.sub(r'\s+([?.,;!)])', r'\1', transcription).strip()
        except AttributeError:
            print('Transcription does not exists in {}'.format(index))
            continue

        list_filenames.append(filename)
        list_transcriptions.append(transcription)

    # Create dataframe with filenames and transcriptions
    df = pd.DataFrame({'path': list_filenames, 'transcription': list_transcriptions})
    df.to_csv(output_filename, sep=';', encoding='utf-8', header=False, index=False, escapechar="\\", quoting=3)
