#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import csv


def export_list_to_file(list_to_export: list, output_filename: str):
    """

    :param list_to_export:
    :param output_filename:
    :return:
    """

    with open(output_filename, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for elem in list_to_export:
            writer.writerow([elem])