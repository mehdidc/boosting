#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import re

clean_regex = ['.pyc$', '~$']

def main():
    parser = argparse.ArgumentParser(description='Clean the directory.')
    parser.add_argument('directory', type=str, help='directory to clean', default='.')
    args = parser.parse_args()

    for root, dirs, files in os.walk(args.directory):
        for f in files:
            if not all([re.search(pattern, f) is None for pattern in clean_regex]):
                if root[-1] != '/':
                    file = "".join([root, '/', f])
                else:
                    file = "".join([root, f])
                print 'Removing :', file
                os.remove(file)

if __name__ == '__main__':
    main()