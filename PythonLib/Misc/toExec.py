#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import stat

accepted_extensions = ['.py']

def main():
    parser = argparse.ArgumentParser(description='Add execute permissions to the files.')
    parser.add_argument('filename', type=str, help='filename of the file')
    args = parser.parse_args()

    if os.path.splitext(args.filename)[1] not in accepted_extensions:
        raise Exception('Incorrect Extensions')

    st = os.stat(args.filename)
    os.chmod(args.filename, st.st_mode | stat.S_IEXEC)

if __name__ == '__main__':
    main()