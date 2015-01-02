#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse

template = """#! /usr/bin/env python
# -*- coding: utf-8 -*-

def main():
    pass

if __name__ == '__main__':
    main()"""

def main():
    parser = argparse.ArgumentParser(description='Create new python files.')
    parser.add_argument('filename', type=str, help='filename of the new file')
    args = parser.parse_args()

    filename = args.filename
    if os.path.splitext(filename)[1] != '.py':
        filename = os.path.splitext(filename)[0] + '.py'
    outFile = open(filename, 'w')
    outFile.write(template)
    outFile.close()

if __name__ == '__main__':
    main()