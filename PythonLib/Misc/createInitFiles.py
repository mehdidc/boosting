#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse

accepted_extensions = ['.py']
excluded_files = ['__init__.py']

def main():
    parser = argparse.ArgumentParser(description='Create __init__.py files of the directory.')
    parser.add_argument('directory', type=str, help='directory of the package')
    args = parser.parse_args()

    def Dirs(dir):
        return [f for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))]
    def Files(dir):
        return [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and os.path.splitext(f)[1] in accepted_extensions and f not in excluded_files]
    def write(dir, array):
        res = '#! /usr/bin/env python\n# -*- coding: utf-8 -*-\n\n__all__ = '
        res += str(array)
        file = open('%s/__init__.py' % dir, 'w')
        file.write(res)
        file.close()
    def recursiveFunction(root='.'):
        if os.path.exists(root):
            dirs = Dirs(root)
            files = Files(root)
            print root, dirs + files
            fun = lambda dir : dir if dir[-1] != '/' else fun(dir[:-1])
            write(fun(root), dirs + files)
            for d in dirs:
                if root[-1] != '/':
                    recursiveFunction(root="".join([root, '/', d]))
                else:
                    recursiveFunction(root="".join([root, d]))

    recursiveFunction(root=args.directory)

if __name__ == '__main__':
    main()