#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Synchronize with the virtual machine.')
    parser.add_argument('--user', type=str, help='user who have the permissions on the destination', default='root')
    parser.add_argument('virual_machine', type=str, help='VM with which to synchronize')
    parser.add_argument('src', type=str, help='source directory')
    parser.add_argument('dest', type=str, help='destination directory')
    args = parser.parse_args()

    cmd = ["rsync", "-avz", args.src, "%s@%s:%s" % (args.user, args.virual_machine, args.dest)]
    subprocess.call(cmd)

if __name__ == '__main__':
    main()