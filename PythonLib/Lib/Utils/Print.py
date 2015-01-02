#! /usr/bin/env python
# -*- coding: utf-8 -*-

def Print(msg, lvl=-1, filename=None):
    if lvl < 0:
        res = msg
    elif lvl == 0:
        res = "---> %s" % msg
    else:
        res = "--->"
        for i in xrange(lvl - 1):
            res = "---!%s" % res
        res = "%s %s" % (res, msg)
    if filename is None:
        print res
    else:
        assert isinstance(filename, str)
        file = open(filename, 'a')
        file.write('%s\n' % res)
        file.close()