#! /usr/bin/env python
# -*- coding: utf-8 -*-

import uuid
import multiprocessing

class ProcessManager(object):
    def __init__(self):
        self.__processes = dict()
        self.__idUsed = []

    def get(self):
        return self.__processes

    def add(self, p):
        assert isinstance(p, multiprocessing.Process)
        id = self.__newId()
        self.__processes[id] = p
        return id

    def remove(self, id):
        if id is not None and self.__processes.has_key(id):
            del self.__processes[id]
            self.__idUsed.remove(id)

    def start(self, id=None):
        if id is not None:
            if isinstance(id, list):
                for id, p in [(key, val) for (key, val) in self.__processes.items() if key in id]:
                    p.start()
            elif self.__processes.has_key(id):
                self.__processes[id].start()
        else:
            for id, p in self.__processes.items():
                p.start()

    def join(self, id=None):
        if id is not None:
            if isinstance(id, list):
                for id, p in [(key, val) for (key, val) in self.__processes.items() if key in id]:
                    p.join()
            elif self.__processes.has_key(id):
                self.__processes[id].join()
        else:
            for id, p in self.__processes.items():
                p.join()

    def run(self, id=None):
        self.start(id)
        self.join(id)

    def __newId(self):
        id = uuid.uuid1()
        while id in self.__idUsed:
            id = uuid.uuid1()
        self.__idUsed.append(id)
        return id