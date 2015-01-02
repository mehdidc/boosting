#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import datetime

from AppEnv import AppEnv
from AppParams import AppParams
from AppResult import AppResult
from AppTrainingMethod import AppTrainingMethod


class App(object):
    def __init__(self, app_env_class=AppEnv, app_params_class=AppParams, app_result_class=AppResult, app_training_method_class=AppTrainingMethod):
        assert  isinstance(app_env_class, type) \
            and isinstance(app_params_class, type) \
            and isinstance(app_result_class, type) \
            and isinstance(app_training_method_class, type)
        self.env = app_env_class()
        self.params = app_params_class()
        self.result = app_result_class()
        self.training_method = app_training_method_class()
        assert  isinstance(self.env, AppEnv) \
            and isinstance(self.params, AppParams) \
            and isinstance(self.result, AppResult) \
            and isinstance(self.training_method, AppTrainingMethod)

    def _Load(self):
        raise Exception("Not Implemented : Abstract class must be derivate")

    def _PreProcess(self):
        raise Exception("Not Implemented : Abstract class must be derivate")

    def _Process(self):
        raise Exception("Not Implemented : Abstract class must be derivate")

    def _PostProcess(self):
        raise Exception("Not Implemented : Abstract class must be derivate")
        
    def Run(self):
        if os.path.exists('%s/%s' % (self.params.output_folder, 'log.txt')):
            os.remove('%s/%s' % (self.params.output_folder, 'log.txt'))
        self.params.DebugPrint('Date Time : %s' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), printLvl=-1)
        self.params.printLvl = 0
        self._Load()
        self.params.printLvl = 0
        self.params.DebugPrint('Pre-processing')
        self.params.IncrPrintLvl()
        self._PreProcess()
        self.params.printLvl = 0
        self.params.DebugPrint('Processing')
        self.params.IncrPrintLvl()
        self._Process()
        self.params.printLvl = 0
        self.params.DebugPrint('Post-processing')
        self.params.IncrPrintLvl()
        self._PostProcess()
