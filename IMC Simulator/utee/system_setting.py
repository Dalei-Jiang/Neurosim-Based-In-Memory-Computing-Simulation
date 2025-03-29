# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:36:59 2024

@author: dalei
"""
import os
import platform
import subprocess

def check_script_execution():
    os_type = platform.system()
    enr = str(os_type)
    return enr


