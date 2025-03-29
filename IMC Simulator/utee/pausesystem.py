# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:47:58 2024

@author: dalei
"""
import sys
def stop():
    continue_password = ""
    exit_password = "stop"  
    while True:
        user_input = input("enter code to continue or stop. \n")
        if user_input == continue_password:
            break
        elif user_input == exit_password:
            sys.exit()
        else:
            print("Error, retry.")