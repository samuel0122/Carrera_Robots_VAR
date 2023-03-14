import numpy as np
from functools import cmp_to_key

class Valores:
    valor = 0
    def __init__(self, valor):
        self.valor = valor
    
    def incrementar(self):
        self.valor += 1

def comparar(e1, e2):
    return -1 if e1.valor > e2.valor else 1
"""
def comparar(e1, e2):
    print(f'{e1} : {e2}')
    if e1 < e2:
        return -1
    else:
        return 1

list = [5, 2, 3, 4]

print(len(list))
for i in range(len(list)):
    print(i)

list.sort(key=cmp_to_key(comparar))

print(list)


fileList = []
with open('/home/samuel/P1_Carrera_de_robots/src/all_listeners/models/prueba/list.txt', 'r') as f:
    fileList = [file for file in f.read().split('\n')]
    number = int(fileList[0])
    del fileList[0]

print(number)


print(fileList)
import os

fileList = [str(3)]
[fileList.append(f'file{i}') for i in range(9)]


# print(fileList)
# path = '/home/samuel/P1_Carrera_de_robots/src/all_listeners/models/prueba'

# if not os.path.exists(path):
#     os.mkdir(path) 

# with open('/home/samuel/P1_Carrera_de_robots/src/all_listeners/models/prueba/list.txt', 'w') as f:
#     f.write('\n'.join(fileList))

from datetime import datetime

# datetime object containing current date and time
now = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
 
print("now =", now)
"""

valores = [Valores(i) for i in range(10)]

print([v.valor for v in valores])

[v.incrementar() for v in valores]

print([v.valor for v in valores])

valores.sort(key=cmp_to_key(comparar))

print([v.valor for v in valores])


[v.incrementar() for v in valores]

print([v.valor for v in valores])
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

print(f'{bcolors.HEADER}HEADER {bcolors.ENDC}')
print(f'{bcolors.OKBLUE}OKBLUE {bcolors.ENDC}')
print(f'{bcolors.OKCYAN}OKCYAN {bcolors.ENDC}')
print(f'{bcolors.OKGREEN}OKGREEN{bcolors.ENDC}')
print(f'{bcolors.WARNING}WARNING{bcolors.ENDC}')
print(f'{bcolors.FAIL}FAIL{bcolors.ENDC}')
print(f'{bcolors.ENDC}ENDC{bcolors.ENDC}')
print(f'{bcolors.BOLD}BOLD{bcolors.ENDC}')
print(f'{bcolors.UNDERLINE}UNDERLINE{bcolors.ENDC}')
print(f'{bcolors.HEADER}{bcolors.BOLD}{bcolors.UNDERLINE}{bcolors.WARNING}EVERYTHIN{bcolors.ENDC}')