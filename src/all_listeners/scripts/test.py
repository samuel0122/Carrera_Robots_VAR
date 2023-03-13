import numpy as np
from functools import cmp_to_key


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
