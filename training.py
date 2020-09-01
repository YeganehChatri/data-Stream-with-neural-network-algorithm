from collections import deque
from scipy.spatial import distance
from numpy import array
from numpy import sum
from numpy import mean
from numpy import std
from Update import UpdateCBCE
import json
import csv
import time
import matplotlib

classLabel = []
ensemble = {}
finalCF = {}
potentialCF = {}
prediction = {}

b = 0.9
i = 1
recvQueue = deque([])

##create file for predicted test values

f = open("prediction.txt", "w+")

##read data

count = 0
with open('data.csv', "rt", encoding="utf8") as f:
    reader = csv.reader(f)
    for row in reader:

        if (i % 100) == 0:
            print ("vared if shodeh")
            UpdateCBCE(recvQueue)
            print("az UpDateCBCE kharej shodeh")
           ## recvQueue = []
            i = i + 1

            print (i)
            time.sleep(4)

        else:
            print("vared else shodeh")
            print(row)
            recvQueue.append(row)
            i = i+1
            print(i)
        if i == 1025:
            UpdateCBCE(recvQueue)

