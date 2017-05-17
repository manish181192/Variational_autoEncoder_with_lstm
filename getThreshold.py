import numpy as np
f = open("/home/mvidyasa/Downloads/Result_Log")

lines = f.readlines()

s = np.zeros(3, dtype= np.float64)
max = np.zeros(3, dtype= np.float64)
min = np.full(3, 10,dtype= np.float64)
count = np.zeros(3, dtype= np.float64)
losses = {}
current_category = 0
for line in lines:

    splt = line.split(" ")
    if len(splt) !=4:
        print line
        continue

    if splt[0] == 'c2:':
        current_category = 1
    elif splt[0] == 'c3:':
        current_category = 2

    loss = float(splt[3])
    if not losses.has_key(current_category):
        losses[current_category] = []
    losses[current_category].append(loss)

    if loss > max[current_category]:
        max[current_category] = loss

    if loss < min[current_category]:
        min[current_category] = loss

    s[current_category] += loss
    count[current_category] +=1

mean = np.zeros(3)
mean[0] = s[0]/count[0]
mean[1] = s[1]/count[1]
mean[2] = s[2]/count[2]

sd = np.zeros(3, dtype= np.float64)
var = np.zeros(3, dtype= np.float64)
sum_dev = np.zeros(3, dtype= np.float64)

for i in range(3):
    for loss in losses.get(i):
        dev_ = pow(loss - mean[i], 2)
        sum_dev[i] += dev_

var[0] = sum_dev[0]/ count[0]
var[1] = sum_dev[1]/ count[1]
var[2] = sum_dev[2]/ count[2]

sd[0] = pow(var[0], 0.5)
sd[1] = pow(var[1], 0.5)
sd[2] = pow(var[2], 0.5)

for i in range(3):
    print " min : "+str(min[i])+", max : "+str(max[i])+", mean : "+str(mean[i])+", sd : "+str(sd[i])



