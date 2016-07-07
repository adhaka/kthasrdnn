# @author:akash
# package:kthasrdnn

# just a useful script which will  parse the training log and plot a graph of the two costs against epochs in pre-training. 


import numpy as np
from matplotlib import pyplot as plt
import argparse
import re 


def number_extractor(strval):
	nums = re.findall(r'\d+', strval)
	return nums

def cost_extractor(strval):
	nums = re.findall('\d+\.\d+', strval)
	nums = map(float, nums)
	return nums


parser = argparse.ArgumentParser(description='mention the log text file')

# parser.add_argument('--filename', '-f', type=string)

args = parser.parse_args()
# filename = args.filename


filename = 'sse_timit_10000hu_10percent_0.015_beta350.txt'

f = open(filename, 'r')

linecount = 0
listlines = []
pretrain_index = 0
epoch_lines = []
cost_lines=[]
pretraining_flag=True

for line in f:
	line = line.lower()
	print line

	listlines.append(line)
	linecount += 1
	if  'pretrainining' in line: 
		pretrain_index = linecount
		print 'lol'

	if 'final' in line:
		finaltrain_index = linecount
		pretraining_flag = False

	if 'epoch' in line and pretraining_flag is True:
		nums = number_extractor(line)
		epoch_lines.append(int(nums[0]))
		continue

	if 'cost' in line and pretraining_flag is True:
		nums = cost_extractor(line)
		cost_lines.append(nums)
		continue
	


print len(listlines)
print epoch_lines
print cost_lines

epoch_mat = np.asarray(epoch_lines)
cost_mat = np.asarray(cost_lines)

print cost_mat
print pretrain_index

# epoch_lines = epoch_lines[pretrain_index:finaltrain_index]

print len(epoch_lines)
print len(cost_lines)

max_epoch = np.max(epoch_mat)
print max_epoch
epoch_mat = epoch_mat[:max_epoch]
cost_mat = cost_mat[:max_epoch,:]
c1 = cost_mat[:,1]
c2 = cost_mat[:,2]

# start plotting functionality here ....

plt.figure(1)
plt.subplot(211)

print epoch_mat.shape
# print cost_mat.shape[:1]
e1 = epoch_mat.tolist()
# plt.plot(epoch_mat, cost_mat[:1,:], '-o', label='reconstruction cost')
plt.plot(e1, c1, '-o', label='reconstruction cost')
plt.legend(loc='best')
plt.xlim((0,80))
plt.xlabel('epochs')
plt.ylabel('reconstruction cost')
plt.title('cost')

plt.subplot(212)
plt.plot(e1, c2, '-^', label='classification cost')
plt.legend(loc='best')
plt.xlim((0, 80))
plt.xlabel('epochs')
plt.ylabel('classification loss')
plt.savefig('cost-timit-10percent.pdf')
# plt.show()






