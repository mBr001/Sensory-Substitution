import random 
import math
from matplotlib import pyplot as plt
import numpy as np

weightvalue = {}
weightUpdate = {}

# First entry is the bias neuron defaulted to 1

inputlayer = [1,1,1,1,1,1,1,1,1,1]
hiddenlayer1 = [1,1,1,1,1,1,1,1,1,1]
hiddenlayer2 = [1,1,1,1,1,1,1,1,1,1]
outputlayer = [1,1,1,1]


for k in range(1,len(hiddenlayer1)):
	for i in range(len(inputlayer)):
		weightvalue[str(k) + '-' + str(i) + '-' + str(1)] = random.uniform(-1,1)
		weightUpdate[str(k) + '-' + str(i) + '-' + str(1)] = 0

for k in range(1,len(hiddenlayer2)):
	for i in range(len(hiddenlayer1)):
		weightvalue[str(k) + '-' + str(i) + '-' + str(2)] = random.uniform(-1,1)
		weightUpdate[str(k) + '-' + str(i) + '-' + str(2)] = 0

for k in range(len(outputlayer)):
	for i in range(len(hiddenlayer2)):
		weightvalue[str(k) + '-' + str(i) + '-' + str(3)] = random.uniform(-1,1)
		weightUpdate[str(k) + '-' + str(i) + '-' + str(3)] = 0


# Sigmoid Definition
def sigmoid(x):
	return 1/(1 + math.exp(-x))


def feedForward(inputx):
	if len(inputx) == (len(inputlayer) - 1):
		for i in range(1,len(inputx) + 1):
			inputlayer[i] = inputx[i - 1]


	for k in range(1,len(hiddenlayer1)):
		sumvalue = 0
		for i in range(len(inputlayer)):
			sumvalue = sumvalue + weightvalue[str(k) + '-' + str(i)+ '-1']*inputlayer[i]
		hiddenlayer1[k] = sigmoid(sumvalue)

	for k in range(1,len(hiddenlayer2)):
		sumvalue = 0
		for i in range(len(hiddenlayer1)):
			sumvalue = sumvalue + weightvalue[str(k) + '-' + str(i)+ '-2']*hiddenlayer1[i]
		hiddenlayer2[k] = sigmoid(sumvalue)
	
	for k in range(len(outputlayer)):
		sumvalue = 0
		for i in range(len(hiddenlayer2)):
			sumvalue = sumvalue + weightvalue[str(k) + '-' + str(i)+ '-3']*hiddenlayer2[i]
		outputlayer[k] = sigmoid(sumvalue)


	return outputlayer


def identifyPattern(patternx):
	plustemplate = [0,1,0,1,1,1,0,1,0]
	patterndarkness = (patternx[1] + patternx[3] + patternx[4] + patternx[5] + patternx[7])/5
	backgrounddarkness = (patternx[0] + patternx[2] + patternx[6] + patternx[8])/4
	plusflag = 1
	if abs(patterndarkness - backgrounddarkness) > 0.4:
		for i in range(len(plustemplate)):
			if plustemplate[i] == 1:
				if abs(patternx[i] - patterndarkness) < 0.2:
					None
				else:
					plusflag = 0
					break
			if plustemplate[i] == 0:
				if abs(patternx[i] - backgrounddarkness) < 0.2:
					None
				else:
					plusflag = 0
					break
	else:
		plusflag = 0

	
	crosstemplate = [1,0,1,0,1,0,1,0,1]
	patterndarkness = (patternx[0] + patternx[2] + patternx[4] + patternx[6] + patternx[8])/5
	backgrounddarkness = (patternx[1] + patternx[3] + patternx[5] + patternx[7])/4
	crossflag = 1
	if abs(patterndarkness - backgrounddarkness) > 0.4:
		for i in range(len(crosstemplate)):
			if crosstemplate[i] == 1:
				if abs(patternx[i] - patterndarkness) < 0.2:
					None
				else:
					crossflag = 0
					break
			if crosstemplate[i] == 0:
				if abs(patternx[i] - backgrounddarkness) < 0.2:
					None
				else:
					crossflag = 0
					break
	else:
		crossflag = 0



	minustemplate = [0,0,0,1,1,1,0,0,0]
	patterndarkness = (patternx[3] + patternx[4] + patternx[5])/3
	backgrounddarkness = (patternx[0] + patternx[1] + patternx[2] + patternx[6] + patternx[7] + patternx[8])/6
	minusflag = 1
	if abs(patterndarkness - backgrounddarkness) > 0.4:
		for i in range(len(minustemplate)):
			if minustemplate[i] == 1:
				if abs(patternx[i] - patterndarkness) < 0.2:
					None
				else:
					minusflag = 0
					break
			if minustemplate[i] == 0:
				if abs(patternx[i] - backgrounddarkness) < 0.2:
					None
				else:
					minusflag = 0
					break
	else:
		minusflag = 0
	
	somethingelseflag = 0

	if minusflag == 0 and plusflag == 0 and crossflag == 0:
		somethingelseflag = 1

	return [plusflag, minusflag, crossflag, somethingelseflag]
			


def ANDfunction(x1,x2):
	x1 = int(x1)
	x2 = int(x2)

	if(x1 == 0 and x2 == 0):
		return 0
	elif(x1 == 1 and x2 == 0):
		return 0
	elif(x1 == 0 and x2 == 1):
		return 0
	elif(x1 == 1 and x2 == 1):
		return 1
	else:
		return 'Error in Inputs.'

def XORfunction(x1,x2):
	x1 = int(x1)
	x2 = int(x2)

	if(x1 == 0 and x2 == 0):
		return 0
	elif(x1 == 1 and x2 == 0):
		return 1
	elif(x1 == 0 and x2 == 1):
		return 1
	elif(x1 == 1 and x2 == 1):
		return 0
	else:
		return 'Error in Inputs.'


def errorValue(actual, ideal):
	return np.ndarray.tolist(actual - ideal)


def sigmoidDerivative(sigmoidValue):
	return sigmoidValue*(1 - sigmoidValue)


def nodeDelta(errorvalue, derivativeValue, neurontype, deltaslist, weightslist):
	if neurontype == 'outputNeuron':
		return -errorvalue*derivativeValue
	if neurontype == 'interiorNeuron':
		sumvalue = 0
		for i in range(len(deltaslist)):
			sumvalue = sumvalue + deltaslist[i]*weightslist[i]
		return sumvalue*derivativeValue

errorGradientOfWeight = {}

identifiedpattern = [0,0,0,0]
identifyoptions = [[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]]
identifyoptionstemporary = identifyoptions[:]

def backProp():
	global identifiedpattern, identifyoptionstemporary, identifyoptions
	patternValues = [[0,1,0,1,1,1,0,1,0],[0,0,0,1,1,1,0,0,0],[1,0,1,0,1,0,1,0,1]]

	# print(len(identifyoptionstemporary))
	
	while identifiedpattern not in identifyoptionstemporary:
		a = patternValues[0][:]
		template = random.choice(patternValues)[:]

		for k in range(len(patternValues[0])):
			a[k] = random.uniform(0,1)
			if (template[k] == 1):
				a[k] = 1

		identifiedpattern = identifyPattern(a)

	identifyoptionstemporary.remove(identifiedpattern)
	if len(identifyoptionstemporary) == 0:
		identifyoptionstemporary = identifyoptions[:]
		identifiedpattern = [0,0,0,0]

	# if random.randint(0,1) == 1:
	# 	a = random.choice(patternValues)

	feedForward(a)
	patternvalue = identifyPattern(a)
	# print(patternvalue)
	error = errorValue(np.asarray(outputlayer),np.asarray(patternvalue))

	nodeDeltaHiddenLayer1 = []
	for i in range(len(hiddenlayer1)):
		nodeDeltaHiddenLayer1.append(0)

	nodeDeltaHiddenLayer2 = []
	for i in range(len(hiddenlayer1)):
		nodeDeltaHiddenLayer2.append(0)

	nodeDeltaOutputLayer = []
	for i in range(len(outputlayer)):
		nodeDeltaOutputLayer.append(0)


	for i in range(len(outputlayer)):
		nodeDeltaOutputLayer[i] = nodeDelta(error[i], sigmoidDerivative(outputlayer[i]), 'outputNeuron',[],[])


	for i in range(1,len(hiddenlayer2)):
		deltas = []
		weights = []
		for k in range(len(outputlayer)):
			deltas.append(nodeDeltaOutputLayer[k])
			weights.append(weightvalue[str(k) + '-' + str(i) + '-3'])
		derivative = sigmoidDerivative(hiddenlayer2[i])
		nodeDeltaHiddenLayer2[i] = nodeDelta([], derivative, 'interiorNeuron',deltas, weights)

	for i in range(1,len(hiddenlayer1)):
		deltas = []
		weights = []
		for k in range(1,len(hiddenlayer2)):
			deltas.append(nodeDeltaHiddenLayer2[k])
			weights.append(weightvalue[str(k) + '-' + str(i) + '-2'])
		derivative = sigmoidDerivative(hiddenlayer1[i])
		nodeDeltaHiddenLayer1[i] = nodeDelta([], derivative, 'interiorNeuron',deltas, weights)

	for k in range(1,len(hiddenlayer1)):
		for i in range(len(inputlayer)):
			errorGradientOfWeight[str(k) + '-' + str(i) + '-' + str(1)] = nodeDeltaHiddenLayer1[k]*inputlayer[i]

	for k in range(1,len(hiddenlayer2)):
		for i in range(len(hiddenlayer1)):
			errorGradientOfWeight[str(k) + '-' + str(i) + '-' + str(2)] = nodeDeltaHiddenLayer2[k]*hiddenlayer1[i]

	for k in range(len(outputlayer)):
		for i in range(len(hiddenlayer2)):
			errorGradientOfWeight[str(k) + '-' + str(i) + '-' + str(3)] = nodeDeltaOutputLayer[k]*hiddenlayer2[i]

	for k in range(1,len(hiddenlayer1)):
		for i in range(len(inputlayer)):
			weightUpdate[str(k) + '-' + str(i) + '-' + str(1)] = (epsilon*errorGradientOfWeight[str(k) + '-' + str(i) + '-' + str(1)]) + (alpha*weightUpdate[str(k) + '-' + str(i) + '-' + str(1)])
			weightvalue[str(k) + '-' + str(i) + '-' + str(1)] = weightUpdate[str(k) + '-' + str(i) + '-' + str(1)] + weightvalue[str(k) + '-' + str(i) + '-' + str(1)]

	for k in range(1,len(hiddenlayer2)):
		for i in range(len(hiddenlayer1)):
			weightUpdate[str(k) + '-' + str(i) + '-' + str(2)] = (epsilon*errorGradientOfWeight[str(k) + '-' + str(i) + '-' + str(2)]) + (alpha*weightUpdate[str(k) + '-' + str(i) + '-' + str(2)])
			weightvalue[str(k) + '-' + str(i) + '-' + str(2)] = weightUpdate[str(k) + '-' + str(i) + '-' + str(2)] + weightvalue[str(k) + '-' + str(i) + '-' + str(2)]

	for k in range(len(outputlayer)):
		for i in range(len(hiddenlayer2)):
			weightUpdate[str(k) + '-' + str(i) + '-' + str(3)] = (epsilon*errorGradientOfWeight[str(k) + '-' + str(i) + '-' + str(3)]) + (alpha*weightUpdate[str(k) + '-' + str(i) + '-' + str(3)])
			weightvalue[str(k) + '-' + str(i) + '-' + str(3)] = weightUpdate[str(k) + '-' + str(i) + '-' + str(3)] + weightvalue[str(k) + '-' + str(i) + '-' + str(3)]

	return error

epsilon = 0.7
alpha = 0.3

patternValues = [[0,1,0,1,1,1,0,1,0],[0,0,0,1,1,1,0,0,0],[1,0,1,0,1,0,1,0,1]]

# print(inputlayer)
# print(outputlayer)

errors = []

for iterC in range(100000):
	er = backProp()
	errors.append(er)
	if iterC % 500 == 0:
		print(iterC)
		print(er)
	# print(er)

# print(a)
# print()
# print(feedForward(a))
# trainingpairs = [[0,0],[0,1],[1,0],[1,1]]

for i in range(20):
	a = patternValues[0][:]
	template = random.choice(patternValues)[:]

	for k in range(len(patternValues[0])):
		a[k] = random.uniform(0,1)
		if (template[k] == 1):
			a[k] = 1

	chosenfromstandard = 0
	if random.randint(0,1) == 1:
		a = random.choice(patternValues)
		chosenfromstandard = 1

	actualValue = feedForward(a)

	idealValue = identifyPattern(a)
	if(idealValue != [0,0,0,1]):
		print("The Output of the Feed Foward Propagation with input %s is %s. Ideal Value is %s." % (a,str(actualValue),str(idealValue)))

# import numpy as np
# from matplotlib.pylab import *

# a = np.array(a)

# for i in range(len(a)):
# 	a[i] = 1 - a[i]
# plt.matshow(a.reshape((3,3)),cmap=cm.gray)
plt.plot([abs(item[0]) for item in errors])
plt.show()
plt.plot([abs(item[1]) for item in errors])
plt.show()
plt.plot([abs(item[2]) for item in errors])
plt.show()
plt.plot([abs(item[3]) for item in errors])
plt.show()