"""LIANNtf_algorithmVICRegANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see LIANNtf_main.py

# Usage:
see LIANNtf_main.py

# Description:
LIANNtf algorithm VICRegANN - define Variance-Invariance-Covariance Regularization artificial neural network - supervised greedy learning implementation (force neural independence)

Emulates unsupervised singular value decomposition (SVD/factor analysis) learning for all hidden layers

"""

import tensorflow as tf
import numpy as np
from ANNtf2_operations import *	#generateParameterNameSeq, generateParameterName, defineNetworkParameters
import ANNtf2_operations
import ANNtf2_globalDefs
import LIANNtf_algorithmLIANN_math
import copy

#learningAlgorithmVICRegSupervisedGreedy = True

trainGreedy = True	#greedy incremental layer training - else backprop through all layers

#debug parameters;
debugFastTrain = False
debugSmallBatchSize = False	#small batch size for debugging matrix output

#intialise network properties (configurable);
supportSkipLayers = False #fully connected skip layer network	#TODO: add support for skip layers	#see ANNtf2_algorithmFBANN for template

#intialise network properties;
largeBatchSize = False
generateLargeNetwork = False	#large number of layer neurons is required for learningAlgorithmHebbian:useZAcoincidenceMatrix
generateNetworkStatic = False


#network/activation parameters;
#forward excitatory connections;
W = {}
B = {}

learningRate = 0.0	#defined by defineTrainingParametersVICRegANN

generateDeepNetwork = False
generateVeryLargeNetwork = False

if(generateVeryLargeNetwork):
	generateLargeNetworkRatio = 100	#100	#default: 10
else:
	if(generateLargeNetwork):
		generateLargeNetworkRatio = 3
	else:
		generateLargeNetworkRatio = 1

Wmean = 0.0
WstdDev = 0.05	#stddev of weight initialisations
	
	
#Network parameters
n_h = []
numberOfLayers = 0
numberOfNetworks = 0
datasetNumClasses = 0


#note high batchSize is required for learningAlgorithmStochastic algorithm objective functions (>= 100)
def defineTrainingParameters(dataset):
	global learningRate
	global weightDecayRate	
	
	learningRate = 0.005
	
	if(debugSmallBatchSize):
		batchSize = 10
	else:
		if(largeBatchSize):
			batchSize = 1000	#current implementation: batch size should contain all examples in training set
		else:
			batchSize = 100	#3	#100
	if(generateDeepNetwork):
		numEpochs = 100	#higher num epochs required for convergence
	else:
		numEpochs = 10	#100 #10
	if(debugFastTrain):
		trainingSteps = batchSize
	else:
		trainingSteps = 10000	#1000
		
	displayStep = 100
			
	return learningRate, trainingSteps, batchSize, displayStep, numEpochs
	


def defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet):

	global n_h
	global numberOfLayers
	global numberOfNetworks
	global datasetNumClasses

	firstHiddenLayerNumberNeurons = num_input_neurons*generateLargeNetworkRatio
	if(generateDeepNetwork):
		numberOfLayers = 3
	else:
		numberOfLayers = 2
	n_h, numberOfLayers, numberOfNetworks, datasetNumClasses = defineNetworkParametersDynamic(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet, numberOfLayers, firstHiddenLayerNumberNeurons, generateNetworkStatic)		
	
	return numberOfLayers
	
def defineNeuralNetworkParameters():

	print("numberOfNetworks", numberOfNetworks)
	
	global randomNormal
	randomNormal = tf.initializers.RandomNormal(mean=Wmean, stddev=WstdDev)
	#randomNormal = tf.initializers.RandomNormal()
	randomNormalFinalLayer = tf.initializers.RandomNormal()
	
	for networkIndex in range(1, numberOfNetworks+1):
		for l1 in range(1, numberOfLayers+1):
			#forward excitatory connections;
			EWlayer = randomNormal([n_h[l1-1], n_h[l1]]) 
			EBlayer = tf.zeros(n_h[l1])
			W[generateParameterNameNetwork(networkIndex, l1, "W")] = tf.Variable(EWlayer)
			B[generateParameterNameNetwork(networkIndex, l1, "B")] = tf.Variable(EBlayer)
					
def neuralNetworkPropagation(x, networkIndex=1):
	return neuralNetworkPropagationVICRegANNtest(x, networkIndex)

def neuralNetworkPropagationVICRegANNtest(x, networkIndex=1, l=None):
	return neuralNetworkPropagationVICRegANNminimal(x, networkIndex, l)

#def neuralNetworkPropagationLayer(x, networkIndex=1, l=None):
#	return neuralNetworkPropagationVICRegANNminimal(x, networkIndex, l)
	
	
def neuralNetworkPropagationVICRegANNminimal(x, networkIndex=1, l=None):

	x = x[:, 0]	#only optimise final layer weights for first experience in matched class pair
	
	if(l == None):
		maxLayer = numberOfLayers
	elif(l == numberOfLayers):
		maxLayer = numberOfLayers
	else:
		print("neuralNetworkPropagationVICRegANNminimal error: layerToTrain == maxLayer")
		exit()
		
	AprevLayer = x
	ZprevLayer = x
	for l in range(1, maxLayer+1):
					
		A, Z = forwardIteration(networkIndex, AprevLayer, ZprevLayer, l)

		A = tf.stop_gradient(A)	#only train final layer weights

		AprevLayer = A
		ZprevLayer = Z

	return tf.nn.softmax(Z)
	
def neuralNetworkPropagationVICRegANNtrain(x, layerToTrain, networkIndex=1):

	x1 = x[:, 0]
	x2 = x[:, 1]
	
	if(layerToTrain == numberOfLayers):
		print("neuralNetworkPropagationVICRegANNtrain error: layerToTrain == numberOfLayers")
		exit()
	else:
		maxLayer = layerToTrain

		
	AprevLayer1 = x1
	ZprevLayer1 = x1
	AprevLayer2 = x2
	ZprevLayer2 = x2
	for l in range(1, maxLayer+1):
					
		trainLayer = False
		if(l == layerToTrain):
			trainLayer = True
	
		A1, Z1 = forwardIteration(networkIndex, AprevLayer1, ZprevLayer1, l)
		A2, Z2 = forwardIteration(networkIndex, AprevLayer2, ZprevLayer2, l)
		
		if(trainLayer):
			#CHECKTHIS: verify learning algorithm (how to modify weights to maximise independence between neurons on each layer)
			#add tech notes here from do list log
			loss = calculatePropagationLossVICRegANN(A1, A2)
			
		if(trainGreedy):
			A1 = tf.stop_gradient(A1)	#only train trainLayer layer weights
			A2 = tf.stop_gradient(A2)	#only train trainLayer layer weights
			
		AprevLayer1 = A1
		ZprevLayer1 = Z1
		AprevLayer2 = A2
		ZprevLayer2 = Z2

	return loss

	
def neuralNetworkPropagationVICRegANNlearningAlgorithm(networkIndex, AprevLayer, ZprevLayer, l1, enableInhibition, randomlyActivateWeights):
	pass


def forwardIteration(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition=False, randomlyActivateWeights=False):
	#forward excitatory connections;
	EW = W[generateParameterNameNetwork(networkIndex, l, "W")]
	print("forwardIteration: EW = ", EW)
	Z = tf.add(tf.matmul(AprevLayer, EW), B[generateParameterNameNetwork(networkIndex, l, "B")])
	A = activationFunction(Z)
	return A, Z



def activationFunction(Z):
	A = tf.nn.relu(Z)
	return A


def generateTFtrainDataFromNParraysVICRegANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses):

	#trainDataList iterator format x: [batchSize][withinClassComparisonSize=2][numberOfInputNeurons]
	#trainDataList iterator format y: [batchSize][withinClassComparisonSize=2] - not one-hot encoded

	#note shuffleSize = datasetNumExamples
	
	trainDataList = []
	
	#a list of size n full of empty lists
	train_xClassPairs = [[] for _ in range(datasetNumClasses)]
	train_yClassPairs = [[] for _ in range(datasetNumClasses)]
	
	for classTarget in range(datasetNumClasses):
		train_xClassFiltered, train_yClassFiltered = filterNParraysByClassTarget(train_x, train_y, classTargetFilterIndex=classTarget)
		#store every unique pair within class;
		for classExampleIndex1 in range(train_xClassFiltered.shape[0]):
			for classExampleIndex2 in range(train_xClassFiltered.shape[0]):
				if(classExampleIndex1 != classExampleIndex2):
					train_xClassPair = np.stack((train_xClassFiltered[classExampleIndex1], train_xClassFiltered[classExampleIndex2]), axis=0)
					train_yClassPair = np.stack((train_yClassFiltered[classExampleIndex1], train_yClassFiltered[classExampleIndex2]), axis=0)
					train_xClassPairs[classTarget].append(train_xClassPair)
					train_yClassPairs[classTarget].append(train_yClassPair)
	
	#collapse 2d list into 1d list
	train_xPairs = [j for sub in train_xClassPairs for j in sub]
	train_yPairs = [j for sub in train_yClassPairs for j in sub]
	train_xPairsNP = np.array(train_xPairs)
	train_yPairsNP = np.array(train_yPairs)
	
	shuffleSizeNew = train_xPairsNP.shape[0]	#shuffle_buffer_size: For perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required - https://www.tensorflow.org/api_docs/python/tf/data/Dataset
	print("train_xPairsNP.shape = ", train_xPairsNP.shape)
	print("train_yPairsNP.shape = ", train_yPairsNP.shape)
	print("shuffleSizeNew = ", shuffleSizeNew)

	trainDataUnbatched = generateTFtrainDataUnbatchedFromNParrays(train_xPairsNP, train_yPairsNP)
	trainData = generateTFtrainDataFromTrainDataUnbatched(trainDataUnbatched, shuffleSizeNew, batchSize)
		
	return trainData	


def calculatePropagationLossVICRegANN(A1, A2):
	batchVariance1 = calculateVarianceBatch(A1)
	batchVariance2 = calculateVarianceBatch(A2)
	batchVariance = (batchVariance1+batchVariance2)/2.0
	matchedClassPairSimilarity = calculateSimilarityPair(A1, A2)
	covariance1 = calculateCovariance(A1)
	covariance2 = calculateCovariance(A2)
	covariance = (covariance1+covariance2)/2.0
	loss = (covariance)/(batchVariance * matchedClassPairSimilarity)
		#loss = (matchedClassPairVariance * covariance)/batchVariance
		#1.0/(batchVariance * (1.0/matchedClassPairVariance) * (1.0/covariance))
	print("loss = ", loss)
	return loss
	
def calculateVarianceBatch(A):
	#batchVariance = tf.math.reduce_variance(A, axis=0)
	print("A = ", A)
	batchVariance = tf.math.reduce_std(A, axis=0)	#VICReg implementation uses stddev not variance
	batchVariance = tf.math.reduce_mean(batchVariance)
	print("batchVariance = ", batchVariance)
	return batchVariance
	
def calculateSimilarityPair(A1, A2):
	#Apair = tf.stack([A1, A2])
	#matchedClassPairVariance = tf.math.reduce_variance(Apair, axis=0)
	#matchedClassPairVariance = tf.math.tf.math.reduce_std(Apair, axis=0)
	#matchedClassPairSimilarity = 1.0/matchedClassPairVariance
	matchedClassPairSimilarity = tf.math.abs(tf.math.subtract(A1, A2))	#CHECKTHIS
	matchedClassPairSimilarity = tf.math.reduce_mean(matchedClassPairSimilarity)
	print("matchedClassPairSimilarity = ", matchedClassPairSimilarity)
	return matchedClassPairSimilarity
	
def calculateCovariance(A):
	covariance = LIANNtf_algorithmLIANN_math.calculateCovarianceMean(A)
	print("covariance = ", covariance)
	return covariance
	
			

