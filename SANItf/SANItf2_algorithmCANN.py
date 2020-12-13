# -*- coding: utf-8 -*-
"""SANItf2_algorithmCANN.py

# Requirements:
Python 3 and Tensorflow 2.1+ 

# License:
MIT License

# Usage:
see SANItf2.py

# Description

Define fully connected classification artificial neural network (CANN) - unsupervised hebbian learning

- Author: Richard Bruce Baxter - Copyright (c) 2020 Baxter AI (baxterai.com)

"""

import tensorflow as tf
import numpy as np
from SANItf2_operations import *	#generateParameterNameSeq, generateParameterName, defineNetworkParameters
import SANItf2_globalDefs
import math

debugHebbianForwardPropOnlyTrainFinalSupervisedLayer = False
applyWmaxCap = True	#max W = 1
applyAmaxCap = True	#max A = 1
enableForgetting = True
#debugSparseActivatedNetwork = False	#creates much larger network


generateFirstLayerSDR = False	#approximates k winners takes all
if(generateFirstLayerSDR):
	maximumNetworkHiddenLayerNeuronsAsFractionOfInputNeurons = 10.0	#100.0
else:
	maximumNetworkHiddenLayerNeuronsAsFractionOfInputNeurons = 0.8	#0.8

if(generateFirstLayerSDR):
	ignoreFirstXlayersTraining = True	#this can be used to significantly increase the network activation sparsity
	if(ignoreFirstXlayersTraining):
		ignoreFirstXlayersTrainingX = 1
else:
	ignoreFirstXlayersTraining = False
		
applyNeuronThresholdBias = False	#this can be used to significantly increase the network activation sparsity
if(applyNeuronThresholdBias):
	applyNeuronThresholdBiasValue = 0.1
	applyNeuronThresholdBiasDuringTrainOnly = True
	if(generateFirstLayerSDR):
		applyNeuronThresholdBiasFirstLayerOnly = True 	#this can be used to significantly increase the network activation sparsity
	
onlyTrainNeuronsIfActivationContributionAboveThreshold = False	#theshold neurons which will be positively biased, and those which will be negatively (above a = 0 as it is currently) 
if(onlyTrainNeuronsIfActivationContributionAboveThreshold):
	onlyTrainNeuronsIfActivationContributionAboveThresholdValue = 0.1
	backpropCustomOnlyUpdateWeightsThatContributedTowardsTarget = True	#as not every neuron which fires contributes to the learning in fully connected network
		#requires trainHebbianBackprop

if(generateFirstLayerSDR):
	generateNetworkOptimumConvergence = True
else:
	generateNetworkOptimumConvergence = False
if(generateNetworkOptimumConvergence):
	networkDivergenceType = "nonLinearConverging"
	if(applyNeuronThresholdBias):
		#this will affect the optimimum convergence angle
		networkOptimumConvergenceAngle = 0.5+applyNeuronThresholdBiasValue
	else:
		networkOptimumConvergenceAngle = 0.5	#if angle > 0.5, then more obtuse triange, if < 0.5 then more acute triangle	#fractional angle between 0 and 90 degrees
	networkDivergence = 1.0-networkOptimumConvergenceAngle 
	#required for Logarithms with a Fraction as Base:
	networkDivergenceNumerator = int(networkDivergence*10)
	networkDivergenceDenominator = 10
else:
	networkDivergenceType = "linearConverging"
	#networkDivergenceType = "linearDivergingThenConverging"	#not yet coded
	

	
if(enableForgetting):
	enableForgettingRestrictToAPrevAndNotAConnections = True	#True	#this ensures that only connections between active lower layer neurons and unactive higher layer neurons are suppressed

W = {}
B = {}


#Network parameters
n_h = []
numberOfLayers = 0
numberOfNetworks = 0

learningRate = 0.0
forgetRate = 0.0
batchSize = 0


def defineTrainingParametersCANN(dataset, trainMultipleFiles):

	global learningRate
	global forgetRate
	global batchSize
	
	if(trainMultipleFiles):
		learningRate = 0.0001
		forgetRate = 0.00001
		if(dataset == "POStagSequence"):
			trainingSteps = 10000
		elif(dataset == "NewThyroid"):
			trainingSteps = 1000
		batchSize = 100
		numEpochs = 10
	else:
		learningRate = 0.001
		forgetRate = 0.001
		if(dataset == "POStagSequence"):
			trainingSteps = 10000
		elif(dataset == "NewThyroid"):
			trainingSteps = 1000
		batchSize = 10	#1	#10	#1000	#temporarily reduce batch size for visual debugging (array length) purposes)
		if(dataset == "NewThyroid"):
			numEpochs = 10
		else:
			numEpochs = 1
	
	displayStep = 100
				
	return learningRate, trainingSteps, batchSize, displayStep, numEpochs
	

def defineNetworkParametersCANN(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworksSet):

	global n_h
	global numberOfLayers
	global numberOfNetworks
	
	numberOfNetworks = numberOfNetworksSet

	if((networkDivergenceType == "linearConverging") or (networkDivergenceType == "nonLinearConverging")):
		firstHiddenLayerNumberNeurons = int(num_input_neurons*maximumNetworkHiddenLayerNeuronsAsFractionOfInputNeurons)
	
	if(generateNetworkOptimumConvergence):
		#(networkDivergenceType == "nonLinearConverging")
		#num_output_neurons = firstHiddenLayerNumberNeurons * networkDivergence^numLayers [eg 5 = 100*0.6^x]
		#if a^c = b, then c = log_a(b)
		b = float(num_output_neurons)/firstHiddenLayerNumberNeurons
		#numLayers = math.log(b, networkDivergence)
		
		#now log_a(x) = log_b(x)/log_b(a)
		#therefore log_a1/a2(b) = log_a2(b)/log_a2(a1/a2) = log_a2(b)/(log_a2(a1) - b)
		numberOfLayers = math.log(b, networkDivergenceDenominator)/math.log(float(networkDivergenceNumerator)/networkDivergenceDenominator, networkDivergenceDenominator)
		numberOfLayers = int(numberOfLayers)+1	#plus input layer
		
		print("numberOfLayers = ", numberOfLayers)
		
	else:
		if(dataset == "POStagSequence"):
			if(trainMultipleFiles):
				numberOfLayers = 6
			else:
				numberOfLayers = 3
		elif(dataset == "NewThyroid"):
			if(trainMultipleFiles):
				numberOfLayers = 6	#trainMultipleFiles should affect number of neurons/parameters in network
			else:
				numberOfLayers = 3

	n_x = num_input_neurons #datasetNumFeatures
	n_y = num_output_neurons  #datasetNumClasses
	n_h_first = n_x
	previousNumberLayerNeurons = n_h_first
	n_h.append(n_h_first)

	for l in range(1, numberOfLayers):	#for every hidden layer
		if(networkDivergenceType == "linearConverging"):
			if(l == 1):
				n_h_x = firstHiddenLayerNumberNeurons
			else:
				n_h_x = int((firstHiddenLayerNumberNeurons-num_output_neurons) * ((l-1)/(numberOfLayers-2)) + num_output_neurons)
			#previousNumberLayerNeurons = n_h_x
			n_h.append(n_h_x)
		elif(networkDivergenceType == "nonLinearConverging"):
			if(l == 1):
				n_h_x = firstHiddenLayerNumberNeurons
			else:
				n_h_x = int(previousNumberLayerNeurons*networkDivergence)
			n_h.append(n_h_x)
			previousNumberLayerNeurons = n_h_x
		elif(networkDivergenceType == "linearDivergingThenConverging"):
			#not yet coded
			print("defineNetworkParametersCANN error: linearDivergingThenConverging not yet coded")
			exit()
		else:
			print("defineNetworkParametersCANN error: unknown networkDivergenceType")
			exit()

	n_h_last = n_y
	n_h.append(n_h_last)
	
	print("defineNetworkParametersCANN, n_h = ", n_h)

	return numberOfLayers
	
	
def neuralNetworkPropagationCANN(x, networkIndex=1):
			
	AprevLayer = x
	 
	for l in range(1, numberOfLayers+1):
	
		#print("l = " + str(l))

		Z = tf.add(tf.matmul(AprevLayer, W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])
		A = reluCustom(Z, train=False)
			
		AprevLayer = A
			
	return tf.nn.softmax(Z)
	
	
def neuralNetworkPropagationCANNtrain(x, y=None, networkIndex=1, trainHebbianForwardprop=False, trainHebbianBackprop=False, trainHebbianLastLayerSupervision=False):

	#print("batchSize = ", batchSize)
	#print("learningRate = ", learningRate)
	
	AprevLayer = x

	Alayers = []
	if(trainHebbianBackprop):
		Alayers.append(AprevLayer)
	
	#print("x = ", x)
	
	for l in range(1, numberOfLayers+1):
	
		#print("\nl = " + str(l))

		Z = tf.add(tf.matmul(AprevLayer, W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])	
		A = reluCustom(Z, train=True)
		
		if(trainHebbianBackprop):
			Alayers.append(A)
					
		if(applyAmaxCap):
			A = tf.clip_by_value(A, clip_value_min=-1.0, clip_value_max=1.0)

		if(trainHebbianForwardprop):
			trainLayerCANN(y, networkIndex, l, AprevLayer, A, Alayers, trainHebbianBackprop=trainHebbianBackprop, trainHebbianLastLayerSupervision=trainHebbianLastLayerSupervision)

		AprevLayer = A
	
	if(trainHebbianBackprop):
		for l in reversed(range(1, numberOfLayers+1)):
			
			#print("Alayers[l] = ", Alayers[l])
			
			AprevLayer = Alayers[l-1]
			A = Alayers[l]
			
			trainLayerCANN(y, networkIndex, l, AprevLayer, A, Alayers, trainHebbianBackprop=trainHebbianBackprop, trainHebbianLastLayerSupervision=trainHebbianLastLayerSupervision)
							
	return tf.nn.softmax(Z)


def trainLayerCANN(y, networkIndex, l, AprevLayer, A, Alayers, trainHebbianBackprop=False, trainHebbianLastLayerSupervision=False):

		#print("train")
		isLastLayerSupervision = False
		if(trainHebbianLastLayerSupervision):
			if(l == numberOfLayers):
				isLastLayerSupervision = True
				#print("isLastLayerSupervision")

		trainLayer = True
		if(isLastLayerSupervision):
			#perform hebbian learning on last layer based on hypothetical correct one hot class activation (Ahypothetical)
			Alearn = y
		else:
			Alearn = A
			if(debugHebbianForwardPropOnlyTrainFinalSupervisedLayer):
				trainLayer = False
		if(ignoreFirstXlayersTraining):
			if(l <= ignoreFirstXlayersTrainingX):
				trainLayer = False

		if(trainLayer):
			#print("Alearn = ", Alearn)
					
			#update weights based on hebbian learning rule
			#strengthen those connections that caused the current layer neuron to fire (and weaken those that did not)

			AprevLayerLearn = AprevLayer
			if(onlyTrainNeuronsIfActivationContributionAboveThreshold):
				#apply threshold to AprevLayer
				AprevLayerAboveThreshold = tf.math.greater(AprevLayer, onlyTrainNeuronsIfActivationContributionAboveThresholdValue)
				AprevLayerAboveThresholdFloat = tf.dtypes.cast(AprevLayerAboveThreshold, dtype=tf.float32)
				AprevLayerLearn = AprevLayer*AprevLayerAboveThresholdFloat
			
			AcoincidenceMatrix = tf.matmul(tf.transpose(AprevLayerLearn), Alearn)
			#Bmod = 0*learningRate	#biases are not currently used
			Wmod = AcoincidenceMatrix/batchSize*learningRate
			#B[generateParameterNameNetwork(networkIndex, l, "B")] = B[generateParameterNameNetwork(networkIndex, l, "B")] + Bmod
			W[generateParameterNameNetwork(networkIndex, l, "W")] = W[generateParameterNameNetwork(networkIndex, l, "W")] + Wmod

			#print("Alearn = ", Alearn)
			#print("AprevLayerLearn = ", AprevLayerLearn)
			#print("A = ", A)
			#print("Alearn = ", Alearn)
			#print("AcoincidenceMatrix = ", AcoincidenceMatrix)
			#print("Wmod = ", Wmod)
			#print("W = ", W[generateParameterNameNetwork(networkIndex, l, "W")])

			if(enableForgetting):
				if(enableForgettingRestrictToAPrevAndNotAConnections):
					AboolNeg = tf.math.equal(Alearn, 0.0)	#Abool = tf.math.greater(Alearn, 0.0), AboolNeg = tf.math.logical_not(Abool)
					#print("Abool = ",Abool)
					#AboolNegInt = tf.dtypes.cast(AboolNeg, tf.int32)
					AboolNegFloat = tf.dtypes.cast(AboolNeg, tf.float32)
					AcoincidenceMatrixForget = tf.matmul(tf.transpose(AprevLayerLearn), AboolNegFloat)
					Wmod2 = tf.square(AcoincidenceMatrixForget*forgetRate)	#square is required to normalise the forget rate relative to the learn rate [assumes input tensor is < 1]
					#print("Wmod2 = ", Wmod2)
					W[generateParameterNameNetwork(networkIndex, l, "W")] = W[generateParameterNameNetwork(networkIndex, l, "W")] - Wmod2
				else:
					AcoincidenceMatrixIsZero = tf.math.equal(AcoincidenceMatrix, 0)
					#AcoincidenceMatrixIsZeroInt = tf.dtypes.cast(AcoincidenceMatrixIsZero, tf.int32)
					AcoincidenceMatrixIsZeroFloat = tf.dtypes.cast(AcoincidenceMatrixIsZero, dtype=tf.float32)
					Wmod2 = tf.square(AcoincidenceMatrixIsZeroFloat*forgetRate)	#square is required to normalise the forget rate relative to the learn rate [assumes input tensor is < 1]
					#print("Wmod2 = ", Wmod2)
					W[generateParameterNameNetwork(networkIndex, l, "W")] = W[generateParameterNameNetwork(networkIndex, l, "W")] - Wmod2

			if(applyWmaxCap):
				#print("W before cap = ", W[generateParameterNameNetwork(networkIndex, l, "W")])
				W[generateParameterNameNetwork(networkIndex, l, "W")] = tf.clip_by_value(W[generateParameterNameNetwork(networkIndex, l, "W")], clip_value_min=-1.0, clip_value_max=1.0)
				#print("W after cap = ", W[generateParameterNameNetwork(networkIndex, l, "W")])
				
			if(trainHebbianBackprop):
				if(backpropCustomOnlyUpdateWeightsThatContributedTowardsTarget):
					Alayers[l-1] = AprevLayerLearn	#deactivate AprevLayer during backprop based on threshold (to prevent non contributing activation paths to be learnt)
		
			


def defineNeuralNetworkParametersCANN():

	randomNormal = tf.initializers.RandomNormal()
	
	for networkIndex in range(1, numberOfNetworks+1):
	
		for l in range(1, numberOfLayers+1):

			W[generateParameterNameNetwork(networkIndex, l, "W")] = tf.Variable(randomNormal([n_h[l-1], n_h[l]]))
			B[generateParameterNameNetwork(networkIndex, l, "B")] = tf.Variable(tf.zeros(n_h[l]))

def reluCustom(Z, train):

	if(applyNeuronThresholdBias):
		applyBias = False
		if(applyNeuronThresholdBiasDuringTrainOnly):
			if(train):
				applyBias = True
		else:
			applyBias = True
		
		#Z = tf.clip_by_value(Z, min=applyNeuronThresholdBiasValue)	#clamp
		Z = Z - applyNeuronThresholdBiasValue
	
	A = tf.nn.relu(Z)
	
	return A

  