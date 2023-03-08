"""LIANNtf_algorithmLIANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see LIANNtf_main.py

# Usage:
see LIANNtf_main.py

# Description:
LIANNtf algorithm LIANN - define local inhibition artificial neural network (force neural independence)

Emulates unsupervised singular value decomposition (SVD/factor analysis) learning for all hidden layers

"""

import tensorflow as tf
import numpy as np
from ANNtf2_operations import *	#generateParameterNameSeq, generateParameterName, defineNetworkParameters
import ANNtf2_operations
import ANNtf2_globalDefs
import LIANNtf_algorithmLIANN_math
import copy


#select learningAlgorithm (unsupervised learning algorithm for intermediate/hidden layers):
learningAlgorithmNone = False	#create a very large network (eg x10) neurons per layer, and perform final layer backprop only
learningAlgorithmCorrelationReset = False	#minimise correlation between layer neurons	#create a very large network (eg x10) neurons per layer, remove/reinitialise neurons that are highly correlated (redundant/not necessary to end performance), and perform final layer backprop only	#orig mode
learningAlgorithmPCA = False	#note layer construction is nonlinear (use ANNtf2_algorithmAEANN/autoencoder for nonlinear dimensionality reduction simulation)	#incomplete
learningAlgorithmCorrelationStocasticOptimise = False	#stochastic optimise weights based on objective function; minimise the correlation between layer neurons
learningAlgorithmIndependenceReset = False	#randomise neuron weights until layer output independence is detected
learningAlgorithmMaximiseAndEvenSignalStochasticOptimise = False	#stochastic optimise weights based on objective functions; #1: maximise the signal (ie successfully uninhibited) across multiple batches (entire dataset), #2: ensure that all layer neurons receive even activation across multiple batches (entire dataset)	
learningAlgorithmUninhibitedImpermanenceReset = False	#increase the permanence of uninhibited neuron weights, and stocastically modify weights based on their impermanence
learningAlgorithmUninhibitedHebbianStrengthen = False	#strengthen weights of successfully activated neurons
learningAlgorithmPerformanceInhibitStocasticOptimise = True	#learn to inhibit neurons in net for a given task, maximising final layer performance	#neurons remain permanently inhibited, not just during training of weights; see Nactive
learningAlgorithmUnnormalisedActivityReset = False	#ensure that average layer activation lies between a minimum and maximum level	#regularise layer neural activity


#intialise network properties (configurable);
positiveExcitatoryWeights = False	#requires testing	#required for biological plausibility of most learningAlgorithms
positiveExcitatoryWeightsActivationFunctionOffsetDisable = False	
supportSkipLayers = False #fully connected skip layer network	#TODO: add support for skip layers	#see ANNtf2_algorithmFBANN for template
supportMultipleNetworks = True


#intialise network properties;
largeBatchSize = False
generateLargeNetwork = False	#large number of layer neurons is required for learningAlgorithmUninhibitedHebbianStrengthen:useZAcoincidenceMatrix
generateNetworkStatic = False
generateDeepNetwork = False
generateVeryLargeNetwork = False


#debug parameters;
debugFastTrain = False
debugSmallBatchSize = False	#small batch size for debugging matrix output


#select learningAlgorithmFinalLayer (supervised learning algorithm for final layer/testing):
learningAlgorithmFinalLayerBackpropHebbian = True	#only apply backprop (effective hebbian) learning at final layer
if(learningAlgorithmFinalLayerBackpropHebbian):
	positiveExcitatoryWeightsFinalLayer = False	#allow negative weights on final layer to emulate standard backprop/hebbian learning


#default sparsity
estNetworkActivationSparsity = 0.5	#50% of neurons are expected to be active during standard propagation (no inhibition)


#intialise algorithm specific parameters;
inhibitionAlgorithmBinary = False	#simplified inhibition algorithm implementation - binary on/off
inhibitionAlgorithmArtificial = False	#simplified inhibition algorithm implementation
inhibitionAlgorithmArtificialMoreThanXLateralNeuronActive = False	#inhibit layer if more than x lateral neurons active
inhibitionAlgorithmArtificialSparsity = False	#inhibition signal increases with number of simultaneously active neurons
#inhibitionAlgorithmSimulation = True	#default inhibition algorithm (propagate signal through simulated inhibitory neurons)

#A thresholding;
Athreshold = False
AthresholdValue = 1.0	#do not allow output signal to exceed 1.0


#LIANN hidden layer vs final layer hebbian execution staging;
supportDimensionalityReductionLimitFrequency = False
supportDimensionalityReductionFirstPhaseOnly = True	#perform LIANN in first phase only (x epochs of training), then apply hebbian learning at final layer
if(supportDimensionalityReductionFirstPhaseOnly):
	supportDimensionalityReductionLimitFrequency = False
	supportDimensionalityReductionFirstPhaseOnlyNumEpochs = 1
else:
	supportDimensionalityReductionLimitFrequency = True
	if(supportDimensionalityReductionLimitFrequency):
		supportDimensionalityReductionLimitFrequencyStep = 1000


#intialise algorithm specific parameters;
enableInhibitionTrainAndInhibitSpecificLayerOnly = True
applyInhibitoryNetworkDuringTest = False
randomlyActivateWeightsDuringTrain = False


#enable shared stocastic optimisation parameters;
learningAlgorithmStochastic = False
if(learningAlgorithmCorrelationStocasticOptimise):
	learningAlgorithmStochastic = True
elif(learningAlgorithmMaximiseAndEvenSignalStochasticOptimise):
	learningAlgorithmStochastic = True


#learning algorithm customisation;
if(learningAlgorithmNone):
	#can pass different task datasets through a shared randomised net
	#positiveExcitatoryWeights = True	#optional
	generateDeepNetwork = True	#optional	#used for algorithm testing
	generateVeryLargeNetwork = True
	generateNetworkStatic = True
elif(learningAlgorithmCorrelationReset):
	#note learningAlgorithmCorrelationReset requires supportSkipLayers - see LIANNtf_algorithmIndependentInput/AEANNtf_algorithmIndependentInput:learningAlgorithmLIANN for similar implementation
	#positiveExcitatoryWeights = True	#optional	

	maxCorrelation = 0.95	#requires tuning
	supportDimensionalityReductionRandomise	= True	#randomise weights of highly correlated neurons, else zero them (effectively eliminating neuron from network, as its weights are no longer able to be trained)
	
	generateDeepNetwork = True	#optional	#used for algorithm testing
	generateVeryLargeNetwork = True
	generateNetworkStatic = True
elif(learningAlgorithmPCA):
	#positiveExcitatoryWeights = True	#optional
	largeBatchSize = True	#1 PCA is performed across entire dataset [per layer]
elif(learningAlgorithmIndependenceReset):
	#positiveExcitatoryWeights = True	#optional
	inhibitionAlgorithmArtificialMoreThanXLateralNeuronActive = True	#mandatory
	fractionIndependentInstancesAcrossBatchRequired = 0.3	#divide by number of neurons on layer	#if found x% of independent instances, then record neuron as independent (solidify weights)	#FUTURE: will depend on number of neurons on current layer and previous layer	#CHECKTHIS: requires calibration
	largeBatchSize = True
elif(learningAlgorithmStochastic):
	inhibitionAlgorithmArtificialMoreThanXLateralNeuronActive = True	#optional
	if(learningAlgorithmCorrelationStocasticOptimise):
		learningAlgorithmStochasticAlgorithm = "correlation"
		#positiveExcitatoryWeights = True	#optional
		#learning objective function: minimise the correlation between layer neurons
	elif(learningAlgorithmMaximiseAndEvenSignalStochasticOptimise):
		learningAlgorithmStochasticAlgorithm = "maximiseAndEvenSignal"
		#positiveExcitatoryWeights = True	#optional?
		#learning objective functions:
			#1: maximise the signal (ie successfully uninhibited) across multiple batches (entire dataset)
			#2: ensure that all layer neurons receive even activation across multiple batches (entire dataset)		
		metric1Weighting = 1.0
		metric2Weighting = 1000.0	#normalise metric2Weighting relative to metric1Weighting; eg metric1 =  0.9575, metric2 =  0.000863842
	numberStochasticIterations = 10
	updateParameterSubsetSimultaneously = False	#current tests indiciate this is not required/beneficial with significantly high batch size
	if(updateParameterSubsetSimultaneously):
		numberOfSubsetsTrialledPerBaseParameter = 10	#decreases speed, but provides more robust parameter updates
		parameterUpdateSubsetSize = 5	#decreases speed, but provides more robust parameter updates
	else:
		numberOfSubsetsTrialledPerBaseParameter = 1
		parameterUpdateSubsetSize = 1	
	NETWORK_PARAM_INDEX_TYPE = 0
	NETWORK_PARAM_INDEX_LAYER = 1
	NETWORK_PARAM_INDEX_H_CURRENT_LAYER = 2
	NETWORK_PARAM_INDEX_H_PREVIOUS_LAYER = 3
	NETWORK_PARAM_INDEX_VARIATION_DIRECTION = 4
elif(learningAlgorithmUninhibitedImpermanenceReset):
	inhibitionAlgorithmArtificialMoreThanXLateralNeuronActive = True	#optional
	#positiveExcitatoryWeights = True	#optional?
	enableInhibitionTrainAndInhibitSpecificLayerOnly = False	#always enable inhibition	#CHECKTHIS
	applyInhibitoryNetworkDuringTest = True	#CHECKTHIS (set False)
	Wpermanence = {}
	Bpermanence = {}
	WpermanenceInitial = 0.1
	BpermanenceInitial = 0.1
	WpermanenceUpdateRate = 0.1
	BpermanenceUpdateRate = 0.1
	permanenceNumberBatches = 10	#if permanenceUpdateRate=1, average number of batches to reset W to random values
	solidificationRate = 0.1
elif(learningAlgorithmUninhibitedHebbianStrengthen):
	tuneInhibitionNeurons = False	#optional
	useZAcoincidenceMatrix = True	#reduce connection weights for unassociated neurons
	positiveExcitatoryWeights = True	#mandatory (requires testing)
	positiveExcitatoryWeightsThresholds = True	#do not allow weights to exceed 1.0 / fall below 0.0 [CHECKTHIS]
	Athreshold = True	#prevents incremental increase in signal per layer
	alwaysApplyInhibition = False	
	if(useZAcoincidenceMatrix):
		alwaysApplyInhibition = True	#inhibition is theoretically allowed at all times with useZAcoincidenceMatrix as it simply biases the network against a correlation between layer k neurons (inhibition is not set up to only allow X/1 neuron to fire)
	if(alwaysApplyInhibition):
		#TODO: note network sparsity/inhibition must be configured such that at least one neuron fires per layer
		positiveExcitatoryWeightsActivationFunctionOffsetDisable = True	#activation function will always be applied to Z signal comprising positive+negative components	#CHECKTHIS
		inhibitionAlgorithmArtificialSparsity = True
		generateLargeNetwork = True 	#large is required because it will be sparsely activated due to constant inhibition
		generateNetworkStatic = True	#equal number neurons per layer for unsupervised layers/testing
		enableInhibitionTrainAndInhibitSpecificLayerOnly = False	#always enable inhibition
		applyInhibitoryNetworkDuringTest = True
	else:
		inhibitionAlgorithmArtificialMoreThanXLateralNeuronActive = True	#optional
		enableInhibitionTrainAndInhibitSpecificLayerOnly = True
		applyInhibitoryNetworkDuringTest = False
	randomlyActivateWeightsDuringTrain = False	#randomly activate x weights (simulating input at simulataneous time interval t)
	if(randomlyActivateWeightsDuringTrain):
		randomlyActivateWeightsProbability = 1.0
	WinitialisationFactor = 1.0	#initialise network with relatively low weights	#network will be trained (weights will be increased) up until point where activation inhibited
	BinitialisationFactor = 1.0	#NOTUSED
	weightDecay = False
	if(useZAcoincidenceMatrix):
		useZAcoincidenceMatrix = True	#reduce connection weights for unassociated neurons
		if(useZAcoincidenceMatrix):
			#inhibitionAlgorithmArtificialMoreThanXLateralNeuronActive = False	#useZAcoincidenceMatrix requires real negative weights
			normaliseWeightUpdates = False
		else:
			normaliseWeightUpdates = True	#unimplemented: strengthen/update weights up to some maxima as determined by input signal strength (prevent runaway increase in weight strength up to 1.0)
	else:
		weightDecay = True	#constant neural net weight decay, such that network can be continuously trained
		weightDecayRate = 0.0	#defined by defineTrainingParametersLIANN		
		useZAcoincidenceMatrix = False
		normaliseWeightUpdates = False
		
	maxWeightUpdateThreshold = False	#max threshold weight updates to learningRate	
	#TODO: ensure learning algorithm does not result in runnaway weight increases
elif(learningAlgorithmPerformanceInhibitStocasticOptimise):
	enableInhibitionTrainAndInhibitSpecificLayerOnly = False	#always enable inhibition
	inhibitionAlgorithmBinary = True
	inhibitionAlgorithmBinaryInitialiseRandom = True
	supportDimensionalityReductionRandomise	= True	#randomise weights of highly correlated neurons, else zero them (effectively eliminating neuron from network, as its weights are no longer able to be trained)
	generateDeepNetwork = True	#optional	#used for algorithm testing
	generateVeryLargeNetwork = True
	generateNetworkStatic = True
elif(learningAlgorithmUnnormalisedActivityReset):
	supportDimensionalityReductionRandomise	= True	#randomise weights of highly correlated neurons, else zero them (effectively eliminating neuron from network, as its weights are no longer able to be trained)
	generateDeepNetwork = True	#optional	#used for algorithm testing
	generateVeryLargeNetwork = True
	generateNetworkStatic = True
	supportDimensionalityReductionRandomise = True
	supportDimensionalityReductionRegulariseActivityMinAvg = 0.01	#requires tuning
	supportDimensionalityReductionRegulariseActivityMaxAvg = 0.99	#requires tuning
	supportDimensionalityReductionRandomise	= True
	

learningRate = 0.0	#defined by defineTrainingParametersLIANN

#network/activation parameters;
#forward excitatory connections;
W = {}
B = {}
if(supportMultipleNetworks):
	WallNetworksFinalLayer = None
	BallNetworksFinalLayer = None
if(learningAlgorithmStochastic):
	Wbackup = {}
	Bbackup = {}
useBinaryWeights = False
					

if(generateVeryLargeNetwork):
	generateLargeNetworkRatio = 100	#100	#default: 10
else:
	if(generateLargeNetwork):
		generateLargeNetworkRatio = 3
	else:
		generateLargeNetworkRatio = 1

positiveExcitatoryWeightsActivationFunctionOffset = False
if(positiveExcitatoryWeights):
	if(positiveExcitatoryWeightsActivationFunctionOffsetDisable):
		positiveExcitatoryWeightsActivationFunctionOffset = False
	else:
		positiveExcitatoryWeightsActivationFunctionOffset = True
	normaliseInput = False	#TODO: verify that the normalisation operation will not disort the code's capacity to process a new data batch the same as an old data batch
	normalisedAverageInput = 1.0	#normalise input signal	#arbitrary
	if(positiveExcitatoryWeightsActivationFunctionOffset):
		positiveExcitatoryThreshold = 0.5	#1.0	#weights are centred around positiveExcitatoryThreshold, from 0.0 to positiveExcitatoryThreshold*2	#arbitrary
	Wmean = 0.5	#arbitrary
	WstdDev = 0.05	#stddev of weight initialisations	#CHECKTHIS
else:
	normaliseInput = False
	Wmean = 0.0
	WstdDev = 0.05	#stddev of weight initialisations

randomUniformMin = 0.0
randomUniformMax = 1.0
randomUniformMid = 0.5

if(inhibitionAlgorithmArtificialMoreThanXLateralNeuronActive or inhibitionAlgorithmArtificialSparsity):
	inhibitionAlgorithmArtificial = True
	
if(inhibitionAlgorithmArtificial):
	if(inhibitionAlgorithmArtificialMoreThanXLateralNeuronActive):
		inhibitionAlgorithmMoreThanXLateralNeuronActiveFraction = True
		if(inhibitionAlgorithmMoreThanXLateralNeuronActiveFraction):
			inhibitionAlgorithmMoreThanXLateralNeuronActiveFractionValue = 0.25	#fraction of the layer active allowed before inhibition
		else:
			inhibitionAlgorithmMoreThanXLateralNeuronActiveValue = 1	#maximum (X) number of neurons activate allowed before inhibition
else:
	inhibitionFactor1 = 1.0	#pass through signal	#positiveExcitatoryThreshold	#CHECKTHIS: requires recalibration for activationFunction:positiveExcitatoryWeights
	inhibitionFactor2 = estNetworkActivationSparsity	#-(WstdDev)	#inhibition contributes a significant (nullifying) effect on layer activation	#CHECKTHIS: requires calibration
	if(randomlyActivateWeightsDuringTrain):
		inhibitionFactor1 = inhibitionFactor1
		inhibitionFactor2 = (inhibitionFactor2*randomlyActivateWeightsProbability)	#the lower the average activation, the lower the inhibition
	#TODO: inhibitionFactor1/inhibitionFactor2 requires recalibration for activationFunction:positiveExcitatoryWeights
	singleInhibitoryNeuronPerLayer = False	#simplified inhibitory layer
	#lateral inhibitory connections (incoming/outgoing);
	IWi = {}
	IBi = {}
	IWo = {}
	IWiWeights = inhibitionFactor1	#need at least 1/IWiWeights active neurons per layer for the inhibitory neuron to become activated	#CHECKTHIS: requires calibration	#WstdDev*2	#0.5	#0.3
	IWoWeights = inhibitionFactor2	#will depend on activation sparsity of network (higher the positive activation, higher the inhibition required)	#requires calibration such that more than x (e.g. 1) active neuron on a layer will inhibit the layer
	In_h = []

if(inhibitionAlgorithmBinary):
  Nactive = {}  #effective bool [1.0 or 0.0]; whether neuron is active/inhibited
  
if(learningAlgorithmIndependenceReset):
	Bindependent = {}	#independent neurons previously identified	#effective boolean (0.0 or 1.0)	#FUTURE: consider making this a continuous variable, such that the higher the independence the less the variable is randomly shuffled per training iteration
	
#Network parameters
n_h = []
numberOfLayers = 0
numberOfNetworks = 0
datasetNumClasses = 0


#note high batchSize is required for learningAlgorithmStochastic algorithm objective functions (>= 100)
def defineTrainingParameters(dataset):
	global learningRate
	global weightDecayRate	
	
	if(learningAlgorithmStochastic):
		learningRate = 0.001
	elif(learningAlgorithmUninhibitedHebbianStrengthen):
		learningRate = 0.001
		weightDecayRate = learningRate/10.0	#CHECKTHIS	#will depend on learningRate
	else:
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
	if(not inhibitionAlgorithmArtificial):
		global In_h

	firstHiddenLayerNumberNeurons = num_input_neurons*generateLargeNetworkRatio
	if(generateDeepNetwork):
		numberOfLayers = 3
	else:
		numberOfLayers = 2
	n_h, numberOfLayers, numberOfNetworks, datasetNumClasses = defineNetworkParametersDynamic(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet, numberOfLayers, firstHiddenLayerNumberNeurons, generateNetworkStatic)		
	#n_h, numberOfLayers, numberOfNetworks, datasetNumClasses = ANNtf2_operations.defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet, generateLargeNetwork=generateLargeNetwork, generateNetworkStatic=generateNetworkStatic)
	
	if(not inhibitionAlgorithmArtificial):
		if(singleInhibitoryNeuronPerLayer):
			In_h = [1] * len(n_h)	#create one inhibitory neuron per layer
		else:
			In_h = copy.copy(n_h)	#create one inhibitory neuron for every excitatory neuron
		
	return numberOfLayers
	

def defineNeuralNetworkParameters():

	print("numberOfNetworks", numberOfNetworks)
	
	global randomNormal
	global randomUniformIndex
	randomNormal = tf.initializers.RandomNormal(mean=Wmean, stddev=WstdDev)
	#randomNormal = tf.initializers.RandomNormal()
	randomNormalFinalLayer = tf.initializers.RandomNormal()
	randomUniformIndex = tf.initializers.RandomUniform(minval=randomUniformMin, maxval=randomUniformMax)	#not available:	minval=0, maxval=numberOfSharedComputationalUnitsNeurons, dtype=tf.dtypes.int32; 
	
	for networkIndex in range(1, numberOfNetworks+1):
		for l1 in range(1, numberOfLayers+1):
			#forward excitatory connections;
			EWlayer = randomNormal([n_h[l1-1], n_h[l1]]) 
			EBlayer = tf.zeros(n_h[l1])
			if(positiveExcitatoryWeights):
				EWlayer = tf.abs(EWlayer)	#ensure randomNormal generated weights are positive
				if((l1 == numberOfLayers) and not positiveExcitatoryWeightsFinalLayer):
					EWlayer = randomNormalFinalLayer([n_h[l1-1], n_h[l1]])
			if(learningAlgorithmUninhibitedHebbianStrengthen):
				EWlayer = tf.multiply(EWlayer, WinitialisationFactor)
				EBlayer = tf.multiply(EBlayer, BinitialisationFactor)
			W[generateParameterNameNetwork(networkIndex, l1, "W")] = tf.Variable(EWlayer)
			B[generateParameterNameNetwork(networkIndex, l1, "B")] = tf.Variable(EBlayer)
	
			if(learningAlgorithmIndependenceReset):
				Bindependent[generateParameterNameNetwork(networkIndex, l1, "Bindependent")] = tf.Variable(EBlayer)		#initialise all neurons to zero (false)
			elif(learningAlgorithmStochastic):
				Wbackup[generateParameterNameNetwork(networkIndex, l1, "W")] = tf.Variable(W[generateParameterNameNetwork(networkIndex, l1, "W")])
				Bbackup[generateParameterNameNetwork(networkIndex, l1, "B")] = tf.Variable(B[generateParameterNameNetwork(networkIndex, l1, "B")])			
			elif(learningAlgorithmUninhibitedImpermanenceReset):
				EWlayerPermanence = tf.multiply(tf.ones([n_h[l1-1], n_h[l1]]), WpermanenceInitial)
				EBlayerPermanence = tf.multiply(tf.ones(n_h[l1]), BpermanenceInitial)
				Wpermanence[generateParameterNameNetwork(networkIndex, l1, "Wpermanence")] = tf.Variable(EWlayerPermanence)
				Bpermanence[generateParameterNameNetwork(networkIndex, l1, "Bpermanence")] = tf.Variable(EBlayerPermanence)
			
			if(not inhibitionAlgorithmArtificial):			
				#lateral inhibitory connections (incoming/outgoing);
				#do not currently train inhibitory weights;
				IWilayer = tf.multiply(tf.ones([n_h[l1], In_h[l1]]), IWiWeights)		#CHECKTHIS: inhibitory neuron firing is a function of current (lateral) layer (not previous layer)
				IBilayer = tf.zeros(In_h[l1])
				if(singleInhibitoryNeuronPerLayer):
					IWoWeightsL = IWoWeights
				else:
					IWoWeightsL = IWoWeights/In_h[l1]	#normalise across number inhibitory neurons
				IWolayer = tf.multiply(tf.ones([In_h[l1], n_h[l1]]), IWoWeightsL)
				IWi[generateParameterNameNetwork(networkIndex, l1, "IWi")] = tf.Variable(IWilayer)
				IBi[generateParameterNameNetwork(networkIndex, l1, "IBi")] = tf.Variable(IBilayer)
				IWo[generateParameterNameNetwork(networkIndex, l1, "IWo")] = tf.Variable(IWolayer)

			if(inhibitionAlgorithmBinary):
				if(inhibitionAlgorithmBinaryInitialiseRandom):
					Nactivelayer = randomUniformIndex([n_h[l1]])	#tf.cast(), dtype=tf.dtypes.bool)
					Nactivelayer = tf.greater(Nactivelayer, randomUniformMid)
					Nactivelayer = tf.cast(Nactivelayer, dtype=tf.dtypes.float32)
				else:
					Nactivelayer = tf.ones(n_h[l1])
				Nactive[generateParameterNameNetwork(networkIndex, l1, "Nactive")] = tf.Variable(Nactivelayer)
				
	if(supportMultipleNetworks):
		if(numberOfNetworks > 1):
			global WallNetworksFinalLayer
			global BallNetworksFinalLayer
			WlayerF = randomNormal([n_h[numberOfLayers-1]*numberOfNetworks, n_h[numberOfLayers]])
			WallNetworksFinalLayer =  tf.Variable(WlayerF)
			Blayer = tf.zeros(n_h[numberOfLayers])
			BallNetworksFinalLayer	= tf.Variable(Blayer)	#not currently used
					
def neuralNetworkPropagation(x, networkIndex=1):
	return neuralNetworkPropagationLIANNtest(x, networkIndex)

def neuralNetworkPropagationLIANNtest(x, networkIndex=1, l=None):
	return neuralNetworkPropagationLIANNminimal(x, networkIndex, l)

def neuralNetworkPropagationLayer(x, networkIndex=1, l=None):
	return neuralNetworkPropagationLIANNminimal(x, networkIndex, l)
	#return neuralNetworkPropagationLIANN(x, None, networkIndex, trainWeights=False)

def neuralNetworkPropagationLIANNtrainIntro(x, y=None, networkIndex=1):
	if(enableInhibitionTrainAndInhibitSpecificLayerOnly):
		for l in range(1, numberOfLayers+1):
			if(l < numberOfLayers):
				return neuralNetworkPropagationLIANNtrain(x, y, networkIndex, layerToTrain=l)
	else:
		return neuralNetworkPropagationLIANNtrain(x, y, networkIndex, layerToTrain=None)	

#if(supportMultipleNetworks):
def neuralNetworkPropagationAllNetworksFinalLayer(AprevLayer):
	Z = tf.add(tf.matmul(AprevLayer, WallNetworksFinalLayer), BallNetworksFinalLayer)	
	#Z = tf.matmul(AprevLayer, WallNetworksFinalLayer)	
	pred = tf.nn.softmax(Z)	
	return pred
	
	
#minimal code extracted from neuralNetworkPropagationLIANN;
def neuralNetworkPropagationLIANNminimal(x, networkIndex=1, l=None):

	randomlyActivateWeights = False

	if(l == None):
		maxLayer = numberOfLayers
	else:
		maxLayer = l
		
	AprevLayer = x
	ZprevLayer = x
	for l in range(1, maxLayer+1):
				
		enableInhibition = False
		if(not enableInhibitionTrainAndInhibitSpecificLayerOnly):
			enableInhibition = True
				
		A, Z, _ = forwardIteration(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights)
		
		if(learningAlgorithmFinalLayerBackpropHebbian):
			A = tf.stop_gradient(A)

		AprevLayer = A
		ZprevLayer = Z

	if(maxLayer == numberOfLayers):
		return tf.nn.softmax(Z)
	else:
		return A
	
def neuralNetworkPropagationLIANNtrain(x, y=None, networkIndex=1, layerToTrain=None):

	if(normaliseInput):
		#TODO: verify that the normalisation operation will not disort the code's capacity to process a new data batch the same as an old data batch
		averageTotalInput = tf.math.reduce_mean(x)
		#print("averageTotalInput = ", averageTotalInput)
		x = tf.multiply(x, normalisedAverageInput/averageTotalInput)	#normalise input wrt positiveExcitatoryThreshold
		#averageTotalInput = tf.math.reduce_mean(x)

	if(layerToTrain is None):
		maxLayer = numberOfLayers
	else:	#ie !enableInhibitionTrainAndInhibitSpecificLayerOnly
		maxLayer = layerToTrain
			
	AprevLayer = x
	ZprevLayer = x
	for l in range(1, maxLayer+1):
					
		trainLayer = False
		enableInhibition = False
		randomlyActivateWeights = False
		if(enableInhibitionTrainAndInhibitSpecificLayerOnly):
			if(l == layerToTrain):
				#enableInhibition = False
				enableInhibition = True
				trainLayer = True
		else:
			if(l < numberOfLayers):
				enableInhibition = True
				trainLayer = True
		if(randomlyActivateWeightsDuringTrain):
			randomlyActivateWeights = True
	
		if(trainLayer):
			#CHECKTHIS: verify learning algorithm (how to modify weights to maximise independence between neurons on each layer)
			if(learningAlgorithmNone):
				neuralNetworkPropagationLIANNlearningAlgorithmNone(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights)
			elif(learningAlgorithmCorrelationReset):
				neuralNetworkPropagationLIANNlearningAlgorithmCorrelationReset(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights)
			elif(learningAlgorithmPCA):
				neuralNetworkPropagationLIANNlearningAlgorithmPCA(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights)
			elif(learningAlgorithmIndependenceReset):
				neuralNetworkPropagationLIANNlearningAlgorithmIndependenceReset(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights)
			elif(learningAlgorithmStochastic):
				neuralNetworkPropagationLIANNlearningAlgorithmStochastic(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights)
			elif(learningAlgorithmUninhibitedImpermanenceReset):
				neuralNetworkPropagationLIANNlearningAlgorithmUninhibitedImpermanenceReset(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights)
			elif(learningAlgorithmUninhibitedHebbianStrengthen):
				neuralNetworkPropagationLIANNlearningAlgorithmUninhibitedHebbianStrengthen(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights)
			elif(learningAlgorithmPerformanceInhibitStocasticOptimise):
				neuralNetworkPropagationLIANNlearningAlgorithmPerformanceInhibitStocasticOptimise(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights, x, y)
			elif(learningAlgorithmUnnormalisedActivityReset):
				neuralNetworkPropagationLIANNlearningAlgorithmUnnormalisedActivityReset(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights)
	
			A, Z, _ = forwardIteration(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition=(not enableInhibitionTrainAndInhibitSpecificLayerOnly), randomlyActivateWeights=False)
		else:
			A, Z, _ = forwardIteration(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights=False)
			
		AprevLayer = A
		ZprevLayer = Z

	return tf.nn.softmax(Z)

def calculatePropagationLoss(x, y, networkIndex=1):
	costCrossEntropyWithLogits = False
	pred = neuralNetworkPropagation(x, networkIndex)
	target = y
	lossCurrent = calculateLossCrossEntropy(pred, target, datasetNumClasses, costCrossEntropyWithLogits)
	#acc = calculateAccuracy(pred, target)	#only valid for softmax class targets
	return lossCurrent
	
def neuralNetworkPropagationLIANNlearningAlgorithmNone(networkIndex, AprevLayer, ZprevLayer, l1, enableInhibition, randomlyActivateWeights):
	pass

def neuralNetworkPropagationLIANNlearningAlgorithmCorrelationReset(networkIndex, AprevLayer, ZprevLayer, l1, enableInhibition, randomlyActivateWeights):
	A, Z, _ = forwardIteration(networkIndex, AprevLayer, ZprevLayer, l1)
	#measure and minimise correlation between layer neurons;
	neuronActivationCorrelationMinimisation(networkIndex, n_h, l1, A, randomNormal, Wf=W, Wfname="W", Wb=None, Wbname=None, updateAutoencoderBackwardsWeights=False, supportSkipLayers=supportSkipLayers, supportDimensionalityReductionRandomise=supportDimensionalityReductionRandomise, maxCorrelation=maxCorrelation)

def neuralNetworkPropagationLIANNlearningAlgorithmPCA(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights):
	#Afinal, Zfinal, _ = forwardIteration(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights)	#batched
	SVDinputMatrix = LIANNtf_algorithmLIANN_math.generateSVDinputMatrix(l, n_h, AprevLayer)
	U, Sigma, VT = LIANNtf_algorithmLIANN_math.calculateSVD(M=SVDinputMatrix, k=n_h[l])
	AW = LIANNtf_algorithmLIANN_math.calculateWeights(l, n_h, SVDinputMatrix, U, Sigma, VT)
	W[generateParameterNameNetwork(networkIndex, l, "W")] = AW

	#weights = U -> Sigma -> VT	[linear]
	#M_reduced = reduce_to_k_dim(M=spikeCoincidenceMatrix, k=n_h[l])
	
def neuralNetworkPropagationLIANNlearningAlgorithmIndependenceReset(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights):
	layerHasDependentNeurons = True
	Bind = Bindependent[generateParameterNameNetwork(networkIndex, l, "Bindependent")]
	if(count_zero(Bind) > 0):	#more than 1 dependent neuron on layer
		layerHasDependentNeurons = True
	else:
		layerHasDependentNeurons = False

	while(layerHasDependentNeurons):
		Afinal, Zfinal, _ = forwardIteration(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights)	#batched

		AnumActive = tf.math.count_nonzero(Afinal, axis=1)	#batched
		Aindependent = tf.equal(AnumActive, 1)	#batched
		Aindependent = tf.dtypes.cast(Aindependent, dtype=tf.dtypes.float32)	#batched
		Aindependent = tf.expand_dims(Aindependent, 1)	#batched
		#print("Afinal = ", Afinal)
		#print("AnumActive = ", AnumActive)
		#print("Aindependent = ", Aindependent)

		Aactive = tf.greater(Afinal, 0)	#2D: batched, for every k neuron
		Aactive = tf.dtypes.cast(Aactive, dtype=tf.dtypes.float32) 	#2D: batched, for every k neuron
		#print("Aactive = ", Aactive)
		#ex

		AactiveAndIndependent = tf.multiply(Aactive, Aindependent)	#2D: batched, for every k neuron	
		AactiveAndIndependent = tf.reduce_sum(AactiveAndIndependent, axis=0) #for every k neuron

		AactiveAndIndependentPass = tf.greater(AactiveAndIndependent, fractionIndependentInstancesAcrossBatchRequired*n_h[l])	 #for every k neuron
		#print("AactiveAndIndependentPass = ", AactiveAndIndependentPass)

		BindBool = tf.dtypes.cast(Bind, dtype=tf.dtypes.bool)
		AactiveAndIndependentPassRequiresSolidifying = tf.logical_and(AactiveAndIndependentPass, tf.logical_not(BindBool))
		#print("AactiveAndIndependentPass = ", AactiveAndIndependentPass)
		#print("BindBool = ", BindBool)
		print("AactiveAndIndependentPassRequiresSolidifying = ", AactiveAndIndependentPassRequiresSolidifying)
		BindNew = tf.logical_or(BindBool, AactiveAndIndependentPassRequiresSolidifying)
		BdepNew = tf.logical_not(BindNew)

		#update layer weights (reinitialise weights for all dependent neurons);
		BindNew = tf.dtypes.cast(BindNew, dtype=tf.dtypes.float32)
		BdepNew = tf.dtypes.cast(BdepNew, dtype=tf.dtypes.float32)
		EWlayerDep = randomNormal([n_h[l-1], n_h[l]]) 
		if(positiveExcitatoryWeights):
			EWlayerDep = tf.abs(EWlayerDep)	#ensure randomNormal generated weights are positive
		EBlayerDep = tf.zeros(n_h[l])
		EWlayerDep = tf.multiply(EWlayerDep, BdepNew)	#requires broadcasting
		EBlayerDep = tf.multiply(EBlayerDep, BdepNew)				
		EWlayerInd = W[generateParameterNameNetwork(networkIndex, l, "W")] 
		EBlayerInd = B[generateParameterNameNetwork(networkIndex, l, "B")]
		EWlayerInd = tf.multiply(EWlayerInd, BindNew)	#requires broadcasting
		EBlayerInd = tf.multiply(EBlayerInd, BindNew)
		EWlayerNew = tf.add(EWlayerDep, EWlayerInd)
		EBlayerNew = tf.add(EBlayerDep, EBlayerInd)					
		W[generateParameterNameNetwork(networkIndex, l, "W")] = EWlayerNew
		B[generateParameterNameNetwork(networkIndex, l, "B")] = EBlayerNew	
		#print("EWlayerNew = ", EWlayerNew)				

		#print("BdepNew = ", BdepNew)
		#print("BindNew = ", BindNew)

		Bindependent[generateParameterNameNetwork(networkIndex, l, "Bindependent")] = BindNew	#update independence record
		Bind = BindNew
		if(count_zero(Bind) > 0):	#more than 1 dependent neuron on layer
			layerHasDependentNeurons = True
			#print("layerHasDependentNeurons: count_zero(Bind) = ", count_zero(Bind))
		else:
			layerHasDependentNeurons = False	
			#print("!layerHasDependentNeurons")
							
def neuralNetworkPropagationLIANNlearningAlgorithmStochastic(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights):

	if(learningAlgorithmStochastic):
		if(useBinaryWeights):
			variationDirections = 1
		else:
			variationDirections = 2
			
	#code from ANNtf2_algorithmLREANN_expSUANN;
	for s in range(numberStochasticIterations):
		for hIndexCurrentLayer in range(0, n_h[l]):
			for hIndexPreviousLayer in range(0, n_h[l-1]+1):
				if(hIndexPreviousLayer == n_h[l-1]):	#ensure that B parameter updates occur/tested less frequently than W parameter updates
					parameterTypeWorB = 0
				else:
					parameterTypeWorB = 1
				for variationDirectionInt in range(variationDirections):

					networkParameterIndexBase = (parameterTypeWorB, l, hIndexCurrentLayer, hIndexPreviousLayer, variationDirectionInt)

					metricBase = learningAlgorithmStochasticCalculateMetric(networkIndex, AprevLayer, ZprevLayer, l)

					for subsetTrialIndex in range(0, numberOfSubsetsTrialledPerBaseParameter):

						accuracyImprovementDetected = False

						currentSubsetOfParameters = []
						currentSubsetOfParameters.append(networkParameterIndexBase)

						for s in range(1, parameterUpdateSubsetSize):
							networkParameterIndex = getRandomNetworkParameter(networkIndex, currentSubsetOfParameters)
							currentSubsetOfParameters.append(networkParameterIndex)

						for s in range(0, parameterUpdateSubsetSize):
							networkParameterIndex = currentSubsetOfParameters[s]

							if(not useBinaryWeights):
								if(networkParameterIndex[NETWORK_PARAM_INDEX_VARIATION_DIRECTION] == 1):
									variationDiff = learningRate
								else:
									variationDiff = -learningRate		

							if(networkParameterIndex[NETWORK_PARAM_INDEX_TYPE] == 1):
								#Wnp = W[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "W")].numpy()
								#currentVal = Wnp[networkParameterIndex[NETWORK_PARAM_INDEX_H_PREVIOUS_LAYER], networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]]
								currentVal = W[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "W")][networkParameterIndex[NETWORK_PARAM_INDEX_H_PREVIOUS_LAYER], networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]].numpy()

								#print("currentVal = ", currentVal)
								#print("W1 = ", W[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "W")])
								if(useBinaryWeights):
									if(useBinaryWeightsReduceMemoryWithBool):
										newVal = not currentVal
									else:
										newVal = float(not bool(currentVal))
										#print("newVal = ", newVal)
								else:
									newVal = currentVal + variationDiff

								if(positiveExcitatoryWeights):
									newVal = max(newVal, 0)	#do not allow weights fall below zero [CHECKTHIS]	

								W[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "W")][networkParameterIndex[NETWORK_PARAM_INDEX_H_PREVIOUS_LAYER], networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]].assign(newVal)

								#print("W2 = ", W[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "W")])
							else:
								#Bnp = B[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "B")].numpy()
								#currentVal = Bnp[networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]]
								currentVal = B[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "B")][networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]].numpy()

								if(useBinaryWeights):
									if(useBinaryWeightsReduceMemoryWithBool):
										newVal = not currentVal
									else:
										newVal = float(not bool(currentVal))
								else:
									newVal = currentVal + variationDiff

								if(positiveExcitatoryWeights):
									newVal = max(newVal, 0)	#do not allow weights fall below zero [CHECKTHIS]	

								B[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "B")][networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]].assign(newVal)

						metricAfterStochasticUpdate = learningAlgorithmStochasticCalculateMetric(networkIndex, AprevLayer, ZprevLayer, l)
						#print("metricBase = ", metricBase)
						#print("metricAfterStochasticUpdate = ", metricAfterStochasticUpdate)

						if(metricAfterStochasticUpdate > metricBase):
							#print("(metricAfterStochasticUpdate > metricBase)")
							accuracyImprovementDetected = True
							metricBase = metricAfterStochasticUpdate
						#else:
							#print("(metricAfterStochasticUpdate < metricBase)")

						if(accuracyImprovementDetected):
							#retain weight update
							Wbackup[generateParameterNameNetwork(networkIndex, l, "W")].assign(W[generateParameterNameNetwork(networkIndex, l, "W")])
							Bbackup[generateParameterNameNetwork(networkIndex, l, "B")].assign(B[generateParameterNameNetwork(networkIndex, l, "B")])	
						else:
							#restore weights
							W[generateParameterNameNetwork(networkIndex, l, "W")].assign(Wbackup[generateParameterNameNetwork(networkIndex, l, "W")])
							B[generateParameterNameNetwork(networkIndex, l, "B")].assign(Bbackup[generateParameterNameNetwork(networkIndex, l, "B")])	

def neuralNetworkPropagationLIANNlearningAlgorithmUninhibitedImpermanenceReset(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights):
	Afinal, Zfinal, _ = forwardIteration(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights)

	#update W/B permanence;
	Afinal2D = tf.reduce_mean(Afinal, axis=0)	#average across batch
	Afinal2D = tf.expand_dims(Afinal2D, axis=0)	#make compatible shape to W
	WpermanenceUpdate = tf.multiply(Afinal2D, WpermanenceUpdateRate)	#verify that broadcasting works
	WpermanenceNew = tf.add(Wpermanence[generateParameterNameNetwork(networkIndex, l, "Wpermanence")], WpermanenceUpdate)	#increase the permanence of neuron weights that successfully fired
	Wpermanence[generateParameterNameNetwork(networkIndex, l, "Wpermanence")] = WpermanenceNew
	print("WpermanenceUpdate = ", WpermanenceUpdate)

	#stochastically modify weights based on permanence values:
	Wupdate = randomNormal([n_h[l-1], n_h[l]])
	Wupdate = tf.divide(Wupdate, Wpermanence[generateParameterNameNetwork(networkIndex, l, "Wpermanence")])
	Wupdate = tf.divide(Wupdate, permanenceNumberBatches)
	Wnew = tf.add(W[generateParameterNameNetwork(networkIndex, l, "W")], Wupdate)
	if(positiveExcitatoryWeights):
		Wnew = tf.maximum(Wnew, 0)	#do not allow weights fall below zero [CHECKTHIS]
	W[generateParameterNameNetwork(networkIndex, l, "W")] = Wnew
	#print("Wupdate = ", Wupdate)

def neuralNetworkPropagationLIANNlearningAlgorithmUninhibitedHebbianStrengthen(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights):
	AW = W[generateParameterNameNetwork(networkIndex, l, "W")] 
	Afinal, Zfinal, EWactive = forwardIteration(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights)
	#print("Zfinal = ", Zfinal)

	if(useZAcoincidenceMatrix):
		AWcontribution = tf.matmul(tf.transpose(ZprevLayer), Afinal)	#increase excitatory weights that contributed to the output signal	#hebbian
	else:
		AWcontribution = tf.matmul(tf.transpose(AprevLayer), Afinal)	#increase excitatory weights that contributed to the output signal	#hebbian

	if(randomlyActivateWeights):
		#do not apply weight updates to temporarily suppressed weights [CHECKTHIS];
		AWcontribution = tf.multiply(AWcontribution, EWactive)		

	if(normaliseWeightUpdates):
		print("LIANNtf_algorithmLIANN:neuralNetworkPropagationLIANN error - normaliseWeightUpdates: normaliseWeightUpdatesReduceConnectionWeightsForUnassociatedNeurons unimplemented")
	else:
		if(maxWeightUpdateThreshold):
			AWcontribution = tf.minimum(AWcontribution, 1.0)

	AWupdate = tf.multiply(AWcontribution, learningRate)
	#print("AWupdate = ", AWupdate)

	AW = tf.add(AW, AWupdate)	#apply weight updates

	if(weightDecay):
		#apply decay to all weights;
		AWdecay = -weightDecayRate
		#print("AWdecay = ", AWdecay)
		AW = tf.add(AW, AWdecay)
		#print("AWdecay = ", AWdecay)

	if(positiveExcitatoryWeightsThresholds):
		AW = tf.minimum(AW, 1.0)	#do not allow weights to exceed 1.0 [CHECKTHIS]
		AW = tf.maximum(AW, 0)	#do not allow weights fall below zero [CHECKTHIS]

	W[generateParameterNameNetwork(networkIndex, l, "W")] = AW

def neuralNetworkPropagationLIANNlearningAlgorithmPerformanceInhibitStocasticOptimise(networkIndex, AprevLayer, ZprevLayer, l1, enableInhibition, randomlyActivateWeights, x=None, y=None):
	A, Z, _ = forwardIteration(networkIndex, AprevLayer, ZprevLayer, l1)
	#randomly select a neuron k on layer to trial inhibition performance;
	lossCurrent = calculatePropagationLoss(x, y, networkIndex)	#moved 15 Mar 2022
	Nactivelayer = Nactive[generateParameterNameNetwork(networkIndex, l1, "Nactive")]
	NactivelayerBackup = Nactivelayer #tf.Variable(Nactivelayer)
	layerInhibitionIndex = tf.cast(randomUniformIndex([1])*n_h[l1], tf.int32)[0].numpy()
	#print("layerInhibitionIndex = ", layerInhibitionIndex)
	if(inhibitionAlgorithmBinary):
		inhibitionValue = randomUniformIndex([1])
		inhibitionValue = tf.greater(inhibitionValue, randomUniformMid)
		inhibitionValue = tf.cast(inhibitionValue, dtype=tf.dtypes.float32)
		inhibitionValue = inhibitionValue[0].numpy()
		#print("inhibitionValue = ", inhibitionValue)
	else:
		inhibitionValue = 0.0
	Nactivelayer = tf.Variable(ANNtf2_operations.modifyTensorRowColumn(Nactivelayer, True, layerInhibitionIndex, inhibitionValue, False))	#tf.Variable added to retain formatting
	#print("NactivelayerBackup = ", NactivelayerBackup)
	#print("Nactivelayer = ", Nactivelayer)
	Nactive[generateParameterNameNetwork(networkIndex, l1, "Nactive")] = Nactivelayer
	loss = calculatePropagationLoss(x, y, networkIndex)
	#acc = calculateAccuracy(pred, target)	#only valid for softmax class targets 
	if(loss < lossCurrent):
		lossCurrent = loss
		#print("loss < lossCurrent; loss = ", loss, ", lossCurrent = ", lossCurrent)
	else:
		Nactive[generateParameterNameNetwork(networkIndex, l1, "Nactive")] = NactivelayerBackup
		#print("loss !< lossCurrent; loss = ", loss, ", lossCurrent = ", lossCurrent)
		
def neuralNetworkPropagationLIANNlearningAlgorithmUnnormalisedActivityReset(networkIndex, AprevLayer, ZprevLayer, l1, enableInhibition, randomlyActivateWeights):
	A, Z, _ = forwardIteration(networkIndex, AprevLayer, ZprevLayer, l1)
	neuronActivationRegularisation(networkIndex, n_h, l1, A, randomNormal, Wf=W, Wfname="W", Wb=None, Wbname=None, updateAutoencoderBackwardsWeights=False, supportSkipLayers=supportSkipLayers, supportDimensionalityReductionRandomise=supportDimensionalityReductionRandomise, supportDimensionalityReductionRegulariseActivityMinAvg=supportDimensionalityReductionRegulariseActivityMinAvg, supportDimensionalityReductionRegulariseActivityMaxAvg=supportDimensionalityReductionRegulariseActivityMaxAvg)


def forwardIteration(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition=False, randomlyActivateWeights=False):
	#forward excitatory connections;
	EWactive = None
	EW = W[generateParameterNameNetwork(networkIndex, l, "W")]
	if(randomlyActivateWeights):
		#print("EW = ", EW)
		EWactive = tf.less(tf.random.uniform(shape=EW.shape), randomlyActivateWeightsProbability)	#initialised from 0.0 to 1.0
		EWactive = tf.dtypes.cast(EWactive, dtype=tf.dtypes.float32) 
		#print("EWactive = ", EWactive)
		#EWactive = tf.dtypes.cast(tf.random.uniform(shape=EW.shape, minval=0, maxval=2, dtype=tf.dtypes.int32), dtype=tf.dtypes.float32)
		EW = tf.multiply(EW, EWactive)
	Z = tf.add(tf.matmul(AprevLayer, EW), B[generateParameterNameNetwork(networkIndex, l, "B")])
	A = activationFunction(Z, n_h[l-1])

	#lateral inhibitory connections (incoming/outgoing);
	if(enableInhibition):
		Afinal, Zfinal = forwardIterationInhibition(networkIndex, AprevLayer, ZprevLayer, l, A, Z)
	else:
		Zfinal = Z
		Afinal = A
		
	return Afinal, Zfinal, EWactive

def forwardIterationInhibition(networkIndex, AprevLayer, ZprevLayer, l, A, Z):
	if(inhibitionAlgorithmBinary):
		Afinal = tf.multiply(A, Nactive[generateParameterNameNetwork(networkIndex, l, "Nactive")])
		Zfinal = tf.multiply(Z, Nactive[generateParameterNameNetwork(networkIndex, l, "Nactive")])
	else:
		if(inhibitionAlgorithmArtificial):
			if(inhibitionAlgorithmArtificialSparsity):
				prevLayerSize = n_h[l-1]
				inhibitionResult = tf.math.reduce_mean(AprevLayer, axis=1)	#or ZprevLayer?	#batched
				#print("inhibitionResult = ", inhibitionResult)
				inhibitionResult = tf.multiply(inhibitionResult, prevLayerSize)	#normalise by prev layer size		#batched
				inhibitionResult = tf.multiply(inhibitionResult, Wmean)	#normalise by average weight
				inhibitionResult = tf.expand_dims(inhibitionResult, axis=1)	#batched
				Zfinal = tf.subtract(Z, inhibitionResult)	#requires broadcasting
				#print("Z = ", Z)
				#print("Zfinal = ", Zfinal)
				Afinal = activationFunction(Zfinal, prevLayerSize=prevLayerSize)
			elif(inhibitionAlgorithmArtificialMoreThanXLateralNeuronActive):
				layerSize = n_h[l]
				numActiveLateralNeurons = tf.math.count_nonzero(A, axis=1)
				if(inhibitionAlgorithmMoreThanXLateralNeuronActiveFraction):
					numberActiveNeuronsAllowed = inhibitionAlgorithmMoreThanXLateralNeuronActiveFractionValue*layerSize
				else:
					numberActiveNeuronsAllowed = inhibitionAlgorithmMoreThanXLateralNeuronActiveValue
				numberActiveNeuronsAllowed = int(numberActiveNeuronsAllowed)
				#print("numActiveLateralNeurons = ", numActiveLateralNeurons)
				#print("numberActiveNeuronsAllowed = ", numberActiveNeuronsAllowed)
				inhibitionResult = tf.greater(numActiveLateralNeurons, numberActiveNeuronsAllowed)

				inhibitionResult = tf.logical_not(inhibitionResult)
				inhibitionResult = tf.dtypes.cast(inhibitionResult, dtype=tf.dtypes.float32)
				inhibitionResult = tf.expand_dims(inhibitionResult, axis=1)
				#print("numActiveLateralNeurons = ", numActiveLateralNeurons)
				#print("inhibitionResult = ", inhibitionResult)
				Zfinal = tf.multiply(Z, inhibitionResult)	#requires broadcasting
				Afinal = tf.multiply(A, inhibitionResult)	#requires broadcasting		
		else:
			#if((l < numberOfLayers) or positiveExcitatoryWeightsFinalLayer):

			#print("AprevLayer = ", AprevLayer)
			#print("Z = ", Z)

			IZi = tf.matmul(A, IWi[generateParameterNameNetwork(networkIndex, l, "IWi")])	#CHECKTHIS: inhibitory neuron firing is a function of current (lateral) layer (not previous layer)
			IAi = activationFunction(IZi, n_h[l-1])
			#print("IZi = ", IZi)
			#print("IAi = ", IAi)
			IZo = tf.matmul(IAi, IWo[generateParameterNameNetwork(networkIndex, l, "IWo")])
			#print("W = ", W[generateParameterNameNetwork(networkIndex, l, "W")])
			#print("IZo = ", IZo)

			#final activations;
			Zfinal = tf.add(Z, IZo)
			#print("Zfinal = ", Zfinal)
			Afinal = activationFunction(Zfinal, n_h[l-1])

	if(Athreshold):
		Afinal = tf.minimum(Afinal, AthresholdValue)
		#print("Afinal = ", Afinal)
		
	return Afinal, Zfinal




#LIANNlearningAlgorithmCorrelation metric:

def neuronActivationCorrelationMinimisation(networkIndex, n_h, l1, A, randomNormal, Wf, Wfname="W", Wb=None, Wbname=None, updateAutoencoderBackwardsWeights=False, supportSkipLayers=False, supportDimensionalityReductionRandomise=True, maxCorrelation=0.95):

	resetNeuronIfSameValueAcrossBatch = False #reset neuron if all values of a neuron k being the same value across the batch
	randomlySelectCorrelatedNeuronToReset = False	#randomly select one of each correlated neuron to reset
	
	useCorrelationMatrix = True	#only implementation currently available
	
	Atransposed = tf.transpose(A)
	if(useCorrelationMatrix):
		correlationMatrix = LIANNtf_algorithmLIANN_math.calculateOffDiagonalCorrelationMatrix(A, nanReplacementValue=0.0, getOffDiagonalCorrelationMatrix=True)	#off diagonal correlation matrix is required so that do not duplicate k1->k2 and k2->k1 correlations	#CHECKTHIS: nanReplacementValue
		#nanReplacementValue=0.0; will set the correlation as 0 if all values of a neuron k being the same value across the batch		
		#print("correlationMatrix = ", correlationMatrix)
		#print("correlationMatrix.shape = ", correlationMatrix.shape)
	
	if(useCorrelationMatrix):
		if(randomlySelectCorrelatedNeuronToReset):
			correlationMatrixRotated = np.transpose(correlationMatrix)
			k1MaxCorrelation = correlationMatrix.max(axis=0)
			k2MaxCorrelation = correlationMatrixRotated.max(axis=0)
			#print("k1MaxCorrelation = ", k1MaxCorrelation)
			#print("k2MaxCorrelation = ", k2MaxCorrelation)
			kSelect = np.random.randint(0, 2, size=k1MaxCorrelation.shape)
			mask1 = kSelect.astype(bool)
			mask2 = np.logical_not(mask1)
			mask1 = mask1.astype(float)
			mask2 = mask2.astype(float)
			k1MaxCorrelation = np.multiply(k1MaxCorrelation, mask1)
			k2MaxCorrelation = np.multiply(k2MaxCorrelation, mask2)
			kMaxCorrelation = np.add(k1MaxCorrelation, k2MaxCorrelation)
			#print("correlationMatrix = ", correlationMatrix)
			#print("correlationMatrixRotated = ", correlationMatrixRotated)
			#print("k1MaxCorrelation = ", k1MaxCorrelation)
			#print("k2MaxCorrelation = ", k2MaxCorrelation)
			#print("mask1 = ", mask1)
			#print("mask2 = ", mask2)
			#print("kMaxCorrelation = ", kMaxCorrelation)
		else:
			k1MaxCorrelation = correlationMatrix.max(axis=0)
			k2MaxCorrelation = correlationMatrix.max(axis=1)
			#k1MaxCorrelation = np.amax(correlationMatrix, axis=0)	#reduce max
			#k2MaxCorrelation = np.amax(correlationMatrix, axis=1)	#reduce max
			kMaxCorrelation = np.maximum(k1MaxCorrelation, k2MaxCorrelation)
		#kMaxCorrelationIndex = correlationMatrix.argmax(axis=0)	#or axis=1
		kMaxCorrelation = tf.convert_to_tensor(kMaxCorrelation, dtype=tf.dtypes.float32)	#make sure same type as A
		#print("kMaxCorrelation;", kMaxCorrelation)
		
		if(resetNeuronIfSameValueAcrossBatch):
			AbatchAllZero = tf.reduce_sum(A, axis=0)
			AbatchAllZero = tf.equal(AbatchAllZero, 0.0)
			AbatchAllZero = tf.cast(AbatchAllZero, tf.float32)
			kMaxCorrelation = tf.add(kMaxCorrelation, AbatchAllZero)	#set kMaxCorrelation[k]=1.0 if AbatchAllZero[k]=True
			#print("AbatchAllZero;", AbatchAllZero)

	else:
		#incomplete;
		for k1 in range(n_h[l1]):
			#calculate maximum correlation;
			k1MaxCorrelation = 0.0
			for k2 in range(n_h[l1]):
				if(k1 != k2):
					Ak1 = Atransposed[k1]	#Ak: 1d vector of batchsize
					Ak2 = Atransposed[k2]	#Ak: 1d vector of batchsize
					k1k2correlation = calculateCorrelation(Ak1, Ak2)	#undefined

	#generate masks (based on highly correlated k/neurons);
	#print("kMaxCorrelation = ", kMaxCorrelation)
	kPassArray = tf.less(kMaxCorrelation, maxCorrelation)
	randomiseLayerNeurons(networkIndex, n_h, l1, kPassArray, randomNormal, Wf, Wfname, Wb, Wbname, updateAutoencoderBackwardsWeights, supportSkipLayers, supportDimensionalityReductionRandomise)

def neuronActivationRegularisation(networkIndex, n_h, l1, A, randomNormal, Wf, Wfname="W", Wb=None, Wbname=None, updateAutoencoderBackwardsWeights=False, supportSkipLayers=False, supportDimensionalityReductionRandomise=True, supportDimensionalityReductionRegulariseActivityMinAvg=0.1, supportDimensionalityReductionRegulariseActivityMaxAvg=0.9):
	#CHECKTHIS: treat any level/intensity of activation the same
	Aactive = tf.cast(A, tf.bool)
	AactiveFloat = tf.cast(Aactive, tf.float32)
	neuronActivationFrequency = tf.reduce_mean(AactiveFloat, axis=0)
	#print("neuronActivationFrequency = ", neuronActivationFrequency)
	kPassArray = tf.logical_and(tf.greater(neuronActivationFrequency, supportDimensionalityReductionRegulariseActivityMinAvg), tf.less(neuronActivationFrequency, supportDimensionalityReductionRegulariseActivityMaxAvg))
	#print("kPassArray = ", kPassArray)
	randomiseLayerNeurons(networkIndex, n_h, l1, kPassArray, randomNormal, Wf, Wfname, Wb, Wbname, updateAutoencoderBackwardsWeights, supportSkipLayers, supportDimensionalityReductionRandomise)
	
def randomiseLayerNeurons(networkIndex, n_h, l1, kPassArray, randomNormal, Wf, Wfname="W", Wb=None, Wbname=None, updateAutoencoderBackwardsWeights=False, supportSkipLayers=False, supportDimensionalityReductionRandomise=True):
	kFailArray = tf.logical_not(kPassArray)
	#print("kPassArray = ", kPassArray)
	#print("kFailArray = ", kFailArray)
	kPassArrayF = tf.expand_dims(kPassArray, axis=0)
	kFailArrayF = tf.expand_dims(kFailArray, axis=0)
	kPassArrayF = tf.cast(kPassArrayF, tf.float32)
	kFailArrayF = tf.cast(kFailArrayF, tf.float32)
	if(updateAutoencoderBackwardsWeights):
		kPassArrayB = tf.expand_dims(kPassArray, axis=1)
		kFailArrayB = tf.expand_dims(kFailArray, axis=1)
		kPassArrayB = tf.cast(kPassArrayB, tf.float32)
		kFailArrayB = tf.cast(kFailArrayB, tf.float32)

	#apply masks to weights (randomise specific k/neurons);					
	if(supportSkipLayers):
		for l2 in range(0, l1):
			if(l2 < l1):
				#randomize or zero
				if(supportDimensionalityReductionRandomise):
					WlayerFrand = randomNormal([n_h[l2], n_h[l1]])
				else:
					WlayerFrand = tf.zeros([n_h[l2], n_h[l1]], dtype=tf.dtypes.float32)
				Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, Wfname)] = applyMaskToWeights(Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, Wfname)], WlayerFrand, kPassArrayF, kFailArrayF)
				if(updateAutoencoderBackwardsWeights):
					if(supportDimensionalityReductionRandomise):
						WlayerBrand = randomNormal([n_h[l1], n_h[l2]])
					else:
						WlayerBrand = tf.zeros([n_h[l1], n_h[l2]], dtype=tf.dtypes.float32)
					Wb[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, Wbname)] = applyMaskToWeights(Wb[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, Wbname)], WlayerBrand, kPassArrayB, kFailArrayB)		
	else:
		if(supportDimensionalityReductionRandomise):
			WlayerFrand = randomNormal([n_h[l1-1], n_h[l1]]) 
		else:
			WlayerFrand = tf.zeros([n_h[l1-1], n_h[l1]], dtype=tf.dtypes.float32)
		Wf[generateParameterNameNetwork(networkIndex, l1, Wfname)] = applyMaskToWeights(Wf[generateParameterNameNetwork(networkIndex, l1, Wfname)], WlayerFrand, kPassArrayF, kFailArrayF)
		if(updateAutoencoderBackwardsWeights):
			if(supportDimensionalityReductionRandomise):
				WlayerBrand = randomNormal([n_h[l1], n_h[l1-1]])
			else:
				WlayerBrand = tf.zeros([n_h[l1], n_h[l1-1]], dtype=tf.dtypes.float32)		
			Wb[generateParameterNameNetwork(networkIndex, l1, Wbname)] = applyMaskToWeights(Wb[generateParameterNameNetwork(networkIndex, l1, Wbname)], WlayerBrand, kPassArrayB, kFailArrayB)

def applyMaskToWeights(Wlayer, WlayerRand, kPassArray, kFailArray):
	WlayerFail = tf.multiply(WlayerRand, kFailArray)
	#print("WlayerFail = ", WlayerFail)
	WlayerPass = tf.multiply(Wlayer, kPassArray)
	#print("WlayerPass = ", WlayerPass)
	Wlayer = tf.add(WlayerPass, WlayerFail)
	return Wlayer



#LIANNlearningAlgorithmStochastic metric:	
						
def learningAlgorithmStochasticCalculateMetric(networkIndex, AprevLayer, ZprevLayer, l):
	randomlyActivateWeights = False
	if(randomlyActivateWeightsDuringTrain):
		randomlyActivateWeights = true
		
	if(learningAlgorithmCorrelationStocasticOptimise):
		enableInhibition = False
		A, Z, _ = forwardIteration(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights)
		metric = learningAlgorithmStochasticCalculateMetricCorrelation(A)
	elif(learningAlgorithmMaximiseAndEvenSignalStochasticOptimise):
		enableInhibition = True
		Afinal, Zfinal, _ = forwardIteration(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights)
		metric = learningAlgorithmStochasticCalculateMetricMaximiseAndEvenSignal(Afinal, metric1Weighting, metric2Weighting)
	return metric
	
def learningAlgorithmStochasticCalculateMetricCorrelation(A):
	#print("A = ", A)	
	meanCorrelation = LIANNtf_algorithmLIANN_math.calculateCorrelationMean(A)
	print("meanCorrelation = ", meanCorrelation)
	metric = 1 - meanCorrelation
	#print("metric = ", metric)
	return metric
	
def learningAlgorithmStochasticCalculateMetricMaximiseAndEvenSignal(Afinal, metric1Weighting, metric2Weighting):	
	#learning objective functions:
	#1: maximise the signal (ie successfully uninhibited) across multiple batches (entire dataset)
	#2: ensure that all layer neurons receive even activation across multiple batches (entire dataset)		
	
	#print("Afinal = ", Afinal) 

	AfinalThresholded = tf.greater(Afinal, 0.0)	#threshold signal such that higher average weights are not preferenced
	AfinalThresholded = tf.dtypes.cast(AfinalThresholded, dtype=tf.dtypes.float32)	
	#print("Afinal = ", Afinal)
	#print("AfinalThresholded = ", AfinalThresholded)
	
	metric1 = tf.reduce_mean(AfinalThresholded)	#average output across batch, across layer
	
	#stdDevAcrossLayer = tf.math.reduce_std(Afinal, axis=1)	#stddev calculated across layer [1 result per batch index]
	#metric2 = tf.reduce_mean(stdDevAcrossLayer)	#average output across batch
	
	stdDevAcrossBatches = tf.math.reduce_mean(Afinal, axis=0)	 #for each dimension (k neuron in layer); calculate the mean across all batch indices
	metric2 = tf.math.reduce_std(stdDevAcrossBatches)	#then calculate the std dev across these values
	
	metric1 = metric1.numpy()
	metric2 = metric2.numpy()
	#print("metric1 = ", metric1)
	#print("metric2 = ", metric2)
				
	metric1 = metric1*metric1Weighting
	metric2 = metric2*metric2Weighting
	#print("metric1 = ", metric1)
	#print("metric2 = ", metric2)
	if(metric2 != 0):
		metric = metric1/metric2
	else:
		metric = 0.0
	
	return metric
	
	

def getRandomNetworkParameter(networkIndex, currentSubsetOfParameters):
	
	foundNewParameter = False
	while not foundNewParameter:
	
		variationDirection = random.randint(2)
		layer = random.randint(1, len(n_h))
		parameterTypeWorBtemp = random.randint(n_h[layer-1]+1)	#ensure that B parameter updates occur/tested less frequently than W parameter updates	#OLD: random.randint(2)	
		if(parameterTypeWorBtemp == n_h[layer-1]):
			parameterTypeWorB = 0
		else:
			parameterTypeWorB = 1
		hIndexCurrentLayer = random.randint(n_h[layer])	#randomNormal(n_h[l])
		hIndexPreviousLayer = random.randint(n_h[layer-1]) #randomNormal(n_h[l-1])
		networkParameterIndex = (parameterTypeWorB, layer, hIndexCurrentLayer, hIndexPreviousLayer, variationDirection)
	
		matches = [item for item in currentSubsetOfParameters if item == networkParameterIndex]
		if len(matches) == 0:
			foundNewParameter = True
			
	return networkParameterIndex				


def activationFunction(Z, prevLayerSize=None):
	return reluCustomPositive(Z, prevLayerSize)
	
def reluCustomPositive(Z, prevLayerSize=None):
	if(positiveExcitatoryWeightsActivationFunctionOffset):
		#CHECKTHIS: consider sigmoid instead of relu
		#offset required because negative weights are not used:
		
		#Zoffset = tf.ones(Z.shape)
		#Zoffset = tf.multiply(Zoffset, normalisedAverageInput)
		#Zoffset = tf.multiply(Zoffset, Wmean)
		#Zoffset = tf.multiply(Zoffset, prevLayerSize)
		Zpred = prevLayerSize*normalisedAverageInput*Wmean
		Zoffset = Zpred
		#print("Zoffset = ", Zoffset)
		
		Z = tf.subtract(Z, Zoffset) 
		A = tf.nn.relu(Z)
		A = tf.multiply(A, 2.0)	#double the slope of A to normalise the input:output signal
		#print("A = ", A)
	else:
		A = tf.nn.relu(Z)
		#A = tf.nn.sigmoid(Z)
	return A
	

def count_zero(M, axis=None):	#emulates count_nonzero behaviour
	if axis is not None:
		nonZeroElements = tf.math.count_nonzero(M, axis=axis)
		totalElements = tf.shape(M)[axis]
		zeroElements = tf.subtract(totalElements, nonZeroElements)
	else:
		totalElements = tf.size(M)
		nonZeroElements = tf.math.count_nonzero(M).numpy()
		zeroElements = tf.subtract(totalElements, nonZeroElements)
		zeroElements = zeroElements.numpy()
	return zeroElements



