from nltk import word_tokenize
import numpy 
import matplotlib.pyplot as plt
import sys

wordCount = dict()
vocabularyList = list()
trainingAnswers = []
testingAnswers = []


#creates the vocabulary by going through the testing data and choosing every word that occurs more than 30 times
def create_vocab(filename):
	#open the file of testing data to creat
	emailFile = open(filename, 'r')
	for line in emailFile:
		# a set to hold words that have been seen in this particular email
		# only want words that occur in at least 30 emails, not 30 times
		wordsSeen = set()
		tokens = word_tokenize(line)
		## through all tokens except for the first (the tag)
		for token in tokens[1:]:
			if token not in wordsSeen:
				wordsSeen.add(token)
				if token in wordCount:
					wordCount[token] += 1
				else:

					wordCount[token] = 1
	#iterate through word count and put all words with more than 30 occurences in the vocabulary list
	for word in wordCount:
		if wordCount[word] >= 30:
			vocabularyList.append(word)

#turn text into feature vectors based off of the vocabulary list
#parameters: filename - the name of the file to extract the data
def vectorize(filename):
	#open the file containg the data to turn into feature vectors
	newFile = open(filename, 'r')
	answerKey = []
	allVectors = []
	for line in newFile:
		tokens = word_tokenize(line)
		answerKey.append(int(tokens[0]))
		newVector = []
		for word in vocabularyList:
			if word in tokens:
				newVector.append(1)			
			else:
				newVector.append(0)
		allVectors.append(newVector)
	return (allVectors, answerKey)
			

def perceptron_train(data, numRows, maxIts):
	#start with feature vector of all 0s
	weightVector = [0] * len(vocabularyList)
	updates = 0
	passes = 0
	madeChange = True
	vectorCount = -1
	#if maxIts == -1, no max number of iterations
	if(maxIts == -1):
		maxIts = sys.maxsize
	#keep iterating through the data until no changes to the weight vector have been made 
	while madeChange and maxIts > passes:
		madeChange = False
		#iterate through each vector in training set
		for vec, answer in zip(data[0:numRows], trainingAnswers[0:numRows]):
			vectorCount += 1
			#dot product of the vector and weight vector
			dotProduct = numpy.dot(vec, weightVector)
			#correct prediction
			if (dotProduct < 0 and answer == 0) or (dotProduct > 0 and answer == 1):
				continue
			#incorrect prediction
			#if the dot product is zero (Treated as spam) and it is spam
			elif (dotProduct == 0 and answer == 1):
				continue
			#not correctly marked
			else:
				madeChange = True
				updates += 1
				if answer == 0:
					weightVector = numpy.add(weightVector, [i*-1 for i in vec])
				else:
					weightVector = numpy.add(weightVector, vec)
		passes += 1
	return (weightVector, updates, passes)

#implements the average perceptron algorithm
def avg_perceptron_train(data, numRows, maxIts):
	#start with feature vector of all 0s
	weightVector = [0] * len(vocabularyList)
	allWeightVectors = [0] * len(vocabularyList)
	updates = 0
	passes = 0
	madeChange = True
	vectorCount = -1
	#if maxIts == -1, no max number of iterations
	if(maxIts == -1):
		maxIts = sys.maxsize
	#keep iterating through the data until no changes to the weight vector have been made 
	while madeChange and maxIts > passes:
		madeChange = False
		#iterate through each vector in training set
		for vec, answer in zip(data[:numRows], trainingAnswers[:numRows]):
			vectorCount += 1
			#dot product of the vector and weight vector
			dotProduct = numpy.dot(vec, weightVector)
			allWeightVectors = numpy.add(weightVector, allWeightVectors)
			#correct prediction
			if (dotProduct < 0 and answer == 0) or (dotProduct > 0 and answer == 1):
				continue
			#incorrect prediction
			#if the dot product is zero (Treated as spam) and it is spam
			elif (dotProduct == 0 and answer == 1):
				continue
			#not correctly marked
			else:
				madeChange = True
				updates += 1
				if answer == 0:
					weightVector = numpy.add(weightVector, [i*-1 for i in vec])
				else:
					weightVector = numpy.add(weightVector, vec)
		passes += 1
	avgWeightVector = [i/vectorCount for i in allWeightVectors]
	return (avgWeightVector, updates, passes)


def perceptron_test(w, data):
	vectorCount = 0
	mislabel = 0
	for vec, answer in zip(data, testingAnswers):
		vectorCount += 1
		dotProduct = numpy.dot(vec, w)
		#correctly classified 
		if(dotProduct < 0 and answer == 0) or (dotProduct > 0 and answer == 1):
			continue
		#corner case of 0 is treated as spam
		elif(dotProduct == 0 and answer == 1):
			continue
		else:
			mislabel += 1
	return (mislabel/vectorCount)

#finds the 15 words with the greatest and the least weights 
def greatest_weights(weightVector):
	greatest = numpy.argsort(-weightVector)
	least = numpy.argsort(weightVector)
	greatestWeight = []
	leastWeight = []
	for num in greatest[0:15]:
		greatestWeight.append(vocabularyList[num])
	for num in least[0:15]:
		leastWeight.append(vocabularyList[num])
	return (greatestWeight, leastWeight)



#create the vocab list first
create_vocab('train.txt')
trainingVectors, trainingAnswers = vectorize('train.txt')

#produces the data for the first few questions of the assignment
if('initTrain' in sys.argv):
	#first test training set on itself
	testingVectors, testingAnswers = vectorize('train.txt')
	weightVector, numUpdates, numPasses = perceptron_train(trainingVectors, 4000, -1)
	print("Number of updates while training: ", numUpdates, "Number of passes: ", numPasses)
	print("Percent error in classification with training set used as test set: ", perceptron_test(weightVector, testingVectors))

	#next test with validation set
	testingVectors, testingAnswers = vectorize('validation.txt')
	print("Percent error in classification with validation set: ", perceptron_test(weightVector, testingVectors))

	#the words with the greatest and least weight
	greatestWords, leastWords = greatest_weights(weightVector)
	print("Higest valued words: ", greatestWords)
	print("Lowest valued words: ", leastWords)

#for testing out different configurations of the perceptron and average perceptron algorithms
elif('config' in sys.argv):
	it = int(sys.argv[2])
	algo = sys.argv[3]
	testingVectors, testingAnswers = vectorize('validation.txt')
	if(algo == "avg"):
		weightVector, numUpdates, numPasses = avg_perceptron_train(trainingVectors, 4000, it)
		print("Updates:", numUpdates, "Iterations:", numPasses)
		valErr = perceptron_test(weightVector, testingVectors)
	else:
		weightVector, numUpdates, numPasses = perceptron_train(trainingVectors, 4000, it)
		print("Updates:", numUpdates, "Iterations:", numPasses)
		valErr = perceptron_test(weightVector, testingVectors)
	print("Validation Error:", valErr)

#final test of training and testing sets 
else:
	it = int(sys.argv[1])
	algo = sys.argv[2]
	trainVectors, trainingAnswers = vectorize('spam_train.txt')
	testingVectors, testingAnswers = vectorize('spam_test.txt')
	if(algo == "avg"):
		weightVector, numUpdates, numPasses = avg_perceptron_train(trainingVectors, 4000, it)
	else:
		weightVector, numUpdates, numPasses = perceptron_train(trainingVectors, 4000, it)
	testingError = perceptron_test(weightVector, testingVectors)
	print("Testing Error:", testingError)