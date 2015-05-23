import config
import os
from extractfeature import readFromDisk 
from sklearn.cross_validation import ShuffleSplit 
from sklearn.linear_model import LogisticRegression
import numpy as np

def train(x,y):
	x = np.array(x)
	y = np.array(y)
	logregr = LogisticRegression()
	for trainIndex,testIndex in ShuffleSplit(len(x), n_iter=1, test_size=0.2) :
		x_train = x[trainIndex]
		y_train = y[trainIndex]
		x_test = x[testIndex]
		y_test = y[testIndex]
		logregr.fit(x_train,y_train)
	
	for x_test,y_test in zip(x_test,y_test):
		print "Predicted = ",logregr.predict(x_test)," Expected = ", y_test

def initializeTrainData():
	classificationTargetValues = dict((value,index) for index,value in enumerate(config.GENRES) )
	x = list()
	y = list()
	for dirpath,subdirs,files in os.walk(config.BASE_DIR):
		head,tail = os.path.split(dirpath)
		if tail not in config.GENRES :
			continue
		target = classificationTargetValues[tail]
		for filename in files:
			filePath = os.path.join(dirpath,filename)
			if not filePath.endswith(".npy") :
				continue
			features = readFromDisk(filePath)
			x.append(features)
			y.append(target)
	return x,y


if __name__ == '__main__':
	x,y = initializeTrainData()
	train(x,y)









