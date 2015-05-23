import config
import os
import scipy.io.wavfile
import scikits.talkbox.features
import numpy as np

def readFromDisk(filepath):
	return np.load(filepath)

def saveToDisk(ceptralCoeffs,filepath):
	meanVector = np.mean(ceptralCoeffs,axis=0)
	np.save(filepath, meanVector)

def createMFCC(filepath):
	samplerate,songdata = scipy.io.wavfile.read(filepath)
	ceptralCoeffs,_,_ = scikits.talkbox.features.mfcc(songdata)
	return ceptralCoeffs

if __name__ == "__main__" :
	base_dir = config.BASE_DIR
	for dirpath,subdirs,files in os.walk(base_dir):
		for filename in files :
			filepath = os.path.join(dirpath,filename)
			ceptralCoeffs = createMFCC(filepath)
			saveToDisk(ceptralCoeffs,filepath)