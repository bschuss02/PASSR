#%%
from __future__ import print_function, division, absolute_import
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')
#%matplotlib inline

import librosa
import librosa.display
import IPython.display

import os
import sys
import re
import shutil
import datetime
import logging
import colorlog
import progressbar
import speech_recognition as sr
from os import path
from pydub import AudioSegment
import wavio
import pyaudio
import wave

import tensorflow as tf
from sklearn.model_selection import train_test_split

# Setting up progressbar and logger
# progressbar.streams.wrap_stderr()
logger = colorlog.getLogger("ASSR")
handler = logging.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s%(levelname)-8s| %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class Dataset:
    def __init__(self, datasetDir, datasetLabelFilename, datasetArrayFilename):
        self.n_features = 80
        logger.info("Number of features: %s", self.n_features)
        self.X = np.empty(shape=(0, self.n_features))
        self.Y = np.empty(shape=(0, 2))
        
        self.datasetArrayFilename = datasetArrayFilename
        logger.debug("Dataset array filename: %s", self.datasetArrayFilename)
        
        if os.path.isfile(self.datasetArrayFilename):    
            self.__readFromFile()
        else:
            self.datasetDir = datasetDir
            logger.debug("Dataset Directory: %s", self.datasetDir)

            self.datasetLabelFilename = datasetLabelFilename
            logger.debug("Dataset labels filename: %s", self.datasetLabelFilename)

            if not os.path.isdir(self.datasetDir) or not os.path.isfile(self.datasetLabelFilename):
                logger.info("%s or %s does not exists", self.datasetDir, self.datasetLabelFilename)
                self.__buildDatasetAndLabels('wav/release1')
                
            self.__build()
            self.__writeToFile()
    
    def __build(self):
        logger.info("Building dataset from directory: %s", self.datasetDir)
        num_lines = sum(1 for line in open(self.datasetLabelFilename, 'r'))
        with open(self.datasetLabelFilename, 'r') as datasetLabelFile:
            filesProcessed=0
            pbar = progressbar.ProgressBar(redirect_stdout=True)
            for line in pbar(datasetLabelFile, max_value=num_lines):
                lineSplit = line.strip().split(' ')
                audiofilename = lineSplit[0]
                label = lineSplit[1]
                try:
                    features = FeatureExtraction()
                    features.loadFile(os.path.join(self.datasetDir, audiofilename))
                    features.melspectrogram()
                    features.extractmfcc()
#                     features.extractmfcc(mode='nearest')
                    features.extractrms()
                except ValueError:
                    logger.warning("Error in extracting features from file %s", audiofilename)
                    continue
                
                featureVector = []
                for feature in features.mfcc:
                    featureVector.append(np.mean(feature))
                    featureVector.append(np.var(feature))
                
                for feature in features.delta_mfcc:
                    featureVector.append(np.mean(feature))
                    featureVector.append(np.var(feature))
                
                for feature in features.delta2_mfcc:
                    featureVector.append(np.mean(feature))
                    featureVector.append(np.var(feature))
                
                featureVector.append(np.mean(features.rms))
                featureVector.append(np.var(features.rms))
                
                self.X = np.vstack((self.X, [featureVector]))
                
                if label == "STUTTER":
                    self.Y = np.vstack((self.Y, [0, 1]))
                elif label == "NORMAL":
                    self.Y = np.vstack((self.Y, [1, 0]))
                else:
                    logger.error("Unexpected label: %s", label)
                    sys.exit()
                
                filesProcessed += 1            
            
            logger.info("Total files processed: %d", filesProcessed)
    
    def __buildDatasetAndLabels(self, audioAndChaFilesDirectory):
        logger.info("Rebuilding the dataset directory and labels")
        if os.path.isdir(self.datasetDir):
            shutil.rmtree(self.datasetDir)
        os.makedirs(self.datasetDir)
        
        labelFile = open(self.datasetLabelFilename, 'w')
        
        splitDuration = 300 # milliseconds
        pbar = progressbar.ProgressBar(redirect_stdout=True)
        for chaFileName in pbar(os.listdir(audioAndChaFilesDirectory)):
            if chaFileName.endswith(".cha"):
                subject = chaFileName.split('.')[0]
                wavFileName = subject + ".wav"
                y, sr = librosa.load(os.path.join(audioAndChaFilesDirectory, wavFileName))

                logger.debug("Parsing file: %s", chaFileName)

                with open(os.path.join(audioAndChaFilesDirectory, chaFileName), 'r') as chaFile:
                    sndFound = False
                    phoFound = False
                    startTime = -1
                    endTime = -1
                    label = None
                    for line in chaFile:
                        if not sndFound:
                            if re.search(r"%snd:", line):
                                lineSplit = line.split("_")
                                startTime = int(lineSplit[-2])
                                endTime = lineSplit[-1]
                                endTime = int(re.sub(r"\u0015\n", '', endTime))
                                sndFound = True
                        else:
                            if re.search(r"%pho:", line):
                                if re.search(r'[A-Z]', line):
                                    label = "STUTTER"
                                else:
                                    label = "NORMAL"
                                phoFound = True
                        if sndFound and phoFound:
                            n_splits = int(np.round((endTime - startTime) / splitDuration))
                            
                            startingSample = int(startTime * sr / 1000)
                            for i in range(1, n_splits):
                                endingSample = int(startingSample + (splitDuration * sr / 1000))
                                audiofilename = subject + ":" + str(startTime) + ":" + str(int(startTime) + splitDuration) + ".wav"
                                labelFile.write(audiofilename + " " + label + "\n")
                                audio = y[startingSample:endingSample]
                                librosa.output.write_wav(os.path.join(self.datasetDir, audiofilename), audio, sr)
                                
                                startingSample = endingSample
                                startTime = int(startTime) + splitDuration
                            
                            endingSample = int(endTime * sr / 1000)
                            audiofilename = subject + ":" + str(startTime) + ":" + str(endTime) + ".wav"
                            labelFile.write(audiofilename + " " + label + "\n")
                            audio = y[startingSample:endingSample]
                            librosa.output.write_wav(os.path.join(self.datasetDir, audiofilename), audio, sr)
                            
                            
                            sndFound = False
                            phoFound = False
                            startTime = -1
                            endTime = -1
                            label = None

        labelFile.close()
    
    def __writeToFile(self, filename=None):
        if filename == None:
            filename = self.datasetArrayFilename
            
        if os.path.exists(filename):
            os.remove(filename)
        np.savetxt(filename, np.hstack((self.X, self.Y)))
        logger.info("Array stored in file %s", filename)
    
    def __readFromFile(self, filename=None):
        if filename == None:
            filename = self.datasetArrayFilename
            
        if not os.path.isfile(filename):
            logger.error("%s does not exists or is not a file", filename)
            sys.exit()
        matrix = np.loadtxt(filename)
        self.X = matrix[:, 0:self.n_features]
        self.Y = matrix[:, self.n_features:]
        logger.info("Array read from file %s", filename)



#%%
dataset = Dataset('dataset', 'datasetLabels.txt', 'datasetArray80.gz')
X_train, X_test, Y_train, Y_test = train_test_split(dataset.X, dataset.Y)

total_batch = int(len(X_train) / 100)
X_batches = np.array_split(X_train, total_batch)
Y_batches = np.array_split(Y_train, total_batch)

for i in range(total_batch):
    temp_batch_x, temp_batch_y = X_batches[i], Y_batches[i]
    batch_x = []
    for i in range(total_batch):
        # print(X_batches[0][0])
        temp_batch_x, temp_batch_y = X_batches[i], Y_batches[i]

        for j in range(len(temp_batch_x)):
            temporary_label = temp_batch_y[j][0]
            if temporary_label == 1.:
                batch_x.append(np.repeat(temp_batch_x[j],2))
    np_batch_x = np.array(batch_x)