# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% [markdown]
# # ASSR: Automatic Stuttered Speech Recoginition

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

import tensorflow as tf
from sklearn.model_selection import train_test_split

# Setting up progressbar and logger
# progressbar.streams.wrap_stderr()
logger = colorlog.getLogger("ASSR")
handler = logging.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s%(levelname)-8s| %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

#%% [markdown]
# ## Data Preparation

#%%
class FeatureExtraction:
    def __init__(self, n_mels=128):
        self.n_mels = n_mels
        self.y = None
        self.sr = None
        self.S = None
        self.log_S = None
        self.mfcc = None
        self.delta_mfcc = None
        self.delta2_mfcc = None
        self.M = None
        self.rms = None
    
    def loadFile(self, filename):
        self.y, self.sr = librosa.load(filename)
        logger.debug('File loaded: %s', filename)
    
    def load_y_sr(self, y, sr):
        self.y = y
        self.sr = sr
    
    def melspectrogram(self):
        self.S = librosa.feature.melspectrogram(self.y, sr=self.sr, n_mels=self.n_mels)
        self.log_S = librosa.amplitude_to_db(self.S, ref=np.max)
    
    def plotmelspectrogram(self):
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(self.log_S, sr=self.sr, x_axis='time', y_axis='mel')
        plt.title('mel Power Spectrogram')
        plt.colorbar(format='%+02.0f dB')
        plt.tight_layout()
    
    def extractmfcc(self, n_mfcc=13):
        self.mfcc = librosa.feature.mfcc(S=self.log_S, n_mfcc=n_mfcc)
#         self.delta_mfcc = librosa.feature.delta(self.mfcc)
        self.delta_mfcc = librosa.feature.delta(self.mfcc,mode='nearest')
        self.delta2_mfcc = librosa.feature.delta(self.mfcc, order=2,mode='nearest')
        self.M = np.vstack([self.mfcc, self.delta_mfcc, self.delta2_mfcc])
    
    def plotmfcc(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(3, 1, 1)
        librosa.display.specshow(self.mfcc)
        plt.ylabel('MFCC')
        plt.colorbar()
        
        plt.subplot(3, 1, 2)
        librosa.display.specshow(self.delta_mfcc)
        plt.ylabel('MFCC-$\Delta$')
        plt.colorbar()
        
        plt.subplot(3, 1, 3)
        librosa.display.specshow(self.delta2_mfcc, sr=self.sr, x_axis='time')
        plt.ylabel('MFCC-$\Delta^2$')
        plt.colorbar()
        
        plt.tight_layout()
    
    def extractrms(self):
        self.rms = librosa.feature.rms(y=self.y)


#%%
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

#%% [markdown]
# ## Tensorflow binary classification

#%%
class NeuralNetwork:
    def __init__(self, X_train=None, Y_train=None, X_test=None, Y_test=None):
        # Data
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        
        # Learning Parameters
        self.learning_rate = 0.001
        self.training_epochs = 1200
        self.batch_size = 100
        self.display_step = 100

        # Model Parameters
        self.n_hidden = [10, 10, 10]
        self.hiddenLayers = len(self.n_hidden)
        self.n_input = 80
        self.n_classes = 2

        logger.debug("Neural network of depth %d", self.hiddenLayers)
        for i in range(self.hiddenLayers):
            logger.debug("Depth of layer %d is %d", (i + 1), self.n_hidden[i])

        self.x = tf.placeholder("float", [None, self.n_input])
        self.y = tf.placeholder("float", [None, self.n_classes])
        self.layer = None
        self.weights = None
        self.biases = None
        # Model
        self.model = self.__network(self.x)
        self.save_path = None

        # Loss function and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.model, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # Initialize the variables
        self.init = tf.global_variables_initializer()
    
    def setTrainData(self, X, Y):
        self.X_train = X
        self.Y_train = Y
        
    def setTestData(self, X, Y):
        self.X_test = X
        self.Y_test = Y
        
    def __network(self, x):
        self.layer = []
        self.weights = []
        self.biases = []

        for n_layer in range(self.hiddenLayers):
            if n_layer == 0:
                self.weights.append(tf.Variable(tf.random_normal([self.n_input, self.n_hidden[n_layer]])))
                self.biases.append(tf.Variable(tf.random_normal([self.n_hidden[n_layer]])))
                self.layer.append(tf.nn.relu(tf.add(tf.matmul(x, self.weights[n_layer]), self.biases[n_layer])))
            else:
                self.weights.append(tf.Variable(tf.random_normal([self.n_hidden[n_layer - 1], self.n_hidden[n_layer]])))
                self.biases.append(tf.Variable(tf.random_normal([self.n_hidden[n_layer]])))
                self.layer.append(tf.nn.relu(tf.add(tf.matmul(self.layer[n_layer - 1], self.weights[n_layer]), self.biases[n_layer])))


        # Output layer
        self.weights.append(tf.Variable(tf.random_normal([self.n_hidden[self.hiddenLayers - 1], self.n_classes])))
        self.biases.append(tf.Variable(tf.random_normal([self.n_classes])))
        self.layer.append(tf.matmul(self.layer[self.hiddenLayers - 1], self.weights[self.hiddenLayers]) + self.biases[self.hiddenLayers])

        return self.layer[self.hiddenLayers]
    
    def train(self, volume_coefficient):
        logger.info("Training the neural network")
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(self.init)
            pbarWidgets = [
                progressbar.Percentage(),
                ' (',
                progressbar.SimpleProgress(),
                ') ',
                progressbar.Bar(),
                ' ',
                progressbar.Timer(),
                ' ',
                progressbar.ETA(),
                ' ',
                progressbar.DynamicMessage('Cost'),
            ]
            with progressbar.ProgressBar(max_value=self.training_epochs, redirect_stdout=True, widgets=pbarWidgets) as pbar:
                for epoch in range(self.training_epochs):
                    avg_cost = 0
                    total_batch = int(len(self.X_train) / self.batch_size)
                    X_batches = np.array_split(self.X_train, total_batch)
                    Y_batches = np.array_split(self.Y_train, total_batch)
                    
        return X_batches, Y_batches

#%%
def audio_to_text(filepath):
    AUDIO_FILE = filepath

    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)  # read the entire audio file                  

    return r.recognize_google(audio)


#%%
def run(train=False, correct=False, louder=False):
    dataset = Dataset('dataset', 'datasetLabels.txt', 'datasetArray80.gz')
    X_train, X_test, Y_train, Y_test = train_test_split(dataset.X, dataset.Y)

    tf.reset_default_graph()
    nn = NeuralNetwork(X_train, Y_train, X_test, Y_test)
    return nn.train(8)

#%%
def invlogamplitude(S):
    """librosa.logamplitude is actually 10_log10, so invert that."""
    return 10.0**(S/10.0)

def to_wav(batch):
    mfccs = batch
    n_mfcc = mfccs.shape[0]
    n_mel = 128
    dctm = librosa.filters.dct(n_mfcc, n_mel)
    n_fft = 2048
    mel_basis = librosa.filters.mel(22000, n_fft)


    # Empirical scaling of channels to get ~flat amplitude mapping.
    bin_scaling = 1.0/np.maximum(0.0005, np.sum(np.dot(mel_basis.T, mel_basis),axis=0))

    # Reconstruct the approximate STFT squared-magnitude from the MFCCs.
    recon_stft = bin_scaling[:, np.newaxis] * np.dot(mel_basis.T,invlogamplitude(np.dot(dctm.T, mfccs)))

    # Impose reconstructed magnitude on white noise STFT.
    excitation = np.random.randn(y.shape[0])
    E = librosa.stft(excitation)
    recon = librosa.istft(E/np.abs(E)*np.sqrt(recon_stft))

    # Output
    librosa.output.write_wav('from_mfcc.wav', recon, sr)

    plt.style.use('seaborn-darkgrid')
    plt.figure(1)
    plt.subplot(211)
    librosa.display.waveplot(y, sr)
    plt.subplot(212)
    librosa.display.waveplot(recon,sr)
    plt.show()


#%%
if __name__ == "__main__":
    X_batches, Y_batches = run(True,False)
    to_wav(X_batches[0])
    print(X_batches, Y_batches)












#%%
"""
import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt


def invlogamplitude(S):
    # librosa.logamplitude is actually 10_log10, so invert that.
    return 10.0**(S/10.0)


# load
filename = u'sample.wav'
y, sr = librosa.load(filename)


# calculate mfcc
Y = librosa.stft(y)
mfccs = librosa.feature.mfcc(y)


# Build reconstruction mappings,
n_mfcc = mfccs.shape[0]
n_mel = 128
dctm = librosa.filters.dct(n_mfcc, n_mel)
n_fft = 2048
mel_basis = librosa.filters.mel(sr, n_fft)


# Empirical scaling of channels to get ~flat amplitude mapping.
bin_scaling = 1.0/np.maximum(0.0005, np.sum(np.dot(mel_basis.T, mel_basis),axis=0))

# Reconstruct the approximate STFT squared-magnitude from the MFCCs.
recon_stft = bin_scaling[:, np.newaxis] * np.dot(mel_basis.T,invlogamplitude(np.dot(dctm.T, mfccs)))

# Impose reconstructed magnitude on white noise STFT.
excitation = np.random.randn(y.shape[0])
E = librosa.stft(excitation)
recon = librosa.istft(E/np.abs(E)*np.sqrt(recon_stft))


# Output
librosa.output.write_wav('output.wav', recon, sr)

plt.style.use('seaborn-darkgrid')
plt.figure(1)
plt.subplot(211)
librosa.display.waveplot(y, sr)
plt.subplot(212)
librosa.display.waveplot(recon,sr)
plt.show()
"""