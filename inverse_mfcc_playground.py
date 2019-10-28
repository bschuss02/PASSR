#%%
import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
import scipy

#%%
def invlogamplitude(S):
    """librosa.logamplitude is actually 10_log10, so invert that."""
    return 10.0**(S/10.0)

#%%
# load
filename = u'sample.wav'
y, sr = librosa.load(filename)

#%%
# calculate mfcc
Y = librosa.stft(y)
mfccs = librosa.feature.mfcc(y)

#%%
# Build reconstruction mappings,
n_mfcc = mfccs.shape[0]
n_mel = 128
dctm = librosa.filters.dct(n_mfcc, n_mel)
n_fft = 2048
mel_basis = librosa.filters.mel(sr, n_fft)

#%%
# Empirical scaling of channels to get ~flat amplitude mapping.
bin_scaling = 1.0/np.maximum(0.0005, np.sum(np.dot(mel_basis.T, mel_basis),axis=0))
#%%
# Reconstruct the approximate STFT squared-magnitude from the MFCCs.
recon_stft = bin_scaling[:, np.newaxis] * np.dot(mel_basis.T,invlogamplitude(np.dot(dctm.T, mfccs)))
#%%
# Impose reconstructed magnitude on white noise STFT.
excitation = np.random.randn(y.shape[0])
E = librosa.stft(excitation)
recon = librosa.istft(E/np.abs(E)*np.sqrt(recon_stft))

#%%
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
audio=np.frombuffer(stream.read(CHUNK))
S = librosa.feature.melspectrogram(audio, sr=RATE)

audio=np.frombuffer(stream.read(CHUNK),dtype=np.int16)
S = librosa.feature.melspectrogram(audio.astype('float32'), sr=RATE)
"""