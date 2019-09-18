# PASSR
By Benji Schussheim and Rishabh Mandayam.

Personalized Automatic Stuttered Speech Recognition: Create a speech recognition algorithm that can understand stuttering by having personalizeable settings where the user can purposefully stutter in different ways (eg: slower, louder, lower), increasing accuracy for a neural net that classifies what is stuttering and what is not.

The system modifies a neural network to recognize and edit out stuttering from the repo anshulgupta0803/ASSR. That NN splits audio into overlapping frames and uses a binary classification neural network with tensorFlow to recognize if the audio frame has stuttering. If it does, the audio frame is edited out. The audio that is removed of stuttering is fed through IBM Watson speech recognition.

There are two approaches to incorperating settings for stuttering differently. One is to modify the stuttering segments of the training data with the desired audio characteristic so that when the user stutters with that audio characteristic, the neural network is more likely to accurately identify it. The second approach is to change the binary classification of the original NN to instead return a confidence interval between 0 and 1. The degree that an audio frame has the charcteristic of interest is multiplied with the confidence interval to potential push it over the breaking point of .5 so that it is either classified or not classified as stuttering once the number is rounded to 0 or 1.
