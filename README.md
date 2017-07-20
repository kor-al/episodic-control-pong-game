# episodic-control-pong-game
Model-Free Episodic Control algorithm written in C++ and tested in Unreal Engine 4 in the Pong Game.

## Algorithm
This algorithm is described in detail in the folowing paper: 
"Model-Free Episodic Control" by Blundell Charles, Uria Benigno, Pritzel Alexander, Li Yazhe et. al. (https://arxiv.org/pdf/1606.04460.pdf)

An epsilon-greedy strategy is added: an epsilon is decreasing with the time.

## Result
The algorithm fails in the PongGame. Another evidence can be found in the folowing paper: https://arxiv.org/pdf/1703.01988.pdf

## Implementation
### Environment processing
Every screenshot is tranformed to a simplified (b/w, smaller size, filters) image. This image is projected to a vector state.
Random projection s.t. x->Ax is exploited, where A is randrom matrix with entries that are from a standart Gaussian distribution. 
### KNN
A KNN search is implemented using a KDtree from the FLANN library from OpenCV 3.0.
* may be it should be replaced with the nanoflann library
### Backup
A calculated QECtable is saved (serialized) to a .xml archive. This file is quite heavy (~47 MB) but still readable. To decrease its size, a binary archive can be used.
A small summary is also written to .txt file simultaneously with the archive.
### VAE
A VAE is implemented using the TensorFlow library and the UnrealEnginePython plagin for Unreal Engine 4 (https://github.com/akashin/UnrealEnginePython). Both the encoder and decoder networks are convolutional. The distribution of latent variables is chosen to be Gaussian while the decoder uses Bernoulli distribution since game screenshots are grayscale. 

## Tuning parameters
* number of neighbors (K - kKNN) [11]
* dimension of a state vector (kDimState) [32]
* size of a transformed screenshot (kTransformedImageDim) [84 x 84]
* buffer size (the size of QEC table for each action - kBufferMaxsize) [1000000]
* epsilon decay from a maximum to a minimum value (kEpsMax, kEpsMin, kEpsDenominator)
* discount coefficient (kECdiscount) [0.99]
* number of episodes before the next save (kSaveEpisodes)
* number of frames to skip (kFrameSkip) [5]

## Dependencies
1. OpenCV plagin for Unreal Engine 4 (https://wiki.unrealengine.com/Integrating_OpenCV_Into_Unreal_Engine_4)
  - screenshots processing
  - KD trees from FLANN library
2. Boost library
  - serialisation of QEC table to load previously obtained results
3. UnrealEnginePython plagin for UE4 
  - VAE and plot of a learning curve 
 
## References
1. HSE_AI_Labs/Lab_4/ by Andrey Kashin (https://github.com/akashin/HSE_AI_Labs)
2. Model-Free-Episodic-Control written in Python by Frank He (https://github.com/ShibiHe/Model-Free-Episodic-Control)
