This is official Python implementation for the paper "AVJnet: Atrioventricular Junction Point Tracking Network in Cardiac Magnetic Resonance" by Jianguo Chen, Xulei Yang, Shuang Leng, Ru-San Tan, Zeng Zeng, and Liang Zhong.

The target of the AVJnet model is to automatically detect and track Atrioventricular junction (AVJ) motion during cardiac cycle. 

The AVJnet model consists of AVJ point detection module and motion tracking module.   

In AVJ point detection, we design the convolutional-based feature extraction and elastic regression to detect AVJ points frame by frame of each CMR video. 

Then, in AVJ tracking, we adopt the Deep_SORT model to capture spatio-temporal continuity between frames and fine-tune the coordinate position of AVJ points.
