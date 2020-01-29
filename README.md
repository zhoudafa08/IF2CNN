# IF2CNN: Towards non-stationary time series feature extraction by integrating iterative filtering and convolutional neural networks. The manuscript has been submitted to IEEE Transactions on Cybernetics on Jan. 25, 2020.

# Requirements
1. Language: Python 2.7.15, Matlab R2014b
2. Libraries that Python programs depend on: Numpy, Keras and Scikit-Learn

# Descriptions of the main code files
1. ./src/if2cnn.py, the main code of IF2CNN that adaptively extracts the features from the non-stationary time series.

2. ./src/fnn.py, Implementation of the prediction algorithm proposed in [1].

3. ./src/FIF, The directory includes the Code of the fast-iterative-filtering(FIF)-based signal decomposition technique, which is proposed in [2-3].

4. ./src/sliding_if.m, Using the FIF-based signal decomposition technique to generate the samples that are fed into the IF2CNN as the inputs.

# References
[1] F. Zhou, H. Zhou, Z. Yang, and L. Yang, “EMD2FNN: A strategy combining empirical mode decomposition and factorization machine based neural network for stock market trend prediction,” Expert Systems With Applications, vol. 115, pp. 136–151, 2019.
[2] A. Cicone, J. Liu, and H. Zhou, “Adaptive local iterative filtering for signal decomposition and instantaneous frequency analysis,” Applied and Computational Harmonic Analysis, vol. 41, no. 2, pp. 384–411, 2016.
[3] A. Cicone and H. Zhou, “Multidimensional iterative filtering method for the decomposition of high-dimensional non-stationary signals,” Cambridge Core in Numerical Mathematics: Theory, Methods and Applications, vol. 10, no. 2, pp. 278–298, 2017.
