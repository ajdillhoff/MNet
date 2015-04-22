function [trainingSet, trainingLabels, testSet, testLabels] = loadMNIST()
    trainingSet = loadMNISTImages( '/home2/Projects/MNIST/train-images-idx3-ubyte' );
    trainingLabels = loadMNISTLabels( '/home2/Projects/MNIST/train-labels-idx1-ubyte' );
    testSet = loadMNISTImages( '/home2/Projects/MNIST/t10k-images-idx3-ubyte' );
    testLabels = loadMNISTLabels( '/home2/Projects/MNIST/t10k-labels-idx1-ubyte' );
end
