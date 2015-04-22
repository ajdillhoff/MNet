function [trainingSet, trainingLabels, testSet, testLabels] = loadMNIST( szFlag )
% function loadMNIST( szFlag )
%
% Loads MNIST training and test sets.
%
% szFlag - If this value is true, the entire data set is loaded. Otherwise a
% small subset (2000) is returned.

    %trainingSet = loadMNISTImages( '/home2/Projects/MNIST/train-images-idx3-ubyte' );
    %trainingLabels = loadMNISTLabels( '/home2/Projects/MNIST/train-labels-idx1-ubyte' );
    %testSet = loadMNISTImages( '/home2/Projects/MNIST/t10k-images-idx3-ubyte' );
    %testLabels = loadMNISTLabels( '/home2/Projects/MNIST/t10k-labels-idx1-ubyte' );
    trainingSet = loadMNISTImages( '~/Documents/Data Sets/MNIST/train-images-idx3-ubyte' );
    trainingLabels = loadMNISTLabels( '~/Documents/Data Sets/MNIST/train-labels-idx1-ubyte' );
    testSet = loadMNISTImages( '~/Documents/Data Sets/MNIST/t10k-images-idx3-ubyte' );
    testLabels = loadMNISTLabels( '~/Documents/Data Sets/MNIST/t10k-labels-idx1-ubyte' );

    % tranpose traininSet and testSet. We want examples x features.
    trainingSet = trainingSet';
    testSet = testSet';

    if szFlag == 0
        % training set
        trainingClassSize = 6000;
        numExamples = size( trainingSet, 1 );
        train = zeros( 0, size( trainingSet, 2 ) );
        targets = zeros( 0, 1 );
        for i = 1 : trainingClassSize : numExamples;
            train( end + 1 : end + 200, : ) = trainingSet( i : i - 1 + 200, : );
            targets( end + 1 : end + 200, 1 ) = trainingLabels( i : i - 1 + 200 );
        end

        trainingSet = train;
        trainingLabels = targets;
        
        % test set
        testClassSize = 1000;
        numExamples = size( testSet, 1 );
        train = zeros( 0, size( testSet, 2 ) );
        targets = zeros( 0, 1 );
        for i = 1 : testClassSize : numExamples;
            train( end + 1 : end + 100, : ) = testSet( i : i - 1 + 100, : );
            targets( end + 1 : end + 100, 1 ) = testLabels( i : i - 1 + 100 );
        end

        testSet = train;
        testLabels = targets;
    end
end
