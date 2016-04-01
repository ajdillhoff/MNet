% load FRGC data set
%trainingSet = buildTrainingSet( [40 40] );
%cl = eye( 20 );
%targets = zeros( 2000, 1 );
%for i = 1 : 20
    %targets( ( i - 1 ) * 100 + 1 : i * 100 ) = i;
%end
%targets = cl( targets, : );
%numInputs = size( trainingSet, 2 );

% load test set
%testSet = buildTestSet( [10 10] );

% Test using MNIST data set
[trainingSet, trainingLabels, testSet, testLabels] = loadMNIST( 0 );
numInputs = size( trainingSet, 2 );
numExamples = size( trainingSet, 1 );

cl = eye( 10 );
trainingLabels = 1 + trainingLabels;
trainingLabels = uint8( trainingLabels );
trainingLabels = cl( trainingLabels, : );

if exist( 'testLabels' )
    testLabels = 1 + testLabels;
    testLabels = uint8( testLabels );
    testLabels = cl( testLabels, : );
end

% Normalize input data
avg = mean( trainingSet, 1 );
trainingSet = bsxfun( @minus, trainingSet, avg );
avg = mean( testSet, 1 );
testSet = bsxfun( @minus, testSet, avg );

batchSize = 10;

% build network
model = MNet();
model.AddLayer( MLinear( numInputs, 1568 ) );
model.AddLayer( MSigmoid() );
model.AddLayer( MLinear( 1568, 10 ) );
model.AddLayer( MSoftMax() );

% add criterion
criterion = MCEError();

% optimization options
w = model.GetParameters();

% randomly select order
idxs = randperm( numExamples );

cm = MConfusionMatrix( 10 );

epoch = 0;
while true
    J = 0;
    for batch = 1 : batchSize : numExamples
        range = idxs( batch : batchSize + batch - 1 );
        currentBatch = trainingSet( range, : );
        currentTargets = trainingLabels( range, : );

        % handle to cost function
        costFunc = @(w) costFunction( w, model, criterion, currentBatch, currentTargets, cm );

        [w, cost] = sgd( costFunc, w );
        J = J + cost;
    end

    fprintf( 'Loss: %f\n', J );
    % print confusion matrix
    cm.Print();
    cm.Reset();

    % test against test set
    if exist( 'testSet' )
        fprintf( 'Testing using test set\n' );
        for batch = 1 : batchSize : size( testSet, 1 );
            range = batch : batchSize + batch - 1;
            currentBatch = testSet( range, : );
            currentTargets = testLabels( range, : );

            % handle to cost function
            costFunction( w, model, criterion, currentBatch, currentTargets, cm );
        end
    end

    % print confusion matrix
    cm.Print();
    cm.Reset();
    epoch = epoch + 1;
end
