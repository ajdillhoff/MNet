% load data set
trainingSet = buildTrainingSet( [16 15] );
cl = eye( 2 );
trainingLabels = zeros( 40, 1 );
for i = 1 : 2
    trainingLabels( ( i - 1 ) * 20 + 1 : i * 20 ) = i;
end
trainingLabels = cl( trainingLabels, : );
numInputs = size( trainingSet, 2 );
numExamples = size( trainingSet, 1 );

% load test set
testSet = buildTestSet( [16 15] );
testLabels = zeros( 10, 1 );
for i = 1 : 2
    testLabels( ( i - 1 ) * 10 + 1 : i * 10 ) = i;
end
testLabels = cl( testLabels, : );

batchSize = 1;

% build network
model = MNet();
model.AddLayer( MLinear( numInputs, 4 ) );
model.AddLayer( MSigmoid() );
model.AddLayer( MLinear( 4, 2 ) );
model.AddLayer( MSoftMax() );

% add criterion
criterion = MCEError();

% optimization options
w = model.GetParameters();

% randomly select order
idxs = randperm( numExamples );

cm = MConfusionMatrix( 2 );

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
