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
[trainingSet, trainingLabels, ~, ~] = loadMNIST();
trainingSet = trainingSet';
numInputs = size( trainingSet, 2 );
numExamples = size( trainingSet, 1 );
cl = eye( 10 );
trainingLabels = 1 + trainingLabels;
trainingLabels = uint8( trainingLabels );
trainingLabels = cl( trainingLabels, : );

batchSize = 10;

% build network
model = MNet();
model.AddLayer( MLinear( numInputs, 1536 ) );
model.AddLayer( MSigmoid() );
model.AddLayer( MLinear( 1536, 10 ) );
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

    % Check validation set
    %o = model.Forward( testSet );
    %o = o > 0.5;

    %cm = zeros( 2 );
    %cm( 1, 1 ) = sum( o( 1 : 70, : ) == 1 );
    %cm( 1, 2 ) = sum( o( 1 : 70, : ) == 0 );
    %cm( 2, 1 ) = sum( o( 71 : end, : ) == 1 );
    %cm( 2, 2 ) = sum( o( 71 : end, : ) == 0 );
    %pre = cm( 1, 1 ) / ( cm( 1, 1 ) + cm( 1, 2 ) );
    %recall = cm( 1, 1 ) / ( cm( 1, 1 ) + cm( 2, 1 ) );
    %fprintf( 'Precision %.20f\n', pre );
    %fprintf( 'recall %.20f\n', recall );

    fprintf( 'Loss: %f\n', J );
    % print confusion matrix
    cm.Print();
    cm.Reset();
    epoch = epoch + 1;
end
