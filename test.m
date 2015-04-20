%% Testing Program with OR problem
data = [0 0; 0 1; 1 0; 1 1];
targets = [0; 1; 1; 0];
numInputs = size( data, 2 );
trainingSet = data;

% load FRGC data set
%data = buildTrainingSet( [10 10] );
%targets = zeros( 2000, 1 );
%targets( 1 : 100 ) = 1;
%numInputs = size( data, 2 );
%data = data( 1 : 200, : );
%targets = targets( 1 : 200 );

% try 10 images 5 face1 5 face2
%trainingSet = data( 1 : 50, : );
%trainingSet = [trainingSet; data( 101 : 150, : )];
%targets = [targets( 1 : 50 ); targets( 101 : 150 )];

batchSize = 4;

% build network
model = MNet();
model.AddLayer( MLinear( numInputs, 4 ) );
model.AddLayer( MSigmoid() );
model.AddLayer( MLinear( 4, 1 ) );
model.AddLayer( MSigmoid() );

% add criterion
criterion = MMSError();

% optimization options
options = optimoptions( @fminunc, 'GradObj', 'on', 'Display', 'iter', 'Tolx', 1e-20, 'TolFun', 1e-20, 'MaxIter', 200 );
w = model.GetParameters();

% randomly select order
idxs = randperm( 4 );

while true
    %for batch = 1 : batchSize : size( trainingSet, 1 )
        %range = idxs( batch : batchSize + batch - 1 );
        range = idxs;
        currentBatch = trainingSet( range, : );
        currentTargets = targets( range );

        fprintf( 'Processing batch\n' );

        % handle to cost function
        costFunc = @(w) costFunction( w, model, criterion, currentBatch, currentTargets );

        [w, grad] = fminunc( costFunc, w, options );

        %o = model.Forward( trainingSet );
        %o = o > 0.5;

        %cm = zeros( 2 );
        %cm( 1, 1 ) = sum( o( 1 : 50, : ) == 1 );
        %cm( 1, 2 ) = sum( o( 1 : 50, : ) == 0 );
        %cm( 2, 1 ) = sum( o( 51 : 100, : ) == 1 );
        %cm( 2, 2 ) = sum( o( 51 : 100, : ) == 0 );
        %pre = cm( 1, 1 ) / ( cm( 1, 1 ) + cm( 1, 2 ) );
        %recall = cm( 1, 1 ) / ( cm( 1, 2 ) + cm( 2, 1 ) );
        %fprintf( 'Precision %.20f\n', pre );
        %fprintf( 'recall %.20f\n', recall );
    %end
end
