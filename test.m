%% Testing Program with OR problem
%data = [0 0; 0 1; 1 0; 1 1];
%targets = [0; 1; 1; 0];
%numInputs = size( data, 2 );

% load FRGC data set
data = buildTrainingSet( [10 10] );
targets = zeros( 2000, 1 );
targets( 1 : 100 ) = 1;
numInputs = size( data, 2 );
%data = data( 1 : 200, : );
%targets = targets( 1 : 200 );

% build network
model = MNet();
model.AddLayer( MLinear( numInputs, 2 ) );
model.AddLayer( MSigmoid() );
model.AddLayer( MLinear( 2, 1 ) );

% add criterion
criterion = MMSError();


for epoch = 1 : 100000
    J = 0;
    for batch = 1 : 10 : size( data, 1 )
        currentBatch = data( batch : 10 + batch - 1, : );
        currentTargets = targets( batch : 10 + batch - 1 );
        % Forward through network
        o = model.Forward( currentBatch );

        %disp( o );

        % Compute error
        J = J + criterion.ComputeOutput( o, currentTargets );

        % Backward!
        fx = criterion.UpdateGradInput( o, currentTargets );

        model.Backward( currentBatch, fx );
    end
    J = J / size( data, 1 );
    fprintf( 'Cost: %.20f\n', J );
end
