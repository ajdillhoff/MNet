% input
X = [0 1 0; 0 0 1; 1 0 0];
numInputs = size( X, 2 );
numHidden = 4;
rho = 3;

% Define model
model = MNet();
model.AddLayer( MRecurrent( numInputs, numHidden, rho ) );

% Output test
o = model.Forward( X )
