function [f, dfdx] = costFunction( parameters, model, criterion, currentBatch, currentTargets, cm )

    model.ZeroGradients();

    % Set the parameters of the network
    model.SetParameters( parameters );

    % Compute error
    o = model.Forward( currentBatch );
    f = criterion.ComputeOutput( o, currentTargets );

    % add to confusion matrix
    [m, k] = max( o, [], 2 );
    [i, j] = find( currentTargets == 1 );
    cm.Add( j, k );

    % Backward propagation to get df/dparameters
    fx = criterion.UpdateGradInput( o, currentTargets );
    model.Backward( currentBatch, fx );

    dfdx = model.GetGradients();
end
