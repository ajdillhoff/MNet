function [f, dfdx] = costFunction( parameters, model, criterion, currentBatch, currentTargets )

    model.ZeroGradients();

    % Set the parameters of the network
    model.SetParameters( parameters );

    % Compute error
    o = model.Forward( currentBatch );
    f = criterion.ComputeOutput( o, currentTargets );

    % Backward propagation to get df/dparameters
    fx = criterion.UpdateGradInput( o, currentTargets );
    model.Backward( currentBatch, fx );

    dfdx = model.GetGradients();
end
