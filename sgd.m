function [w, f] = sgd( costFunc, w )
    momentum = 0.9;
    lr = 0.3;

    % Evaluate the network output
    [f, dfdx] = costFunc( w );

    % Apply momentum
    if momentum ~= 0
        dfdx = dfdx * momentum;
    end

    % Apply learning rate
    dfdx = dfdx * -lr;

    % Update weight
    w = w + dfdx;
end