classdef MNLLError < MLayer
    properties
    end

    methods
        function obj = MNLLError()
            obj.Type = 'MNLLError';
        end

        % ----------------------------------------------------------------------
        % MNLLError.ComputeOutput
        %
        % Computes the negative log likelihood given the output of the network and
        % the target (true values).
        function result = ComputeOutput( obj, output, target )
            result = -target .* log( output ) + ( 1 - target ) .* log( 1 - output );
            result = sum( result ) / size( output, 1 );
        end
    
        % ----------------------------------------------------------------------
        % MNLLError.UpdateGradInput
        %
        % Computes the negative log likelihood derivative and returns the result.
        function result = UpdateGradInput( obj, output, target )
            result = output - target;
        end

    end
end
