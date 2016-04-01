classdef MCEError < MLayer
    properties
    end

    methods
        function obj = MCEError()
            obj.Type = 'MCEError';
        end

        % ----------------------------------------------------------------------
        % MCEError.ComputeOutput
        %
        % Computes the Cross Entropy Error given the output of the network and
        % the target (true values).
        function result = ComputeOutput( obj, output, target )
            result = -sum( sum( target .* log( output ) ) ) / size( output, 1 );
        end
    
        % ----------------------------------------------------------------------
        % MCEError.UpdateGradInput
        %
        % Computes the Cross Entropy Error derivative and returns the result.
        function result = UpdateGradInput( obj, output, target )
            result = target - output;
        end

    end
end
