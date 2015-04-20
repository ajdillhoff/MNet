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
            result = -target .* log( output ) + ( 1 - target ) .* log( 1 - output );
            result = sum( result ) / size( output, 1 );
        end
    
        % ----------------------------------------------------------------------
        % MCEError.UpdateGradInput
        %
        % Computes the Cross Entropy Error derivative and returns the result.
        function result = UpdateGradInput( obj, output, target )
            result = output - target;
        end

    end
end
