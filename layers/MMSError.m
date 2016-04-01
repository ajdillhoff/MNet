classdef MMSError < MLayer
    properties
    end

    methods
        function obj = MMSError()
            obj.Type = 'MMSError';
        end

        % ----------------------------------------------------------------------
        % MMSError.ComputeOutput
        %
        % Computes the Mean Squared Error given the output of the network and
        % the target (true values).
        function result = ComputeOutput( obj, output, target )
            result = sum( output - target )^2;
            result = result / size( output, 1 );
        end
    
        % ----------------------------------------------------------------------
        % MMSError.UpdateGradInput
        %
        % Computes the Mean Squared Error derivative and returns the result.
        function result = UpdateGradInput( obj, output, target )
            result = ( 2 / size( output, 1 ) ) * ( output - target );
        end

    end
end
