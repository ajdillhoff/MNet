classdef MTanh < MLayer
    methods
        % ----------------------------------------------------------------------
        % MTanh ComputeOutput
        %
        % Computes the hyperbolic tangent function. Optimized version used here.
        function result = ComputeOutput( obj, inputs )
            result = 1.7159 * tanh( (2/3) .* inputs );
            obj.Output = result;
        end

        % ----------------------------------------------------------------------
        % MTanh UpdateGradInput
        %
        % Computes the derivative of the hyperbolic tangent function.
        function result = UpdateGradInput( obj, inputs, gradOut )
            result = 1.7159 * ( 2 / 3 ) * ( 1 - 1 / (1.7159)^2 * obj.Output.^2 );
            result = result .* gradOut;
            obj.GradInput = result;
        end

    end
end
