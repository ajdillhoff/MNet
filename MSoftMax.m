classdef MSoftMax < MLayer
    methods

        function obj = MSoftMax()
            obj.Type = 'SoftMax';
        end

        % ----------------------------------------------------------------------
        % MSoftMax ComputeOutput
        %
        % Computes the softmax function for multiclass problems.
        function result = ComputeOutput( obj, inputs )
            result = exp( bsxfun( @minus, inputs, max( inputs, [], 2 ) ) );
            result = bsxfun( @rdivide, result, sum( result, 2 ) );
            obj.Output = result;
        end

        % ----------------------------------------------------------------------
        % MSoftMax UpdateGradInput
        %
        % Computes the derivative of the softmax function
        function result = UpdateGradInput( obj, inputs, gradOut )
            obj.GradInput = -gradOut;
            result = obj.GradInput;
        end
    end
end
