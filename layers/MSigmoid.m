classdef MSigmoid < MLayer
    properties
    end

    methods
        function obj = MSigmoid()
            obj.Type = 'Sigmoid';
        end

        function result = ComputeOutput( obj, inputs )
            result = 1 ./ ( 1 + exp( -inputs ) );
            obj.Output = result;
        end

        function result = UpdateGradInput( obj, inputs, gradOut )
            z = obj.Output;
            result = z .* ( 1 - z );
            result = result .* gradOut;
            obj.GradInput = result;
        end
    end
end
