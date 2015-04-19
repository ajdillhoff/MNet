classdef MLayer < handle
    properties
        Type
        GradInput
        Output
    end

    methods
        function obj = MLayer()
        end

        function result = Forward( obj, inputs )
            result = obj.ComputeOutput( inputs );
        end

        function result = Backward( obj, inputs, gradOut )
            obj.UpdateGradInput( inputs, gradOut );
            obj.UpdateParameters( inputs, gradOut );
            result = obj.GradInput;
        end

        function result = ComputeOutput( obj, inputs )
            result = obj.Output;
        end

        function result = UpdateGradInput( obj, inputs, gradOut )
            result = obj.GradInput;
        end

        function result = UpdateParameters( obj, inputs, gradOut )
        end
    end
end
