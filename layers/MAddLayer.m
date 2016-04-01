% TODO: Support more than 2 layers
classdef MAddLayer < MLayer
    properties
        Layers
        Output
        GradInput
    end

    methods
        % ----------------------------------------------------------------------
        % MAddLayer
        %
        % Adds the outputs of two layers together.
        function obj = MAddLayer( layers )
            obj.Layers = layers;
        end

        function result = ComputeOutput( obj, )
    end
end
