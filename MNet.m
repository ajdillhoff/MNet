classdef MNet < MLayer
    properties
        NumLayers
        Layers
        Weights
    end

    methods
        function obj = MNet()
            obj.NumLayers = 0;
            obj.Layers = cell( 0, 1 );
            obj.Weights = cell( 0, 1 );
        end

        % ----------------------------------------------------------------------
        % MNet ComputeOutput
        %
        % Begins the feed forward process of the neural network.
        function result = ComputeOutput( obj, inputs )
            result = inputs;
            for l = 1 : obj.NumLayers
                result = obj.Layers{ l }.ComputeOutput( result );
            end

            obj.Output = result;
        end
        
        % ----------------------------------------------------------------------
        % MNet UpdateGradInput
        %
        % Updates the gradient input across the entire network.
        function result = UpdateGradInput( obj, inputs, gradOut )
            result = gradOut;
            currentLayer = obj.Layers{ end };
            for l = obj.NumLayers - 1 : -1 : 1
                previousLayer = obj.Layers{ l };
                result = currentLayer.UpdateGradInput( previousLayer.Output, result );
                currentLayer = previousLayer;
            end

            result = currentLayer.UpdateGradInput( inputs, result );
            obj.GradInput = result;
        end

        % ----------------------------------------------------------------------
        % MNet UpdateParameters
        %
        % Updates the weights of the network.
        function result = UpdateParameters( obj, inputs, gradOut )
            result = gradOut;
            currentLayer = obj.Layers{ end };
            for l = obj.NumLayers - 1 : -1 : 1
                previousLayer = obj.Layers{ l };
                currentLayer.UpdateParameters( previousLayer.Output, result );
                result = currentLayer.GradInput;
                currentLayer = previousLayer;
            end

            currentLayer.UpdateParameters( inputs, result );
        end

        % ----------------------------------------------------------------------
        % MNet AddLayer
        %
        % Adds a new layer to the model. If the layer is linear, a new weight
        % matrix will need to be added as well.
        function AddLayer( obj, layer )
            obj.Layers{ end + 1 } = layer;

            if strcmp( layer.Type, 'Linear' )
                w = zeros( layer.NumIn, layer.NumOut );
                obj.Weights{ end + 1 } = w;
            end

            obj.NumLayers = obj.NumLayers + 1;
        end
    end
end
