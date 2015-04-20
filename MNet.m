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
        
        % ----------------------------------------------------------------------
        % MNet GetParameters
        %
        % Returns the unrolled weight and gradient of the network
        function result = GetParameters( obj )
            result = zeros( 0, 1 );
            for i = 1 : obj.NumLayers
                currentLayer = obj.Layers{ i };
                if strcmp( currentLayer.Type, 'Linear' )
                    result = [result; currentLayer.GetParameters()];
                end
            end
        end

        % ----------------------------------------------------------------------
        % MNet SetParameters
        %
        % Takes in a large 1D vector of all parameters of the network. The
        % parameters are rolled up and assigned to each appropriate layer.
        function SetParameters( obj, parameters )
            idx = 1;
            for i = 1 : obj.NumLayers
                currentLayer = obj.Layers{ i };
                if strcmp( currentLayer.Type, 'Linear' )
                    layerIn = currentLayer.NumIn;
                    layerOut = currentLayer.NumOut;
                    layerSize = layerIn * layerOut + layerOut;
                    layerParameters = parameters( idx : (idx - 1) + layerSize );
                    currentLayer.SetParameters( layerParameters );
                    idx = layerSize + 1;
                end
            end
        end

        % ----------------------------------------------------------------------
        % MNet GetGradients
        %
        % Unrolls the gradients of the networks and returns them.
        % TODO: In the future, Linear layers may not be the only ones with
        % gradients. Be sure to account for new types of layers.
        function result = GetGradients( obj )
            result = zeros( 0, 1 );
            for i = 1 : obj.NumLayers
                currentLayer = obj.Layers{ i };
                if strcmp( currentLayer.Type, 'Linear' )
                    result = [result; currentLayer.GetGradients()];
                end
            end
        end

        % ----------------------------------------------------------------------
        % MNet ZeroGradients
        %
        % Resets all gradients to 0.
        function ZeroGradients( obj )
            for i = 1 : obj.NumLayers
                currentLayer = obj.Layers{ i };
                if strcmp( currentLayer.Type, 'Linear' )
                    currentLayer.ZeroGradients();
                end
            end
        end

    end
end
