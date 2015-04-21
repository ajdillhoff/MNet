classdef MLinear < MLayer
    properties
        NumIn
        NumOut
        Weight
        Bias
        GradWeight
        GradBias
    end

    methods
        function obj = MLinear( numIn, numOut )
            obj.Type = 'Linear';
            obj.NumIn = numIn;
            obj.NumOut = numOut;
            obj.GradWeight = zeros( numOut, numIn );
            obj.GradBias = zeros( numOut, 1 );

            obj = obj.Init();
        end

        % ----------------------------------------------------------------------
        % MLinear.Init
        %
        % Initializes weights randomly.
        function obj = Init( obj )
            obj.Weight = rand( obj.NumOut, obj.NumIn ) - 0.5;
            obj.Bias = rand( obj.NumOut, 1 ) - 0.5;
        end

        % ----------------------------------------------------------------------
        % MLinear.ComputeOutput
        %
        % Computes the output of a linear layer by multiplying the inputs by the
        % weights and adding the bias terms.
        function result = ComputeOutput( obj, inputs )
            ComputeOutput@MLayer( obj );
            dims = min( size( inputs ) );

            if dims == 1
                result = obj.Bias;
                result = result + obj.Weight * inputs;
            elseif dims > 1
                result = inputs * obj.Weight';

                % Add bias term
                biasBuf = repmat( obj.Bias', size( result, 1 ), 1 );
                result = result + biasBuf;
            elseif ndims( inputs ) > 2
                fprintf( 'ERROR: MLinear.ComputeOutput: Illegal number of ' ...
                    + 'dimensions.' );
            end

            obj.Output = result;
        end

        % ----------------------------------------------------------------------
        % MLinear.UpdateGradInput
        %
        % Performs backpropagation with respect to this layer. Returns the 
        % calculated gradient.
        function result = UpdateGradInput( obj, inputs, gradOut )
            obj.GradInput = gradOut * obj.Weight;
            result = obj.GradInput;
        end

        % ----------------------------------------------------------------------
        % MLinear.UpdateParameters
        %
        % Updates the current layer's Weight and Bias parameters.
        function UpdateParameters( obj, inputs, gradOut )
            % reset gradients
            obj.GradWeight = 0;
            obj.GradBias = 0;

            deltaG = gradOut' * inputs;
            deltaB = gradOut' * ones( size( inputs, 1 ), 1 );

            % apply regularization
            lambda = 0;
            m = 4;
            deltaG = deltaG / m;
            deltaB = deltaB / m;
            
            deltaG = deltaG + ( lambda / m ) * obj.Weight;
            %deltaB = deltaB + ( lambda / m ) * obj.Bias;

            obj.GradWeight = deltaG;
            obj.GradBias = deltaB;
        end
        
        % ----------------------------------------------------------------------
        % MLinear.GetParameters
        %
        % Returns the layers current parameters
        function result = GetParameters( obj )
            result = [obj.Bias(:); obj.Weight(:)];
        end

        % ----------------------------------------------------------------------
        % MLinear.SetParameters
        %
        % Sets the layer's bias and weight
        function SetParameters( obj, parameters )
            bias = reshape( parameters( 1 : obj.NumOut ), size( obj.Bias ) );
            obj.Bias = bias;

            weight = reshape( parameters( obj.NumOut + 1 : end ), size( obj.Weight ) );
            obj.Weight = weight;
        end

        % ----------------------------------------------------------------------
        % MLinear.GetGradients
        %
        % Returns the unrolled bias and weight gradients
        function result = GetGradients( obj )
            result = [obj.GradBias(:); obj.GradWeight(:)];
        end
        
        % ----------------------------------------------------------------------
        % MLinear.ZeroGradients
        %
        % Resets gradWeight and gradBias to 0
        function ZeroGradients( obj )
            obj.GradWeight = zeros( obj.NumOut, obj.NumIn );
            obj.GradBias = zeros( obj.NumOut, 1 );
        end
        
    end
end
