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
            obj.GradBias = zeros( 1, numOut );

            obj = obj.Init();
        end

        % ----------------------------------------------------------------------
        % MLinear.Init
        %
        % Initializes weights randomly.
        % TODO: Understand the bias term more before implementing any
        % initialization for it.
        function obj = Init( obj )
            obj.Weight = rand( obj.NumOut, obj.NumIn );
            obj.Bias = rand( 1, obj.NumOut );
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
                biasBuf = repmat( obj.Bias, size( result, 1 ), 1 );
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
            deltaG = gradOut' * inputs;
            deltaB = ones( 1, size( inputs, 1 ) ) * gradOut;

            % apply regularization
            lambda = 0.1;
            m = 200;
            deltaG = deltaG / m;
            deltaB = deltaB / m;
            
            deltaG = deltaG + ( lambda / m ) * obj.Weight;
            deltaB = deltaB + ( lambda / m ) * obj.Bias;

            % apply learning rate
            lr = 0.1;
            deltaG = deltaG * lr;
            deltaB = deltaB * lr;

            % apply momentum
            momentum = 0.9;
            deltaG = deltaG + momentum * obj.GradWeight;
            deltaB = deltaB + momentum * obj.GradBias;

            obj.GradWeight = deltaG;
            obj.GradBias = deltaB;

            % weight update with learning rate
            obj.Weight = obj.Weight - deltaG;
            obj.Bias = obj.Bias - deltaB;
        end
        
    end
end
