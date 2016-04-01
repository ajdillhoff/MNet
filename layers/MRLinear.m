classdef MRLinear < MLayer
    properties
        BPTTSteps
        NumIn
        NumOut
        W
        U
        HiddenStates
        Bias
        GradW
        GradU
        GradBias
    end

    methods
        function obj = MRLinear( numIn, numOut, bpttSteps )
            obj.Type = 'Linear';
            obj.NumIn = numIn;
            obj.NumOut = numOut;
            obj.GradU = zeros( numOut );
            obj.GradW = zeros( numOut, numIn );
            obj.GradBias = zeros( numOut, 1 );
            obj.BPTTSteps = bpttSteps;

            obj = obj.Init();
        end

        % ----------------------------------------------------------------------
        % MRLinear.Init
        %
        % Initializes weights randomly.
        % TODO: From -1/sqrt(n) to 1/sqrt(n)
        function obj = Init( obj )
            rng( 'shuffle' );
            obj.U = 2 * rand( obj.NumOut ) / sqrt( obj.NumOut ) - ...
                1 / sqrt( obj.NumOut );
            obj.W = 2 * rand( obj.NumOut, obj.NumIn ) / sqrt( obj.NumIn ) - ...
                1 / sqrt( obj.NumIn );
            obj.Bias = 2 * rand( obj.NumOut, 1 ) - 1;
        end

        % ----------------------------------------------------------------------
        % MRLinear.ComputeOutput
        %
        % Computes the output of a linear layer by multiplying the input X by 
        % the weights and adding the bias terms.
        % TODO: tanh function is used here, but this goes against
        % the modularity of MNet.
        function result = ComputeOutput( obj, X )
            ComputeOutput@MLayer( obj );

            T = size( X, 1 );

            hiddenStates = zeros( T + 1, obj.NumOut );

            for t = 1 : T
                hiddenStates(t + 1, :) = tanh( X(t, :) * obj.W' + ...
                    hiddenStates(t, :) * obj.U' );
            end

            % TODO: Add bias term
            result = hiddenStates;

            obj.Output = result;
        end

        % ----------------------------------------------------------------------
        % MRLinear.UpdateGradInput
        %
        % Performs backpropagation with respect to this layer. Returns the 
        % calculated gradient.
        function [dLdW, dLdU, dLdV] = UpdateGradInput( obj, inputs, gradOut )
            T = size( gradOut, 1 );
            
            for t = T : -1 : 1
                dLdV = dLdV + gradOut(t, :) * obj.HiddenStates(t, :)';

                dT = obj.V * gradOut(t, :) .* (1 - obj.HiddenStates(t, :).^2);

                % BPTT
                endTime = max( 1, t - obj.BPTTSteps + 1 );
                
                for idx = t + 1 : -1 : endTime
                    dLdW = dLdW + dT * obj.HiddenStates(idx - 1, :);
                    dLdU(:, inputs(idx, :)) = dLdU(:, inputs(idx, :)) + dT;

                    dT = obj.W * dT .* (1 - obj.HiddenStates(idx - 1, :).^2);
                end
            end
        end

        % ----------------------------------------------------------------------
        % MRLinear.UpdateParameters
        %
        % Updates the current layer's Weight and Bias parameters.
        function UpdateParameters( obj, inputs, gradOut )
            % reset gradients
            obj.GradW = 0;
            obj.GradBias = 0;

            deltaG = gradOut' * inputs;
            deltaB = gradOut' * ones( size( inputs, 1 ), 1 );

            % apply regularization
            lambda = 0;
            m = 4;
            deltaG = deltaG / m;
            deltaB = deltaB / m;
            
            deltaG = deltaG + ( lambda / m ) * obj.W;
            %deltaB = deltaB + ( lambda / m ) * obj.Bias;

            obj.GradW = deltaG;
            obj.GradBias = deltaB;
        end
        
        % ----------------------------------------------------------------------
        % MRLinear.GetParameters
        %
        % Returns the layers current parameters
        % TODO: Update for additional parameters
        function result = GetParameters( obj )
            result = [obj.Bias(:); obj.W(:)];
        end

        % ----------------------------------------------------------------------
        % MRLinear.SetParameters
        %
        % Sets the layer's bias and weight
        % TODO: Update for additional parameters
        function SetParameters( obj, parameters )
            bias = reshape( parameters( 1 : obj.NumOut ), size( obj.Bias ) );
            obj.Bias = bias;

            weight = reshape( parameters( obj.NumOut + 1 : end ), size( obj.Weight ) );
            obj.Weight = weight;
        end

        % ----------------------------------------------------------------------
        % MRLinear.GetGradients
        %
        % Returns the unrolled bias and weight gradients
        % TODO: Update for additional parameters
        function result = GetGradients( obj )
            result = [obj.GradBias(:); obj.GradWeight(:)];
        end
        
        % ----------------------------------------------------------------------
        % MRLinear.ZeroGradients
        %
        % Resets gradients to 0
        % TODO: Update for additional parameters
        function ZeroGradients( obj )
            obj.GradW = zeros( obj.NumOut, obj.NumIn );
            obj.GradBias = zeros( obj.NumOut, 1 );
        end
        
    end
end
