classdef MRecurrent < MLayer
    properties
        NumIn
        NumOut
        Rho
        StartModule
        TransferModule
        FeedbackModule
        InitialModule
        RecurrentModule
        Step
        Outputs
    end

    methods
        function obj = MRecurrent( numIn, numOut, rho )
            obj.Type = 'Recurrent';
            obj.NumIn = numIn;
            obj.NumOut = numOut;
            obj.Rho = rho;
            obj.Step = 1;
            obj.StartModule = MLinear( numIn, numOut );
            obj.TransferModule = MSigmoid();
            obj.FeedbackModule = MLinear( numOut, numOut ); 
            obj.Outputs = {};

            % Build the intial module for t = 1
            obj.BuildInitialModule();
        end

        function BuildInitialModule( obj )
            obj.InitialModule = MNet();
            obj.InitialModule.AddLayer( obj.StartModule );
            obj.InitialModule.AddLayer( obj.TransferModule );
        end

        function result = ComputeOutput( obj, inputs )
            for i = 1 : size( inputs, 1 );
                result = [];
                x = inputs(i, :);
                if i == 1
                    result = obj.InitialModule.ComputeOutput( x );
                else
                    % recurrent calculations - input, hidden
                    x_output = obj.StartModule.ComputeOutput( x );
                    h_output = obj.FeedbackModule.ComputeOutput( obj.Outputs{i - 1} );
                    sum_output = x_output + h_output;
                    result = obj.TransferModule.ComputeOutput( sum_output );
                end

                obj.Outputs{i} = result;
            end
            
            obj.Output = result;
        end
    end
end
