classdef MConfusionMatrix < handle
    properties
        NumClasses
        Matrix
    end

    methods
        function obj = MConfusionMatrix( numClasses )
            obj.NumClasses = numClasses;
            obj.Matrix = zeros( numClasses );
        end

        % ----------------------------------------------------------------------
        % MConfusionMatrix Reset
        %
        % Resets the matrix to all zeros. Uses the number of classes to 
        % re-initialize the matrix.
        function Reset( obj )
            obj.Matrix = zeros( obj.NumClasses );
        end

        % ----------------------------------------------------------------------
        % MConfusionMatrix Print
        %
        % Displays the current matrix.
        function Print( obj )
            disp( obj.Matrix );
            acc = sum( diag( obj.Matrix ) ) / sum( sum( obj.Matrix ) );
            fprintf( 'Accuracy: %f\n', acc );
        end

        % ----------------------------------------------------------------------
        % MConfusionMatrix Add
        %
        % Increments the value or values at the given indices. The indicies can
        % be ranges if there are multiple values being updated simultaneously.
        function Add( obj, rowIdx, colIdx )
            %obj.Matrix( rowIdx, colIdx ) = obj.Matrix( rowIdx, colIdx ) + 1;
            for i = 1 : size( rowIdx )
                obj.Matrix( rowIdx( i ), colIdx( i ) ) = ...
                    obj.Matrix( rowIdx( i ), colIdx( i ) ) + 1;
            end
        end
    end
end
