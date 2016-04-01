function result = buildTestSet( windowSize )
% result = buildTestSet( windowSize )
%
% Loads training images from the given data path and constructs a training set
% to be used for SRC.
%
% windowSize - The desired downsampling size.

    numClasses = 1;
    result = zeros( 0, windowSize( 1 ) * windowSize( 2 ) );
    
    for i = 1 : numClasses
        %dataPath = sprintf( '~/Documents/Data Sets/FRGC_Partial/class_%d/test/', i );
        dataPath = sprintf( '~/Documents/School/Graduate/CSE6363/Project1/images/Test/');
        dataType = '*.pgm';
        directory = dir( [dataPath dataType] );
        dataCount = length( directory );
        for idx = 1:dataCount
            % FRGC data set has 100x100 images, so we will be static
            face = double( imread( [dataPath directory(idx).name] ) );
            face = imresize( face, windowSize, 'bilinear' );
            face = face(:);

            % Normalize each column to have unit l^2-norm
            face = face / norm( face );

            result( end + 1, : ) = face';
        end
    end
end
