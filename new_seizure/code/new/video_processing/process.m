%%
clc
fprintf('##############################################################\n')
fprintf('Code from papers:\n\n - J.R.R. Uijlings, I.C. Duta, E. Sangineto, and N. Sebe\n   "Video Classification with Densely Extracted HOG/HOF/MBH Features:\n   An Evaluation of the Accuracy/Computational Efficiency Trade-off"\n   In International Journal of Multimedia Information Retrieval (IJMIR), 2015.\n - I.C. Duta, J.R.R. Uijlings, T.A. Nguyen, K. Aizawa, A.G. Hauptmann, B. Ionescu, N. Sebe\n   "Histograms of Motion Gradients for Real-time Video Classification" \n   In International Workshop on Content-based Multimedia Indexing (CBMI), 2016.\n\n');
fprintf('This code requires the Matlab vision toolbox.\n');
fprintf('##############################################################\n\n')
% pause(1);
% Setup a global variable which contains the path of video files
global DATAopts
DATAopts.videoPath = '%s';

path = '/home/taliah/Documents/Course/Project/new_seizure/';

%settings
blockSize = [8 8 1]; % block size is 8 by 8 pixels by 6 frames, but we will vary the number of frames
numBlocks = [2 2 1]; % 3 x3 spatial blocks and 2 temporal blocks
numOr = 8; % Quantization in 8 orientations
flowMethod = 'Horn-Schunck'; % the optical flow choice

files = dir([path 'video/6464_chunks/*/*.avi']);

for i = 1:size(files,1)


    % Load the video
    vidName = [files(i).folder '/' files(i).name];
    
    
    fprintf('Video ---- %s\n\n',vidName); 

    if exist('mmread', 'file')
        % Under Linux, Fedora 20, mmread was almost 5x faster. Just download mmread from:
        %       http://www.mathworks.co.uk/matlabcentral/fileexchange/8028-mmread
        % and make sure to set your path correctly
        fprintf('Using mmread to load video');
        tic;
        vid = VideoRead(vidName);
        videoReadTime = toc;
        fprintf('... took %.2f seconds\n', videoReadTime);
    else
        fprintf('Using VideoReader from Matlab to load in video.\nWarning: We found that loading videos using native Matlab code (under Fedora 20) took more \ntime than the HOG features sampled at every frame. Instead, using the external library\nmmread loading a video is 8x faster. See comments in demo.m\n');
        tic
        vid = VideoReadNative(vidName);
        videoReadTime = toc;
        fprintf('Loaded video in %.2f seconds\n', videoReadTime);
    end


    %%
    % For-loop over the sampling rate for HOG
    fprintf('\nNow extracting HOG features. Timings below include loading the video (as in our paper):\n');

    tic
    % Subsample framerate of video
    sampledVid = vid;


    % Get HOG descriptors
    [hogDesc, hogInfo] = Video2DenseHOGVolumes(sampledVid, blockSize, numBlocks, numOr);

    % Print statistics
    extractionTimeHOG = toc;
    totalDescriptorTime = extractionTimeHOG + videoReadTime;
    fprintf('HOG: frames/block: %d sample rate: %d sec/vid: %.2f frame/sec: %.2f\n', ...
        blockSize(3), 1, totalDescriptorTime, size(vid,3)/totalDescriptorTime);

    %
    % For-loop over the sampling rate for HOF
    fprintf('\nNow extracting HOF features. Timings below include loading the video (as in our paper):\n');
    tic
        % Subsample framerate of video


        % Get HOG descriptors
    [hofDesc, hofInfo] = ...
        Video2DenseHOFVolumes(sampledVid, blockSize, numBlocks, numOr, flowMethod);

        % Print statistics
    extractionTimeHOF = toc;
    totalDescriptorTime = extractionTimeHOF + videoReadTime;
    fprintf('HOF: frames/block: %d sample rate: %d sec/vid: %.2f frame/sec: %.2f\n', ...
        blockSize(3), 1, totalDescriptorTime, size(vid,3)/totalDescriptorTime);

    %
    % For-loop over the sampling rate for MBH
    fprintf('\nNow extracting MBH features. Timings below include loading the video (as in our paper):\n');
    tic
    % Subsample framerate of video

    % Get HOG descriptors
    [MBHRowDesc, MBHColDesc, mbhInfo] = ...
            Video2DenseMBHVolumes(sampledVid, blockSize, numBlocks, numOr, flowMethod);

    % Print statistics
    extractionTimeMBH = toc;
    totalDescriptorTime = extractionTimeMBH + videoReadTime;
    fprintf('MBH: frames/block: %d sample rate: %d sec/vid: %.2f frame/sec: %.2f\n', ...
        blockSize(3), 1, totalDescriptorTime, size(vid,3)/totalDescriptorTime);


    hog = reshape(hogDesc', 32*49,[])';
    hof = reshape(hofDesc', 32*49,[])';
    mbhCol = reshape(MBHColDesc', 32*49,[])';
    mbhRow = reshape(MBHRowDesc', 32*49,[])';

    dataFull = [hog hof mbhRow mbhCol];
    
    save([path 'data/6464/mats/' files(i).folder(69:end) '/' files(i).name(1:end-4) '.mat'],'dataFull')

end


fprintf('\nDone!\n');