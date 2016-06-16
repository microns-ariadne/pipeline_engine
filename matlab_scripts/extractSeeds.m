function extractSeeds(probPath,probOutput,kernelSize,regionalDepthReduce,stdSmooth,IMSIZE_D1,IMSIZE_D2,NImages,DemoMode)
% calculate seeds from distance channels
% see the Demo mode paramteres to see potential usages
% Examples,
% extractSeeds('/Users/yaronm/Dropbox (MIT)/MIT/yv_connectome/results/AC3/AC3_fully_conv_49x49_w0_dist_4_4M_3D_32f/test_3D/probs',13,0.04,2,976,73)
% extractSeeds([],[],[],[],[],[],1)

% make these input parameters
% regionalDepthReduce = 0.04
%smoothing distance transform: stdSmooth=2, kernelSmooth=[13 13];
%
%%% should be default values
% stdSmooth=2,
% kernelSize=13;
% regionalDepthReduce=0.04;

kernelSize = str2num(kernelSize);
regionalDepthReduce = str2num(regionalDepthReduce);
stdSmooth = str2num(stdSmooth);
IMSIZE_D1 = str2num(IMSIZE_D1);
IMSIZE_D2 = str2num(IMSIZE_D2);
NImages = str2num(NImages);

if ~exist('probPath','var')
    DemoMode = 1;
end


if  exist('DemoMode','var') && DemoMode
    probPath = '/Users/yaronm/Dropbox (MIT)/MIT/yv_connectome/results/AC3/AC3_fully_conv_49x49_w0_dist_4_4M_3D_32f/test_3D/probs';
    ac3 = '/Users/yaronm/Dropbox (MIT)/MIT/yv_connectome/results/AC3/';
    prob_output = fullfile(ac3,'prob-distSeeds');
    lb='/Users/yaronm/Dropbox (MIT)/MIT/yv_connectome/results/AC3/AC3_fully_conv_49x49_w0_dist_4_4M_3D_32f/test_3D/pipeline/stacked-labels-49x49-w0-dist-4-4M-ac3-3D-only-0.tif';
    
    
    lbfiles = fullfile(fileparts(lb),'labels'); mkdir(lbfiles);
    lbv=readvolume(lb);
    NImages = 73;
    IMSIZE = 976; % 976 928
    
    regionalDepthReduce=0.04;
    kernelSize = 13;
    stdSmooth = 2;
    
else
    DemoMode = 0;
    prob_output = probOutput
    %prob_output = fullfile(probPath,'seeds');
end

if ~exist(prob_output,'file'), mkdir(prob_output); end

if DemoMode
    fig = figure; 
    fig2 = figure;
end

kernelSmooth=[kernelSize kernelSize];

Dist = zeros(IMSIZE_D1,IMSIZE_D2,NImages);
Prob = zeros(IMSIZE_D1,IMSIZE_D2,NImages);
Memb = zeros(IMSIZE_D1,IMSIZE_D2,NImages);

files_probs = dir(fullfile(probPath, '*.png'));
files_probs = files_probs(~[files_probs.isdir]);
files_probs = sort({files_probs.name});
n_files_probs = length(files_probs);

assert(n_files_probs == (NImages * 4));

for iz=1:NImages
    
    % probability method 2-3 / 2-4
    
    start_idx = (iz - 1) * 4 + 1;
    end_idx = start_idx + 4 - 1;
    
    fl = files_probs(start_idx:end_idx);
    
    for k=1:length(fl)
        fprintf(' -- [%d] %s\n', k, fl{k});
    end
    
    sz = [size(imread(fullfile(probPath,fl{1}))) length(fl)];
    fI = zeros(sz);
    for k=1:length(fl)
        fI(:,:,k)=imread(fullfile(probPath,fl{k}));
    end
    
    Dist(:,:,iz)=convn(double(fI)/255,reshape(log((size(fI,3):-1:1)),...
        [1 1 size(fI,3)]),'valid');
     
    Prob(:,:,iz)=convn(double(fI)/255,reshape(log((1:size(fI,3))),...
        [1 1 size(fI,3)]),'valid');
    
    Memb(:,:,iz)=mat2gray(fI(:,:,1));
    
    I = -imfilter(Dist(:,:,iz),fspecial('gaussian',kernelSmooth,stdSmooth));
    seed = imregionalmin(imhmin(I,regionalDepthReduce));
    
    %ws_prob = uint8(255*Memb(:,:,iz));
    ws_prob = uint8(255*Memb(:,:,iz)+1); ws_prob(seed)=0;
    
    full_out_path = fullfile(prob_output,strcat(fl{1}, '_0-4-combined-with-seeds.png'));
    
    imwrite(ws_prob,full_out_path,'png','compression','lzw');
    
    fprintf(' -- writing: %s\n\n', full_out_path);
    
end

%%% Need to combine membrane channel with combined probabilities to for
%%% better seeds for 2D watershed.

