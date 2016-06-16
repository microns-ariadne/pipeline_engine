function mergeH5_1core(npPath,subdirectory,ylen,xlen,depth,video,emPath)
%%% Glues segmentation folders into one h5 and generating
%%% a video
% label 0 is dedicated to the wall/error segment
% Yaron

% subdirectory (default 'block_*') subdirectory pattern 
% video (default 0), video generation flag. 
%
% There are no log files indicating block dimensions. Hence it is either passed to the function or defaults are used 
% Since no log files exist, the indexing is done based on the filenames. 

ylen = str2num(ylen);
xlen = str2num(xlen);
depth = str2num(depth);
video = str2num(video);

if ~exist('npPath','var') || isempty(npPath)
	npPath = '/mnt/disk1/armafire/datasets/P3/blocks_new/tile_1_1_2048x2048x100_np';
end

if ~exist('subdirectory','var') || isempty(subdirectory)
	subdirectory = 'block_*';
end
if ~exist('depth','var') || isempty(depth)
	depth = 100; % number of slices in a segmetnation
end
if ~exist('ylen','var') || isempty (ylen)
	ylen = 2048; 
end
if ~exist('xlen','var') || isempty (xlen)
	xlen = 2048;
end
if ~exist('video','var') 
	video = 0;
end


blockFolders = dir(fullfile(npPath,subdirectory))
blockFolders = blockFolders([blockFolders.isdir])
NBlocks = length(blockFolders)

chunk = []; % support pool of workers

% going over np blocks (each has sub-blocks)
for blocki=1:NBlocks

    blocki
    
    folderPath = fullfile(npPath,blockFolders(blocki).name);
    
    subblockFolders = dir(fullfile(folderPath,'block_*'));
    subblockFolders = subblockFolders([subblockFolders.isdir]);
    
    segmentation = zeros(ylen,xlen,depth,'uint32');
    
    
    max_label = 0;
    for subblocki=1:length(subblockFolders)
        subblockPath = fullfile(folderPath,subblockFolders(subblocki).name);
        
        segfile = dir(fullfile(subblockPath, '*segmentation.h5'));
        [voli,chunk] = readvolume(fullfile(subblockPath,segfile.name));
        [~,~,vol_unq] = unique(voli); % here min val in vol_unq is 1 max is number of unique elements
        vol_unq = reshape(vol_unq,size(voli)) + max_label;
        max_label = max(vol_unq(:));
        
        %%% NOTE: counting on the file format to have only three numbers
        %%% with exactly four digits -- then the last two numbers of the
        %%% three indicate the range
        
	range_T=str2double(regexp(subblockFolders(subblocki).name,['\d\d\d\d'],'match'));
        range = range_T([5 6])+1;
    
    range
	size(vol_unq)
	size(segmentation)

	szSegs = size(segmentation); 
	szVol  = size(vol_unq);
	szSegs(1:2), szVol(1:2) 

	if ~isequal(szSegs(1:2), szVol(1:2))
		'change size'
		segmentation = zeros(szVol(1),szVol(2),depth,'uint32');
		
	end	
	segmentation(:,:,range(1):range(2)) = vol_unq;
        
        
    end

    
    h5Segmentation = fullfile(folderPath,strcat(regexprep(subdirectory, '*', '_'), 'segmentation_union_subblocks.h5'));
    if exist(h5Segmentation,'file'), delete(h5Segmentation); end
    
    Df = 9; % maximal compression (SLOW)
    
    h5create(h5Segmentation,'/stack', [depth szVol(1) szVol(2)],'ChunkSize',chunk,'Deflate',Df);
    h5write(h5Segmentation, '/stack', permute(segmentation,[3 1 2]));
    
    if video
	emFolder = fullfile(emPath,blockFolders(blocki).name(1:end-3)); %%% coutning on specific defininitions of the filename 	
    	generateVideo(h5Segmentation, emFolder);
    end
end

function generateVideo(dataset1,dataset2)

data = readvolume(dataset1);
em_vol = readvolume(dataset2);

[path,dataset1]=fileparts(dataset1);

disp(dataset2)
gtSz = size(em_vol);
sz = size(data);

ddim=gtSz-sz;
Iy = 1+ddim(1)/2:gtSz(1)-ddim(1)/2; Ix = 1+ddim(2)/2:gtSz(2)-ddim(2)/2; Iz = 1+ddim(3)/2:gtSz(3)-ddim(3)/2;
em_vol=em_vol(:,:,Iz);
% crop gt to agree with dim of prob

c=unique(data);
l=ceil(length(c)/sz(1))*sz(1);
c=reshape([c' zeros(1,l-length(c))],sz(1),[]);
c=repmat(c,[1 1 sz(3)]);
data_aug = cat(2,data,c);

datasetF = fullfile(path,strcat('VidSumm_', dataset1, '.avi'));
if exist(datasetF,'file'), delete(datasetF); end
%%% slow frame by frame rendering
figure('color','w','position',[492    46   847   759]);
v = VideoWriter(datasetF); % any quality and format can be defined here
v.Quality=100; v.FrameRate=4;  open(v);
K = [1:size(data,3) size(data,3)-1:-1:2];
for k=K
    em=em_vol(:,:,k);
    slice = data_aug(:,:,k);
    border=imdilate(slice(:,1:size(data,2)),ones(3,3))~=imerode(slice(:,1:size(data,2)),ones(3,3));
    Li=label2rgb(slice,'hsv','k','shuffle');
    %Li(repmat(border,[1 1 3])) = 0;
    
    
    imagesc(em(Iy,Ix)); colormap(gray), axis image, axis off, hold on;
    h=imagesc(Li(:,1:size(data,2),:)); axis image; h.AlphaData=.20;
    [yb,xb]=ind2sub(size(border),find(border));
    plot(xb,yb,'.k','markersize',1);
    text(50,100,['segmentation slice #(1-N): ',num2str(k)],'color','w','fontsize',20);
    set(gca,'nextplot','replacechildren');
    frame = getframe;
    writeVideo(v,frame);
end
close(v);
