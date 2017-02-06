function [vi, vi_split, vi_merge] = VI3D(seg_vol,gt_vol, range_gt)
% Segmentation and gt are paths to 2D images folder, to stacked tiff or h5
% This function will calculate VI by pushing the two volume and their
% label-product (Cartesian product) into RAM and then calculate entropy of
% the labeling distribution whose difference known as VI/VOI. (see Miela)
% Function assumes that in any case label class is \le uint32
% Examples:
%   gt = 'yv_connectome/datasets/isbi/train-labels.tif'
%   segmentation = 'yv_connectome/pipeline/datasets/isbi_orig/segmentation.h5'
%
%
% 'range_gt' is an optional parameter to slect a subect of images (slices/pages)
% to select from gt

if ~exist('range_gt','var') || isempty(range_gt) 
    range_gt = 0;
end

if ischar(gt_vol)  && exist(gt_vol,'file'), gt_vol = readvolume(gt_vol); end
if ischar(seg_vol) && exist(seg_vol,'file'), seg_vol = readvolume(seg_vol); end
 
if numel(range_gt)==2, range_gt=range_gt(1):range_gt(2);
else range_gt=1:size(gt_vol,3); end

gt_vol = gt_vol(:,:,range_gt);

% 
 gtSz=size(gt_vol);
 sgSz=size(seg_vol);
if ~isequal(gtSz,sgSz)
   
    ddim=gtSz-size(seg_vol);
    if all(ddim>=0)
        Iy = 1+ddim(1)/2:gtSz(1)-ddim(1)/2; Ix = 1+ddim(2)/2:gtSz(2)-ddim(2)/2;
        gt_vol = gt_vol(Iy,Ix,:);
        gtSz = size(gt_vol);
    elseif all(ddim<=0)
        Iy = 1-ddim(1)/2:sgSz(1)+ddim(1)/2; Ix = 1-ddim(2)/2:sgSz(2)+ddim(2)/2;
        seg_vol = seg_vol(Iy,Ix,:);
        sgSz=size(seg_vol); %#ok<NASGU>
    else
        error('Cannot figure out how to crop the two datasets to match');
    end
     
end


% get the ditribution of labels
tic, D_gt=histcounts(gt_vol,[unique(gt_vol); intmax('uint32')]); toc
tic, D_seg=histcounts(uint32(seg_vol),[unique(seg_vol); intmax('uint32')]); toc
tic, D_mutual=histcounts(uint64(uint64(seg_vol)*2^32+uint64(gt_vol)),...
    [unique(uint64(seg_vol)*2^32+uint64(gt_vol)); intmax('uint64')]); toc


H_mutual = -sum(log(D_mutual/prod(gtSz)).*D_mutual/prod(gtSz));
H_gt = -sum(log(D_gt/prod(gtSz)).*D_gt/prod(gtSz));
H_seg = -sum(log(D_seg/prod(gtSz)).*D_seg/prod(gtSz));

vi_split = H_mutual - H_gt;
vi_merge = H_mutual - H_seg;

vi = vi_split + vi_merge;
