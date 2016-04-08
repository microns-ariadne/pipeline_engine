% CC_black_resiliant

function [outmask_alt, border_alt] = IdentifyLargeBorder(em_cc, border_width, close_width, eccentricity)
% assuming em_cc is colore corrected 
% returning mask for the boundary area and edge 

% long almost linear boundaries will be treated as walls. Other objects wil
% be treated as one image noise. 

if ~exist('border_width','var')
	border_width = 5;
end

if ~exist('close_width','var')
	close_width = 13;
end

if ~exist('eccentricity','var')
	eccentricity = 2; % long almost linear boundaries will be treated as walls  
end

if ischar(em_cc) 
    em_cc=imread(em_cc);
end

% em = imread('~/K11_S1_block_id_0000_0002_0002.png');
% %em = imread('~/K11_S1_block_id_0000_0003_0001.png');
% 
% mask=imdilate(imdilate(em,ones(3,3))==0,ones(3,3));
% low = prctile(em(~mask),.9);
% med = median(em(~mask));
% hig = prctile(em(~mask),95);
% em_standard = em;
% em_standard(mask) = med;
% em_cc = imadjust(em_standard,double([low hig])/255);
% em_cc(mask) = em(mask);


em_cc_close = imclose(em_cc,ones(close_width,close_width));
mask = imdilate(em_cc_close,ones(101,101))==0;
outmask_alt =uint8(imfill(em_cc_close==0,find(mask)));
border_alt = uint8(imdilate(outmask_alt,ones(border_width,border_width)) ~= imerode(outmask_alt,ones(border_width,border_width)));
%outmask = em_cc_close == 0;
%border = imdilate(outmask,ones(border_width,border_width)) ~= imerode(outmask,ones(border_width,border_width));

%J = em_cc;
%J(border_alt) = 255;

%figure; im(J)

cc = bwconncomp(border_alt,8);

for iobj=1:cc.NumObjects
    % slow -- can be made fast with a static matrix of indices (x,y) that
    % is referenced here with the pixel indices 
    [y,x]=ind2sub(size(border_alt),cc.PixelIdxList{iobj});
    %figure; plot(x,y);
    eg=eig(cov([x y]));
    if eg(2)/eg(1) > eccentricity
        border_alt(cc.PixelIdxList{iobj}) = 2;
        
        outmask_boundPixels = outmask_alt(cc.PixelIdxList{iobj});
        common_pnt_ind = find(outmask_boundPixels,1);
        [ys,xs] = ind2sub(size(border_alt),cc.PixelIdxList{iobj}(common_pnt_ind));
        outmask_alt(bwselect(outmask_alt,xs,ys)) = 2;
    end
end

if nargout == 0
    figure; 
    subplot(221); im(em_cc);
    subplot(222); im(outmask_alt); caxis([0 2])
    subplot(223); im(border_alt);  caxis([0 2])
end

