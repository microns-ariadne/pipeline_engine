function cc_block(in_blockpath, out_blockpath)
%

files = dir(in_blockpath);
files = files(~[files.isdir]);
n_files = length(files);

ems = cell(1, n_files);

for file_id=1:n_files
    
    file_id
    
    em = imread(fullfile(in_blockpath, files(file_id).name));
    
    if sum(sum(em)) == 0
        fprintf('Found a black image [%d] => Return\n', file_id);
        return
    end
    
    ems{file_id} = em;
end

for file_id=1:n_files

    file_id
    
%     orig_im = imread(fullfile(in_blockpath, files(file_id).name));
%     %cc_im = uint8(255*mat2gray(orig_im));
%     
%     m_val = mean(mean(orig_im));
%     
%     mask = uint8((orig_im == 0) * m_val);
%     
%     orig_im = orig_im + mask;
%     
%     low = prctile(orig_im(:),.9);
% 
%     hig = prctile(orig_im(:),95);
% 
%     cc_im = imadjust(orig_im,double([low hig])/255);
%     
%     %cc_im = uint8(imadjust(cc_im));
    
    em = ems{file_id};
    
    mask=imdilate(imdilate(em,ones(3,3))==0,ones(3,3));
    
    low = prctile(em(~mask),.9);
    
    med = median(em(~mask));
    
    hig = prctile(em(~mask),95);
    
    em_standard = em;
    
    em_standard(mask) = med;
    
    P = imadjust(em_standard,double([low hig])/255);
    
    P(mask) = em(mask);
    
    imwrite(P, fullfile(out_blockpath, strcat(files(file_id).name, '_cc.png'))); 
    
end
