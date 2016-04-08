function ws_np_prepare(cnn_crop_leg_str, ...   
                       in_outer_mask_filename, ...
                       in_border_mask_filename, ...
                       in_blockpath_meta, ...
                       in_blockpath_probs, ...
                       out_blockpath_ws, ...
                       out_blockpath_np)
%

cnn_crop_leg = str2num(cnn_crop_leg_str);
fprintf('cnn_crop_leg: %d\n', cnn_crop_leg);
fprintf('in_outer_mask_filename: %s\n', in_outer_mask_filename);
fprintf('in_border_mask_filename: %s\n', in_border_mask_filename);
fprintf('in_blockpath_meta: %s\n', in_blockpath_meta);
fprintf('in_blockpath_probs: %s\n', in_blockpath_probs);
fprintf('out_blockpath_ws: %s\n', out_blockpath_ws);
fprintf('out_blockpath_np: %s\n', out_blockpath_np);

files_probs = dir(fullfile(in_blockpath_probs, '*.png'));
files_probs = files_probs(~[files_probs.isdir]);
files_probs = sort({files_probs.name});
n_files_probs = length(files_probs);

outer_mask_all = imread(in_outer_mask_filename);
border_mask_all = imread(in_border_mask_filename);

dims = size(outer_mask_all);
dims_all = size(border_mask_all);

assert(isequal(dims, dims_all));

outer_mask_all = outer_mask_all(1+cnn_crop_leg : dims(1)-cnn_crop_leg , 1+cnn_crop_leg : dims(2)-cnn_crop_leg);
border_mask_all = border_mask_all(1+cnn_crop_leg : dims(1)-cnn_crop_leg , 1+cnn_crop_leg : dims(2)-cnn_crop_leg);

images_probs = cell(1, n_files_probs);
outer_masks = cell(1, n_files_probs);
border_masks = cell(1, n_files_probs);

for file_id = 1 : n_files_probs
    
    fprintf('Read file_id = %d\n', file_id);
    
    probs_filename = files_probs{file_id};
    im_probs = imread(fullfile(in_blockpath_probs, probs_filename));
    images_probs{file_id} = im_probs;
    
    outer_mask_filename = sprintf('outer_mask_%.4d.png', file_id);
    border_mask_filename = sprintf('border_mask_%.4d.png', file_id);
    
    outer_mask = imread(fullfile(in_blockpath_meta, outer_mask_filename));
    border_mask = imread(fullfile(in_blockpath_meta, border_mask_filename));
    
    dims = size(outer_mask);
    dims2 = size(border_mask);
    assert(isequal(dims, dims2));
    assert(isequal(dims, dims_all));
    
    outer_masks{file_id} = outer_mask(1+cnn_crop_leg : dims(1)-cnn_crop_leg , 1+cnn_crop_leg : dims(2)-cnn_crop_leg);
    border_masks{file_id} = border_mask(1+cnn_crop_leg : dims(1)-cnn_crop_leg , 1+cnn_crop_leg : dims(2)-cnn_crop_leg);
    
end

for file_id=1:n_files_probs
    
    fprintf('Process file_id = %d\n', file_id);
    
    im_probs = images_probs{file_id};
    
    outer_mask = outer_masks{file_id};
    %border_mask = border_masks{file_id};
    
    cur_outer_mask = outer_mask_all | outer_mask;
    %cur_border_mask = border_mask_all | border_mask;
    
    im_probs_new = im_probs .* uint8(~cur_outer_mask);
    
    im_probs_new(im_probs_new == 255) = 254;
    
    im_probs_new = im_probs_new + (uint8(cur_outer_mask) * 255);
	im_probs_ws = im_probs_new;
    im_probs_np = im_probs_new;
    
    probs_filename = files_probs{file_id};
    imwrite(im_probs_ws, fullfile(out_blockpath_ws, probs_filename)); 
    imwrite(im_probs_np, fullfile(out_blockpath_np, probs_filename)); 
    
end

