function ws_np_prepare(cnn_crop_leg_str, ...
                       in_blockpath_meta, ...
                       in_blockpath_probs, ...
                       out_blockpath_ws, ...
                       out_blockpath_np)
%

cnn_crop_leg = str2num(cnn_crop_leg_str);
fprintf('cnn_crop_leg       : %d\n', cnn_crop_leg);
fprintf('in_blockpath_meta  : %s\n', in_blockpath_meta);
fprintf('in_blockpath_probs : %s\n', in_blockpath_probs);
fprintf('out_blockpath_ws   : %s\n', out_blockpath_ws);
fprintf('out_blockpath_np   : %s\n', out_blockpath_np);

files_probs = dir(fullfile(in_blockpath_probs, '*-1.png'));
files_probs = files_probs(~[files_probs.isdir]);
files_probs = sort({files_probs.name});
n_files_probs = length(files_probs);

all_outer_mask_filename = sprintf('t_outer_mask_all.png');
all_border_mask_filename = sprintf('t_border_mask_all.png');

outer_mask_all = imread(fullfile(in_blockpath_meta, all_outer_mask_filename));
border_mask_all = imread(fullfile(in_blockpath_meta, all_border_mask_filename));

dims = size(outer_mask_all);
dims_all = size(border_mask_all);

assert(isequal(dims, dims_all));

outer_mask_all = outer_mask_all(1+cnn_crop_leg : dims(1)-cnn_crop_leg , 1+cnn_crop_leg : dims(2)-cnn_crop_leg);
outer_mask_all = imresize(outer_mask_all, 0.5);
outer_mask_all = outer_mask_all ~= 0;

border_mask_all = border_mask_all(1+cnn_crop_leg : dims(1)-cnn_crop_leg , 1+cnn_crop_leg : dims(2)-cnn_crop_leg);
border_mask_all = imresize(border_mask_all, 0.5);
border_mask_all = border_mask_all ~= 0;

images_probs = cell(1, n_files_probs);
outer_masks = cell(1, n_files_probs);
border_masks = cell(1, n_files_probs);

for file_id = 1 : n_files_probs
    
    probs_filename = files_probs{file_id};
    probs_filepath = fullfile(in_blockpath_probs, probs_filename);
    
    fprintf('Read[%d]: %s\n', file_id, probs_filepath);
    im_probs = imread(probs_filepath);
    
    images_probs{file_id} = im_probs;
    
    outer_mask_filename = sprintf('outer_mask_%.4d.png', file_id);
    border_mask_filename = sprintf('border_mask_%.4d.png', file_id);
    
    outer_mask_filepath = fullfile(in_blockpath_meta, outer_mask_filename);
    border_mask_filepath = fullfile(in_blockpath_meta, border_mask_filename);
    
    fprintf('  -- Read mask[%d][outer]: %s\n', file_id, outer_mask_filepath);
    outer_mask = imread(outer_mask_filepath);
    
    fprintf('  -- Read mask[%d][border]: %s\n', file_id, border_mask_filepath);
    border_mask = imread(border_mask_filepath);
    
    dims = size(outer_mask);
    dims2 = size(border_mask);
    assert(isequal(dims, dims2));
    assert(isequal(dims, dims_all));
    
    outer_mask = outer_mask(1+cnn_crop_leg : dims(1)-cnn_crop_leg , 1+cnn_crop_leg : dims(2)-cnn_crop_leg);
    outer_mask = imresize(outer_mask, 0.5);
    outer_mask = outer_mask ~= 0;
    
    border_mask = border_mask(1+cnn_crop_leg : dims(1)-cnn_crop_leg , 1+cnn_crop_leg : dims(2)-cnn_crop_leg);
    border_mask = imresize(border_mask, 0.5);
    border_mask = border_mask ~= 0;
    
    outer_masks{file_id} = outer_mask;
    border_masks{file_id} = border_mask;
    
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
    
    probs_ws_filepath = fullfile(out_blockpath_ws, probs_filename);
    probs_np_filepath = fullfile(out_blockpath_np, probs_filename);
    
    fprintf('  -- write_ws_probs[%d]: %s\n', file_id, probs_ws_filepath);
    imwrite(im_probs_ws, probs_ws_filepath);
    
    fprintf('  -- write_np_probs[%d]: %s\n', file_id, probs_np_filepath);
    imwrite(im_probs_np, probs_np_filepath); 
    
end

fprintf('-- finish\n');
