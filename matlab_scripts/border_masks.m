function border_masks(border_width, ...
                      close_width, ...    
                      in_blockpath_data, ...
                      out_blockpath_meta)
%

border_width = str2num(border_width);
close_width = str2num(close_width);

fprintf('border_width: %d\n', border_width);
fprintf('close_width: %d\n', close_width);
fprintf('in_blockpath_data: %s\n', in_blockpath_data);
fprintf('out_blockpath_meta: %s\n', out_blockpath_meta);

files_data = dir(in_blockpath_data);
files_data = files_data(~[files_data.isdir]);
files_data = sort({files_data.name});
n_files_data = length(files_data);

images_data = cell(1, n_files_data);

for file_id = 1 : n_files_data
    
    fprintf('Read file_id = %d\n', file_id);
    
    data_filename = files_data{file_id};
    im_data = imread(fullfile(in_blockpath_data, data_filename));
    
    images_data{file_id} = im_data;
    
end

t_outer_mask_all = zeros(size(images_data{1}));
t_border_mask_all = zeros(size(images_data{1}));

for file_id=1:n_files_data
    
    fprintf('Mask file_id = %d\n', file_id);
    
    im_data = images_data{file_id};
    
    [out_mask, border_mask] = IdentifyLargeBorder(im_data, border_width, close_width);
    
    out_mask_image = (out_mask == 1);
    out_mask_all = (out_mask == 2);
    
    border_mask_image = (border_mask == 1);
    border_mask_all = (border_mask == 2);
    
    outer_filename = sprintf('outer_mask_%.4d.png', file_id);
    border_filename = sprintf('border_mask_%.4d.png', file_id);
    
    imwrite(uint8(out_mask_image) * 255, fullfile(out_blockpath_meta, outer_filename));
    imwrite(uint8(border_mask_image) * 255, fullfile(out_blockpath_meta, border_filename));

    t_outer_mask_all = t_outer_mask_all | out_mask_all;
    t_border_mask_all = t_border_mask_all | border_mask_all;

end

imwrite(uint8(t_outer_mask_all) * 255, fullfile(out_blockpath_meta, 't_outer_mask_all.png'));
imwrite(uint8(t_border_mask_all) * 255, fullfile(out_blockpath_meta, 't_border_mask_all.png'));

