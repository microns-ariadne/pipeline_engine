function [output] = demoForward(input,layers,features, bias)
%%%% The use of 'single' precision or 'double' precision is optional 
global CONV_TIMER FUN_TIMER;

if nargin==0
    input = rand(1076,1076,3);
end


if nargin <= 1
    layers= {       'conv-maxout2-maxpooling2',       'conv-maxout2-maxpooling2',  'conv-maxout2-maxpooling2',  'conv'    };
       
    w1 = '/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11_S1_AC3_256_cc_53_dist_4_3D_PAD/conv1A_1B_kernels.txt';
    w2 = '/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11_S1_AC3_256_cc_53_dist_4_3D_PAD/conv2A_2B_kernels.txt';
    w3 = '/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11_S1_AC3_256_cc_53_dist_4_3D_PAD/conv3A_3B_kernels.txt';
    w4 = '/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11_S1_AC3_256_cc_53_dist_4_3D_PAD/ip_conv_kernels.txt';
    
    [features{1},bias{1}] = get_net(w1);
    [features{2},bias{2}] = get_net(w2);
    [features{3},bias{3}] = get_net(w3);
    [features{4},bias{4}] = get_net(w4); 
end


%%%%%%%%% Recursion Condition

if numel(layers) == 4
    CONV_TIMER =0;
    FUN_TIMER = clock;
end

if numel(layers) == 0
    output = input;
    return
end

%%%%%%%%%% Expose Info

%  variables dims
numel(layers)
size(input)

%%%%%%%%%%%%%%

switch(layers{1})
    case 'conv'
        NFeatures = size(features{1},4);
        
        tic;
        l1 = ...
            arrayfun(@(k) convn((input),(features{1}(:,:,:,k)) ,'valid') + bias{1}(k),NFeatures:-1:1,... % end:-1:1,end:-1:1
            'uniformoutput',false);
        CONV_TIMER = CONV_TIMER + toc;
        
        output=demoForward(   permute(cat(4,l1{:}),[1 2 4 3])  , ... % compute convolutions take sum
            layers(2:end),features(2:end),bias(2:end));
        
    case 'conv-maxout2-maxpooling2'
        NFeatures = size(features{1},4);

        tic;
        left = ...
            arrayfun(@(k) convn((input),(features{1}(:,:,:,k)) ,'valid') + bias{1}(k),NFeatures-1:-2:1,...
            'uniformoutput',false);
        CONV_TIMER = CONV_TIMER + toc;
       
%         tic
%         f = single(features{1}(:,:,:,:));
%         i = single(input);
%         left = ...
%             arrayfun(@(k) convn(i,f(:,:,:,k) ,'valid') + bias{1}(k),NFeatures-1:-2:1,...
%             'uniformoutput',false);
%         toc
    
        tic;
        right = ...
            arrayfun(@(k) convn((input),(features{1}(:,:,:,k)) ,'valid') + bias{1}(k),NFeatures:-2:1,...
            'uniformoutput',false);
        CONV_TIMER = CONV_TIMER + toc; 
        
       
        
        % replacing max-out with dilation 
        %tic;
        %activ = ...
        %    arrayfun(@(k) convn((input),(features{1}(:,:,:,k)) ,'valid') + bias{1}(k),NFeatures:-1:1,...
        %    'uniformoutput',false);
        %CONV_TIMER = CONV_TIMER + toc; 
        %a = cat(4,activ{:});
        %b = imdilate(a,ones(1,1,1,2)); b=b(:,:,1,1:2:end);
        %mx=max(cat(4,left{:}),cat(4,right{:}));
        %b = imdilate(b,ones(2,2,1,1));
        %b=permute(b,[1 2 4 3]);
        
        % slower
        %f =  convn((input),rot90(features{1}(:,:,:,5)));
        
        % slower
        %input_t = permute(input,[3 1 2]);
        %features_t = permute(features{1},[3 1 2 4]);
        % right_t = ...
        %    arrayfun(@(k) convn((input_t),(features_t(:,:,:,k)) ,'valid') + bias{1}(k),NFeatures/2+1:NFeatures,...
        %    'uniformoutput',false);
        
        %for ifeature5=1:size(features{1},4)
        
        %end
        
        % there is no dilate 'valid', so last element should be erased
        %maxOutAndRunMaxima =  imdilate(permute((cat(4,left{:})+cat(4,right{:}))/2,[1 2 4 3]),ones(2,2));
        maxOutAndRunMaxima =  imdilate(permute(max(cat(4,left{:}),cat(4,right{:})),[1 2 4 3]),ones(2,2));
        %maxOutAndRunMaxima =  imdilate(permute((cat(4,left{:})),[1 2 4 3]),ones(2,2));
        
        if numel(layers) == 4 && 0
            mx = permute(max(cat(4,left{:}),cat(4,right{:})),[1 2 4 3]);
            for i=1:32
                imwrite( left{i},fullfile('~/Dump_images',['left_' num2str(i,'%5d'),'.png']),'png');
                imwrite(right{i},fullfile('~/Dump_images',['rght_' num2str(i,'%5d'),'.png']),'png');
                imwrite(mx(:,:,i),fullfile('~/Dump_images',['maxi_' num2str(i,'%5d'),'.png']),'png');
                imwrite(maxOutAndRunMaxima(:,:,i),fullfile('~/Dump_images',['dila_' num2str(i,'%5d'),'.png']),'png');
            end
        end
        
        output1=demoForward(maxOutAndRunMaxima(1:2:end-1,1:2:end-1,:)    , ... % compute convolutions take sum
            layers(2:end),features(2:end),bias(2:end));
        output2=demoForward(maxOutAndRunMaxima(2:2:end-1,1:2:end-1,:)    , ... % compute convolutions take sum
            layers(2:end),features(2:end),bias(2:end));
        output3=demoForward(maxOutAndRunMaxima(1:2:end-1,2:2:end-1,:)    , ... % compute convolutions take sum
            layers(2:end),features(2:end),bias(2:end));
        output4=demoForward(maxOutAndRunMaxima(2:2:end-1,2:2:end-1,:)    , ... % compute convolutions take sum
            layers(2:end),features(2:end),bias(2:end));
        
        if numel(layers) == 4
            
            output1=exp(output1);
            output1=output1./repmat(sum(output1,3),[1 1 size(output1,3)]);
            
            output2=exp(output2);
            output2=output2./repmat(sum(output2,3),[1 1 size(output2,3)]);
            
            output3=exp(output3);
            output3=output3./repmat(sum(output3,3),[1 1 size(output3,3)]);
            
            output4=exp(output4);
            output4=output4./repmat(sum(output4,3),[1 1 size(output4,3)]);
        end
        
        outputDim = size(output1,3);
        
        output=zeros(size(output1,1)+size(output2,1),size(output2,2)+size(output4,2),outputDim);

        output(1:2:end,1:2:end,:,:) = output1;
        output(2:2:end,1:2:end,:,:) = output2;
        output(1:2:end,2:2:end,:,:) = output3;
        output(2:2:end,2:2:end,:,:) = output4;
end

%%%%%%%%%%% Expose Stats 

if numel(layers) == 4 
    CONV_TIMER %#ok<NOPRT>
    etime(clock,FUN_TIMER)
end

%%%%%%%%%%%


function [kernels,bias] = get_net(weights_file)

filetext = fileread(weights_file);
file_data = regexp(filetext,'\(((-)?\d+(\.[0-9]{1,2})?(e-\d+)?\s?)+\)','match');
nMatrcies_line = regexp(filetext,'total-matrices: (\d)+','match');
nMatrcies = textscan(nMatrcies_line{1}, '%s %d');
nMatrcies = nMatrcies{2};

fileread_info = regexp(filetext,'\[(\d+\s?){3,3}\]','match');
kernelSize = str2num(fileread_info{1}(2:end-1));  %#ok<*ST2NM>


KernelsLinSize = kernelSize(1)*kernelSize(2);

bias_str = file_data(1:KernelsLinSize+1:end);
kernels_str = file_data(mod(1:length(file_data),KernelsLinSize+1)~=1);
kernels=cell2mat(cellfun(@(s) str2num(s(2:end-1)), kernels_str, 'UniformOutput',false));
kernels=reshape(kernels,kernelSize(3),kernelSize(1),kernelSize(2),nMatrcies); % [depth,x,y,ifeature]
bias=single(cellfun(@(s) str2num(s(2:end-1)), bias_str));


kernels = single(permute(kernels, [2 3 1 4]));

