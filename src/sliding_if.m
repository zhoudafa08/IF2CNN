function sliding_if(x, k, h, w, o)
% x:time series (eg. nasdaq values)
% k:minimum length of time series (eg. 50)
% h:window size (height, eg. 10)
% w:width (eg. 10)
% o:output size (eg. 1)
n=length(x);
num=n-h-o;
all_labels=[];
all_features=[];
for i=k:num
    i
    xx = x(1:i);
    label = x(i+1:i+o)'; 
    all_labels=[all_labels;label];
	max_data = max(xx);
	xx = xx / max_data;
%     opts=Settings_IF_v1('IF.Xi',2,'IF.alpha','ave');
    opts=Settings_IF_v1('IF.delta',10^-2,'IF.Xi',1.9);
    [IMF,] = FIF_v1(xx, opts);
    imf_num = size(IMF, 1);
    if imf_num > w
        IMF=IMF(end-w+1:end,:);
        imf_num = size(IMF, 1);
    end 
	feature = [IMF(:, i-h+1:i)'*max_data,zeros(h, w-imf_num)];
    all_features=[all_features; reshape(feature, 1, h*w)];
end

csvwrite('nq_label_for_cnn.csv',all_labels);
csvwrite('nq_feat_for_cnn.csv', all_features);
