% 本程序用于考试测试过程
% 仅仅对保存的训练好的模型进行调用，循环输入图片，即可获得最终的结果


% 加载模型
clc;close all;clear;
load('-mat','F:\模式识别\my_net');

% 加载测试集
Location = 'F:\模式识别\测试数据';
tic
result = [];

for i=1:100
    Path = strcat([Location, '\'], [num2str(i), '.jpg']);
    imds = imageDatastore(Path);
    inputSize = net_train.Layers(1).InputSize;
    imds = augmentedImageDatastore(inputSize(1:2),imds);   % 输入自动变成inputSize(1:2)所规定的
    
    YPred = classify(net_train, imds);

    A = str2num(cell2mat(cellstr(YPred)));
    result = [result; A];
end

% 使用训练好的模型对测试集进行分类
disp(['分类所用时间为：', num2str(toc), '秒']);

% save result.dat -ascii result
fp=fopen('张一川11.dat','a');   % 自动生成.dat文件
fwrite(fp,result,'int');       % 将指代的内容写入.dat文件
fclose(fp);



