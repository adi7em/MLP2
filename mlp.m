%file = fopen('steering/data.txt','rt');
count = 0;
while (fgets(file) ~= -1)
	count = count + 1;
end
count 
myfile = importdata('steering/data.txt');
inp = zeros(21999,1024);
label = zeros(21999,1);
% out1 = zeros(17536,1024);
% label1 = zeros(17536,1);
% out2 = zeros(4463,1024);
% label2 = zeros(4463,1);

for i = 1:21999
	A = imread(strcat('steering/',myfile.textdata{i}(3:end)));
	label(i) = myfile.data(i);
	A = rgb2gray(A);
	inp(i,:)= reshape(A,1,[]);
	y = label;
end
% %%%%%%%%%%

m = 32;
n = 32;
N = 21999;
X = inp;
out = label;
S = bsxfun(@minus,X,mean(X));
X = bsxfun(@rdivide,S,std(X));
X=[X out];
X=X(randperm(end),:);
out=X(:,(m*n)+1);
X=X(:,1:m*n);
t=floor((4*N)/5);
Xtrain=X(1:t,:);
Xvalid=X(t+1:N,:);
outtrain=out(1:t,:);
outvalid=out(t+1:N,:);
epochs=10000;
rate=0.0005;
Bsize=50;
[w, u, v, trainerror, validerror,wgood,ugood,vgood,error_min]=mlpTrain(Xtrain,outtrain,Xvalid,outvalid,512,64,rate,epochs,Bsize);
%ydash=mlptest(X,w,u,v);
%disp(trainerror);
figure, plot(trainerror), hold on,
plot(validerror)
hold off
title(sprintf('mean squared train error with learning rate=%d and Batch size=%d',rate,Bsize));


