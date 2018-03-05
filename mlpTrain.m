function [w, u, v, trainerror, validerror,wgood,ugood,vgood,error_min] = mlpTrain(X, Y, Xvalid, Yvalid, H1, H2, eta, nEpochs,Bsize)
% X - training data of size NxD
% Y - training labels of size NxK
% H - the number of hiffe
% eta - the learning rate
% nEpochs - the number of training epochs
% define and initialize the neural network parameters
error_min = 10;
% number of training data points
N = size(X,1);
Nvalid=size(Xvalid,1);
% number of inputs
D = size(X,2); % excluding the bias term
% number of outputs
K = size(Y,2);
% weights for the connections between input and hidden layer
% random values from the interval [-0.01 0.01]
% w is a H1x(D+1) matrix
w = -0.01+(0.02)*rand(H1,D);
w=[zeros(H1,1) w];
% weights for the connections between input and hidden layer
% random values from the interval [-0.01 0.01]
% v is a H2x(H1+1) matrix
u = -0.01+(0.02)*rand(H2,H1);
u=[zeros(H2,1) u];
% weights for the connections between input and hidden layer
% random values from the interval [-0.01 0.01]
% v is a Kx(H2+1) matrix
v = -0.01+(0.02)*rand(K,H2);
v=[zeros(K,1) v];
trainerror=zeros(nEpochs,1);
validerror=zeros(nEpochs,1);
% randomize the order in which the input data points are presented to the
% MLP
%iporder = randperm(N);
% mlp training through stochastic gradient descent
for epoch = 1:nEpochs
    start=1;
    End=start+Bsize-1;
    n=End-start+1;
    while End<N
        x=[ones(n,1) X(start:End,:)];
        % the current training point is X(iporder(n), :)
        % forward pass
        % --------------
        % input to hidden layer
        % calculate the output of the hidden layer units - z1
        % ---------
        z1=x*transpose(w);
        z1=1./(1.+exp(-z1));
        %z1=sigmf(z1,[0 1]);
        % ---------
        % calculate the output of the hidden layer units - z2
        % ---------
        z1temp=[ones(n,1) z1];
        z2=z1temp*transpose(u);
        z2=1./(1.+exp(-z2));
        %z2=sigmf(z2,[0 1]);
        % hidden to output layer
        % calculate the output of the output layer units - ydash
        % ---------
        z2temp=[ones(n,1) z2];
        ydash=z2temp*transpose(v);
        %disp(ydash);
        % ---------
        % backward pass
        % ---------------
        % update the weights for the connections between hidden and
        % outlayer units
        % ---------
        y=Y(start:End,:);
        gradv=ydash-y;
        gradu=(gradv*v(:,2:H2+1)).*z2.*(1-z2);
        gradw=(gradu*u(:,2:H1+1)).*z1.*(1-z1);
        v=v-(eta/Bsize)*(transpose(gradv)*z2temp);
        % ---------
        % update the weights for the connections between the hidden layer and
        % hidden later 2 units
        % -----`----
        u=u-(eta/Bsize)*(transpose(gradu)*z1temp);
        %disp(u);
        % ------
        % ---------
        % update the weights for the connections between the input and
        % hidden later 1 units
        % -----`----
        w=w-(eta/Bsize)*(transpose(gradw)*x);
        start=End+1;
        End=min(start+Bsize-1,N);
        n=End-start+1;
    end
    ydash = mlptest(X, w, u, v);
    ydashvalid = mlptest(Xvalid, w, u, v);
   % disp(ydash);
   disp([ydash Y]);
    % compute the training error
    % ---------
    %'TO DO'% uncomment the next line after adding the necessary code
    trainerror(epoch)=(transpose(ydash-Y)*(ydash-Y))/(2*N);
    validerror(epoch)=(transpose(ydashvalid-Yvalid)*(ydashvalid-Yvalid))/(2*Nvalid);
    if(validerror(epoch) < error_min)
        error_min = validerror(epoch);
        wgood = w;
        ugood = u;
        vgood = v;
        disp(error_min);
    end
    % ---------
    disp(sprintf('training error and valid error after epoch %d: %f and %f\n',epoch,trainerror(epoch),validerror(epoch)));
end
end
