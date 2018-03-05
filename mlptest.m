function ydash = mlptest(X, w, u, v)
        [N,D]=size(X);
        X=[ones(N,1) X];
        % the current training point is X(iporder(n), :)
        % forward pass
        % --------------
        % input to hidden layer
        % calculate the output of the hidden layer units - z1
        % ---------
        z1=X*transpose(w);
        z1=1./(1+exp(-z1));
        %z1=sigmf(z1,[0 1]);
        % ---------
        % calculate the output of the hidden layer units - z2
        % ---------
        z1temp=[ones(N,1) z1];
        z2=z1temp*transpose(u);
        z2=1./(1+exp(-z2));
        %z2=sigmf(z2,[0 1]);
        % hidden to output layer
        % calculate the output of the output layer units - ydash
        % ---------
        z2temp=[ones(N,1) z2];
        ydash=z2temp*transpose(v);
end