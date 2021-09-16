clc
clear all

% load data

data = load('data.txt');

%user input

% getinput = 'enter number of input neuron : '
% L=input('enter number of input neuron : ',L)
% getoutput = 'enter number of output neuron : '
% N=input(getoutput);
% gettrain = 'enter number of training pattern : '
% P=input(gettrain);
% gettest = 'enter number of testing pattern : '
% P_test =input(gettest);
L=2;N=2;p=2;p_test=1;eta=0.3;alpha=0.2;

%%%%% saperate input and output data

% training data

X=data(1:p,1:L);
Y=data(1:p,(L+1):(L+N));
X=normalization(X);
X=[ones(size(X,1),1) X]; % add bias to hidden layer

% testing data

% X_test = data(p+1:p+p_test,1:L);
% Y_test = data(p+1:p+p_test,(L+1):(L+N));
% X_test = normalization(X_test);
% X_test = [ones(size(X_test,1),1) X_test]; % add bias to hidden layer

% weight matrix

M = L+1; % number of hidden layer

v = rand(L+1,M);
w = rand(M+1,N);

% initialize all parameter

IH=zeros(M,p);
OH=zeros(M,p);
IO=zeros(N,p);
OO=zeros(N,p);
E=zeros(N,p);
Eavg=zeros(N,1);
delv=zeros(L+1,M);
delw=zeros(M+1,N);

%%%%%%%%%%%% training of neural network 

% iteration loop

E_vec = zeros(200,1);
E_vec_test = zeros(200,1);
iteration_vec = zeros(200,1);
for iter_limit = 1:10:2000
   
    count = 1;
    
for iter=1:iter_limit
    
    %%%%% forward path calculation
    
    IH = v'*X';
    OH = sigmoid(IH);
    OH = [ones(1,size(OH,2));OH];  % add bias to o/p layer
    IO = w'*OH;
    OO = tansigmoid(IO);
    E = 0.5*(Y' - OO).^2;
    Eavg = (1/p)*sum(E,2);
    
    E_absolute = log10((1/N)*sum(Eavg));
      
    %%%%% back-propogation loop
    
    % W update loop
    
        delw_1 = zeros(M+1,N);
        delw_1 = (eta/p)*(OH)*((Y - OO').*(1 - OO'.^2));
        w = w +  delw_1 + alpha*delw;
        delw = delw_1;
    
    % V update loop
    
   b=zeros(L+1,M);
   for i=1:L+1
       for j=1:M
           sum1 = 0;
           xx=0;
           for k=1:p
               sum2=0;
               for m=1:N
                   sum2 = sum2 +(Y(k,m) - OO(m,k))*(1 - OO(m,k)^2)*...
                                (w(j,m))*(OH(j,k))*(1 - OH(j,k))*(X(k,i));
                   xx= sum2;
               end
               sum1 = sum1 + xx;
               yy=sum1;
           end
           b(i,j) = (eta/(N*p))*yy + alpha*delv(i,j);
       end
   end
   delv = b;
   v = v + delv;
end

%%%%%%%%%%  testing of neural network

% IH = v'*X_test';
% OH = sigmoid(IH);
% OH = [ones(1,size(OH,2));OH];  % add bias to o/p layer
% IO = w'*OH;
% OO = tansigmoid(IO);
% E = 0.5*(Y_test' - OO).^2;    
% Eavg = (1/p)*sum(E,2);

% E_absolute_test = log10((1/N)*sum(Eavg));

E_vec(count) = [E_absolute];
% E_vec_test(count) = [E_absolure_test];
iteration_vec(count) = [iter_limit];

count = count + 1;

end

plot(iteration_vec,E_vec)
hold on
%plot(iteration_vec,E_vec_test)







