clc
clear all

% load data

data = load('data.txt');

% randomly short the data

random = randi([1,size(data,1)],size(data,1),1);
for i=1:length(random)
    data1(i,:) = data(random(i),:);
end

%user input

% getinput = 'enter number of input neuron : '
% L=input('enter number of input neuron : ',L)
% getoutput = 'enter number of output neuron : '
% N=input(getoutput);
% gettrain = 'enter number of training pattern : '
% P=input(gettrain);
% gettest = 'enter number of testing pattern : '
% P_test =input(gettest);
L=8;N=1;p=50;p_test=9;eta=0.01;alpha=0.005;

%%%%% saperate input and output data

% training data

X=data1(1:p,1:L);
Y=data1(1:p,(L+1):(L+N));
X_original = X;
Y_original = Y;
X=[ones(size(X,1),1) X]; % add bias to hidden layer

% testing data

X_test = data1(p+1:p+p_test,1:L);
Y_test = data1(p+1:p+p_test,(L+1):(L+N));
X_test_original = X_test;
Y_test_original = Y_test;
X_test = [ones(size(X_test,1),1) X_test]; % add bias to hidden layer

% weight matrix

M = 3; % number of hidden layer

v = -1 + (2).*rand(L+1,M);
w = -1 + (2).*rand(M+1,N);

% initialize all parameter

IH=zeros(p,M);
OH=zeros(p,M);
IO=zeros(p,N);
OO=zeros(p,N);
E=zeros(p,N);
Eavg=zeros(1,N);
delv=zeros(L+1,M);
delw=zeros(M+1,N);

%%%%%%%%%%%% training of neural network 

% iteration loop
% E_vec = zeros(100,1);
% iter_vec = zeros(100,1);
for iter=1:100000
    
    %%%%% forward path calculation
    
    IH = X*v;
    OH = sigmoid(IH);
    OH = [ones(size(OH,1),1) OH];  % add bias to o/p layer
    IO = OH*w;
    OO = tansigmoid(IO);
    E = 0.5*(Y - OO).^2;
    
    Eavg = (1/p)*sum(E,1);
    
    error1 = (1/p)*norm(abs(Y - OO));
   
    E_vec(iter) = error1;
    iter_vec(iter) = iter;
      
    %%%%% back-propogation loop
    
    % W update loop
    
        delw_1 = zeros(M+1,N);
        delw_1 = (eta/p)*(OH)'*((Y - OO).*(1 - OO.^2));
        w = w +  delw_1 + alpha*delw;
        delw = delw_1;
    
    % V update loop
    
    OH = sigmoid(IH);
   b=zeros(L+1,M);
   for i=1:L+1
       for j=1:M
           sum1 = 0;
           xx=0;
           for k=1:p
               sum2=0;
               for m=1:N
                   sum2 = sum2 +(Y(k,m) - OO(k,m))*(1 - OO(k,m)^2)*...
                                (w(j,m))*(OH(k,j))*(1 - OH(k,j))*(X(k,i));
                                  
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
   
    IH_test = X_test*v;
    OH_test = sigmoid(IH_test);
    OH_test = [ones(size(OH_test,1),1) OH_test];  % add bias to o/p layer
    IO_test = OH_test*w;
    OO_test = tansigmoid(IO_test);
    E_test = 0.5*(Y_test - OO_test).^2;
       
    Eavg_test = (1/p_test)*sum(E_test,1);

     error2 = (1/p_test)*norm(abs(Y_test - OO_test));
     E_vec_2(iter,1) = error2;
  
end



error_norm = error1
error_norm_test = error2

plot(iter_vec,E_vec)
hold on
plot(iter_vec,E_vec_2)
hold off

training = [Y_original OO]
testing = [Y_test_original OO_test]

%%%%%%%%%%  testing of neural network

%  IH_test = X_test*v;
%  OH_test = sigmoid(IH_test);
%  OH_test = [ones(size(OH_test,1),1) OH_test];  % add bias to o/p layer
%  IO_test = OH_test*w;
%  OO_test = tansigmoid(IO_test);
%  E_test = 0.5*(Y_test - OO_test).^2;      
%  Eavg_test = (1/p_test)*sum(E_test,1);







