function [x1,y1] = normalization(X,Y)

 for i=1:size(X,2)
     x1(:,i)=((X(:,i)-min(X(:,i)))/(max(X(:,i)) - min(X(:,i))));
     
 end
 
 for i=1:size(Y,2)
     y1(:,i)=((Y(:,i)-min(Y(:,i)))/(max(Y(:,i)) - min(Y(:,i))))*2 - 1;
     
 end
%  y1=((Y-min(Y))/((max(Y)-min(Y))))*2 - 1;


end