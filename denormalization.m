function y1 = denormalization(Y,Z)

for i=1:size(Y,2)
    
    y1 = ((Y(:,i) + 1)./2).*(max(Z(:,i)) - min(Z(:,i))) + min(Z(:,i));
    
end

end