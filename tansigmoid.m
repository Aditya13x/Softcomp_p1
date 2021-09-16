function x= tansigmoid(y)

    x=(exp(y) - exp(-y))./(exp(y) + exp(-y));
end