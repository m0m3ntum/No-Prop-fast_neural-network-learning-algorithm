%%%%%%%%%%%%%%%%%% 5. K neurons (Hidden layer) %%%%%%%%%%%%%%%%%%

%R=inputs, s=num of neurons,x=num of inputs , b=bias of network
function [A] = kneuron(R,s,x,w)
%http://www.mathworks.com/help/nnet/ug/neural-network-architectures.html
%For all neurons

for i=1:s
    %For all input R
    for j=1:x
        sum=0;
        for k=1:length(R(:,1))
            sum=sum+(w(i,k)*R(k,j));
        end
        n(i,j) = sum;
    end
end    
    %Activation function tanh    
    A=tanh(n);
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
