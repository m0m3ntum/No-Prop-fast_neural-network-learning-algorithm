clear;
clc;

encoding = [1 0 0 0 0 0 0 0 0 0;
            0 1 0 0 0 0 0 0 0 0;
            0 0 1 0 0 0 0 0 0 0;
            0 0 0 1 0 0 0 0 0 0;
            0 0 0 0 1 0 0 0 0 0;
            0 0 0 0 0 1 0 0 0 0;
            0 0 0 0 0 0 1 0 0 0;
            0 0 0 0 0 0 0 1 0 0;
            0 0 0 0 0 0 0 0 1 0;
            0 0 0 0 0 0 0 0 0 1];


trlabelidx1 = 'train-labels.idx1-ubyte';
trimageidx3 = 'train-images.idx3-ubyte';
t10kimageidx3 = 't10k-images.idx3-ubyte';
t10klabelidx1 = 't10k-labels.idx1-ubyte';

%%%%%%%%%%%%%%%%% 1.READING DATA %%%%%%%%%%%%%%%%%
out = 0;
while out ~= 1 
    x=input('How many train images?');

    if x<=60000 %if value less of 60000
        %Load train files
        trainimages = loadMNISTImages(trimageidx3, x);
        trainlabels = loadMNISTLabels(trlabelidx1, x);
        out=1;
    end
end

out = 0;
while out ~= 1 
    s=input('How many neurons?');

    if s<=60000 %if value less of 60000
        out=1;
    end
end


out = 0;
while out ~= 1 
    numoftestimages=input('How many test images?');

    if numoftestimages<=10000 %if value less of 10000
        %Load test files
        testimages = loadMNISTImages(t10kimageidx3, numoftestimages);
        testlabels = loadMNISTLabels(t10klabelidx1, numoftestimages);
        out=1;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%% 2.CUT 4 LAST ROWS AND 4 LAST LINES %%%%%%%
disp('Training System!Please wait...')
tic %Starting counting times
images = trainimages(1:20,1:20,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%% 3. TABLE FROM 20X20XN TO 400XN %%%%%%%%%%
images2 = reshape(images,20*20,x);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%% 4. NORMALIZE IMAGE %%%%%%%%%%%%%%%%%%%
images2 = double(images2);
normimages = images2(:,:,:)/255;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%% 5. K neurons %%%%%%%%%%%%%%%%%%
%%% SEE FILE kneuron.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%% 6. Change R to R+1 %%%%%%%%%%%%%%%
%Add 1 row with aces
newR = [normimages;ones(1,x)];

%Calculate b for every neuron
%b = -0.25 + (0.25+0.25)*rand(s,1);%%%%% we need it?????

%Calculate w for every input & neuron
w = -0.25 + (0.25+0.25)*rand(s,401);
temp = -0.25 + (0.25+0.25)*rand(s,1);%%%%%%%
wh = [w,temp;];

S=kneuron(newR,s,x,wh);%use neurons
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%% 7. Calculating WOUT %%%%%%%%%%%%%%%
%Creating D matrix
D=[];
for i=1:x
   %Change train label number to encoding ex. 0 = 1 0 0 0 0 0 0 0 0 0
   getlabel=trainlabels(i);
   encodedlabel=encoding(:,(getlabel+1));
   D=[D,encodedlabel]; 
end

PS=pinv(S);
%PS=inv(S);
WOUT=D*PS;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Finish Training:')
toc
%%%%%%%%%%%%%%% 8. TEST SETS %%%%%%%%%%%%%%%
disp('Testing System!Please wait...')
testimages = testimages(1:20,1:20,:);
testimages2 = reshape(testimages,20*20,numoftestimages);
testimages2 = double(testimages2);
Z = testimages2(:,:,:)/255;
%use neurons (hidden layer) for test set
T = kneuron(Z,s,numoftestimages,w);
%Here is out layer (Y matrix)
Y = WOUT * T;

%Create
nD=[];
for i=1:numoftestimages
   %Change test label number to encoding ex. 0 = 1 0 0 0 0 0 0 0 0 0
   getlabel=testlabels(i);
   encodedlabel=encoding(:,(getlabel+1));
   nD=[nD,encodedlabel]; 
end

%Getting the max value,position of Y matrix for every column 
[maxvalY,iinY] = max(Y,[],1);
%Getting the max value,position of Y matrix for every column 
[maxvalD,iinD] = max(nD,[],1);


counter=0;
%Checking Y matrix attributes with test labels
for i=1:numoftestimages
    if (iinY(i))==(iinD(i))
        counter=counter+1;        
    end
end
%counter %how many found
disp('Percentage of found')
(counter/numoftestimages)*100
disp('Finished Test')
toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
