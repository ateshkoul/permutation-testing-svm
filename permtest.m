% Script to perform a permutation testing on the classification testing

% Select the data randomly
initial_col = [8 18 40 50 60 70 80 90 100 170 180 190];
initial_col=initial_col-1;


for i=1:10
posIns = Data(Data(:,2)==1,[2 initial_col+i]);
negIns = Data(Data(:,2)==2,[2 initial_col+i]);
allIns = [posIns;negIns];
accObs(i) = kinematicsvm_param(allIns);
end


clear i;

% i = 10 : time points
% j = 1000 : no. of runs


for i=1:10
    for j=1:1000
        posIns = Data(Data(:,2)==1,[2 initial_col+i]);
        negIns = Data(Data(:,2)==2,[2 initial_col+i]);
        allIns = [posIns;negIns];
        allIns(:,1) = Shuffle(allIns(:,1));
        accRan(j,i) = kinematicsvm_param(allIns);
    end
end

clear i j;

% calculate p-value
for i=1:10
    pValue(i)= sum(accRan(:,i)>=accObs(i))./1000;
end
