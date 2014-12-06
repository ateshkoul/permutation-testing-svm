% function to automatically select the kinematics data, predict the
% accuracy of the classification, select the best features/ provide best
% features using F-score
function [accCv, accTest] = kinematicsvm_param(Data,param)


% Get the current folder will be used to save the files filename.fscore,
% filename.select
current_loc = pwd;
%% old remanant: have to consider
% Loading the File and reading the data
% Currently the pro
% select the file in a double format with first column as predicted label
if nargin<1
    [a,b] = uigetfile('*.xlsx','Multiselect','on');
    filename = fullfile(b,a);

    %load the file 
    % if the file already is interpolated
    %A = importdata(filename);

    % if the file neads to be interpolated
    A = interpol_kin(filename);

    Data = cat(1,A{:});

    % Removing the trial no. detail from the kinematic data
    % not needed as the origional column names would be used to avoid
    % confusion
    %Data = Data(:,2:end);
end


% create fixed parameters that are not foing to change during the updating
% od c and gamma (g).

if exist('param','var')
    paramFix = ['-m 2000 -q ',param];
else
    paramFix = ['-m 2000 -q -t 2 '];
end

% Separate Data as X and Y

% X is 3rd column onwards in origional Data file.
%% Data cleaning and selecting

% there is no manipulation here
% this function is not specific to the work
X = Data(:,2:end);

% Y is the 2nd column in the origional Data, so first column of Data as
% Data = Data(:,2:end) was done.
y = Data(:,1);


% change labels to 1 and -1 (from 1 and 2) (may not change the analysis significantly)
y(y==2)= -1;

%% Feature scaling

% normalise the features (mean centering), other otions can be done as well
for i=1:size(X,2)
    X_norm(:,i) = X(:,i)./mean(X(:,i));
end

clear i

%% Feature selection using fselect.py

% Libsvm needs data to be in sparse format so have to convert variables to
% a sparse format (only the X values not y)
X_norm_sparse=sparse(X_norm);


% % change to the libsvm tools folder location 
% path = '/usr/local/MATLAB/R2012b/toolbox/libsvm-3.17/tools/';
% 
% % change to the directory
% cd(path);
% 
% 
% if ~exist('filename')
%     filename = 'feature_select.txt';
% end

% Libsvmwrite to write the data to a libsvm friendly sparse format for
% feature selection file

% libsvmwrite(filename,y,X_norm_sparse);


%X_norm = X_norm(:,[1:3,6:13,15,16,17,25:end])
%X_norm = X_norm(:,[4,24,23,22,21,20,5,19,18,14]);

% using the fselect.py to select the features. The features will be stored
% in the file feature_select.txt.fscore file.

% Setting up python path 
% necessary to run the following script using python to select the features
% Configuration can be used to set PYTHONPATH for linux systems so that the
% script fselect.py can be called from the same folder. However, the script
% calls the function grid.py. so to use this, one has to modify the code of
% the fseelct.py script. I choose the simple way of chanding directory to
% libsvm and moving the file to the current location.

% specific code so that the python can choose a variable filename
% the location of space after fselect.py is important
% This is because the command concatanates as python fselect <file>
% if the space is not present, it makes it as python fselect.py<file> which
% python does not understand.

% system(['python fselect.py' ' ' filename]);
% movefile(strcat(filename,'.fscore'),current_loc)
% movefile(strcat(filename,'.select'),current_loc)
% movefile(filename,current_loc)
% 
% cd(current_loc)

%% partition the data using cvpartition into test and training sets
% the entire data is divided for 10 fold crossvalidation.
% the entire dataset is divided into 10 parts, train on 9 and test on one
% left.
    
% using a default value for no. of folds = 10 for now.
nfold = 10;
% the parameter containing values of best c and g preallocated here.
Par = zeros(nfold,3);
%[train,test] = crossvalind('LeaveMOut',N,10)

% getting the index values from the function
index = crossvalidation(y,nfold);
for folds=1:nfold
    
    testIdx = (index == folds); trainIdx = ~testIdx;   
    % divide the data into 10 folds one part is selected in the training
    predtrainData = X_norm(trainIdx,:);
    predtrainData_sparse = sparse(predtrainData);
    predtrainLabel = y(trainIdx);
    
    predtestData = X_norm(testIdx,:);
    predtestData_sparse = sparse(predtestData);
    predtestLabel = y(testIdx);
        
    %separating the data for training for model optimisation and for
    %actual prediction.
    % This is done to keep the process of optimisation and actual training
    % separate and independend. The data is divided into two complimentary parts
    
    % in each iteration, different part of the 10 parts will be selected
    % and of this part, one will be used for getting optimal c and g and
    % the other for actual training.
    
    % using the crossvalidation function to get equal partition of the data
    % with almost equal probability of two options.
    optimIdx = crossvalidation(predtrainLabel,2);    
    trainData= predtrainData((optimIdx==1),:);
    trainData_sparse = sparse(trainData);
    trainLabel = predtrainLabel(optimIdx==1);
    
    %keep the test and train data separate
    testData = predtrainData(optimIdx==2,:);
    testData_sparse = sparse(testData);
    testLabel = predtrainLabel(optimIdx==2);
    sizeoftestdata(folds) = size(testData,2);
        



    % ###################################################################
    % cross validation scale 1
    % This is the big scale (rough)
    % ###################################################################
    
    % can choose the list from -20 to 20
    stepSize = 1;
    log2c_list = -2:stepSize:10;
    log2g_list = -2:stepSize:10;

    numLog2c = length(log2c_list);
    numLog2g = length(log2g_list);
    cvMatrix = zeros(numLog2c,numLog2g);
    bestcv = 0;
    for i = 1:numLog2c
        log2c = log2c_list(i);
        for j = 1:numLog2g
            log2g = log2g_list(j);
            % -v 3 --> 3-fold cross validation
            param = [paramFix,' -v 3',' -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
            cv = svmtrain(trainLabel, trainData_sparse, param);
            cvMatrix(i,j) = cv;
            if (cv >= bestcv),
                bestcv = cv; bestLog2c = log2c; bestLog2g = log2g;
            end
            % fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
        end
    end
    
    clear i j log2g_list log2c_list log2c log2g cvMatrix cv

    disp(['CV scale1: best log2c:',num2str(bestLog2c),' best log2g:',num2str(bestLog2g),' accuracy:',num2str(bestcv),'%']);

    % % Plot the results
    % figure;
    % imagesc(cvMatrix); colormap('jet'); colorbar;
    % set(gca,'XTick',1:numLog2g)
    % set(gca,'XTickLabel',sprintf('%3.1f|',log2g_list))
    % xlabel('Log_2\gamma');
    % set(gca,'YTick',1:numLog2c)
    % set(gca,'YTickLabel',sprintf('%3.1f|',log2c_list))
    % ylabel('Log_2c');

    
    % ###################################################################
    % cross validation scale 2
    % This is the medium scale
    % ###################################################################
    prevStepSize = stepSize;
    stepSize = prevStepSize/2;
    log2c_list = bestLog2c-prevStepSize:stepSize:bestLog2c+prevStepSize;
    log2g_list = bestLog2g-prevStepSize:stepSize:bestLog2g+prevStepSize;
    
    numLog2c = length(log2c_list);
    numLog2g = length(log2g_list);
    cvMatrix = zeros(numLog2c,numLog2g);
    bestcv = 0;
    for i = 1:numLog2c
        log2c = log2c_list(i);
        for j = 1:numLog2g
            log2g = log2g_list(j);
            % -v 3 --> 3-fold cross validation
            param = [paramFix,' -v 3',' -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
            cv = svmtrain(trainLabel, trainData_sparse, param);
            cvMatrix(i,j) = cv;
            if (cv >= bestcv),
                bestcv = cv; bestLog2c = log2c; bestLog2g = log2g;
            end
            % fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
        end
    end
    
    clear i j log2g_list log2c_list log2c log2g cvMatrix cv
    
    disp(['CV scale2: best log2c:',num2str(bestLog2c),' best log2g:',num2str(bestLog2g),' accuracy:',num2str(bestcv),'%']);
    
    % % Plot the results
    % figure;
    % imagesc(cvMatrix); colormap('jet'); colorbar;
    % set(gca,'XTick',1:numLog2g)
    % set(gca,'XTickLabel',sprintf('%3.1f|',log2g_list))
    % xlabel('Log_2\gamma');
    % set(gca,'YTick',1:numLog2c)
    % set(gca,'YTickLabel',sprintf('%3.1f|',log2c_list))
    % ylabel('Log_2c');
 

    % 
    % ###################################################################
    % cross validation scale 3
    % This is the small scale
    % ###################################################################
    prevStepSize = stepSize;
    stepSize = prevStepSize/2;
    log2c_list = bestLog2c-prevStepSize:stepSize:bestLog2c+prevStepSize;
    log2g_list = bestLog2g-prevStepSize:stepSize:bestLog2g+prevStepSize;
    
    numLog2c = length(log2c_list);
    numLog2g = length(log2g_list);
    cvMatrix = zeros(numLog2c,numLog2g);
    bestcv = 0;
    for i = 1:numLog2c
        log2c = log2c_list(i);
        for j = 1:numLog2g
            log2g = log2g_list(j);
            % -v 3 --> 3-fold cross validation
            param = [paramFix,' -v 3',' -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
            cv = svmtrain(trainLabel, trainData_sparse, param);
            cvMatrix(i,j) = cv;
            if (cv >= bestcv),
                bestcv = cv; bestLog2c = log2c; bestLog2g = log2g;
            end
            % fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
        end
    end
    
    clear prevStepSize stepSize cvMatrix log2g_list log2c_list log2c log2g i j cv
    
    disp(['CV scale3: best log2c:',num2str(bestLog2c),' best log2g:',num2str(bestLog2g),' accuracy:',num2str(bestcv),'%']);
    
    % % Plot the results
    % figure;
    % imagesc(cvMatrix); colormap('jet'); colorbar;
    % set(gca,'XTick',1:numLog2g)
    % set(gca,'XTickLabel',sprintf('%3.1f|',log2g_list))
    % xlabel('Log_2\gamma');
    % set(gca,'YTick',1:numLog2c)
    % set(gca,'YTickLabel',sprintf('%3.1f|',log2c_list))
    % ylabel('Log_2c');

    paramCV = [paramFix,' -c ', num2str(2^bestLog2c), ' -g ',num2str(2^bestLog2g), ' -b 1'];
    Par(folds,1:2) = [2^bestLog2c 2^bestLog2g];
    % for the cross validation use the testing data for which the parameters
    % have not been optimised.

    % This is done  to avoid overfitting the data with the best parameters
    % see
    % http://stackoverflow.com/questions/9047459/retraining-after-cross-validation-with-libsvm 

    nfoldtrain = 10;
    %nfoldtrain = size(testLabel,1); % leave one out approach for now

    % the following does the kfold cross validation
    acc = libsvmcrossval_ova(testLabel, testData, paramCV, nfoldtrain);
    % in case of probability estimates, use 100* mean
    fprintf('Cross Validation Accuracy = %.2f%%\n', mean(acc));

    %# compute final model over the entire dataset
    %faccuracy = svmtrain(predtrainLabel,predtrainData, strcat(param,' -v 5'));
    %[~,accuracy,~] = svmpredict(predtestLabel,predtestData,fmodel,'-b 1')
    % for nmdl=1:size(mdl,1)
    % [pred,acc{i},prob] = libsvmpredict_ova(y, X, mdl{nmdl})
    % end
    Par(folds,3) = mean(acc);
    size(predtestLabel);
    Para = [param ' -v 3'];
    fmodelacc(folds) = svmtrain(predtestLabel, predtestData_sparse,Para);  
end
    
    accTest = mean(fmodelacc);
    accCv = mean(Par(:,3));
    fprintf('Final accuracy on the test data set = %.2f%%\n', accTest);
    fprintf('Final accuracy from cross validation set = %.2f%%\n', accCv);   
     
   
    %% feature selection using sequentialfs
    
%     % using the best accuracy parameters for svmtrain
%     [p,q] = find(Par(:,3)==max(Par(:,3)));
%     
%     % use of p(1,1) because max can return 2 similar values of c and g (for 100% accuracy)
%     % defaulting to the first value. In case, there is only one optimum,
%     % that value is chosen.
% 
%     finc = Par(p(1,1),1);
%     fing = Par(p(1,1),2);
%        
%     % -v option is not used for svmtrain as svmpredict requires models (not accuracies which -v option gives)
%     fPar = ['-q -c ', num2str(finc), ' -g ', num2str(fing), ' -b 1'];
%     
%     X_norm
%     svfun = @(xtrain,ytrain,xtest,ytest)sum(svmpredict(ytest,xtest,svmtrain(ytrain,xtrain,fPar)) ~= ytest);
%     [fs history] = sequentialfs(svfun,X_norm,y);    
    
   
    
%% Old kfold cross procedure




% get the indices

%     cv = cvpartition(trainLabel, 'kfold',nfold);          %# Statistics toolbox
%     % prealloting for the indices
%     indices = zeros(size(trainLabel));
%     for i=1:nfold
%         indices(cv.test(i)) = i;
%     end
% %% # N-fold cross-validation testing
%     acc = zeros(nfold,1);
%     for i=1:nfold
%         testIdx = (indices == i)
%         trainIdx = ~testIdx;
%         mdl = libsvmtrain_ova(trainLabel(trainIdx), trainData(trainIdx,:), param);
%         [~,acc(i)] = libsvmpredict_ova(trainLabel(testIdx), trainData(testIdx,:), mdl);
%     end
%     acc = mean(acc);   %# average accuracy
%     
%  fprintf('Cross Validation Accuracy = %.4f%%\n', 100*mean(acc));
%  
%  %# compute final model over the entire dataset
% %mdl = libsvmtrain_ova(labels, data, opts);

%% Old Way
% for j=1:100
%     [train test] = crossvalind('LeaveMOut',N);
%     Xtrain = X_norm([train]);
%     Ytrain = Y([train]);
%     Xtest = X_norm([test]);
%     Ytest = Y([test]);
%     % train the model
%     model = svmtrain(Ytrain,Xtrain,options)
%     % get predictions from the model
%     [label,Pre_acc,prob] = svmpredict(Ytrain,Xtrain,model,'-b 1');
%     % trying to use matlab function for feature selection. Didnt work!! so
%     % using python based libsvm tool deature selection.
%     %wrap = @(Xtrain, Ytrain, Xtest, Ytest)sum(svmpredict(Ytest, Xyest, svmtrain(Ytrain, Xtrain,options) ~= Ytest));
%     %[fs,history] = sequentialfs(@wrap,Xtrain,Ytrain)
%     P_acc(j,1) = Pre_acc(1);
% end
% %P_acc
% %final_acc  = mean(P_acc)
% 
% history
% final_acc  = mean(P_acc)
% % select the best features

