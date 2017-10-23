delimiter = ' ';
formatSpec = '%f%f%[^\n\r]';

fileID = fopen('GMD_F17_Train.dat','r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true, 'TextType', 'string',  'ReturnOnError', false);
fclose(fileID);
X_train = table(dataArray{1:end-1}, 'VariableNames', {'e00','e1'});

fileID = fopen('GMD_F17_Valid.dat','r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true, 'TextType', 'string',  'ReturnOnError', false);
fclose(fileID);
X_test = table(dataArray{1:end-1}, 'VariableNames', {'e00','e1'});

clearvars filename delimiter formatSpec fileID dataArray ans;
