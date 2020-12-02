% ECE 532 Machine Learning
% Project
% Source of data Fashion MNIST
% Boyuan Lu

% import the matrix of data
filesName = 'D:\Research\CourseWork\ECE532\Assignment\Project\archive\fashion-mnist_train.csv';
num = xlsread(filesName);
filesName2 = 'D:\Research\CourseWork\ECE532\Assignment\Project\archive\fashion-mnist_test.csv';
num2 = xlsread (filesName2);

% doing a first principle component analysis
y = num(:,1);
A = num(:,2:end);
Ac =A - ones(60000,1) * mean(A); % centralize the data
[U,s,V] =svd(Ac,'econ'); % svd the matrix
PCA = U(:,1)*V(:,1)'*s(1,1);
figure (1)
plot (s)
title("singular value")



% doing a least square problem
y = num(:,1);
A = num(:,2:end);
w = inv(A'*A) * A' * y;
y2 = num2(:,1);
A2 = num2(:,2:end);
y_v = round(A2*w);
erRate1 = 1- sum(y_v == y2)/length(y_v);


% doing a ridge regression using decendent gradient
[U,s,V] = svd(A,'econ');
tau = 0.5*1/s(1,1).^2;
lam = 2;
w0 = ones(784,1);
w = zeros(784,1);
z = zeros(784,1);
while (norm(w-w0)> 1e-6)
    w = w0;
    z = w0 - tau * A' * (A * w0 - y);
    w0 = z / (1 + lam * tau);
end 
y_v = round(A2*w0);
erRate2 = 1- sum(y_v == y2)/length(y_v);


% doing a 10-fold CV
% setindices =[1 6000;6001 12000;12001 18000;18001 24000;24001 30000;30001 36000;36001 42000;42001 48000;48001 54000;54001 60000];
% holdoutindices = [1 2;2 3; 3 4; 4 5;5 6; 7 8; 9 10; 10 1];
% cases = length(holdoutindices);
% for j =1:cases
%     v1_ind = A
% end

cvp = cvpartition(60000,'KFold',6);
% doing a KNN classification
for i = 1:10
mdl = fitcknn(A,y,'NumNeighbor',i,'distance','euclidean','Standardize',1);
label = predict(mdl,A2);
err1(i) = 1- sum(label == y2)/length(label);
end

figure
plot(i,err1)
xlabel("K-value")
ylabel("error rate")
title("Euclidean Distance")
for i = 1:10
mdl = fitcknn(A,y,'NumNeighbor',i,'distance','cityblock','Standardize',1);
label = predict(mdl,A2);
err2(i) = 1- sum(label == y2)/length(label);
end
for i = 1:10
mdl = fitcknn(A,y,'NumNeighbor',i,'distance','correlation','Standardize',1);
label = predict(mdl,A2);
err3(i) = 1- sum(label == y2)/length(label);
end
for i = 1:10
mdl = fitcknn(A,y,'NumNeighbor',i,'distance','minkowski','Standardize',1);
label = predict(mdl,A2);
err4(i) = 1- sum(label == y2)/length(label);
end
figure
plot(i,err1,i,err2,i,err3,i,err4)
xlabel("K-value")
ylabel("error rate")
legend("Euclidian","Cityblock","correlation","Minkowski")