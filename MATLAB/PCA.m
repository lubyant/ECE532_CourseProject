pred% ECE 532 Machine Learning
% Project
% Source of data Fashion MNIST
% Boyuan Lu

% import the matrix of data
filesName = 'D:\Research\CourseWork\ECE532\Assignment\Project\archive\fashion-mnist_train.csv';
num = xlsread(filesName);
filesName2 = 'D:\Research\CourseWork\ECE532\Assignment\Project\archive\fashion-mnist_test.csv';
num2 = xlsread (filesName2);

% doing a first principle component analysis
% y = num(:,1);
% A = num(:,2:end);
% Ac =A - ones(60000,1) * mean(A); % centralize the data
% Ac = Ac';
% [U,s,V] =svd(Ac,'econ'); % svd the matrix
% Xe = U(:,1)*V(:,1)'*s(1,1);
% figure (1)
% plot (diag(s))
% title("singular value")
% 
% A_dp = Ac * U(:,1:300)*s;

A = num(:,2:end)'; % extracts the features matrix
y = num(:,1); % extracts the label vector
Ac = A; % centerized the data
for i=1:784
    Ac(i,:) = A(i,:) - mean(A);
end

% svd 
[U,s,V] =svd(Ac,'econ'); % svd the matrix
% A_d = U(1:300,:)*s(:,1:300)*V(:,1:300)';

A_d = U(1:300,:)* Ac;

[coeff,score,latent]  = pca(A,'NumComponents',300);
A_d = score'*Ac;
