[num_records, num_features] = size(X_lrw);
affinity_matrix = zeros(num_records);
distance2_threshold = 1.5 ^2 ; % epsilon ^2
gaussian_param = 2; % sigma ^2
for i=1:num_records
    for j=1:num_records
        distance2 = sum((X_lrw(i,:)-X_lrw(j,:)) .^ 2);
        if distance2 < distance2_threshold
            affinity_matrix(i,j) = exp(-0.5*distance2/gaussian_param);
        else
            affinity_matrix(i,j) = 0;
        end
    end
end

diagD = sum(affinity_matrix);
matD = diag(diagD);
matL = matD - affinity_matrix;
[eigen_vectors, eigen_values] = eig(matL, matD);
eigen_values = diag(eigen_values);
cluster_labels = eigen_vectors(:,2) > 5e-3;
figure;
plot(1:num_records, eigen_values);
figure;
plot(1:num_records, eigen_vectors(:,2));
figure;
gscatter(X_lrw(:,1), X_lrw(:,2), cluster_labels, 'rb');
