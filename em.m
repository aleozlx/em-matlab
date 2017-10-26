num_components = 3; % k
[num_records, num_features] = size(X_train); % nd

% parameters initialization
mixture_weights = ones(num_components, 1);
covariance_matrices = reshape(repmat(eye(num_features), 1, num_components), num_features, num_features, num_components);
mean_init = [14,10,-4,-4;6,-1,6,-1];
mean_vectors = mean_init(:, 1:num_components); % dk

% EM algorithm
for num_iterations = 1:10
    % E step
    probs = zeros(num_records, num_components);
    for j=1:num_components
        probs(:,j) = mixture_weights(j, 1) * mvnpdf(X_train, mean_vectors(:, j)', covariance_matrices(:, :, j));
    end
    probs = probs ./ sum(probs, 2); % nk / n1

    % M step
    sum_probs = sum(probs); % 1k
    mean_vectors = X_train' * probs ./ sum_probs; % dot(dn, nk) / 1k
    for j=1:num_components
        X_centered = X_train - mean_vectors(:, j)';
        covariance_matrices(:,:,j) = X_centered' * (X_centered .* probs(:, j)) ./ sum_probs(j); % dot(dn, (nd * n1))
    end
    mixture_weights = probs' ./ num_records;
end
