num_components = 3; % k
[num_records, num_features] = size(X_train); % nd
num_iterations = 10;
covariance_structure = 'full';
mean_init = [ 14, 10, -4, -4;
               6, -1,  6, -1 ]';

% parameters initialization
mixture_weights = ones(num_components, 1);
covariance_matrices = reshape(repmat(eye(num_features), 1, num_components), num_features, num_features, num_components);
mean_vectors = mean_init(1:num_components, :); % kd
log_likelihood = zeros(num_iterations, 1);

% EM algorithm
for iteration = 1:num_iterations
    % E step
    probs = zeros(num_records, num_components);
    for j=1:num_components
        probs(:,j) = mixture_weights(j, 1) * mvnpdf(X_train, mean_vectors(j, :), covariance_matrices(:, :, j));
    end
    sum_probs2 = sum(probs, 2);
    probs = probs ./ sum_probs2; % nk / n1
    log_likelihood(iteration) = sum(log(sum_probs2)); % log likelihood evaluation for previous iteration

    % M step
    sum_probs = sum(probs); % 1k
    mean_vectors = probs' * X_train ./ sum_probs'; % dot(kn, nd) / k1
    switch covariance_structure
    case 'full'
        for j=1:num_components
            X_centered = X_train - mean_vectors(j, :);
            covariance_matrices(:,:,j) = X_centered' * (X_centered .* probs(:, j)) ./ sum_probs(j); % dot(dn, (nd * n1))
        end
    case 'diag'
        covariance_vector = probs' * (X_train .^ 2) ./ sum_probs' - 2 * (mean_vectors .* (probs' * X_train) ./ sum_probs') + mean_vectors .^ 2; % kd
        for j=1:num_components
            covariance_matrices(:,:,j) = diag(covariance_vector(j, :))
        end
    case 'spherical'
        covariance_vector = probs' * (X_train .^ 2) ./ sum_probs' - 2 * (mean_vectors .* (probs' * X_train) ./ sum_probs') + mean_vectors .^ 2; % kd
        for j=1:num_components
            covariance_matrices(:,:,j) = eye(num_features) .* mean(covariance_vector(j, :));
        end
    end
    mixture_weights = sum_probs' ./ num_records;
end
figure;
plot(1:num_iterations,log_likelihood);
title('Log Likelihood over time');

probs = zeros(num_records, num_components);
for j=1:num_components
    probs(:,j) = mixture_weights(j, 1) * mvnpdf(X_train, mean_vectors(j, :), covariance_matrices(:, :, j));
end
[~, labels] = max(probs, [], 2);
figure;
gscatter(X_train(:,1), X_train(:,2), labels, 'rbg');

