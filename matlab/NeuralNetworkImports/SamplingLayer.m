
classdef SamplingLayer < nnet.layer.Layer
    % SamplingLayer   Reparameterization sampling layer for VAE
    % Expects input of size [1 1 2*latentDim N] and outputs [latentDim 1 1 N]
    methods
        function layer = SamplingLayer(name)
            layer.Name = name;
            layer.Description = "Sample z ~ N(mu, sigma^2) using reparameterization";
        end

        function Z = predict(~, X)
            % X size: 1 x 1 x (2*latentDim) x N or (2*latentDim) x N
            sX = size(X);
            if numel(sX) >= 3 && sX(1)==1 && sX(2)==1
                C = sX(3);
                N = sX(4);
                latentDim = C/2;
                mu = reshape(X(:,:,1:latentDim,:), [latentDim, N]);
                logVar = reshape(X(:,:,latentDim+1:end,:), [latentDim, N]);
            else
                % fallback if X is (C x N)
                C = sX(1);
                N = sX(2);
                latentDim = C/2;
                mu = X(1:latentDim, :);
                logVar = X(latentDim+1:end, :);
            end
            sigma = exp(0.5 * logVar);
            eps = randn(size(mu), 'like', mu);
            Z = mu + sigma .* eps;            % size latentDim x N
            Z = reshape(Z, [1 1 latentDim N]); % return 1x1xlatentDimxN for compatibility
        end

        function [Z, memory] = forward(layer, X)
            Z = predict(layer, X);
            memory = [];
        end
    end
end