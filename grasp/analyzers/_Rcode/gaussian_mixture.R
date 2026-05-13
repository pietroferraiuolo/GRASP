library(mclust)

GMMTrainAndFit <- function(
  train_data,
  fit_data,
  G = 2,
  modelNames = "VII",
  nstart = 10,
  maxiter = 1000,
  tol = 1e-6,
  verbose = FALSE
) {
  # Fit the model
  model <- Mclust(train_data,
                  G = G,
                  modelNames = modelNames,
                  nstart = nstart,
                  maxiter = maxiter,
                  tol = tol,
                  verbose = verbose)
  # Predict the cluster membership probabilities
  cluster <- predict.Mclust(object = model, newdata = fit_data)
  return(list(model = model, cluster = cluster))
}

GaussianMixtureModel <- function(
  train_data,
  G = 2,
  modelNames = "VII",
  nstart = 10,
  maxiter = 1000,
  tol = 1e-6,
  verbose = FALSE
) {
  # Fit the model
  model <- Mclust(train_data,
                  G = G,
                  modelNames = modelNames,
                  nstart = nstart,
                  maxiter = maxiter,
                  tol = tol,
                  verbose = verbose)
  # Return the model and its parameters
  return(model)
}

GMMPredict <- function(
  model,
  fit_data
) {
  # Predict the cluster membership probabilities
  cluster <- predict.Mclust(model, fit_data)
  return(cluster)
}

KFoldGMM <- function(data, K, G = 2, ...) {
  #' K-Fold Cross-Validation for Mclust GMM
  #'
  #' @param data Matrix or data.frame of features (n_samples x n_features)
  #' @param G Number of mixture components (or vector of possible G)
  #' @param K Number of folds (default: 5)
  #' @param ... Additional arguments passed to Mclust
  #' @return List with average log-likelihood, BIC, and per-fold results
  #'
  # FIXME: incorrect CV score.
  # The per-fold "log-likelihood" computed below is
  #     sum(log(rowSums(pred$z * fit$parameters$pro)))
  # which mixes ``pred$z`` (already a posterior membership distribution) with
  # the prior weights ``pro``. The correct marginal log-likelihood is
  #     sum_i log sum_k pi_k * N(x_i | mu_k, Sigma_k)
  # i.e. ``sklearn.mixture.GaussianMixture.score(test_data) * n_test``.
  # The Python backend (grasp.analyzers.backends._python.KFoldGMM) does this
  # correctly; this R implementation is kept for parity testing only.
  n <- nrow(data)
  # Shuffle the data along the instance axis
  data <- data[sample(n), , drop = FALSE]
  folds <- sample(rep(1:K, length.out = n))
  logliks <- numeric(K)
  bics <- numeric(K)
  models <- vector("list", K)
  for (k in 1:K) {
    print(paste("Fold", k))
    train_idx <- which(folds != k)
    test_idx <- which(folds == k)
    train_data <- data[train_idx, , drop = FALSE]
    test_data <- data[test_idx, , drop = FALSE]
    fit <- Mclust(train_data, G = G, ...)
    fit$test_data <- test_data
    fit$n_test <- nrow(test_data)
    # Predict on test set
    pred <- predict(fit, test_data)
    # FIXME: see header. This is *not* the test-set log-likelihood; treat
    # the returned mean_loglik with caution and prefer the Python backend.
    logliks[k] <- sum(log(rowSums(pred$z * fit$parameters$pro)))
    bics[k] <- fit$bic
    models[[k]] <- fit
  }
  list(
    mean_loglik = mean(logliks),
    mean_bic = mean(bics),
    logliks = logliks,
    bics = bics,
    models = models
  )
}