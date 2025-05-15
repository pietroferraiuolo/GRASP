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

#' K-Fold Cross-Validation for Mclust GMM
#'
#' @param data Matrix or data.frame of features (n_samples x n_features)
#' @param G Number of mixture components (or vector of possible G)
#' @param K Number of folds (default: 5)
#' @param ... Additional arguments passed to Mclust
#' @return List with average log-likelihood, BIC, and per-fold results
KFoldGMM <- function(data, K, G = 2, ...) {
  library(mclust)
  n <- nrow(data)
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
    # Predict on test set
    pred <- predict(fit, test_data)
    # Log-likelihood of test set under fitted model
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