library(mclust)

GMMTrainAndFit <- function(
  train_data,
  fit_data,
  n_clusters = 2,
  model_name = "VII",
  n_init = 10,
  max_iter = 1000,
  tol = 1e-6,
  verbose = FALSE
) {
  # Fit the model
  model <- Mclust(train_data,
                  G = n_clusters,
                  modelNames = model_name,
                  nstart = n_init,
                  maxiter = max_iter,
                  tol = tol,
                  verbose = verbose)
  # Predict the cluster membership probabilities
  cluster <- predict.Mclust(model, fit_data)
  return(list(model = model, cluster = cluster))
}

GaussianMixtureModel <- function(
  train_data,
  n_clusters = 2,
  model_name = "VII",
  n_init = 10,
  max_iter = 1000,
  tol = 1e-6,
  verbose = FALSE
) {
  # Fit the model
  model <- Mclust(train_data,
                  G = n_clusters,
                  modelNames = model_name,
                  nstart = n_init,
                  maxiter = max_iter,
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