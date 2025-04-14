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