library("minpack.lm")

regression <- function(data, method, verb = FALSE) {
  N <- seq(min(data), max(data), length.out = ceiling(1.5 * sqrt(length(data))))
  hist_data <- hist(data, breaks = N, plot = FALSE)
  x <- hist_data$mids  # The midpoint of histogram bins
  y <- hist_data$counts  # The count in each bin
  maxiter <-  nls.lm.control(
    maxiter = 1000,   # Increase iterations to 1000
    ftol = 1e-12,     # Stricter tolerance on function changes (RSS)
    ptol = 1e-10,     # Stricter tolerance on parameter changes
    gtol = 1e-8       # Adjust gradient tolerance
  )
  if (method == "gaussian") {
    # Define Gaussian function
    gaussian <- function(x, a, mean, sd) {
      a * exp(-(x - mean)^2 / (2 * sd^2))
    }
    # Fit the Gaussian function to the histogram data
    fit_gaussian <- nlsLM(y ~ gaussian(x, a, mean, sd),
                          start = list(a = max(y),
                                       mean = mean(x),
                                       sd = sd(x)),
                          control = maxiter,
                          trace = verb)
    out <- predict(fit_gaussian)
    coefficients <- coef(fit_gaussian)
  } else if (method == "boltzmann") {
    # Define the Boltzmann function
    boltzmann <- function(x, A1, A2, x0, dx) {
      (A1 - A2) / (1 + exp((x - x0) / dx)) + A2
    }
    fit_boltzmann <- nlsLM(y ~ boltzmann(x, A1, A2, x0, dx),
                           start = list(A1 = max(y),
                                        A2 = min(y),
                                        x0 = 0,
                                        dx = 1),
                           control = maxiter,
                           trace = verb)
    out <- predict(fit_boltzmann)
    coefficients <- coef(fit_boltzmann)
  } else if (method == "exponential") {
    exponential <- function(x, a, b) {
      a * exp(-b * x)
    }
    fit_exponential <- nlsLM(y ~ exponential(x, a, b),
                             start = list(a = max(y), b = 0),
                             control = maxiter,
                             trace = verb)
    out <- predict(fit_exponential)
    coefficients <- coef(fit_exponential)
  } else if (method == "king") {
    king <- function(v, A, sigma, ve) {
      ifelse(v <= ve,
             A * (exp(-v^2 / (2 * sigma^2)) - exp(-ve^2 / (2 * sigma^2))),
             0)
    }
    fit_king <- nlsLM(y ~ king(x, A, sigma, ve),
                      start = list(A = max(y),
                                   sigma = sd(x),
                                   ve = min(x)),
                      control = maxiter,
                      trace = verb)
    out <- predict(fit_king)
    coefficients <- coef(fit_king)
  } else if (method == "maxwell") {
    maxwell <- function(v, A, sigma) {
      A * v^2 * exp(-v^2 / (2 * sigma^2))
    }
    fit_maxwell <- nlsLM(y ~ maxwell(x, A, sigma),
                         start = list(A = max(y),
                                      sigma = sd(x)),
                         control = maxiter,
                         trace = verb)
    out <- predict(fit_maxwell)
    coefficients <- coef(fit_maxwell)
  } else if (method == "rayleigh") {
    rayleigh <- function(v, A, sigma) {
      A * v * exp(-v^2 / (2 * sigma^2))
    }
    fit_rayleigh <- nlsLM(y ~ rayleigh(x, A, sigma),
                          start = list(A = max(y),
                                       sigma = sd(x)),
                          control = maxiter,
                          trace = verb)
    out <- predict(fit_rayleigh)
    coefficients <- coef(fit_rayleigh)
  } else if (method == "lorentzian") {
    lorentzian <- function(x, A, x0, gamma) {
      A * gamma / (2 * pi) / ((x - x0)^2 + (gamma / 2)^2)
    }
    fit_lorentzian <- nlsLM(y ~ lorentzian(x, A, x0, gamma),
                            start = list(A = max(y),
                                         x0 = mean(x),
                                         gamma = sd(x)),
                            control = maxiter,
                            trace = verb)
    out <- predict(fit_lorentzian)
    coefficients <- coef(fit_lorentzian)
  } else if (method == "power") {
    power <- function(x, a, b) {
      a * x^b
    }
    fit_power <- nlsLM(y ~ power(x, a, b),
                       start = list(a = max(y), b = 1),
                       control = maxiter,
                       trace = verb)
    out <- predict(fit_power)
    coefficients <- coef(fit_power)
  } else if (method == "lognormal") {
    lognormal <- function(x, A, mu, sigma) {
      A / (x * sigma * sqrt(2 * pi)) * exp(-(log(x) - mu)^2 / (2 * sigma^2))
    }
    fit_lognormal <- nlsLM(y ~ lognormal(x, A, mu, sigma),
                           start = list(A = max(y),
                                        mu = mean(x),
                                        sigma = sd(x)),
                           control = maxiter,
                           trace = verb)
    out <- predict(fit_lognormal)
    coefficients <- coef(fit_lognormal)
  } else if (method == "poisson") {
    poisson <- function(x, A, lambda) {
      A * exp(-lambda) * lambda^x / factorial(x)
    }
    fit_poisson <- nlsLM(y ~ poisson(x, A, lambda),
                         start = list(A = max(y),
                                      lambda = mean(x)),
                         control = maxiter,
                         trace = verb)
    out <- predict(fit_poisson)
    coefficients <- coef(fit_poisson)
  } else {
    stop("Unknown method")
  }
  residuals <- y - out
  return(
    list(data = data,
         x = x,
         y = out,
         coeffs = coefficients,
         residuals = residuals,
         kind = method)
  )
}

linear_regression <- function(data, method = "linear", verb = FALSE) {
  # `method` is a dummy argument for the python interface
  fit <- lm(y ~ x, data)
  return(
    list(data = data,
         x = data$x,
         y = fitted(fit),
         coeffs = coef(fit),
         residuals = resid(fit),
         kind = method)
  )
}
