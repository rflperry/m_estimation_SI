p <- 19
n <- 100
logit <- function(x) 1 / (1 + exp(-x))

# Simulate design matrix X
X <- matrix(runif(n * p, 0, 1), nrow = n, ncol = p)

# Simulate true coefficients
beta_true <- rnorm(p, 0, 1)

# Simulate Bernoulli outcomes Y
eta <- X %*% beta_true
prob <- logit(eta)
Y <- rbinom(n, 1, prob)
Y_var_true <- as.vector(prob * (1 - prob))

logistic_loss <- function(beta, X, Y) {
    eta <- X %*% beta
    log_likelihood <- Y * eta - log(1 + exp(eta))
    return(-sum(log_likelihood))
}

fit_logistic_regression <- function(X, Y, beta_init = NULL) {
    if (is.null(beta_init)) beta_init <- rep(0, ncol(X))
    opt <- optim(beta_init, logistic_loss, X = X, Y = Y, method = "BFGS")
    return(opt$par)
}




# Define CVXR variable


# Non-noisy 
beta_est <- fit_logistic_regression(X, Y)

# Noise
beta_noisy <- fit_logistic_regression(X, Y + rnorm(n, 0, Y_var_true))



glm(Y ~ X, family = binomial(make.link("logit")))

fam <- binomial(make.link("logit"))


