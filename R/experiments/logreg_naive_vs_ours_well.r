source("selection.R")
source("inference.R")

###################################
# Define functions
###################################

logit <- function(x) 1 / (1 + exp(-x))

selective_conf_int <- function(X, Y, W, gamma, penalty, Y_var, noise = "outcome", loss = "logistic") {
    n <- nrow(X)

    if (noise == "outcome") {
        selected <- penalized_regression(X, Y + gamma * W, loss = loss, l1_lambda = penalty, return_est = FALSE)
    } else if (noise == "gradient") {
        selected <- penalized_regression(X, Y, loss = loss, l1_lambda = penalty, affine_penalty = gamma * W, return_est = FALSE)
    } else {
        stop("noise must be 'outcome' or 'gradient'")
    }

    if (length(selected) == 0) {
        return(list(selected, NULL))
    }

    # Fit regression on selected variables
    X_selected <- X[, selected]
    if (noise == "outcome") {
        beta_hat <- fit_glm(X_selected, Y - W / gamma, loss = loss)
    } else if (noise == "gradient") {
        W_selected <- W[selected]
        beta_hat <- fit_glm(X_selected, Y, loss = loss, affine_penalty = -W_selected / gamma)
    }

    # Compute confidence intervals using true variance of Y
    # Calculate Fisher information using true variance
    Sigma <- t(X_selected) %*% diag(Y_var) %*% X_selected / n
    Y_var_est <- as.vector(logit(X_selected %*% beta_hat) * (1 - logit(X_selected %*% beta_hat)))
    H_est_inv <- solve(t(X_selected) %*% diag(Y_var_est) %*% X_selected / n)
    cov_beta_hat <- H_est_inv %*% Sigma %*% H_est_inv / n
    se_beta_hat <- sqrt(diag(cov_beta_hat))
    # 95% CI
    conf_int <- cbind(beta_hat - 1.96 * se_beta_hat, beta_hat + 1.96 * se_beta_hat)

    return(list(selected, conf_int))
}

###################################
# Simulation study
###################################

set.seed(123)

p <- 20
n_values <- c(50, 100, 500)#, 1000)
n_reps <- 100
gamma <- 1
signal_frac <- 0.1
sparse_frac <- 0.2

results <- list()

for (n in n_values) {
    coverage_classic <- c()
    coverage_thin_outcomes <- c()
    coverage_thin_gradient <- c()

    for (rep in 1:n_reps) {
        # Simulate design matrix X
        X <- matrix(runif(n * p, 0, 1), nrow = n, ncol = p)
        X <- scale(X)

        # Simulate true coefficients
        beta_true <- rep(0, p)
        signal = sqrt(signal_frac * 2 * log(p))
        beta_true[1:(ceiling(sparse_frac * p)+1)] <- seq(signal, 0, length.out = ceiling(sparse_frac * p) + 1)

        # Simulate Bernoulli outcomes Y
        eta <- X %*% beta_true
        prob <- logit(eta)
        Y <- rbinom(n, 1, prob)
        Y_var_true <- as.vector(prob * (1 - prob))

        # Lasso logistic regression
        penalty <- 10*n^(-1 / 2) * sqrt(log(p))

        ##### Classical approach #####
        sci <- selective_conf_int(X, Y, 0, 1, penalty, Y_var_true)
        selected <- sci[[1]]
        conf_int <- sci[[2]]
        if (is.null(conf_int)) {
            next
        }

        # True coefficients for selected variables
        beta_selected <- beta_true[selected]

        # Coverage: check if true beta falls within CI
        coverage <- (beta_selected >= conf_int[, 1]) & (beta_selected <= conf_int[, 2])
        coverage_classic <- c(coverage_classic, coverage)

        ##### Thinning outcomes approach approach #####
        W <- rnorm(n, 0, Y_var_true)

        sci <- selective_conf_int(X, Y, W, gamma, penalty, (1 + 1 / gamma^2) * Y_var_true, noise = "outcome")
        selected <- sci[[1]]
        conf_int <- sci[[2]]
        if (is.null(conf_int)) {
            next
        }

        beta_selected <- beta_true[selected]
        coverage <- (beta_selected >= conf_int[, 1]) & (beta_selected <= conf_int[, 2])
        coverage_thin_outcomes <- c(coverage_thin_outcomes, coverage)

        ##### Thinning gradient approach approach #####
        W <- as.vector(t(X) %*% rnorm(n, 0, Y_var_true)) / sqrt(n)

        sci <- selective_conf_int(X, Y, W, gamma, penalty, (1 + 1 / gamma^2) * Y_var_true, noise = "gradient")
        selected <- sci[[1]]
        conf_int <- sci[[2]]
        if (is.null(conf_int)) {
            next
        }

        beta_selected <- beta_true[selected]
        coverage <- (beta_selected >= conf_int[, 1]) & (beta_selected <= conf_int[, 2])
        coverage_thin_gradient <- c(coverage_thin_gradient, coverage)
    }

    # Average coverage for each parameter
    results[[as.character(n)]] <- list(
        classical = mean(coverage_classic),
        outcomes = mean(coverage_thin_outcomes),
        gradient = mean(coverage_thin_gradient)
    )
}


