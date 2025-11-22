library(glmnet)
library(MASS)
library(sandwich)

###################################
# Simulation study
###################################

# # Fit the lasso
W <- rnorm(n, 0, Y_sd)
fit_lasso <- glmnet(X, Y + W, family = "binomial", lambda = penalty)
sel <- which(coef(fit_lasso)[-1] != 0)

# Confidence inter
fit_glm <- glm((Y - W) ~ X[, sel], family = "binomial")
ci <- confint(fit_glm)

set.seed(123)

p <- 40
n_values <- c(100, 200, 500)#, 1000)
n_reps <- 100
gamma <- 1
signal_frac <- 1
sparsity <- 10 # 20
rho <- 0.5
lam <- 0.1
Y_var <- 1
level <- 0.90

coverage_results <- list()
width_results <- list()

Sigma12 <- (rho ^ abs(outer(1:p, 1:p, "-")))^(1 / 2)

for (n in n_values) {
    coverage_classic <- c()
    coverage_thin_outcomes <- c()
    coverage_sample_split <- c()
    width_classic <- c()
    width_thin_outcomes <- c()
    width_sample_split <- c()

    for (rep in 1:n_reps) {
        # Simulate design matrix X
        X <- matrix(rnorm(n * p, 0, 1), nrow = n, ncol = p) %*% Sigma12
        X <- scale(X, center = FALSE)
        X <- X / sqrt(n)

        # Simulate true coefficients
        beta_true <- rep(0, p)
        signal <- sqrt(signal_frac)# * 2 * log(p))
        # signal <- sparsity # sqrt(signal_frac * 2 * log(p))
        beta_true[1:(sparsity+1)] <- seq(signal, 0, length.out = ceiling(sparsity) + 1)
        beta_true <- beta_true * sample(c(-1, 1), size = p, replace = TRUE)
        beta_true <- beta_true

        # Simulate outcomes Y
        mu <- X %*% beta_true
        # prob <- 1 / (1 + exp(-mu))
        # Y <- rbinom(n, 1, prob)
        # Y_var <- prob * (1 - prob)
        Y <- rnorm(n, 0, sqrt(Y_var)) + mu

        # Lasso regression
        penalty <- lam * sqrt(2 * log(p) / n) * sqrt(Y_var)

        ##### Classical approach #####
        fit_glmnet <- glmnet(X, Y, family = "gaussian", lambda = penalty, standardize = FALSE, intercept = FALSE)
        selected <- which(coef(fit_glmnet)[-1] != 0)

        if(length(selected) > 0) {
            X_sel <- X[, selected]
            fit_glm <- glm(Y ~ X_sel, family = "gaussian")
            se <- sqrt(diag(vcovHC(fit_glm, type = "HC1")))
            ci <- cbind(coef(fit_glm) - qnorm(1 - (1 - level) / 2) * se, coef(fit_glm) + qnorm(1 - (1 - level) / 2) * se)
            beta_sel_true <- coef(glm(mu ~ X_sel, family = "gaussian"))
            coverage <- as.integer(beta_sel_true >= ci[, 1] & beta_sel_true <= ci[, 2])
            coverage_classic <- c(coverage_classic, coverage)
            width_classic <- c(width_classic, mean(ci[, 2] - ci[, 1]))
        }

        ##### Thinning outcomes approach #####
        W <- rnorm(n, 0, sqrt(Y_var))
        Y_train <- Y + gamma * W
        Y_test <- Y - W / gamma
        fit_glmnet <- glmnet(X, Y_train, family = "gaussian", lambda = penalty, standardize=FALSE, intercept=FALSE )
        selected <- which(coef(fit_glmnet)[-1] != 0)

        if(length(selected) > 0) {
            X_sel <- X[, selected]
            fit_glm <- glm(Y_test ~ X_sel, family = "gaussian")
            se <- sqrt(diag(vcovHC(fit_glm, type = "HC1")))
            ci <- cbind(coef(fit_glm) - qnorm(1 - (1 - level) / 2) * se, coef(fit_glm) + qnorm(1 - (1 - level) / 2) * se)
            beta_sel_true <- coef(glm(mu ~ X_sel, family = "gaussian"))
            coverage <- as.integer(beta_sel_true >= ci[, 1] & beta_sel_true <= ci[, 2])
            coverage_thin_outcomes <- c(coverage_thin_outcomes, coverage)
            width_thin_outcomes <- c(width_thin_outcomes, mean(ci[, 2] - ci[, 1]))
        }

        ##### Sample splitting approach #####
        n_half <- n %/% 2
        X1 <- X[1:n_half, ]
        X2 <- X[(n_half + 1):n, ]
        Y1 <- Y[1:n_half]
        Y2 <- Y[(n_half + 1):n]
        mu2 <- mu[(n_half + 1):n]

        fit_glmnet <- glmnet(X1, Y1, family = "gaussian", lambda = penalty, standardize=FALSE, intercept=FALSE )
        selected <- which(coef(fit_glmnet)[-1] != 0)

        if(length(selected) > 0) {
            X_sel <- X2[, selected]
            fit_glm <- glm(Y2 ~ X_sel, family = "gaussian")
            se <- sqrt(diag(vcovHC(fit_glm, type = "HC1")))
            ci <- cbind(coef(fit_glm) - qnorm(1 - (1 - level) / 2) * se, coef(fit_glm) + qnorm(1 - (1 - level) / 2) * se)
            beta_sel_true <- coef(glm(mu2 ~ X_sel, family = "gaussian"))
            coverage <- as.integer(beta_sel_true >= ci[, 1] & beta_sel_true <= ci[, 2])
            coverage_sample_split <- c(coverage_sample_split, coverage)
            width_sample_split <- c(width_sample_split, mean(ci[, 2] - ci[, 1]))
        }
    }

    # Average coverage for each parameter
    coverage_results[[as.character(n)]] <- list(
        classical = mean(coverage_classic),
        outcomes = mean(coverage_thin_outcomes),
        sample_split = mean(coverage_sample_split)
    )
    width_results[[as.character(n)]] <- list(
        classical = median(width_classic),
        outcomes = median(width_thin_outcomes),
        sample_split = median(width_sample_split)
    )
}

print(coverage_results)
print(width_results)
