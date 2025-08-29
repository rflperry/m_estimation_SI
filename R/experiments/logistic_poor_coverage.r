set.seed(123)

library(glmnet)

simulate_once <- function(n=200, p=20, s=3, beta_signal=2, alpha=0.05) {
  # true sparse coefficients
  beta_true <- rep(0, p)
  beta_true[1:s] <- beta_signal
  
  # covariates
  X <- matrix(rnorm(n*p), n, p)
  
  # logistic outcomes
  linpred <- X %*% beta_true
  prob <- 1 / (1 + exp(-linpred))
  y <- rbinom(n, 1, prob)
  
  # fit l1 logistic regression with cv
  cvfit <- cv.glmnet(X, y, family="binomial", alpha=1, intercept=FALSE)
  fit <- glmnet(X, y, family="binomial", alpha=1, lambda=cvfit$lambda.min, intercept=FALSE)
  
  selected <- which(coef(fit)[-1] != 0)  # drop intercept

  if (length(selected) == 0) {
    return(NA)  # no selection
  }

  # refit logistic regression on selected variables
  Xsel <- X[, selected, drop=FALSE]
  refit <- glm(y ~ Xsel, family=binomial)

  # Wald CI for first selected true variable
  ci <- confint(refit, level=1-alpha)
  beta_sel <- coef(glm(prob ~ Xsel, family=binomial))

  covered <- as.integer(beta_sel >= ci[,1] & beta_sel <= ci[,2])

  return(covered)
}

# run many reps
n_reps <- 500
results <- replicate(n_reps, simulate_once())

mean(results, na.rm=TRUE)
