library(CVXR)

#' @param X Numeric matrix of predictors (features), with rows as observations and columns as variables.
#' @param Y Numeric response vector.
#' @param l1_lambda Non-negative numeric value. Regularization strength for the L1 penalty.
#' @param affine_penalty Optional numeric vector of length equal to the number of columns in X. If provided, adds an affine penalty term to the coefficients.
#' @param loss Character string, either "logistic" or "squared". Specifies the loss function to use.
#' @param intercept Logical. If TRUE, includes an intercept in the model.
#' @param tol Numeric. Tolerance for the CVXR solver.
#' @param return_est Logical. If TRUE, returns the estimated coefficients instead of active coordinates.
penalized_regression <- function(
    X,
    Y,
    l1_lambda = NULL,
    affine_penalty = NULL,
    loss = "logistic",
    intercept = FALSE,
    tol = 1e-12,
    verbose = FALSE,
    return_est = TRUE) {
    if (intercept) {
        X <- cbind(1, X)
    }
    n <- nrow(X)
    p <- ncol(X)
    beta_cvxr <- Variable(p)

    if (!is.null(l1_lambda) & loss == "squared") {
        sd_Y <- drop(sqrt(var(Y) * (n - 1) / n))
    } else {
        sd_Y = 1
    }

    if (loss == "squared") {
        obj <- sum_squares((Y / sd_Y) - X %*% beta_cvxr)
    } else if (loss == "logistic") {
        # CVXR defines logistic(x) = log(1 + exp(x))
        obj <- sum(-(Y / sd_Y) * (X %*% beta_cvxr) + logistic(X %*% beta_cvxr))
    } else {
        stop("Loss must be 'squared' or 'logistic'")
    }

    if (!is.null(l1_lambda)) {
        if (l1_lambda <= 0) {
            stop("l1_lambda must be non-negative")
        }
        obj <- obj + l1_lambda * norm1(beta_cvxr) / sd_Y
    }
    if (!is.null(affine_penalty)) {
        if (length(affine_penalty) != p) {
            stop("Length of affine_penalty must equal number of columns in X")
        }
        obj <- obj + sum_entries(affine_penalty * beta_cvxr)
    }

    prob <- Problem(Minimize(obj))
    result <- solve(prob, verbose = verbose, ABSTOL = tol, RELTOL = tol, FEASTOL = tol)

    beta_cvxr_est <- drop(result$getValue(beta_cvxr) * sd_Y)
    beta_cvxr_est[abs(beta_cvxr_est) < (tol * 10 * sd_Y)] <- 0

    active_coords <- which(beta_cvxr_est != 0)
    if (return_est) {
        return(beta_cvxr_est)
    } else {
        return(active_coords)
    }
}

