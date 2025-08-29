source("selection.R")

#' Estimates the coefficients of a Generalized Linear Model (GLM) regression.
#'
#' This function fits a GLM to the provided data using L1 and affine penalized regression.
#' The Latter is for valid post-selection inference via score-thinning.
#'
#' @param X A numeric matrix of predictors.
#' @param Y A numeric vector of responses.
#' @param family A family object specifying the canonical GLM type.
#' @param affine_penalty Optional affine penalty matrix for regularization (default is NULL).
#' @param tol Numeric tolerance for convergence (default is 1e-12).
#' @param intercept Logical; whether to include an intercept in the model (default is TRUE).
#' @param verbose Logical; whether to print progress messages (default is FALSE).
#'
#' @return A numeric vector of estimated GLM coefficients.
#' @export
fit_glm <- function(X, Y, loss, affine_penalty = NULL, tol = 1e-12, intercept = FALSE, verbose = FALSE) {
    # if (family == "binomial") {
    #     loss <- "logistic"
    # } else if (family == "gaussian") {
    #     loss <- "squared"
    # } else {
    #     stop("Family not supported")
    # }

    beta_est <- penalized_regression(
        X = X,
        Y = Y,
        l1_lambda = NULL,
        affine_penalty = affine_penalty,
        loss = loss,
        intercept = intercept,
        tol = tol,
        verbose = verbose,
        return_est = FALSE
    )

    return(beta_est)
}