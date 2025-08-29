library(testthat)
source("../../selection.R")
# devtools::load_all()

set.seed(123)
n <- 50
p <- 5
X <- matrix(runif(n * p, 0, 1), nrow = n, ncol = p)
Y <- rbinom(n, 1, 0.5)
W <- rnorm(n, 0, 0.5^2)

test_that("penalized_regression runs without penalties", {
  result <- penalized_regression(X, Y, l1_lambda = NULL, affine_penalty = NULL, loss = "logistic")
  expect_type(result, "double")
  expect_length(result, ncol(X))
})

test_that("penalized_regression runs without penalties, noisy case", {
  result <- penalized_regression(X, Y + W, l1_lambda = NULL, affine_penalty = NULL, loss = "logistic")
  expect_type(result, "double")
  expect_length(result, ncol(X))
})

test_that("penalized_regression runs with L1 penalty", {
  result <- penalized_regression(X, Y + W, l1_lambda = 0.1, affine_penalty = NULL, loss = "logistic")
  expect_type(result, "double")
  expect_length(result, ncol(X))
})

test_that("penalized_regression runs with affine penalty", {
  W_affine <- t(X) %*% W / sqrt(n)
  result <- penalized_regression(X, Y, l1_lambda = NULL, affine_penalty = W_affine, loss = "logistic")
  expect_type(result, "double")
  expect_length(result, ncol(X))
})

test_that("penalized_regression runs with both L1 and affine penalties", {
  W_affine <- t(X) %*% W / sqrt(n)
  result <- penalized_regression(X, Y, l1_lambda = 0.1, affine_penalty = W_affine, loss = "logistic")
  expect_type(result, "double")
  expect_length(result, ncol(X))
})