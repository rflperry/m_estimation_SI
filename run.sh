# Debugging
# python experiments/glm_confints.py \
#   --p 40 \
#   --level 0.90 \
#   --n 100\
#   --n_reps 1 \
#   --gamma 1 \
#   --sparsity 10 \
#   --lam 1 \
#   --verbose \
#   --debug \
#   --error_model heterogeneous \
#   --family linear

# Linear
python experiments/glm_confints.py \
  --p 40 \
  --level 0.90 \
  --n 50 100 200\
  --n_reps 100 \
  --gamma 1 \
  --sparsity 5 \
  --lam 5 \
  --n_jobs 4 \
  --family linear \
  --error_model homogeneous \
  > logs/glm_confint_$(date +%Y%m%d-%H%M%S).log 2>&1 &

python experiments/glm_confints.py \
  --p 40 \
  --level 0.90 \
  --n 50 100 200\
  --n_reps 100 \
  --gamma 1 \
  --sparsity 5 \
  --lam 5 \
  --n_jobs 4 \
  --family linear \
  --error_model homogeneous \
  --true_noise_var \
  > logs/glm_confint_$(date +%Y%m%d-%H%M%S).log 2>&1 &
