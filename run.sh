###################################################
# Linear model, homogenous errors, varying n
###################################################
python experiments/glm_confints.py \
  --p 40 \
  --level 0.90 \
  --n 100 200 500 1000\
  --n_reps 1000 \
  --gamma 1 \
  --signal 1 \
  --sparsity 10 \
  --lam 0.02 \
  --n_jobs 4 \
  --family linear \
  --error_model homogeneous \
  > logs/glm_confint_$(date +%Y%m%d-%H%M%S).log 2>&1 &

python experiments/plot_coverage.py \
  --fname results/glm_confints_p=40_level=0.9_n=100,200,500,1000_reps=1000_gamma=1.0_s=10.0_lam=0.02_fam=linear_errors=homogeneous_mis=None_cluster_size=None_signal=1_dispersion=1_true_noise_var=False_discrete=False.csv \
  --out figures/linear.png \
  --logx

###################################################
# Logistic model, varying n
###################################################

python experiments/glm_confints.py \
  --p 40 \
  --level 0.90 \
  --n 100 200 500 1000\
  --n_reps 1000 \
  --gamma 1 \
  --sparsity 10 \
  --lam 0.02 \
  --n_jobs 4 \
  --family logistic \
  > logs/glm_confint_$(date +%Y%m%d-%H%M%S).log 2>&1 &

python experiments/plot_coverage.py \
  --fname results/glm_confints_p=40_level=0.9_n=100,200,500,1000_reps=1000_gamma=1.0_s=10.0_lam=0.02_fam=logistic_errors=None_mis=None_cluster_size=None_signal=1_dispersion=1_true_noise_var=False.csv \
  --out figures/logistic_temp.png \
  --ncol 4 \
  --logx

###################################################
# Linear model, clustered errors, varying n
###################################################

python experiments/glm_confints.py \
  --p 40 \
  --level 0.90 \
  --n 100 200 500 1000\
  --n_reps 5000 \
  --gamma 1 \
  --sparsity 10 \
  --lam 0.02 \
  --n_jobs 4 \
  --family linear \
  --error_model clustered \
  --misspecified variance \
  --cluster_size 10 \
  > logs/glm_confint_$(date +%Y%m%d-%H%M%S).log 2>&1 &

python experiments/plot_coverage.py \
  --fname results/glm_confints_p=40_level=0.9_n=100,200,500,1000_reps=5000_gamma=1.0_s=10.0_lam=0.02_fam=linear_errors=clustered_mis=variance_cluster_size=10_signal=1_dispersion=1_true_noise_var=False_discrete=False.csv \
  --out figures/linear_clustered.png \
  --logx


###################################################
# Poisson model, varying n
###################################################

python experiments/glm_confints.py \
  --p 40 \
  --level 0.90 \
  --n 100 200 500 1000 \
  --n_reps 1000 \
  --gamma 1 \
  --sparsity 10 \
  --lam 0.02 \
  --n_jobs 4 \
  --family poisson \
  > logs/glm_confint_$(date +%Y%m%d-%H%M%S).log 2>&1 &

python experiments/plot_coverage.py \
  --fname results/glm_confints_p=40_level=0.9_n=100,200,500,1000_reps=1000_gamma=1.0_s=10.0_lam=0.02_fam=poisson_errors=None_mis=None_cluster_size=None_signal=1_dispersion=1_true_noise_var=False_discrete=False.csv \
  --out figures/poisson.png \
  --logx

###################################################
# Poisson model, varying dispersion
###################################################

python experiments/glm_confints.py \
  --p 40 \
  --level 0.90 \
  --n 500 \
  --n_reps 1000 \
  --gamma 1 \
  --sparsity 10 \
  --lam 0.02 \
  --n_jobs 4 \
  --dispersion 1 2 4 8 10 \
  --family poisson \
  > logs/glm_confint_$(date +%Y%m%d-%H%M%S).log 2>&1 &

python experiments/plot_coverage.py \
  --fname results/glm_confints_p=40_level=0.9_n=500_reps=1000_gamma=1.0_s=10.0_lam=0.02_fam=poisson_errors=None_mis=None_cluster_size=None_signal=1_dispersion=1,2,4,8,10_true_noise_var=False_discrete=False.csv \
  --out figures/poisson_overdispersed.png \
  --x "dispersion" \
  --x_label "Overdispersion" \
  --ncol 3 \
  --exclude "poisson_thinning" \
  --logx


###################################################
# Logistic model in discrete, varying n
###################################################

python experiments/glm_confints.py \
  --p 40 \
  --level 0.90 \
  --n 100 200 500 1000\
  --n_reps 1000 \
  --gamma 1 \
  --sparsity 10 \
  --lam 0.02 \
  --n_jobs 4 \
  --discrete \
  --family logistic \
  > logs/glm_confint_$(date +%Y%m%d-%H%M%S).log 2>&1 &

python experiments/plot_coverage.py \
  --fname results/glm_confints_p=40_level=0.9_n=100,200,500,1000_reps=1000_gamma=1.0_s=10.0_lam=0.02_fam=logistic_errors=None_mis=None_cluster_size=None_signal=1_dispersion=1_true_noise_var=False_discrete=True.csv \
  --out figures/logistic_discrete.png \
  --logx
