# Debugging
python experiments/glm_confints.py \
  --p 40 \
  --level 0.90 \
  --n 200 \
  --n_reps 1 \
  --gamma 1 \
  --sparsity 10 \
  --lam 0.02 \
  --verbose \
  --debug \
  --family poisson

python experiments/glm_confints.py \
  --p 40 \
  --level 0.90 \
  --n 200\
  --n_reps 1 \
  --gamma 1 \
  --sparsity 10 \
  --lam 0.02 \
  --family negative_binomial \
  --debug \
  --dispersion 2

# Debugging
python experiments/glm_confints.py \
  --p 40 \
  --level 0.90 \
  --n 100 \
  --n_reps 1 \
  --gamma 1 \
  --sparsity 10 \
  --signal 1 \
  --lam 0.02 \
  --verbose \
  --debug \
  --family linear \
  --error_model clustered \
  --cluster_size 5



###################################################
# Linear model, homogenous errors, varying n
###################################################
python experiments/glm_confints.py \
  --p 40 \
  --level 0.90 \
  --n 50 100 200 500 1000\
  --n_reps 500 \
  --gamma 1 \
  --sparsity 10 \
  --lam 0.02 \
  --n_jobs 4 \
  --family linear \
  --error_model homogeneous \
  > logs/glm_confint_$(date +%Y%m%d-%H%M%S).log 2>&1 &

python experiments/glm_confints.py \
  --p 40 \
  --level 0.90 \
  --n 50 100 200 500 1000\
  --n_reps 500 \
  --gamma 1 \
  --sparsity 10 \
  --lam 0.02 \
  --n_jobs 4 \
  --family linear \
  --error_model homogeneous \
  --true_noise_var \
  > logs/glm_confint_$(date +%Y%m%d-%H%M%S).log 2>&1 &

python experiments/plot_coverage.py \
  --fname results/glm_confints_p=40_level=0.9_n=50,100,200,500,1000_reps=500_gamma=1.0_s=10.0_lam=0.02_fam=linear_errors=homogeneous_mis=None_cluster_size=None_dispersion=1_true_noise_var=False.csv \
  --ref results/glm_confints_p=40_level=0.9_n=50,100,200,500,1000_reps=500_gamma=1.0_s=10.0_lam=0.02_fam=linear_errors=homogeneous_mis=None_cluster_size=None_dispersion=1_true_noise_var=True.csv \
  --out figures/linear_homogeneous_with_bias.png


###################################################
# Linear model, homogenous errors, missepcified, varying n
###################################################
python experiments/glm_confints.py \
  --p 40 \
  --level 0.90 \
  --n 50 100 200 500\
  --n_reps 100 \
  --gamma 1 \
  --sparsity 10 \
  --lam 0.02 \
  --n_jobs 4 \
  --family linear \
  --misspecified \
  --error_model homogeneous \
  > logs/glm_confint_$(date +%Y%m%d-%H%M%S).log 2>&1 &

python experiments/glm_confints.py \
  --p 40 \
  --level 0.90 \
  --n 50 100 200 500\
  --n_reps 100 \
  --gamma 1 \
  --sparsity 10 \
  --lam 0.02 \
  --n_jobs 4 \
  --family linear \
  --misspecified \
  --error_model homogeneous \
  --true_noise_var \
  > logs/glm_confint_$(date +%Y%m%d-%H%M%S).log 2>&1 &

python experiments/plot_coverage.py \
  --fname results/glm_confints_p=40_level=0.9_n=50,100,200,500_reps=100_gamma=1.0_s=10.0_lam=0.02_fam=linear_errors=homogeneous_mis=True_dispersion=1_true_noise_var=False.csv \
  --ref results/glm_confints_p=40_level=0.9_n=50,100,200,500_reps=100_gamma=1.0_s=10.0_lam=0.02_fam=linear_errors=homogeneous_mis=True_dispersion=1_true_noise_var=True.csv \
  --out figures/linear_homogeneous_misspecified_mean.png

###################################################
# Linear model, correlated errors, varying n
###################################################


###################################################
# Linear model, homogenous, varying p
###################################################

python experiments/glm_confints.py \
  --p 50 100 200 500 \
  --level 0.90 \
  --n 100\
  --n_reps 500 \
  --gamma 1 \
  --sparsity 10 \
  --lam 5 \
  --n_jobs 4 \
  --family linear \
  --error_model homogeneous \
  > logs/glm_confint_$(date +%Y%m%d-%H%M%S).log 2>&1 &

python experiments/glm_confints.py \
  --p 50 100 200 500 \
  --level 0.90 \
  --n 100 \
  --n_reps 500 \
  --gamma 1 \
  --sparsity 10 \
  --lam 5 \
  --n_jobs 4 \
  --family linear \
  --error_model homogeneous \
  --true_noise_var \
  > logs/glm_confint_$(date +%Y%m%d-%H%M%S).log 2>&1 &

python experiments/plot_coverage.py \
  --fname results/glm_confints_p=50,100,200,500_level=0.9_n=100_reps=500_gamma=1.0_s=10.0_lam=5.0_fam=linear_errors=homogeneous_true_noise_var=False.csv \
  --ref results/glm_confints_p=50,100,200,500_level=0.9_n=100_reps=500_gamma=1.0_s=10.0_lam=5.0_fam=linear_errors=homogeneous_true_noise_var=True.csv \
  --out figures/linear_homogeneous_high_dim.png \
  --x p \
  --x_label "# features (p)"

###################################################
# Linear model, homogenous, varying sparsity
###################################################


###################################################
# Linear model, misspecified mean with laplace noise
###################################################

python experiments/glm_confints.py \
  --p 40 \
  --level 0.90 \
  --n 200\
  --n_reps 100 \
  --gamma 1 \
  --sparsity 10 \
  --lam 0.02 \
  --n_jobs 4 \
  --family linear \
  --error_model homogeneous \
  --misspecified \
  --dispersion 1 9 25 \
  > logs/glm_confint_$(date +%Y%m%d-%H%M%S).log 2>&1 &

python experiments/glm_confints.py \
  --p 40 \
  --level 0.90 \
  --n 200 \
  --n_reps 100 \
  --gamma 1 \
  --sparsity 10 \
  --lam 0.02 \
  --n_jobs 4 \
  --family linear \
  --error_model homogeneous \
  --misspecified \
  --dispersion 1 9 25 \
  --true_noise_var \
  > logs/glm_confint_$(date +%Y%m%d-%H%M%S).log 2>&1 &

python experiments/plot_coverage.py \
  --fname results/glm_confints_p=40_level=0.9_n=200_reps=100_gamma=1.0_s=10.0_lam=0.02_fam=linear_errors=homogeneous_mis=True_dispersion=1,9,25_true_noise_var=False.csv \
  --ref results/glm_confints_p=40_level=0.9_n=200_reps=100_gamma=1.0_s=10.0_lam=0.02_fam=linear_errors=homogeneous_mis=True_dispersion=1,9,25_true_noise_var=True.csv \
  --out figures/linear_homogeneous_misspecified_mean.png \
  --x dispersion \
  --x_label "Noise variance" \
  --logx

###################################################
# Logistic model, varying n
###################################################

python experiments/glm_confints.py \
  --p 40 \
  --level 0.90 \
  --n 100 200 500 1000\
  --n_reps 500 \
  --gamma 1 \
  --sparsity 10 \
  --lam 0.02 \
  --n_jobs 4 \
  --family logistic \
  > logs/glm_confint_$(date +%Y%m%d-%H%M%S).log 2>&1 &

python experiments/glm_confints.py \
  --p 40 \
  --level 0.90 \
  --n 100 200 500 1000\
  --n_reps 500 \
  --gamma 1 \
  --sparsity 10 \
  --lam 0.02 \
  --n_jobs 4 \
  --family logistic \
  --true_noise_var \
  > logs/glm_confint_$(date +%Y%m%d-%H%M%S).log 2>&1 &

python experiments/plot_coverage.py \
  --fname results/glm_confints_p=40_level=0.9_n=100,200,500,1000_reps=500_gamma=1.0_s=10.0_lam=0.02_fam=logistic_errors=None_mis=None_cluster_size=None_dispersion=1_true_noise_var=False.csv \
  --ref results/glm_confints_p=40_level=0.9_n=100,200,500,1000_reps=500_gamma=1.0_s=10.0_lam=0.02_fam=logistic_errors=None_mis=None_cluster_size=None_dispersion=1_true_noise_var=True.csv \
  --out figures/logistic_temp.png


###################################################
# Logistic model, varying p
###################################################

python experiments/glm_confints.py \
  --p 50 100 200 500 \
  --level 0.90 \
  --n 100\
  --n_reps 100 \
  --gamma 1 \
  --sparsity 10 \
  --lam 5 \
  --n_jobs 4 \
  --family logistic \
  > logs/glm_confint_$(date +%Y%m%d-%H%M%S).log 2>&1 &

python experiments/glm_confints.py \
  --p 50 100 200 500 \
  --level 0.90 \
  --n 100 \
  --n_reps 100 \
  --gamma 1 \
  --sparsity 10 \
  --lam 5 \
  --n_jobs 4 \
  --family logistic \
  --true_noise_var \
  > logs/glm_confint_$(date +%Y%m%d-%H%M%S).log 2>&1 &

python experiments/plot_coverage.py \
  --fname results/glm_confints_p=50,100,200,500_level=0.9_n=100_reps=100_gamma=1.0_s=10.0_lam=5.0_fam=logistic_errors=None_true_noise_var=False.csv \
  --ref results/glm_confints_p=50,100,200,500_level=0.9_n=100_reps=100_gamma=1.0_s=10.0_lam=5.0_fam=logistic_errors=None_true_noise_var=True.csv \
  --out figures/logistic_high_dim.png \
  --x p \
  --x_label "# features (p)"


###################################################
# Linear model, clustered errors, varying n, misspecified variance
###################################################

python experiments/glm_confints.py \
  --p 40 \
  --level 0.90 \
  --n 100 200 500 1000\
  --n_reps 500 \
  --gamma 1 \
  --sparsity 10 \
  --lam 0.02 \
  --n_jobs 4 \
  --family linear \
  --error_model clustered \
  --misspecified variance \
  --cluster_size 20 \
  > logs/glm_confint_$(date +%Y%m%d-%H%M%S).log 2>&1 &

python experiments/glm_confints.py \
  --p 40 \
  --level 0.90 \
  --n 100 200 500 1000 \
  --n_reps 500 \
  --gamma 1 \
  --sparsity 10 \
  --lam 0.02 \
  --n_jobs 4 \
  --family linear \
  --error_model clustered \
  --misspecified variance \
  --cluster_size 20 \
  --true_noise_var \
  > logs/glm_confint_$(date +%Y%m%d-%H%M%S).log 2>&1 &

python experiments/plot_coverage.py \
  --fname results/glm_confints_p=40_level=0.9_n=100,200,500,1000_reps=500_gamma=1.0_s=10.0_lam=0.02_fam=linear_errors=clustered_mis=Variance_cluster_size=20_dispersion=1_true_noise_var=False.csv \
  --ref results/glm_confints_p=40_level=0.9_n=100,200,500,1000_reps=500_gamma=1.0_s=10.0_lam=0.02_fam=linear_errors=clustered_mis=Variance_cluster_size=20_dispersion=1_true_noise_var=True.csv \
  --out figures/linear_clustered_equi.png

python experiments/plot_coverage.py \
  --fname results/glm_confints_p=40_level=0.9_n=100,200,500,1000_reps=500_gamma=1.0_s=10.0_lam=0.02_fam=linear_errors=clustered_mis=Variance_cluster_size=20_dispersion=1_true_noise_var=False.csv \
  --out figures/linear_clustered_equi.png \
  --exclude rsc

###################################################
# Linear model, clustered errors, varying cluster_size
###################################################

python experiments/glm_confints.py \
  --p 40 \
  --level 0.90 \
  --n 100 \
  --n_reps 100 \
  --gamma 1 \
  --sparsity 10 \
  --lam 0.02 \
  --n_jobs 4 \
  --family linear \
  --error_model clustered \
  --cluster_size 2 5 10 20 \
  > logs/glm_confint_$(date +%Y%m%d-%H%M%S).log 2>&1 &

python experiments/glm_confints.py \
  --p 40 \
  --level 0.90 \
  --n 100 \
  --n_reps 100 \
  --gamma 1 \
  --sparsity 10 \
  --lam 0.02 \
  --n_jobs 4 \
  --family linear \
  --error_model clustered \
  --cluster_size 2 5 10 20 \
  --true_noise_var \
  > logs/glm_confint_$(date +%Y%m%d-%H%M%S).log 2>&1 &

python experiments/plot_coverage.py \
  --fname results/glm_confints_p=40_level=0.9_n=100_reps=100_gamma=1.0_s=10.0_lam=0.02_fam=linear_errors=clustered_mis=None_cluster_size=2,5,10,20_dispersion=1_true_noise_var=False.csv \
  --ref results/glm_confints_p=40_level=0.9_n=100_reps=100_gamma=1.0_s=10.0_lam=0.02_fam=linear_errors=clustered_mis=None_cluster_size=2,5,10,20_dispersion=1_true_noise_var=True.csv \
  --x_label "Cluster size" \
  --x "cluster_size" \
  --out figures/linear_clustered_varied.png