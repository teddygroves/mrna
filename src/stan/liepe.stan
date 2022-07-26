/* Model that attempts to copy the one in Liepe et al */
functions {
#include custom_functions.stan
}
data {
  int<lower=1> N;
  int<lower=1> N_train;
  int<lower=1> N_test;
  array[2] real sigma_loc_and_scale;
  vector[4] p_upper_bounds;
  vector[N] y_m;
  array[N] real sim_times;
  array[N_train] int<lower=1,upper=N> ix_train;
  array[N_test] int<lower=1,upper=N> ix_test;
  int<lower=0,upper=1> likelihood;
}
transformed data {
  matrix[3, 7] S = [[-1, 0, 1, 0, 0,-1,-1],
                    [ 1,-1, 0, 0,-1, 0, 0],
                    [ 0, 1, 0,-1, 0, 0, 0]];
  vector[3] conc_init = [10, 5, 0]';
  real t_init = 0;
  real abstol = 1e-12;
  real reltol = 1e-12;
  int steps = 1000000;
}
parameters {
  real<lower=0> sigma;
  vector<lower=0>[4] p_unknown;
}
transformed parameters {
  vector[5] p = append_row(p_unknown, 1);
  array[N] vector[3] conc =
    ode_bdf_tol(dPdt, conc_init, t_init, sim_times, abstol, reltol, steps, S, p);
}
model {
  sigma ~ lognormal(sigma_loc_and_scale[1], sigma_loc_and_scale[2]);
  p_unknown ~ uniform(rep_vector(0, 4), p_upper_bounds);
  if (likelihood){
    y_m[ix_train] ~ lognormal(log(conc[ix_train, 1]), sigma);
  }
}
generated quantities {
  vector[N_test] yrep;
  vector[N_test] llik;
  for (n in 1:N_test){
    if (conc[ix_test[n]][1] > 0){
      yrep[n] = lognormal_rng(log(conc[ix_test[n]][1]), sigma);
      llik[n] = lognormal_lpdf(y_m[ix_test[n]] | log(conc[ix_test[n]][1]), sigma);
    }
  }
}
