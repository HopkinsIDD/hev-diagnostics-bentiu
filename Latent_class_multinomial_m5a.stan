data {
  int<lower=1> N; // Number of individuals
  int<lower=1> N_obs; // Number of test result combinations
  matrix[N,3] observed_results_pos; // Test results. Forced all missing RDT to be pos
  array[N,N_obs] int observed_results_array_pos; // Test profiles. Forced all missing RDT to be pos
  matrix[N,3] observed_results_neg; // Test results. Forced all missing RDT to be neg
  array[N,N_obs] int observed_results_array_neg; // Test profiles. Forced all missing RDT to be neg
  int<lower=0, upper=1> indicat[N];  
  matrix[N, 3] spline_basis; // Spline basis for days since jaundice onset
  matrix[N,1] days_since_onset; // Days since jaundice onset group of each individual
  real prior_hev_missing; // prior probabilty of having HEV for those with missing RDTs
  }
  

parameters {
  real logit_phi;     // True infection probability 
  array[3] real intercept_sens; //Intercept for sensitivity regression for each test. 
  array[2] real <upper=0> onset_coef_sens; // Coefficients for days_since_onset on sensitivity for each test
  vector <upper=0> [3]  spline_coef_sens;       // Coefficients for spline terms on sensitivity for each test
  array[3] real <lower=0> logit_spec; 
}

transformed parameters {
  array[N, N_obs] real<lower=0, upper=1> p; // Probabilities for each test result combination for each individual
  real phi = inv_logit(logit_phi); // True infection probability
  array[N,3] real logit_sens; 
  array[N,3] real<lower=0> sens_all;
  array[3] real<lower=0, upper=1> spec_all = inv_logit(logit_spec);


  // Compute sensitivities for each individual for each test
  for (i in 1:N) {
    // PCR uses spline basis
    logit_sens[i, 1] = intercept_sens[1] + dot_product(spline_basis[i], spline_coef_sens);
    sens_all[i, 1] = inv_logit(logit_sens[i, 1]);

    // RDT use days_since_onset
    logit_sens[i, 2] = intercept_sens[2] + onset_coef_sens[1] * days_since_onset[i, 1];
    sens_all[i, 2] = inv_logit(logit_sens[i, 2]);

    //  ELISA use days_since_onset
    logit_sens[i, 3] = intercept_sens[3] + onset_coef_sens[2] * days_since_onset[i, 1];
    sens_all[i, 3] = inv_logit(logit_sens[i, 3]);
  }
  // Loop over individuals
  for (i in 1:N) {
    row_vector[3] sens = [sens_all[i][1], sens_all[i][2], sens_all[i][3]];
    row_vector[3] spec = [spec_all[1], spec_all[2], spec_all[3]];
    
    // Calculate probabilities for each test result combination for the individual
    p[i, 1] = prod(1-sens) * phi + prod(spec) * (1 - phi);
    p[i, 2] = (1-sens[1]) * (1-sens[2]) * (sens[3]) * phi + (spec[1]) * (spec[2]) * (1-spec[3]) * (1 - phi);
    p[i, 3] = (1-sens[1]) * (sens[2]) * (1-sens[3]) * phi + (spec[1]) * (1-spec[2]) * (spec[3]) * (1 - phi);
    p[i, 4] = (1-sens[1]) * (sens[2]) * (sens[3]) * phi + (spec[1]) * (1-spec[2]) * (1-spec[3]) * (1 - phi);
    p[i, 5] = (sens[1]) * (1-sens[2]) * (1-sens[3]) * phi + (1-spec[1]) * (spec[2]) * (spec[3]) * (1 - phi);
    p[i, 6] = (sens[1]) * (1-sens[2]) * (sens[3]) * phi + (1-spec[1]) * (spec[2]) * (1-spec[3]) * (1 - phi);
    p[i, 7] = (sens[1]) * (sens[2]) * (1-sens[3]) * phi + (1-spec[1]) * (1-spec[2]) * (spec[3]) * (1 - phi);
    p[i, 8] = prod(sens) * phi + prod(1-spec) * (1 - phi);
  }
}
  
model {
 // Define the priors for intercept and coefficients 
  intercept_sens[1] ~ normal(1.5,0.5); // PCR 
  intercept_sens[2] ~ normal(2.2,0.5); // RDT
  intercept_sens[3] ~ normal(2.3,0.5); // ELISA

  logit_spec[1] ~ normal(3.5,0.2); // PCR 
  logit_spec[2] ~ normal(2.5,0.2); // RDT
  logit_spec[3] ~ normal(2.5,0.2); // ELISA
  
// PCR
spline_coef_sens[1] ~ normal(0,0.25); 
spline_coef_sens[2] ~ normal(0,0.001); 
spline_coef_sens[3] ~ normal(0,0.001); 
  
// Only for RDT and ELISA
 onset_coef_sens[1] ~ normal(0, 0.01);
 onset_coef_sens[2] ~ normal(0, 0.01);

  // Prior for logit phi 
   logit_phi ~ normal(-1.386294, 0.7); // True infection probability
  
 // Multinomial Likelihood
  // Loop over individuals
  for (i in 1:N) {
    vector[N_obs] p_vec = to_vector(p[i]);
    array[2] real t_rdt;
    
    // Normalize probabilities to sum to 1
    p_vec = p_vec / sum(p_vec);
    
   if (indicat[i] == 1) { // When RDT is missing
       t_rdt[1] = multinomial_lpmf(observed_results_array_pos[i] | p_vec);
       t_rdt[2] = multinomial_lpmf(observed_results_array_neg[i] | p_vec);

       target += log_mix(prior_hev_missing, t_rdt[1], t_rdt[2]); 

  } else if (indicat[i] == 0) { // When RDT is not missing
      target += multinomial_lpmf(observed_results_array_pos[i] | p_vec); 
    }

}
}

generated quantities {
  real log_lik[N];                  // Log-likelihood for each individual
  real posterior_infection_prob[N];  // Posterior probability of infection for each individual
  array[N, 3] real ppv;              // Positive Predictive Value (PPV) for each test and individual
  array[N, 3] real npv;              // Negative Predictive Value (NPV) for each test and individual
  array[2] real t_rdt;

  for (i in 1:N) {
    vector[N_obs] p_vec = to_vector(p[i]);
    p_vec = p_vec / sum(p_vec); 

    // Log-likelihood accounting for marginalization. Same as in model section. 
    if (indicat[i] == 1) {  // When RDT is missing
      t_rdt[1] = multinomial_lpmf(observed_results_array_pos[i] | p_vec);
      t_rdt[2] = multinomial_lpmf(observed_results_array_neg[i] | p_vec);
      log_lik[i] = log_mix(prior_hev_missing, t_rdt[1], t_rdt[2]);
    } else {  // When RDT is observed
      log_lik[i] = multinomial_lpmf(observed_results_array_pos[i] | p_vec);
    }

    // Estimate PPV and NPV
    for (j in 1:3) {
      ppv[i, j] = (sens_all[i][j] * phi) / ((sens_all[i][j] * phi) + (1 - spec_all[j]) * (1 - phi));
      npv[i, j] = (spec_all[j] * (1 - phi)) / ((spec_all[j] * (1 - phi)) + (1 - sens_all[i][j]) * phi);
    }

    // Posterior probability of infection, adjusted for marginalization
    row_vector[3] sens = [sens_all[i][1], sens_all[i][2], sens_all[i][3]];
    row_vector[3] spec = [spec_all[1], spec_all[2], spec_all[3]];

     real p_infected_pos = phi;       // Prior probability of infection (for RDT+ scenario)
     real p_infected_neg = phi;       // Prior probability of infection (for RDT- scenario)
     real p_infected = phi;       

     real p_not_infected_pos = 1 - phi; // Prior probability of no infection (RDT+ scenario)
     real p_not_infected_neg = 1 - phi; // Prior probability of no infection (RDT- scenario)
     real p_not_infected = 1 - phi; 
     
    for (k in 1:3) {
  if (indicat[i] == 1) {  // If RDT is missing
    if (observed_results_pos[i, k] == 1) {
      p_infected_pos *= sens[k];
      p_not_infected_pos *= 1 - spec[k];
    } else {
      p_infected_pos *= 1 - sens[k];
      p_not_infected_pos *= spec[k];
    }

    if (observed_results_neg[i, k] == 1) {
      p_infected_neg *= sens[k];
      p_not_infected_neg *= 1 - spec[k];
    } else {
      p_infected_neg *= 1 - sens[k];
      p_not_infected_neg *= spec[k];
    }

  } else if (indicat[i] == 0) { // When RDT is not missing
     
    if (observed_results_pos[i, k] == 1) { 
      p_infected *= sens[k];
      p_not_infected *= 1 - spec[k];
    } else {
      p_infected *= 1 - sens[k];
      p_not_infected *= spec[k];
    }
  }
}

// Posterior probability of infection, marginalizing over missing RDT
if (indicat[i] == 1) {
  posterior_infection_prob[i] = (p_infected_pos + p_infected_neg) / (p_infected_pos + p_infected_neg + p_not_infected_pos + p_not_infected_neg);
} else {
  posterior_infection_prob[i] = p_infected / (p_infected + p_not_infected);
}
}
}
