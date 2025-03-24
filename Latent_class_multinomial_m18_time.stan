data {
  int<lower=1> N; // Number of individuals
  int<lower=1> N_obs; // Number of test result combinations
  matrix[N,3] observed_results_pos; // Test results. Forced all missing RDT to be pos
  array[N,N_obs] int observed_results_array_pos; // Test profiles. Forced all missing RDT to be pos
  matrix[N,3] observed_results_neg; // Test results. Forced all missing RDT to be neg
  array[N,N_obs] int observed_results_array_neg; // Test profiles. Forced all missing RDT to be neg
  int<lower=0, upper=1> indicat[N];  
  matrix[N,1] days_since_onset; // Days since jaundice onset group of each individual
  matrix[N,1] month1; //month of AJS onset
  matrix[N,1] month2; //month of AJS onset
  matrix[N,1] month3; //month of AJS onset
  matrix[N,1] month4; //month of AJS onset
  matrix[N,1] month5; //month of AJS onset
  matrix[N,1] month6; //month of AJS onset
  matrix[N,1] month7; //month of AJS onset
  matrix[N,1] month8; //month of AJS onset
  matrix[N,1] month9; //month of AJS onset
  matrix[N,1] month10; //month of AJS onset
  real prior_hev_missing; // prior probabilty of having HEV for those with missing RDTs
  matrix[N,1] age_bin; // Binary variable indicating whether someone is <6 or >=6
  }

parameters {
  real phi_base; // Intercept for estimating time varying logit_phi. Same for everyone, give it a prior
  real coef_month1;
  real coef_month2;
  real coef_month3;
  real coef_month4;
  real coef_month5;
  real coef_month6;
  real coef_month7;
  real coef_month8;
  real coef_month9;
  real coef_month10;
  array[3] real age_coef_sens;      // Coefficients for age on sensitivity for each test
  array[3] real intercept_sens; //Intercept for sensitivity regression for each test. 
  array[3] real <upper=0> onset_coef_sens; // Coefficients for days_since_onset on sensitivity for each test
  array[3] real <lower=0> logit_spec; 
}

transformed parameters {
  array[N, N_obs] real<lower=0, upper=1> p; // Probabilities for each test result combination for each individual
  array[N,3] real logit_sens; 
  array[N,3] real<lower=0, upper=1> sens_all;
  array[3] real<lower=0, upper=1> spec_all = inv_logit(logit_spec);

  // Time-varying phi for each individual based on their week or month
  array[N] real phi;
  array[N] real logit_phi;
  for (i in 1:N) {
    logit_phi[i] = phi_base + (coef_month1 * month1[i,1]) + (coef_month2 * month2[i,1]) + (coef_month3 * month3[i,1]) + (coef_month4 * month4[i,1]) + (coef_month5 * month5[i,1]) + (coef_month6 * month6[i,1]) + (coef_month7 * month7[i,1]) + (coef_month8 * month8[i,1]) + (coef_month9 * month9[i,1]) + (coef_month10 * month10[i,1]);
    phi[i] = inv_logit(logit_phi[i]);
  }

  // Compute sensitivities for each individual for each test
  for (i in 1:N) {
    for (j in 1:3) {
       logit_sens[i, j] = (intercept_sens[j] + onset_coef_sens[j] * days_since_onset[i,1] + age_coef_sens[j] * age_bin[i,1]); 
       sens_all[i, j] = inv_logit(logit_sens[i, j]);
    }
  }

  // Loop over individuals
    for (i in 1:N) {
    row_vector[3] sens = [sens_all[i][1], sens_all[i][2], sens_all[i][3]];
    row_vector[3] spec = [spec_all[1], spec_all[2], spec_all[3]];
    // Calculate probabilities for each test result combination for the individual
    p[i, 1] = prod(1-sens) * phi[i] + prod(spec) * (1 - phi[i]);
    p[i, 2] = (1-sens[1]) * (1-sens[2]) * (sens[3]) * phi[i] + (spec[1]) * (spec[2]) * (1-spec[3]) * (1 - phi[i]);
    p[i, 3] = (1-sens[1]) * (sens[2]) * (1-sens[3]) * phi[i] + (spec[1]) * (1-spec[2]) * (spec[3]) * (1 - phi[i]);
    p[i, 4] = (1-sens[1]) * (sens[2]) * (sens[3]) * phi[i] + (spec[1]) * (1-spec[2]) * (1-spec[3]) * (1 - phi[i]);
    p[i, 5] = (sens[1]) * (1-sens[2]) * (1-sens[3]) * phi[i] + (1-spec[1]) * (spec[2]) * (spec[3]) * (1 - phi[i]);
    p[i, 6] = (sens[1]) * (1-sens[2]) * (sens[3]) * phi[i] + (1-spec[1]) * (spec[2]) * (1-spec[3]) * (1 - phi[i]);
    p[i, 7] = (sens[1]) * (sens[2]) * (1-sens[3]) * phi[i] + (1-spec[1]) * (1-spec[2]) * (spec[3]) * (1 - phi[i]);
    p[i, 8] = prod(sens) * phi[i] + prod(1-spec) * (1 - phi[i]);
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
  

onset_coef_sens[1] ~ normal(0, 0.05);
onset_coef_sens[2] ~ normal(0, 0.01);
onset_coef_sens[3] ~ normal(0, 0.01);

age_coef_sens[1] ~ normal(0,1); 
age_coef_sens[2] ~ normal(0,1); 
age_coef_sens[3] ~ normal(0,1); 

phi_base ~ normal(-1.386294, 0.7); // True infection probability 

// Need to add priors for all the coefficients
 coef_month1 ~ normal(0,2);
 coef_month2 ~ normal(0,2);
 coef_month3 ~ normal(0,2);
 coef_month4 ~ normal(0,2);
 coef_month5 ~ normal(0,2);
 coef_month6 ~ normal(0,2);
 coef_month7 ~ normal(0,2);
 coef_month8 ~ normal(0,2);
 coef_month9 ~ normal(0,2);
 coef_month10 ~ normal(0,2);

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
      ppv[i, j] = (sens_all[i][j] * phi[i]) / ((sens_all[i][j] * phi[i]) + (1 - spec_all[j]) * (1 - phi[i]));
      npv[i, j] = (spec_all[j] * (1 - phi[i])) / ((spec_all[j] * (1 - phi[i])) + (1 - sens_all[i][j]) * phi[i]);
    }

    // Posterior probability of infection, adjusted for marginalization
    row_vector[3] sens = [sens_all[i][1], sens_all[i][2], sens_all[i][3]];
    row_vector[3] spec = [spec_all[1], spec_all[2], spec_all[3]];

     real p_infected_pos = phi[i];       // Prior probability of infection (for RDT+ scenario)
     real p_infected_neg = phi[i];       // Prior probability of infection (for RDT- scenario)
     real p_infected = phi[i];       

     real p_not_infected_pos = 1 - phi[i]; // Prior probability of no infection (RDT+ scenario)
     real p_not_infected_neg = 1 - phi[i]; // Prior probability of no infection (RDT- scenario)
     real p_not_infected = 1 - phi[i]; 
     
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
