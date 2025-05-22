
# Implementation of the SAE-WM method using a simulated dataset

# Here, the mediator is continuous and the exposure variable, A, represents some simulated exposure

# Load in simulated dataset, neighborhood matrix, and population weights
load('simulatedDataset.RData')

# Load libraries
library(INLA)
library(dplyr)

# Note that the columns and rows in neighborhoodMat are in the same order as the data
colnames(neighborhoodMat) <- rownames(neighborhoodMat) <- unique(simulatedData$zcta)

# Count number of ZIP codes and age groups
nZCTA <- length(unique(simulatedData$zcta))
nAge <- length(unique(simulatedData$age_group))

# Add a unique identifier for each ZIP code to the dataset
simulatedData$zcta_id <- rep(1:nZCTA, each = nAge)

# Add a unique identifier for each observation to the dataset
simulatedData$obs_id <- 1:(nZCTA*nAge)

# Create mediator dataset 
# Extract M, A, and C
# Ensure that there is one mediator value per ZIP code
med_dat <- filter(simulatedData, age_group == 1) %>% 
  select(M, A, C)

# Create data frame with values to predict mediators
# Since the mediators will be predicted, we set their values to be NA
# We predict the mediator value for each ZIP code under A = 0 and A = 1
predict_mediator_values <- data.frame(M = rep(NA, nZCTA*2),
                                      A = c(rep(0, nZCTA), rep(1, nZCTA)),
                                      C = rep(med_dat$C, 2))

# Augment the mediator dataset with the prediction dataset
med_dat_aug <- rbind(med_dat, predict_mediator_values)

# Define half-Cauchy(5) prior on standard deviation in INLA
HC.prior  = "expression:
  sigma = exp(-theta/2);
  gamma = 5;
  log_dens = log(2) - log(pi) - log(gamma);
  log_dens = log_dens - log(1 + (sigma / gamma)^2);
  log_dens = log_dens - log(2) - theta / 2;
  return(log_dens);
"

# Fit the mediator model
fit2 <- inla(M ~ A + C, 
             num.threads = 1, # set for reproducibility
             data = med_dat_aug, family = 'gaussian', 
             control.family = list(hyper = list(prec = list(prior = HC.prior))),
             control.compute = list(config = T)) # allows us to obtain samples of the linear predictors

# Set the number of posterior samples to draw
nSamp <- 1000

# Draw posterior samples of the predicted mediator values
set.seed(421)
mediatorSamps <- inla.posterior.sample(nSamp, fit2, seed = 421,
                                       selection = list(Predictor = (nZCTA + 1):(3*nZCTA)), 
                                       num.threads = 1)

# Create matrices to store the predicted mediator values
# (Predicted under A = 0 and A = 1)
mediators0 <- matrix(NA, nrow = nZCTA, ncol = 1000)
mediators1 <- matrix(NA, nrow = nZCTA, ncol = 1000)

# Extract the linear predictors when A = 0 and extract the hyperparameter values
temp0 <- lapply(mediatorSamps, function(x) x$latent[1:nZCTA,])
hyper <- sapply(mediatorSamps, function(x) x$hyperpar[1])

# Generate mediator predictions under A = 0
for(j in 1:1000){
  mediators0[,j] <- rnorm(nZCTA, mean = temp0[[j]], sd = 1 / sqrt(hyper[j]))
}

# Extract the linear predictors when A = 1
temp1 <- lapply(mediatorSamps, function(x) x$latent[(nZCTA + 1):(2*nZCTA),])

# Generate mediator predictions under A = 1
for(j in 1:1000){
  mediators1[,j] <- rnorm(nZCTA, mean = temp1[[j]], sd = 1 / sqrt(hyper[j]))
}

rm(temp0, temp1)

# Fit the outcome model
fit1 <- inla(Y ~ 0 + factor(age_group) + A + M + C + 
               f(zcta_id, model = 'besag', graph = neighborhoodMat, hyper = list(prec = list(prior = HC.prior))) + 
               f(obs_id, model = 'iid', hyper = list(prec = list(prior = HC.prior))), 
             E = N, num.threads = 1,
             data = simulatedData, family = 'poisson',
             control.compute = list(config = T))

# Extract posterior samples of the model parameters and random effects
set.seed(422)
parameterSamps <- inla.posterior.sample(nSamp, fit1, seed = 422,
                                        selection = list('factor(age_group)1' = 1, 'factor(age_group)2' = 1, 'factor(age_group)3' = 1, 
                                                         'factor(age_group)4' = 1, 'factor(age_group)5' = 1, 'factor(age_group)6' = 1, 
                                                         A = 1, M = 1, C = 1,
                                                         obs_id = 1:(nZCTA*nAge), zcta_id = 1:nZCTA)) %>% 
  lapply(function(x) x$latent)

# Create empty matrices to store the counterfactual AARs
rate_00 <- rep(NA, 1000)
rate_10 <- rep(NA, 1000)
rate_11 <- rep(NA, 1000)

# Obtain 1000 posterior samples of each lambda_i(0, \tilde{M}_i(0)) and average over all ZIP codes
for(j in 1:1000){
  samps <- parameterSamps[[j]]
  rn <- rownames(samps)
  zcta_id <- rep(samps[which(grepl('zcta_id', rn)),1], each = 6)
  obs_id <- samps[which(grepl('obs_id', rn)),1]
  alpha <- samps[which(grepl('age', rn)),1]
  beta1 <- samps[which(grepl('A:1', rn)),1]
  beta2 <- samps[which(grepl('M:1', rn)),1]
  beta3 <- samps[which(grepl('C', rn)),1]
  meds <- rep(mediators0[,j], each = 6)
  lin_pred <-  rep(alpha, nZCTA) + beta1*0 + beta2*meds + beta3*simulatedData$C + zcta_id + obs_id
  rate <- data.frame(rate = 100000 * exp(lin_pred), zcta = simulatedData$zcta)
  rate$rate <- rate$rate * rep(weights$weight, nZCTA)
  rate_00[j] <- mean(aggregate(rate ~ zcta, rate, sum)[,2])
} 

# Obtain 1000 posterior samples of each lambda_i(1, \tilde{M}_i(0)) and average over all ZIP codes
for(j in 1:1000){
  samps <- parameterSamps[[j]]
  rn <- rownames(samps)
  zcta_id <- rep(samps[which(grepl('zcta_id', rn)),1], each = 6)
  obs_id <- samps[which(grepl('obs_id', rn)),1]
  alpha <- samps[which(grepl('age', rn)),1]
  beta1 <- samps[which(grepl('A:1', rn)),1]
  beta2 <- samps[which(grepl('M:1', rn)),1]
  beta3 <- samps[which(grepl('C', rn)),1]
  meds <- rep(mediators0[,j], each = 6)
  lin_pred <-  rep(alpha, nZCTA) + beta1*1 + beta2*meds +
    beta3*simulatedData$C + zcta_id + obs_id
  rate <- data.frame(rate = 100000 * exp(lin_pred), zcta = simulatedData$zcta)
  rate$rate <- rate$rate * rep(weights$weight, nZCTA)
  rate_10[j] <- mean(aggregate(rate ~ zcta, rate, sum)[,2])
}  

# Obtain 1000 posterior samples of each lambda_i(1, \tilde{M}_i(1)) and average over all ZIP codes
for(j in 1:1000){
  samps <- parameterSamps[[j]]
  rn <- rownames(samps)
  zcta_id <- rep(samps[which(grepl('zcta_id', rn)),1], each = 6)
  obs_id <- samps[which(grepl('obs_id', rn)),1]
  alpha <- samps[which(grepl('age', rn)),1]
  beta1 <- samps[which(grepl('A:1', rn)),1]
  beta2 <- samps[which(grepl('M:1', rn)),1]
  beta3 <- samps[which(grepl('C', rn)),1]
  meds <- rep(mediators1[,j], each = 6)
  lin_pred <-  rep(alpha, nZCTA) + beta1*1 + beta2*meds + 
    beta3*simulatedData$C + zcta_id + obs_id
  rate <- data.frame(rate = 100000 * exp(lin_pred), zcta = simulatedData$zcta)
  rate$rate <- rate$rate * rep(weights$weight, nZCTA)
  rate_11[j] <- mean(aggregate(rate ~ zcta, rate, sum)[,2])
}  

# Create matrix of TE, DE, and IE estimates and lower/ upper bounds of 95% credible intervals
eff <- c(mean(rate_11 - rate_00), quantile(rate_11 - rate_00, probs = c(0.025, 0.975)),
         mean(rate_10 - rate_00), quantile(rate_10 - rate_00, probs = c(0.025, 0.975)),
         mean(rate_11 - rate_10), quantile(rate_11 - rate_10, probs = c(0.025, 0.975)))

names(eff) <- c('TE', 'TE - lower CI', 'TE - upper CI',
                'DE', 'DE - lower CI', 'DE - upper CI',
                'IE', 'IE - lower CI', 'IE - upper CI')         

round(eff, 3)
