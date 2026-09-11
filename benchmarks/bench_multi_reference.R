# Copyright 2026 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Benchmark the ORIGINAL C++ package, installed in a separate R library.
# Usage: Rscript bench_multi_reference.R DATA_NPZ OUTPUT_DIR REFERENCE_R_LIBRARY
# Fixed protocol: 4 independent seeds, 750 total sweeps / 500 discarded per fit.
# C++ defaults otherwise, except lambda_knl=2 and serial execution. This is NOT
# an identical-prior or equal-runtime comparison with the JAX sampler.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==3L)
suppressPackageStartupMessages(library(longbet,lib.loc=args[3]))
stopifnot('longbet_multi' %in% getNamespaceExports('longbet'),
          exists('longbet_cpp',asNamespace('longbet'),inherits=FALSE))
if (file.exists(file.path(args[2],'reference-results.rds'))) {
  stop('Output already contains a benchmark; choose a new directory.')
}
dir.create(args[2],recursive=TRUE,showWarnings=FALSE)
np <- reticulate::import('numpy',convert=FALSE)
d <- np$load(args[1])
get <- function(k) reticulate::py_to_r(d$`__getitem__`(k))
x <- get('x'); z <- get('z'); ze <- get('ze'); tt <- as.numeric(get('t'))
column <- as.integer(get('col'))+1L
ys <- lapply(c('gmv','hours','complaint'),get)
types <- c('continuous','continuous','binary')
groups <- list(good=get('listings')>=80 & get('fulfilled')==1,
               bad=get('listings')<=25 & get('fulfilled')==0)
burn <- 500L; total <- 750L; results <- list()
saveRDS(list(package_version=as.character(packageVersion('longbet')),
  data_md5=unname(tools::md5sum(args[1])),seeds=314159:314162,
  total_sweeps=total,burnin=burn,lambda_knl=2,pcat=0L,parallel=FALSE),
  file.path(args[2],'reference-metadata.rds'))
for (chain in 1:4) {
  started <- proc.time()[['elapsed']]
  cat('C++ reference chain',chain,'start',format(Sys.time()),'\n'); flush.console()
  fit <- longbet_multi(y=ys,x=x,x_trt=x,z=z,t=tt,pcat=0L,outcome=types,
    num_sweeps=total,num_burnin=burn,num_trees_pr=20L,num_trees_trt=20L,
    lambda_knl=2,random_intercept=TRUE,random_seed=314158L+chain,parallel=FALSE)
  draws <- public_draws <- list()
  for (m in 1:3) {
    pr <- predict(fit$fits[[m]],x=x,x_trt=x,z=ze,t=tt,random_seed=271829L)
    public_draws[[m]] <- effect_draws(pr,outcome=types[m])[,column,]
    if (m==3L) {
      # Reference prediction omits gamma_i. Restore the MATCHED training-unit
      # posterior draws to use the same in-sample risk-difference estimand.
      gamma <- fit$fits[[m]]$gamma_draws[,(burn+1L):total,drop=FALSE]
      mu0 <- pr$muhats0[,column,]+gamma
      draws[[m]] <- pnorm(mu0+pr$tauhats[,column,])-pnorm(mu0)
    } else draws[[m]] <- expm1(public_draws[[m]])
    rm(pr); gc(FALSE)
  }
  names(draws) <- names(public_draws) <- c('gmv','hours','complaint')
  results[[chain]] <- list(draws=draws,public_draws=public_draws,
    elapsed=proc.time()[['elapsed']]-started,seed=314158L+chain)
  saveRDS(results,file.path(args[2],'reference-results.rds'))
  cat('C++ reference chain',chain,'finished in',results[[chain]]$elapsed,'seconds\n')
  flush.console(); rm(fit); gc(FALSE)
}
rows <- list(); kept <- list()
for (nm in c('gmv','hours','complaint')) {
  truth <- as.numeric(get(paste0('truth_',nm)))
  if (nm!='complaint') truth <- expm1(truth)
  a <- do.call(cbind,lapply(results,function(v) v$draws[[nm]]))
  kept[[nm]] <- a
  for (label in names(groups)) {
    mask <- groups[[label]]
    values <- vapply(results,function(v) colMeans(v$draws[[nm]][mask,,drop=FALSE]),numeric(250))
    interval <- quantile(values,c(.025,.975),names=FALSE)
    rows[[length(rows)+1L]] <- data.frame(outcome=nm,group=label,
      rmse=sqrt(mean((rowMeans(a)-truth)^2)),truth=mean(truth[mask]),mean=mean(values),
      lower=interval[1],upper=interval[2],p_correct=mean(values*sign(mean(truth[mask]))>0),
      rhat=posterior::rhat(values),ess_bulk=posterior::ess_bulk(values),
      ess_tail=posterior::ess_tail(values))
  }
}
write.csv(do.call(rbind,rows),file.path(args[2],'reference-summary.csv'),row.names=FALSE)
do.call(np$savez,c(list(file.path(args[2],'reference-draws.npz')),kept))
