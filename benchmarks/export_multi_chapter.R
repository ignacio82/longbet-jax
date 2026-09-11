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

# Export only simulation chunks, not any model fit or rejected draft prose.
# Usage: Rscript export_multi_chapter.R BOOK_DIR OUTPUT_NPZ [JAX_R_SOURCE_DIR]
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) %in% c(2L,3L))
if (length(args)==3L) suppressMessages(pkgload::load_all(args[3],quiet=TRUE))
chapter <- readLines(file.path(args[1],'longbet.qmd'))
draft <- readLines(file.path(args[1],'multi-draft.md'))
run_chunk <- function(lines, label) {
  start <- which(lines == paste0('```{r ', label, '}'))
  stopifnot(length(start)==1L)
  end <- start + which(lines[(start+1L):length(lines)] == '```')[1]
  stopifnot(!is.na(end))
  eval(parse(text=lines[(start+1L):(end-1L)]), envir=.GlobalEnv)
}
for (label in c('setup','calendar','sellers','potential-outcomes',
                'treatment-effect','covariates')) run_chunk(chapter,label)
for (label in c('multi-fit-settings','multi-prediction-settings',
                'multi-decision-settings','multi-dgp','multi-dgp-checks',
                'multi-target')) run_chunk(draft,label)
np <- reticulate::import('numpy')
np$savez(args[2], x=multi_x, z=multi_z, ze=multi_z_eval,
  gmv=multi_y$gmv, hours=multi_y$seller_hours, complaint=multi_y$complaint,
  t=as.numeric(week_study), col=as.integer(multi_target_col-1L),
  truth_gmv=log1p(multi_truth$gmv), truth_hours=log1p(multi_truth$seller_hours),
  truth_complaint=multi_truth$complaint,
  listings=listings[multi_idx], fulfilled=fulfilled[multi_idx])
cat('Exported unchanged chapter simulation to', args[2], '\n')
