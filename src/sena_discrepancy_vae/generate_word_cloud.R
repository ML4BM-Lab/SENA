#!/usr/bin/env Rscript
# ============================================================================
# Analysis: activation functions & interventional encoder results
# - Loads causal graph, activation function scores (fc1), and encoder outputs
# - Assigns interventions to latent factors
# - Tests GO activation differences vs control (subsampled t-tests)
# - Selects GO terms per intervention using effect-size and significance criteria
# - Aggregates GO per latent factor, exports lists & wordclouds
# - Extracts and exports top edges from causal subgraph
# ============================================================================

# ------------------------------- Setup --------------------------------------

# Control panel
rm(list = ls())
library(tidyverse)
library(ggplot2)
library(data.table)
library(ComplexHeatmap)
library(GO.db)
library(wordcloud)
library(tm)
library(reticulate)

diff_fc_perc   <- 0.01
sign_threhold  <- -log10(0.05)  # NOTE: keep original name/typo to preserve logic
subsampling    <- 100
n_latent_factors <- 105

res_folder <- paste0('results_LF_', n_latent_factors)
dir.create(res_folder, showWarnings = FALSE, recursive = TRUE)

# ------------------------------ Load data -----------------------------------

# Causal graph
causal_graph <- read.csv(
  paste0('causal_graph_', n_latent_factors, '.csv'),
  row.names = 1
)

# Activation function data
fc1 <- fread(paste0('fc1_', n_latent_factors, '.csv'), data.table = FALSE)
fc1[1:3, 1:3]
colnames(fc1)[1] <- 'condition'
fc1[1:3, 1:3]
fc1[['GO:0000305']][1:3]  # activation functions for the GO nodes

# Interventions -> latent factors (interventional encoder output)
bc_temp1000 <- fread(
  paste0('bc_temp1000_', n_latent_factors, '.csv'),
  data.table = FALSE, header = TRUE
)
rownames(bc_temp1000) <- bc_temp1000$V1
bc_temp1000$V1 <- NULL
n_latent_factors <- ncol(bc_temp1000)
colnames(bc_temp1000) <- paste0('Latent_factor_', 1:n_latent_factors)
bc_temp1000[1:3, 1:3]  # output of the interventional encoder

# --------------- Assign each intervention to a latent factor ----------------

# Binarize (values are ~0.998 or 1e-10; any threshold works here)
bc_bin <- bc_temp1000 > 0.5

# Assignment: LF with maximum (binary) load per intervention
latent_factor_2_intervention <- apply(bc_bin, 1, function(x) { which.max(x) })

# -------- Assign GO nodes to interventions via t-test on activations --------

# Mean activation per condition
fc_stats <- fc1 %>%
  group_by(condition) %>%
  summarise(across(where(is.numeric), mean)) %>%
  as.data.frame()
rownames(fc_stats) <- fc_stats$condition
fc_stats$condition <- NULL

# Differences vs the LAST row 
diff_fc_stats <- fc_stats[1:(nrow(fc_stats) - 1), ]
for (i in 1:nrow(diff_fc_stats)) {
  diff_fc_stats[i, ] <- abs(fc_stats[i, ] - fc_stats[nrow(fc_stats), ])
}

# Subsampled t-tests: each condition vs control rows in fc1
set.seed(12345)
pvalue_fc_stats <- diff_fc_stats

ctrl_idx <- which(fc1$condition == 'ctrl')
ctrl_idx <- ctrl_idx[sample(1:length(ctrl_idx), subsampling, replace = FALSE)]

for (i in 1:nrow(pvalue_fc_stats)) {
  condition_idx <- which(fc1$condition == rownames(pvalue_fc_stats)[i])
  condition_idx <- condition_idx[sample(1:length(condition_idx), subsampling, replace = FALSE)]

  for (j in 1:ncol(pvalue_fc_stats)) {
    pvalue_fc_stats[i, j] <- tryCatch(
      t.test(
        fc1[condition_idx, colnames(pvalue_fc_stats)[j]],
        fc1[ctrl_idx,      colnames(pvalue_fc_stats)[j]]
      )$p.value,
      error = function(e) { return(1) }
    )
  }
}

# FDR adjust -> -log10
adj_pvalue_fc_stats <- pvalue_fc_stats
for (i in 1:nrow(adj_pvalue_fc_stats)) {
  adj_pvalue_fc_stats[i, ] <- -log10(p.adjust(pvalue_fc_stats[i, ], method = 'fdr'))
}

# -------------------------------- Plots -------------------------------------

suppressMessages(
  Heatmap(as.matrix(fc_stats),
          row_title = 'interventions', column_title = 'GO',
          show_column_names = FALSE, show_row_names = FALSE)
)

suppressMessages(
  Heatmap(as.matrix(diff_fc_stats),
          row_title = 'interventions', column_title = 'GO',
          show_column_names = FALSE, show_row_names = FALSE)
)

suppressMessages(
  Heatmap(as.matrix(adj_pvalue_fc_stats),
          row_title = 'interventions', column_title = 'GO',
          show_column_names = FALSE, show_row_names = FALSE)
)

# ---------- Select GO per intervention by effect size & significance ---------

# Global threshold for effect size
diff_fc_threshold <- quantile(as.matrix(diff_fc_stats), 1 - diff_fc_perc)

# Zero-out entries failing either criterion
diff_fc_stats <- as.matrix(diff_fc_stats)
to_delete <- (diff_fc_stats < diff_fc_threshold) |  # effect size
             (adj_pvalue_fc_stats < sign_threhold)  # significance (keep variable name)
diff_fc_stats[to_delete] <- 0

# GO -> intervention: pick intervention with max remaining diff per GO
GO_2_intervention <- vector('list', dim(diff_fc_stats)[2])
names(GO_2_intervention) <- colnames(diff_fc_stats)
for (i in 1:dim(diff_fc_stats)[2]) {
  if (!all(diff_fc_stats[, i] == 0)) {
    GO_2_intervention[[i]] <- rownames(diff_fc_stats)[which.max(diff_fc_stats[, i])]
  }
}
table(sapply(GO_2_intervention, length))

# Intervention -> list of GO that selected it
intervention_2_GO_list <- vector('list', dim(diff_fc_stats)[1])
names(intervention_2_GO_list) <- rownames(diff_fc_stats)
GO_2_intervention <- unlist(GO_2_intervention)
for (i in 1:length(intervention_2_GO_list)) {
  idx <- GO_2_intervention == names(intervention_2_GO_list)[i]
  if (any(idx)) {
    intervention_2_GO_list[[i]] <- names(GO_2_intervention)[idx]
  }
}

# ----------------------- Aggregate GO per latent factor ----------------------

used_latent_factor <- unique(latent_factor_2_intervention)
latent_factor_2_GO_list <- vector('list', length(used_latent_factor))
names(latent_factor_2_GO_list) <- paste0('Latent_factor_', used_latent_factor)

for (k in used_latent_factor) {
  idx <- which(latent_factor_2_intervention == k)
  latent_factor_2_GO_list[[paste0('Latent_factor_', k)]] <-
    unique(unlist(intervention_2_GO_list[names(latent_factor_2_intervention)[idx]]))
}

# Save counts per LF
sink(paste0(res_folder, '/latent_factor_2_GO_list.txt'))
print(sapply(latent_factor_2_GO_list, length))
sink()

# --------------------------- GO IDs -> GO terms -----------------------------

latent_factor_2_GO_list <- sapply(latent_factor_2_GO_list, function(x) {
  gsub('.', ':', x, fixed = TRUE)
})
latent_factor_2_GO_terms <- latent_factor_2_GO_list
for (i in 1:length(latent_factor_2_GO_terms)) {
  latent_factor_2_GO_terms[[i]] <- select(
    GO.db,
    keys = latent_factor_2_GO_list[[i]],
    columns = c('DEFINITION', 'TERM'),
    keytype = 'GOID'
  )$TERM
}

# --------------- Export GO lists + Wordclouds per latent factor -------------

for (i in 1:length(latent_factor_2_GO_terms)) {
  set.seed(12345)

  # Save GO IDs + terms
  to_print <- data.frame(
    GO_ID  = latent_factor_2_GO_list[[i]],
    GO_TERM = latent_factor_2_GO_terms[[i]]
  )
  write.csv(
    to_print, row.names = FALSE,
    file = paste0(res_folder, '/', names(latent_factor_2_GO_terms)[i], '.csv')
  )

  # Skip wordcloud if no terms
  if (length(latent_factor_2_GO_terms[[i]]) == 0) next

  # Build corpus
  go_corpus <- SimpleCorpus(VectorSource(latent_factor_2_GO_terms[[i]]))
  go_corpus <- tm_map(go_corpus, content_transformer(tolower))
  go_corpus <- tm_map(
    go_corpus, removeWords,
    c(
      stopwords("english"), 'regulation', 'process',
      'positive', 'negative', 'pathways', 'pathway',
      'reaction', 'process', 'activity', 'involving',
      'metabolic', 'protein', 'involved', 'activation'
    )
  )

  # Wordcloud
  png(
    filename = file.path(res_folder, paste0(names(latent_factor_2_GO_terms)[i], '.png')),
    width = 2100, height = 2100, res = 300
  )
  wordcloud(
    words = go_corpus,
    min.freq = length(go_corpus) / 100,
    random.order = FALSE,
    rot.per = 0,
    colors = c("#5E4FA2", "#66C2A5", "#E6F598", '#ABDDA4'),
    scale = c(1.5, 0.75)
  )
  dev.off()
}

# -------------------------- Causal graph export -----------------------------

# Select LFs present in the GO-term mapping
idx <- as.numeric(gsub('Latent_factor_', '', names(latent_factor_2_GO_terms)))

selected_graph <- causal_graph[idx, idx]
selected_graph[lower.tri(selected_graph, diag = TRUE)] <- 0
rownames(selected_graph) <- colnames(selected_graph) <- names(latent_factor_2_GO_terms)

selected_graph <- unique(reshape2::melt(as.matrix(selected_graph)))
selected_graph <- selected_graph[order(abs(selected_graph$value), decreasing = TRUE), ]
colnames(selected_graph) <- c('from', 'to', 'coefficient')

# Write top edges
write.csv(selected_graph[1:10, ], row.names = FALSE,
          file = paste0(res_folder, '/causal_graph_', 10, '.csv'))
write.csv(selected_graph[1:15, ], row.names = FALSE,
          file = paste0(res_folder, '/causal_graph_', 15, '.csv'))
write.csv(selected_graph[1:20, ], row.names = FALSE,
          file = paste0(res_folder, '/causal_graph_', 20, '.csv'))
