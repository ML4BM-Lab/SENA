#!/usr/bin/env Rscript
# =============================================================================
# Analysis of activation functions and interventional encoder results
# - Assign interventions -> latent factors
# - Link GO nodes to interventions via t-tests on activation functions
# - Map GO terms, export lists, generate wordclouds
# - Build/prune causal subgraph and export top edges
# =============================================================================

# --------------------------- Setup & Parameters ------------------------------


suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
  library(data.table)
  library(ComplexHeatmap)
  library(GO.db)
  library(wordcloud)
  library(tm)
  library(reticulate)   # (loaded in original script; not used directly here)
  library(reshape2)
})

# ---- Control panel ----
SEED                <- 12345
diff_fc_perc        <- 0.01            # global effect-size threshold quantile
sign_threshold_log10 <- -log10(0.05)   # significance threshold (FDR-adjusted)
subsampling_n       <- 100             # number of samples per condition/control for t-tests
n_latent_factors    <- 105             # base LF count (also used in filenames)
res_folder          <- paste0("results_LF_", n_latent_factors)
dir.create(res_folder, showWarnings = FALSE, recursive = TRUE)

# ---- Input paths ----

"""
This requires the following input files:
- causal_graph_105.csv
- bc_temp1000_105.csv
- fc1_105.csv

All generated from the `generate_activation_scores.py` using the `causal_graph`, `bc_temp1000` and `fc1` keys, respectively.
"""

PATH_CAUSAL_GRAPH   <- paste0("causal_graph_", n_latent_factors, ".csv")
PATH_FC1            <- paste0("fc1_", n_latent_factors, ".csv")             # activation functions table
PATH_BC_TEMP1000    <- paste0("bc_temp1000_", n_latent_factors, ".csv")     # interventional encoder

set.seed(SEED)

# --------------------------- Utilities --------------------------------------

stop_if_missing <- function(path) {
  if (!file.exists(path)) stop(sprintf("File not found: %s", path), call. = FALSE)
}

safe_sample_idx <- function(idx, k) {
  # sample without replacement up to available length
  k_eff <- min(length(idx), k)
  if (k_eff == 0) return(integer(0))
  sample(idx, k_eff, replace = FALSE)
}

heatmap_quiet <- function(mat, title_rows = "rows", title_cols = "columns") {
  suppressMessages(
    Heatmap(
      as.matrix(mat),
      row_title = title_rows,
      column_title = title_cols,
      show_column_names = FALSE,
      show_row_names = FALSE
    )
  )
}

# --------------------------- Load Data --------------------------------------

stop_if_missing(PATH_CAUSAL_GRAPH)
stop_if_missing(PATH_FC1)
stop_if_missing(PATH_BC_TEMP1000)

# Causal graph (square matrix indexed by latent factors)
causal_graph <- read.csv(PATH_CAUSAL_GRAPH, row.names = 1, check.names = FALSE)

# Activation functions: rows = samples, columns = GO features, plus 'condition'
fc1 <- fread(PATH_FC1, data.table = FALSE)
if (ncol(fc1) < 2) stop("fc1 has fewer than 2 columns; expected 'condition' + GO features.")
colnames(fc1)[1] <- "condition"

# Interventional encoder outputs (bc_temp1000): rows = interventions, cols = latent factors
bc <- fread(PATH_BC_TEMP1000, data.table = FALSE, header = TRUE)
rownames(bc) <- bc$V1
bc$V1 <- NULL
n_latent_factors <- ncol(bc)
colnames(bc) <- paste0("Latent_factor_", seq_len(n_latent_factors))

# --------------------------- Map interventions -> latent factor --------------

# Binarize (values are ~0.998 or 1e-10 in your data; threshold 0.5 suffices)
bc_bin <- bc > 0.5

# Assign each intervention to the latent factor with maximum (binary) load
latent_factor_of_intervention <- apply(bc_bin, 1, which.max)
names(latent_factor_of_intervention) <- rownames(bc) # interventions as names

# --------------------------- Activation stats & tests ------------------------

# 1) Per-condition mean activation per GO feature
fc_stats <- fc1 %>%
  group_by(condition) %>%
  summarise(across(where(is.numeric), mean), .groups = "drop") %>%
  as.data.frame()
rownames(fc_stats) <- fc_stats$condition
fc_stats$condition <- NULL

# 2) Absolute differences vs control (assumed last row in your original code)
#    Safer: explicitly pick "ctrl" if present, else last row as fallback
ctrl_name <- if ("ctrl" %in% rownames(fc_stats)) "ctrl" else tail(rownames(fc_stats), 1)
diff_fc_stats <- fc_stats[setdiff(rownames(fc_stats), ctrl_name), , drop = FALSE]
for (i in seq_len(nrow(diff_fc_stats))) {
  diff_fc_stats[i, ] <- abs(fc_stats[rownames(diff_fc_stats)[i], ] - fc_stats[ctrl_name, ])
}

# 3) t-tests: per (condition, GO), comparing condition vs control
#    Use subsampling_n per group where possible
pvalue_fc_stats <- diff_fc_stats
ctrl_idx_all <- which(fc1$condition == ctrl_name)
ctrl_idx <- safe_sample_idx(ctrl_idx_all, subsampling_n)

if (length(ctrl_idx) == 0) {
  stop(sprintf("No control samples found for condition '%s' in fc1.", ctrl_name))
}

for (i in seq_len(nrow(pvalue_fc_stats))) {
  cond_name <- rownames(pvalue_fc_stats)[i]
  cond_idx_all <- which(fc1$condition == cond_name)
  cond_idx <- safe_sample_idx(cond_idx_all, subsampling_n)

  if (length(cond_idx) == 0) {
    # If a condition has no samples, set p-values to 1 (non-significant)
    pvalue_fc_stats[i, ] <- 1
    next
  }

  for (j in seq_len(ncol(pvalue_fc_stats))) {
    go_col <- colnames(pvalue_fc_stats)[j]
    # Use tryCatch to handle potential constant/NA columns gracefully
    pvalue_fc_stats[i, j] <- tryCatch(
      t.test(fc1[cond_idx, go_col], fc1[ctrl_idx, go_col])$p.value,
      error = function(e) 1
    )
  }
}

# 4) FDR adjust and convert to -log10
adj_pvalue_fc_stats <- pvalue_fc_stats
for (i in seq_len(nrow(adj_pvalue_fc_stats))) {
  adj_pvalue_fc_stats[i, ] <- -log10(p.adjust(pvalue_fc_stats[i, ], method = "fdr"))
}

# --------------------------- Visualizations (heatmaps) -----------------------

message("Rendering heatmaps (silently)...")
invisible(heatmap_quiet(fc_stats, "interventions", "GO"))
invisible(heatmap_quiet(diff_fc_stats, "interventions", "GO"))
invisible(heatmap_quiet(adj_pvalue_fc_stats, "interventions", "GO"))

# --------------------------- Select GO per intervention ----------------------

# Global effect-size threshold from diff_fc_stats distribution
diff_fc_threshold <- as.numeric(quantile(as.matrix(diff_fc_stats), 1 - diff_fc_perc))

# Apply thresholds: zero-out entries that fail either effect-size or significance
diff_mat <- as.matrix(diff_fc_stats)
sig_mat  <- as.matrix(adj_pvalue_fc_stats)
to_zero  <- (diff_mat < diff_fc_threshold) | (sig_mat < sign_threshold_log10)
diff_mat[to_zero] <- 0

# For each GO, pick the intervention with the largest (thresholded) difference
GO_to_intervention <- vector("list", ncol(diff_mat))
names(GO_to_intervention) <- colnames(diff_mat)
for (j in seq_len(ncol(diff_mat))) {
  col_vals <- diff_mat[, j]
  if (!all(col_vals == 0)) {
    GO_to_intervention[[j]] <- names(which.max(col_vals))
  }
}
# Count how many GO mapped
mapping_count <- table(sapply(GO_to_intervention, length))
print(mapping_count)

# For each intervention, list all GO that selected it
intervention_to_GO_list <- vector("list", nrow(diff_mat))
names(intervention_to_GO_list) <- rownames(diff_mat)
GO_to_intervention_vec <- unlist(GO_to_intervention)
for (i in seq_along(intervention_to_GO_list)) {
  target_intervention <- names(intervention_to_GO_list)[i]
  idx <- GO_to_intervention_vec == target_intervention
  if (any(idx)) {
    intervention_to_GO_list[[i]] <- names(GO_to_intervention_vec)[idx]
  }
}

# --------------------------- GO lists per latent factor ----------------------

used_lf <- unique(latent_factor_of_intervention)
latent_factor_to_GO <- vector("list", length(used_lf))
names(latent_factor_to_GO) <- paste0("Latent_factor_", used_lf)

for (k in used_lf) {
  int_names <- names(latent_factor_of_intervention)[latent_factor_of_intervention == k]
  gos <- unique(unlist(intervention_to_GO_list[int_names]))
  latent_factor_to_GO[[paste0("Latent_factor_", k)]] <- gos
}

# Save counts per LF
sink(file.path(res_folder, "latent_factor_2_GO_list.txt"))
print(sapply(latent_factor_to_GO, length))
sink()

# --------------------------- Map GO IDs -> Terms -----------------------------

# Normalize GO IDs (replace '.' with ':' if present)
latent_factor_to_GO <- lapply(latent_factor_to_GO, function(x) gsub(".", ":", x, fixed = TRUE))

latent_factor_to_GO_terms <- lapply(latent_factor_to_GO, function(go_ids) {
  if (length(go_ids) == 0) character(0) else {
    out <- suppressMessages(select(GO.db, keys = go_ids,
                                   columns = c("DEFINITION", "TERM"),
                                   keytype = "GOID"))
    unique(out$TERM[match(go_ids, out$GOID, nomatch = 0)])
  }
})

# --------------------------- Save GO CSVs + Wordclouds -----------------------

for (lf_name in names(latent_factor_to_GO_terms)) {
  set.seed(SEED)

  go_ids  <- latent_factor_to_GO[[lf_name]]
  go_terms <- latent_factor_to_GO_terms[[lf_name]]
  out_csv <- file.path(res_folder, paste0(lf_name, ".csv"))

  # Save list (IDs and Terms)
  write.csv(data.frame(GO_ID = go_ids, GO_TERM = go_terms),
            file = out_csv, row.names = FALSE)

  # Skip wordcloud if no terms
  if (length(go_terms) == 0) next

  # Build & filter corpus
  go_corpus <- SimpleCorpus(VectorSource(go_terms))
  go_corpus <- tm_map(go_corpus, content_transformer(tolower))
  go_corpus <- tm_map(go_corpus, removeWords,
                      c(stopwords("english"),
                        "regulation", "process", "positive", "negative",
                        "pathways", "pathway", "reaction", "activity",
                        "involving", "metabolic", "protein", "involved",
                        "activation"))

  # Render wordcloud
  png(filename = file.path(res_folder, paste0(lf_name, ".png")),
      width = 2100, height = 2100, res = 300)
  wordcloud(
    words     = go_corpus,
    min.freq  = max(1, length(go_corpus) %/% 100),
    random.order = FALSE,
    rot.per   = 0,
    colors    = c("#5E4FA2", "#66C2A5", "#E6F598", "#ABDDA4"),
    scale     = c(1.5, 0.75)
  )
  dev.off()
}

# --------------------------- Causal subgraph export --------------------------

# Use only LFs that actually appear in the GO lists (names in latent_factor_to_GO_terms)
lf_idx <- as.numeric(gsub("Latent_factor_", "", names(latent_factor_to_GO_terms)))
subgraph <- causal_graph[lf_idx, lf_idx, drop = FALSE]

# Remove lower triangle & diagonal to avoid duplicates/self loops
subgraph[lower.tri(subgraph, diag = TRUE)] <- 0

# Pretty labels
rownames(subgraph) <- colnames(subgraph) <- names(latent_factor_to_GO_terms)

# Melt and sort by absolute weight
subgraph_long <- reshape2::melt(as.matrix(subgraph))
colnames(subgraph_long) <- c("from", "to", "coefficient")
subgraph_long <- subgraph_long[order(abs(subgraph_long$coefficient), decreasing = TRUE), ]

# Export top-k edges
write.csv(head(subgraph_long, 10), row.names = FALSE,
          file = file.path(res_folder, "causal_graph_10.csv"))
write.csv(head(subgraph_long, 15), row.names = FALSE,
          file = file.path(res_folder, "causal_graph_15.csv"))
write.csv(head(subgraph_long, 20), row.names = FALSE,
          file = file.path(res_folder, "causal_graph_20.csv"))

message("Done. Outputs written to: ", res_folder)
