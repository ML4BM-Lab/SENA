# Script for choosing the pathways to include
# We must select pathways such that:
# 1: each intervention targets at most one pathway
# 2: each selected pathway must be targeted by one intervention
# 3: we maximize the number of gene 'covered' by at least one pathway

#### set up ####

# memory and libraries
rm(list = ls())
library(lpsymphony)
library(biomaRt)
library(tidyverse)
library(reshape)
library(GO.db)

# control panel
min_go_size <- 5 # requiring that GO process must have at least 5 genes that are also present in the gene expression matrix
interventions_to_eliminate <-  c('ENSG00000205189', 'ENSG00000184384', 'ENSG00000118058') # this interventions destabilize the model. To investigate

# loading data
ensembl_ids <- read.csv('data/Norman2019_gene_names.csv', stringsAsFactors = FALSE)
obs <- read.csv('data/Norman2019_obs.csv', stringsAsFactors = FALSE)

# extracting gene names
ensembl_ids <- ensembl_ids$gene_names

# refining obs
obs <- obs %>% filter(good_coverage == 'True', number_of_cells == 1)

# extracting the interventions
gene_name_interventions <- unique(obs$guide_ids)
gene_name_interventions <- gene_name_interventions[-grep(',', gene_name_interventions, fixed = TRUE)]
gene_name_interventions <- gene_name_interventions[gene_name_interventions != '']

#### retrieving GO and ENSEMBL information ####

# # connecting to biomart
# mart = useMart('ENSEMBL_MART_ENSEMBL', 
#                dataset = 'hsapiens_gene_ensembl') 

# # retrieving the ensembl names for the interventions
# ensembl_gene_name_interventions <- getBM(attributes = c('ensembl_gene_id', 
#                                               'external_gene_name'), 
#                                filters = 'external_gene_name', 
#                                values = gene_name_interventions, mart = mart)
# saveRDS(ensembl_gene_name_interventions, file = 'ensembl_gene_name_interventions.rds')
ensembl_gene_name_interventions <- readRDS('data/ensembl_gene_name_interventions.rds')
ensembl_interventions <- ensembl_gene_name_interventions$ensembl_gene_id

# interestingly, not all interventions are in the measured values
sum(ensembl_interventions %in% ensembl_ids) / length(ensembl_interventions)

# # retrieving the GO elements targeted by each intervention
# go_attributes <- c('go_id', 'name_1006', 'definition_1006', 'go_linkage_type', 
#                    'namespace_1003', 'goslim_goa_accession', 'goslim_goa_description')
# go_attributes <- c('go_id', 'name_1006')
# targeted_go <- getBM(attributes = c('ensembl_gene_id', go_attributes), 
#                      filters = 'ensembl_gene_id', 
#                      values = ensembl_interventions, mart = mart)
# targeted_go <- targeted_go[targeted_go$go_id != '', ]
# saveRDS(targeted_go, file = 'targeted_go.rds') # run on 14/02/2024
targeted_go <- readRDS('data/targeted_go.rds')

# focusing only on BP processes
bp_df <- select(GO.db, columns = 'GOID', keys = 'BP', keytype = 'ONTOLOGY')
targeted_go <- targeted_go[targeted_go$go_id %in% bp_df$GOID, ]

# check: how many GO are targeted by each intervention?
tmp <- targeted_go %>% group_by(ensembl_gene_id) %>% summarise(num_targets = n())
summary(tmp$num_targets) # each Go is targeted by many interventions

# # retrieving the list of genes covered by each GO term
# # "Essentially BioMart is not very clever" cit. 
# # See: https://bioinformatics.stackexchange.com/questions/15346/retrieving-all-genes-for-a-gene-ontology-term
# # So, we go for a foor loop
# unique_targeted_go <- unique(targeted_go$go_id)
# go_to_ensembl_id_list <- vector('list', length(unique_targeted_go))
# names(go_to_ensembl_id_list) <- unique_targeted_go
# for(i in 1:length(unique_targeted_go)){
#   print(paste0(i, ' out of ', length(unique_targeted_go)))
#   tmp_res <- tryCatch({
#     getBM(attributes = c('go_id', 'ensembl_gene_id'), 
#           filters = 'go', 
#           values = unique_targeted_go[i], 
#           mart = mart)
#     }, 
#     error = function(e){NULL})
#   if(!is.null(tmp_res)){
#     tmp_res <- unique(tmp_res[tmp_res$go_id == unique_targeted_go[i], ])
#     go_to_ensembl_id_list[[i]] <- tmp_res$ensembl_gene_id
#   }else{
#     sink(paste0('error_', i, '.txt'))
#     sink()
#   }
# }
# 
# # saving..
# saveRDS(go_to_ensembl_id_list, file = 'go_to_ensembl_id_list.rds')
go_to_ensembl_id_list <- readRDS('data/go_to_ensembl_id_list.rds')

# again, focusing only on BP
go_to_ensembl_id_list <- go_to_ensembl_id_list[names(go_to_ensembl_id_list) %in% bp_df$GOID]

# let's find in which sets each intervention is present
intervention_to_go <- vector('list', length(ensembl_interventions))
names(intervention_to_go) <- ensembl_interventions
for(i in 1:length(ensembl_interventions)){
  idx <- which(sapply(go_to_ensembl_id_list, function(x){
    ensembl_interventions[i] %in% x
  }))
  intervention_to_go[[i]] <- names(go_to_ensembl_id_list)[idx]
}

# the following code (up to "solving the lp problem") is quite messy.
# we would need a data structure for better managing the relationships 
# between interventions and GO terms to clean up the code

# keeping only intervention that have at least 1 GO... 
intervention_to_go <- intervention_to_go[sapply(intervention_to_go, length) > 0]

# and now we only need to keep the gene names that are in our measured genes 
go_to_ensembl_id_list <- sapply(go_to_ensembl_id_list, function(x){
  intersect(x, ensembl_ids)
})

# excluding GO with few genes
go_to_ensembl_id_list <- go_to_ensembl_id_list[sapply(go_to_ensembl_id_list, length) >= min_go_size]

# refining again the list of GO for each intervention now that we eliminate some of them...
intervention_to_go <- sapply(intervention_to_go, function(x){
  intersect(x, names(go_to_ensembl_id_list))
})

# removing problematic interventions
# all(interventions_to_eliminate %in% names(intervention_to_go))
if(length(interventions_to_eliminate) > 0){
  intervention_to_go <- intervention_to_go[-which(names(intervention_to_go) %in% 
                                                    interventions_to_eliminate)]
}

# refining again the go_to_ensembl_id_list...
tmp <- intervention_to_go[[1]]
for(i in 2:length(intervention_to_go)){tmp <- union(tmp, intervention_to_go[[i]])}
all(tmp %in% names(go_to_ensembl_id_list))
go_to_ensembl_id_list <- go_to_ensembl_id_list[tmp]

#### solving the lp problem ####

# Problem: we want to select GO processes so that each intervention targets only one
# GO process. We also want to maximize the number of genes that are "covered" by 
# the selected GO processes
#
# X_i: gene i. Binary variable equal to 1 if gene i is covered by at least one GO process
# Y_j: GO process j. Binary variable equal to 1 if gene set j is selected
# T_k: set of GO processes that are candidate targets for intervention k
# Objective function: max sum_i{ X_i } # We want to cover as many genes as possible
# Constraints C1: X_i >=  Y_j, for all j, and for each i in Y_j. # Equivalent to X_i - Y_j >= 0. 
# It means that if Y_j is selected, the corresponding genes are covered
# Constraints C2: sum{ Y_j, j in T_k } = 1, for all T_k. # Meaning that for each intervention only one gene set must be selected
# X_i, Y_J binary

# we can already assign interventions that target a single GO process
idx <- which(sapply(intervention_to_go, length) == 1)
if(length(idx) > 0){
  already_assigned_GO <- intervention_to_go[idx]
  intervention_to_go <- intervention_to_go[-idx]
}

# objective function
obj_fun <- c(rep(1, length(ensembl_ids)), rep(0, length(go_to_ensembl_id_list)))
names(obj_fun) <- c(ensembl_ids, names(go_to_ensembl_id_list))
n_vars <- length(obj_fun)

# directionality of optimization
max <- TRUE

# all binary variables
types <- rep('B', length(obj_fun))
names(types) <- names(obj_fun)

##### C1 constraints #####

# number of constraints
n_C1 <- sum(sapply(go_to_ensembl_id_list, length))

# constraint matrix 
C1_matrix <- matrix(0, n_C1, n_vars)
colnames(C1_matrix) <- names(obj_fun)
count <- 1
for(j in 1:length(go_to_ensembl_id_list)){
  for(i in 1:length(go_to_ensembl_id_list[[j]])){
    tmp_gene <- go_to_ensembl_id_list[[j]][i]
    tmp_go <- names(go_to_ensembl_id_list)[j]
    C1_matrix[count, tmp_gene] <- 1
    C1_matrix[count, tmp_go] <- -1
    count <- count + 1
  }
}

# checking that each row has exactly only a single value equal to 1 
# and a single value equal to -1
all(rowSums(C1_matrix == 1) == 1)
all(rowSums(C1_matrix == -1) == 1)

# constraints directionality and right hand values
C1_dir <- rep('>=', n_C1)
C1_rhv <- rep(0, n_C1)

##### C2 constraints #####

# number of constraints
n_C2 <- length(intervention_to_go)

# constraint matrix 
C2_matrix <- matrix(0, n_C2, n_vars)
colnames(C2_matrix) <- names(obj_fun)
rownames(C2_matrix) <- names(intervention_to_go)
for(i in 1:length(intervention_to_go)){
  tmp_gene_sets <- intervention_to_go[[i]]
  tmp_gene_sets <- intersect(tmp_gene_sets, colnames(C2_matrix))
  C2_matrix[i, tmp_gene_sets] <- 1
}

# constraints directionality and right hand values
C2_dir <- rep('==', n_C2)
C2_rhv <- rep(1, n_C2)

##### running the lp model #####

# full constraint matrix
# C_matrix <- rbind(C1_matrix, C2_matrix[c(1:74, 76:91, 94:100), ])
# C_dir <- c(C1_dir, C2_dir[c(1:74, 76:91, 94:100)])
# C_rhv <- c(C1_rhv, C2_rhv[c(1:74, 76:91, 94:100)])
C_matrix <- rbind(C1_matrix, C2_matrix)
C_dir <- c(C1_dir, C2_dir)
C_rhv <- c(C1_rhv, C2_rhv)

# running the model!
solution <- lpsymphony_solve_LP(obj_fun, C_matrix, C_dir, C_rhv, type = types, max = max)

# this ratio is < 1, so some pathways are hit more than once...
sum(solution$solution[5001:length(solution$solution)]) / length(intervention_to_go)

# assigning pathways
selected_pathways_bin <- solution$solution[(length(ensembl_ids) + 1):length(solution$solution)]
selected_pathways <- colnames(C2_matrix)[which(selected_pathways_bin == 1)]

# multiplying each row of C2 matrix for the pathway vector 
# each row then should sum up to 1
GO_assignment <- C2_matrix[ , (length(ensembl_ids) + 1):length(solution$solution)]
GO_assignment <- t(apply(GO_assignment, 1, function(x){x * selected_pathways_bin}))
# all(apply(GO_assignment, 1, sum) == 1)
suppressWarnings(GO_assignment <- melt(GO_assignment))
GO_assignment <- GO_assignment[GO_assignment$value == 1, ]
GO_assignment$value <- NULL
colnames(GO_assignment) <- c('intervention', 'pathway')

# adding the already assigned pathways...
for(i in 1:length(already_assigned_GO)){
  GO_assignment <- rbind(GO_assignment, 
                         data.frame(intervention = names(already_assigned_GO)[i], 
                                    pathway = already_assigned_GO[[i]]))
}

# merging interventions' ensembl ids and interventions' gene names
GO_assignment <- merge(GO_assignment, ensembl_gene_name_interventions, 
                       by.x = 'intervention', by.y = 'ensembl_gene_id', 
                       all = FALSE)
colnames(GO_assignment) <- c('intervention_ensembl_id', 'GO_id', 'intervention_gene_name')
GO_assignment <- GO_assignment[order(GO_assignment$intervention_gene_name), ]
rownames(GO_assignment) <- 1:dim(GO_assignment)[1]

# dealing with duplicated ensemble names
which(duplicated(GO_assignment$intervention_gene_name))
GO_assignment <- GO_assignment %>% group_by(intervention_gene_name) %>%
  summarise(intervention_ensembl_id = paste0(intervention_ensembl_id, collapse = ' // '), 
            GO_id = unique(GO_id), intervention_gene_name = unique(intervention_gene_name)) %>%
  as.data.frame()
which(duplicated(GO_assignment$intervention_gene_name))

# transforming go_to_ensembl_id_list to a data frame
go_to_ensembl_id_df <- data.frame(GO_id = c(), ensembl_id = c())
for(i in 1:length(go_to_ensembl_id_list)){
  if(names(go_to_ensembl_id_list)[i] %in% GO_assignment$GO_id){
    tmp <- data.frame(GO_id = names(go_to_ensembl_id_list)[i], 
                      ensembl_id = go_to_ensembl_id_list[[i]])
    go_to_ensembl_id_df <- rbind(go_to_ensembl_id_df, tmp)
  }
}

# writing 
write.csv(GO_assignment, row.names = FALSE, file = 'data/intervention_to_GO_assignment_gosize5.csv')
write.csv(go_to_ensembl_id_df, row.names = FALSE, file = 'data/GO_to_ensembl_id_assignment_gosize5.csv')