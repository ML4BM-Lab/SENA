library(ontologyIndex)
library(ontologySimilarity)
library(anndata)
data(go)
data(gene_GO_terms)

## load perturbseq dataset
dataset <- read_h5ad('datasets/Norman2019_raw.h5ad')

#get genes with only 1 knockdown
knockdown_genes <- as.character(dataset$'obs'$'guide_ids')
single_kd_cells <- dataset$'obs'[!grepl(",", knockdown_genes) & as.logical(sapply(knockdown_genes, nchar) > 0),]
single_kd_genes <- sort(unique(as.character(single_kd_cells$'guide_ids')))

## get all children of biological processes
## GO:0008150 <- biological process

category <- 'GO:0008150'
manage_tree <- function(gene){

    go_terms <- gene_GO_terms[gene][[1]]
    ref <- go$children[category][[1]]

    build_tree <- function(goterm){

    }

}

get_onechildren_go <- function(selected_genes){

    ## get level 2
    lev2 <- go$children[category][[1]]
    gene_lower_onechildren_node <- list()

    for (gene in selected_genes){

        #print(gene)
        gene_goterms <- gene_GO_terms[gene][[1]]
        bp_boolean <- intersect(lev2, gene_goterms)
        level_reference <- lev2
        level <- 1
        num_intersects <- c()

        ## go down if there is only one children
        while (TRUE){

            genes_intersect <- intersect(level_reference, gene_goterms)
            num_intersects <- c(num_intersects, length(genes_intersect))

            if(length(genes_intersect) == 1){
                level_reference <- go$children[genes_intersect][[1]]
                term <- genes_intersect
            }
            else{
                
                if (!(identical(bp_boolean, character(0)))){
                    gene_lower_onechildren_node[[gene]] = list("term" = term, "level" = level, "num_intersects" = num_intersects)
                }
                break

            }
            level <- level + 1
        }
    }

    # print(table(sapply(gene_lower_onechildren_node, function(x) x$level)))
    # print(table(sapply(gene_lower_onechildren_node, function(x) x$num_intersects)))

    return (gene_lower_onechildren_node)
}

selected_genes <- single_kd_genes #sample(names(gene_GO_terms),100) #example
onechildren_l <- get_onechildren_go(selected_genes)
print(onechildren_l)

