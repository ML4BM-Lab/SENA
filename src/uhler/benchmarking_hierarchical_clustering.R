# Load the libraries
library(gplots)
library(colorspace)
library(stats)
library(data.table)
library(ggplot2)
library(viridis)
library(pheatmap)
library(seriation)

# Define the file path
modeltype <- "full_go_NA+deltas"
layername <- "z"
mode <- 'logFC'
file_path <- paste0("./result/",modeltype,"/activation_scores_",mode,"_DEA_layer_",layername,"_matrix.tsv")

# Load the matrix
matrix_data <- read.delim(file_path, sep = "\t", header = TRUE)
rownames(matrix_data) <- matrix_data[,1]
matrix_data <- matrix_data[,-1]
hm.colors <- viridis(100)
th <- 0.95
matrix_data[matrix_data<th] = 0

#
distance.row = dist(as.matrix(matrix_data), method = "euclidean")
distance.col = dist(t(as.matrix(matrix_data)), method = "euclidean")
cluster.row = hclust(distance.row, method = "average")
cluster.col = hclust(distance.col, method = "average")

# Use seriation to reorder the matrix
seriation_row <- seriate(distance.row, method = "OLO")
seriation_col <- seriate(distance.col, method = "OLO")
ordered_row <- get_order(seriation_row, margin=1)
ordered_col <- get_order(seriation_col, margin=2)
ordered_data <- matrix_data[ordered_row, ordered_col]

# Create a heatmap using heatmap.2 function
png(filename = paste0("./figures/uhler_paper/",modeltype,"/activation_scores_layer_",layername,"_heatmap_",mode,"_th_",toString(th),".png"), width = 500, height = 1000)
nrows <- nrow(matrix_data)
ncols <- ncol(matrix_data)
heatmap.2(as.matrix(ordered_data), trace = "none", dendrogram = "both",
           #Rowv = reorder(as.dendrogram(cluster.row), nrows:1),
           #Colv = reorder(as.dendrogram(cluster.col), ncols:1),
           Rowv = FALSE,
           Colv = FALSE,
           margins = c(7, 7), col = hm.colors, na.color = "white", srtCol = 45)
dev.off()