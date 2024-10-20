import os
import random
from collections import Counter, defaultdict
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from torch.utils.data.sampler import Sampler


class Norman2019DataLoader:
    def __init__(
        self, num_gene_th=5, batch_size=32, dataname="Norman2019_raw"
    ):
        self.num_gene_th = num_gene_th
        self.batch_size = batch_size
        self.datafile = os.path.join('data',f"{dataname}.h5ad")

        # Initialize variables
        self.adata = None
        self.double_adata = None
        self.ptb_targets = None
        self.ptb_targets_affected = None
        self.gos = None
        self.rel_dict = None
        self.gene_go_dict = None
        self.ensembl_genename_mapping_rev = None
        self.gene_var="guide_ids"

        # Load the dataset
        self.load_norman_2019_dataset()

    def load_norman_2019_dataset(self):
        # Define file path
        fpath = self.datafile

        # Keep only single interventions
        adata = sc.read_h5ad(fpath)
        adata = adata[(~adata.obs["guide_ids"].str.contains(","))]

        # Build gene sets
        gos, GO_to_ensembl_id_assignment, gene_go_dict = self.load_gene_go_assignments(
            adata
        )

        # Compute perturbations with at least 1 gene set
        ptb_targets_affected, _, ensembl_genename_mapping_rev = (
            self.compute_affecting_perturbations(adata, GO_to_ensembl_id_assignment)
        )

        # Build gene-GO relationships
        rel_dict = self.build_gene_go_relationships(
            adata, gos, GO_to_ensembl_id_assignment
        )

        # Load double perturbation data
        ptb_targets = sorted(adata.obs["guide_ids"].unique().tolist())[1:]
        double_adata = sc.read_h5ad(fpath).copy()
        double_adata = double_adata[
            (double_adata.obs["guide_ids"].str.contains(","))
            & (
                double_adata.obs["guide_ids"].map(
                    lambda x: all([y in ptb_targets for y in x.split(",")])
                )
            )
        ]

        # Assign instance variables
        self.adata = adata
        self.double_adata = double_adata
        self.ptb_targets = ptb_targets
        self.ptb_targets_affected = ptb_targets_affected
        self.gos = gos
        self.rel_dict = rel_dict
        self.gene_go_dict = gene_go_dict
        self.ensembl_genename_mapping_rev = ensembl_genename_mapping_rev

    def load_gene_go_assignments(self, adata):
        # Filter genes not in any GO
        GO_to_ensembl_id_assignment = pd.read_csv(
            os.path.join("data", "go_kegg_gene_map.tsv"), sep="\t"
        )
        GO_to_ensembl_id_assignment.columns = ["GO_id", "ensembl_id"]

        # Reduce GOs to the genes we have in adata
        GO_to_ensembl_id_assignment = GO_to_ensembl_id_assignment[
            GO_to_ensembl_id_assignment["ensembl_id"].isin(adata.var_names)
        ]

        # Define GOs and filter
        gos = sorted(
            set(
                pd.read_csv(os.path.join("data", "topGO_uhler.tsv"), sep="\t")[
                    "PathwayID"
                ].values.tolist()
            )
        )
        GO_to_ensembl_id_assignment = GO_to_ensembl_id_assignment[
            GO_to_ensembl_id_assignment["GO_id"].isin(gos)
        ]

        # Keep only gene sets containing more than num_gene_th genes
        counter_genesets_df = pd.DataFrame(
            Counter(GO_to_ensembl_id_assignment["GO_id"]), index=[0]
        ).T
        genesets_in = counter_genesets_df[
            counter_genesets_df.values >= self.num_gene_th
        ].index
        GO_to_ensembl_id_assignment = GO_to_ensembl_id_assignment[
            GO_to_ensembl_id_assignment["GO_id"].isin(genesets_in)
        ]

        # Redefine GOs
        gos = sorted(GO_to_ensembl_id_assignment["GO_id"].unique())

        # Generate gene-GO dictionary
        gene_go_dict = defaultdict(list)
        for go, ens in GO_to_ensembl_id_assignment.values:
            gene_go_dict[ens].append(go)

        return gos, GO_to_ensembl_id_assignment, gene_go_dict

    def compute_affecting_perturbations(self, adata, GO_to_ensembl_id_assignment):
        # Filter interventions not in any GO
        ensembl_genename_mapping = pd.read_csv(
            os.path.join("data", "ensembl_genename_mapping.tsv"), sep="\t"
        )
        ensembl_genename_mapping_dict = dict(
            zip(
                ensembl_genename_mapping.iloc[:, 0], ensembl_genename_mapping.iloc[:, 1]
            )
        )
        ensembl_genename_mapping_rev = dict(
            zip(
                ensembl_genename_mapping.iloc[:, 1], ensembl_genename_mapping.iloc[:, 0]
            )
        )

        # Get intervention targets
        intervention_genenames = map(
            lambda x: ensembl_genename_mapping_dict.get(x, None),
            GO_to_ensembl_id_assignment["ensembl_id"],
        )
        ptb_targets = list(
            set(intervention_genenames).intersection(
                set([x for x in adata.obs["guide_ids"] if x != "" and "," not in x])
            )
        )
        ptb_targets_ens = list(
            map(lambda x: ensembl_genename_mapping_rev[x], ptb_targets)
        )

        return ptb_targets, ptb_targets_ens, ensembl_genename_mapping_rev

    def build_gene_go_relationships(self, adata, gos, GO_to_ensembl_id_assignment):
        # Get genes
        genes = adata.var.index.values
        go_dict = dict(zip(gos, range(len(gos))))
        gen_dict = dict(zip(genes, range(len(genes))))
        rel_dict = defaultdict(list)
        gene_set, go_set = set(genes), set(gos)

        for go, gen in zip(
            GO_to_ensembl_id_assignment["GO_id"],
            GO_to_ensembl_id_assignment["ensembl_id"],
        ):
            if (gen in gene_set) and (go in go_set):
                rel_dict[gen_dict[gen]].append(go_dict[go])

        return rel_dict

    def get_data(self, mode="train"):
        assert mode in ["train", "test"], "mode not supported!"

        if mode == "train":
            dataset = SCDataset(
                adata=self.adata,
                double_adata=self.double_adata,
                ptb_targets=self.ptb_targets,
                perturb_type="single",
            )
            train_idx, test_idx = self.split_scdata(
                dataset,
                split_ptbs=[
                    "ETS2",
                    "SGK1",
                    "POU3F2",
                    "TBX2",
                    "CBL",
                    "MAPK1",
                    "CDKN1C",
                    "S1PR2",
                    "PTPN1",
                    "MAP2K6",
                    "COL1A1",
                ],
            )  # Leave out some cells from the top 12 single target-gene interventions

            ptb_genes = dataset.ptb_targets

            dataset1 = Subset(dataset, train_idx)
            ptb_name = dataset.ptb_names[train_idx]
            dataloader = DataLoader(
                dataset1,
                batch_sampler=SCDATA_sampler(dataset1, self.batch_size, ptb_name),
                num_workers=0,
            )

            dim = dataset[0][0].shape[0]
            cdim = dataset[0][2].shape[0]

            dataset2 = Subset(dataset, test_idx)
            ptb_name = dataset.ptb_names[test_idx]
            dataloader2 = DataLoader(
                dataset2,
                batch_sampler=SCDATA_sampler(dataset2, 8, ptb_name),
                num_workers=0,
            )

            return dataloader, dataloader2, dim, cdim, ptb_genes

        elif mode == "test":
            dataset = SCDataset(
                adata=self.adata,
                double_adata=self.double_adata,
                ptb_targets=self.ptb_targets,
                perturb_type="double",
            )
            ptb_genes = dataset.ptb_targets

            dataloader = DataLoader(
                dataset,
                batch_sampler=SCDATA_sampler(dataset, self.batch_size),
                num_workers=0,
            )

            dim = dataset[0][0].shape[0]
            cdim = dataset[0][2].shape[0]

            return dataloader, dim, cdim, ptb_genes

    def split_scdata(self, scdataset, split_ptbs, pct=0.2):
        # Split data into training and testing
        test_idx = []
        for ptb in split_ptbs:
            idx = np.where(scdataset.ptb_names == ptb)[0]
            test_idx.append(np.random.choice(idx, int(len(idx) * pct), replace=False))
        test_idx = np.hstack(test_idx)
        train_idx = np.array([l for l in range(len(scdataset)) if l not in test_idx])
        return train_idx, test_idx

class Wessel2023HEK293DataLoader:
    def __init__(
        self, num_gene_th=5, batch_size=32, dataname="wessel_dataset/HEK293FT_carpool_processed"
    ):
        self.num_gene_th = num_gene_th
        self.batch_size = batch_size
        self.datafile = os.path.join('data',f"{dataname}.h5ad")

        # Initialize variables
        self.adata = None
        self.double_adata = None
        self.ptb_targets = None
        self.ptb_targets_affected = None
        self.gos = None
        self.rel_dict = None
        self.gene_go_dict = None
        self.ensembl_genename_mapping_rev = None
        self.gene_var='TargetGenes'

        #initialize dataset
        self.load_wessel2023_dataset()

    def load_wessel2023_dataset(self):

        # Define file path
        fpath = self.datafile

        # Keep only single interventions
        adata = sc.read_h5ad(fpath)
        adata.obs['TargetGenes'] = adata.obs['TargetGenes'].str.replace("_",",")
        adata.obs['TargetGenes'] = adata.obs['TargetGenes'].str.replace("NT", "")
        adata = adata[~adata.obs['TargetGenes'].str.contains(",")]

        # Build gene sets
        gos, GO_to_ensembl_id_assignment, gene_go_dict = self.load_gene_go_assignments(
            adata
        )

        # Compute perturbations with at least 1 gene set
        ptb_targets_affected, _, ensembl_genename_mapping_rev = (
            self.compute_affecting_perturbations(adata, GO_to_ensembl_id_assignment)
        )

        # Build gene-GO relationships
        rel_dict = self.build_gene_go_relationships(
            adata, gos, GO_to_ensembl_id_assignment
        )

        # Load double perturbation data
        ptb_targets = sorted(adata.obs["TargetGenes"].unique().tolist())[1:]
        double_adata = sc.read_h5ad(fpath).copy()
        double_adata.obs['TargetGenes'] = double_adata.obs['TargetGenes'].str.replace("_",",")
        double_adata = double_adata[
            (double_adata.obs["TargetGenes"].str.contains(","))
            & (
                double_adata.obs["TargetGenes"].map(
                    lambda x: all([y in ptb_targets for y in x.split(",")])
                )
            )
        ]

        # Assign instance variables
        self.adata = adata
        self.double_adata = double_adata
        self.ptb_targets = ptb_targets
        self.ptb_targets_affected = ptb_targets_affected
        self.gos = gos
        self.rel_dict = rel_dict
        self.gene_go_dict = gene_go_dict
        self.ensembl_genename_mapping_rev = ensembl_genename_mapping_rev

    def load_gene_go_assignments(self, adata):

        # Filter genes not in any GO
        GO_to_ensembl_id_assignment = pd.read_csv(
            os.path.join("data", "go_kegg_gene_map.tsv"), sep="\t"
        )
        GO_to_ensembl_id_assignment.columns = ["GO_id", "ensembl_id"]

        # Reduce GOs to the genes we have in adata
        GO_to_ensembl_id_assignment = GO_to_ensembl_id_assignment[
            GO_to_ensembl_id_assignment["ensembl_id"].isin(adata.var_names)
        ]

        # Define GOs and filter
        gos = sorted(
            set(
                pd.read_csv(os.path.join("data", "topGO_uhler.tsv"), sep="\t")[
                    "PathwayID"
                ].values.tolist()
            )
        )
        GO_to_ensembl_id_assignment = GO_to_ensembl_id_assignment[
            GO_to_ensembl_id_assignment["GO_id"].isin(gos)
        ]

        # Keep only gene sets containing more than num_gene_th genes
        counter_genesets_df = pd.DataFrame(
            Counter(GO_to_ensembl_id_assignment["GO_id"]), index=[0]
        ).T
        genesets_in = counter_genesets_df[
            counter_genesets_df.values >= self.num_gene_th
        ].index
        GO_to_ensembl_id_assignment = GO_to_ensembl_id_assignment[
            GO_to_ensembl_id_assignment["GO_id"].isin(genesets_in)
        ]

        # Redefine GOs
        gos = sorted(GO_to_ensembl_id_assignment["GO_id"].unique())

        # Generate gene-GO dictionary
        gene_go_dict = defaultdict(list)
        for go, ens in GO_to_ensembl_id_assignment.values:
            gene_go_dict[ens].append(go)

        return gos, GO_to_ensembl_id_assignment, gene_go_dict

    def compute_affecting_perturbations(self, adata, GO_to_ensembl_id_assignment):
        # Filter interventions not in any GO
        ensembl_genename_mapping = pd.read_csv(
            os.path.join("data", "ensembl_genename_mapping.tsv"), sep="\t"
        )
        ensembl_genename_mapping_dict = dict(
            zip(
                ensembl_genename_mapping.iloc[:, 0], ensembl_genename_mapping.iloc[:, 1]
            )
        )
        ensembl_genename_mapping_rev = dict(
            zip(
                ensembl_genename_mapping.iloc[:, 1], ensembl_genename_mapping.iloc[:, 0]
            )
        )

        # Get intervention targets
        intervention_genenames = map(
            lambda x: ensembl_genename_mapping_dict.get(x, None),
            GO_to_ensembl_id_assignment["ensembl_id"],
        )
        ptb_targets = list(
            set(intervention_genenames).intersection(
                set([x for x in adata.obs["TargetGenes"] if x != "" and "_" not in x])
            )
        )
        ptb_targets_ens = list(
            map(lambda x: ensembl_genename_mapping_rev[x], ptb_targets)
        )

        return ptb_targets, ptb_targets_ens, ensembl_genename_mapping_rev

    def build_gene_go_relationships(self, adata, gos, GO_to_ensembl_id_assignment):
        # Get genes
        genes = adata.var.index.values
        go_dict = dict(zip(gos, range(len(gos))))
        gen_dict = dict(zip(genes, range(len(genes))))
        rel_dict = defaultdict(list)
        gene_set, go_set = set(genes), set(gos)

        for go, gen in zip(
            GO_to_ensembl_id_assignment["GO_id"],
            GO_to_ensembl_id_assignment["ensembl_id"],
        ):
            if (gen in gene_set) and (go in go_set):
                rel_dict[gen_dict[gen]].append(go_dict[go])

        return rel_dict

    def get_data(self, mode="train"):
        
        assert mode in ["train", "test"], "mode not supported!"

        if mode == "train":
            dataset = SCDataset(
                adata=self.adata,
                double_adata=self.double_adata,
                ptb_targets=self.ptb_targets,
                perturb_type="single",
                gene_var='TargetGenes'
            )
            train_idx, test_idx = self.split_scdata(
                dataset,
                split_ptbs=[
                    "CD46",
                    "CD55",
                    "CD71",
                ],
            )  # Leave out some cells from the top 12 single target-gene interventions

            ptb_genes = dataset.ptb_targets
            dataset1 = Subset(dataset, train_idx)
            ptb_name = dataset.ptb_names[train_idx]
            dataloader = DataLoader(
                dataset1,
                batch_sampler=SCDATA_sampler(dataset1, self.batch_size, ptb_name),
                num_workers=0,
            )

            dim = dataset[0][0].shape[0]
            cdim = dataset[0][2].shape[0]

            dataset2 = Subset(dataset, test_idx)
            ptb_name = dataset.ptb_names[test_idx]
            dataloader2 = DataLoader(
                dataset2,
                batch_sampler=SCDATA_sampler(dataset2, 8, ptb_name),
                num_workers=0,
            )

            return dataloader, dataloader2, dim, cdim, ptb_genes

        elif mode == "test":
            dataset = SCDataset(
                adata=self.adata,
                double_adata=self.double_adata,
                ptb_targets=self.ptb_targets,
                perturb_type="double",
                gene_var="TargetGenes"
            )
            ptb_genes = dataset.ptb_targets

            dataloader = DataLoader(
                dataset,
                batch_sampler=SCDATA_sampler(dataset, self.batch_size),
                num_workers=0,
            )

            dim = dataset[0][0].shape[0]
            cdim = dataset[0][2].shape[0]

            return dataloader, dim, cdim, ptb_genes

    def split_scdata(self, scdataset, split_ptbs, pct=0.2):
        # Split data into training and testing
        test_idx = []
        for ptb in split_ptbs:
            idx = np.where(scdataset.ptb_names == ptb)[0]
            test_idx.append(np.random.choice(idx, int(len(idx) * pct), replace=False))
        test_idx = np.hstack(test_idx)
        train_idx = np.array([l for l in range(len(scdataset)) if l not in test_idx])
        return train_idx, test_idx

class SCDataset(Dataset):
    def __init__(
        self,
        adata,
        double_adata,
        ptb_targets,
        perturb_type="single",
        gene_var = "guide_ids"
    ):
        super().__init__()
        assert perturb_type in ["single", "double"], "perturb_type not supported!"

        self.genes = adata.var.index.tolist()
        self.ptb_targets = ptb_targets

        if perturb_type == "single":
            ptb_adata = adata[
                (~adata.obs[gene_var].str.contains(","))
                & (adata.obs[gene_var] != "")
            ].copy()

            # Keep only cells containing perturbed genes
            ptb_adata = ptb_adata[ptb_adata.obs[gene_var].isin(ptb_targets), :]

            self.ptb_samples = ptb_adata.X
            self.ptb_names = ptb_adata.obs[gene_var].values
            self.ptb_ids = self.map_ptb_features(
                self.ptb_targets, ptb_adata.obs[gene_var].values
            )

        elif perturb_type == "double":
            ptb_adata = double_adata[
                (double_adata.obs[gene_var].str.contains(","))
                & (double_adata.obs[gene_var] != "")
            ].copy()

            # Keep only cells containing perturbed genes
            ptb_adata = ptb_adata[
                ptb_adata.obs[gene_var].apply(
                    lambda x: all([y in ptb_targets for y in x.split(",")])
                ),
                :,
            ]

            self.ptb_samples = ptb_adata.X
            self.ptb_names = ptb_adata.obs[gene_var].values
            self.ptb_ids = self.map_ptb_features(
                self.ptb_targets, ptb_adata.obs[gene_var].values
            )

        self.ctrl_samples = adata[adata.obs[gene_var] == ""].X.copy()
        self.rand_ctrl_samples = self.ctrl_samples[
            np.random.choice(
                self.ctrl_samples.shape[0], self.ptb_samples.shape[0], replace=True
            )
        ]

    def __getitem__(self, item):
        x = torch.from_numpy(
            self.rand_ctrl_samples[item].toarray().flatten()
        ).double()
        y = torch.from_numpy(self.ptb_samples[item].toarray().flatten()).double()
        c = torch.from_numpy(self.ptb_ids[item]).double()
        return x, y, c

    def __len__(self):
        return self.ptb_samples.shape[0]

    def map_ptb_features(self, all_ptb_targets, ptb_ids):
        ptb_features = []
        for id in ptb_ids:
            feature = np.zeros(len(all_ptb_targets))
            feature[[all_ptb_targets.index(i) for i in id.split(",")]] = 1
            ptb_features.append(feature)
        return np.vstack(ptb_features)

class SCDATA_sampler(Sampler):
    def __init__(self, scdataset, batchsize, ptb_name=None):
        self.intervindices = []
        self.len = 0
        if ptb_name is None:
            ptb_name = scdataset.ptb_names
        for ptb in set(ptb_name):
            idx = np.where(ptb_name == ptb)[0]
            self.intervindices.append(idx)
            self.len += len(idx) // batchsize
        self.batchsize = batchsize

    def __iter__(self):
        comb = []
        for indices in self.intervindices:
            random.shuffle(indices)
            interv_batches = self.chunk(indices, self.batchsize)
            if interv_batches:
                comb += interv_batches

        combined = [batch.tolist() for batch in comb]
        random.shuffle(combined)
        return iter(combined)

    def __len__(self):
        return self.len

    @staticmethod
    def chunk(indices, chunk_size):
        split = torch.split(torch.tensor(indices), chunk_size)
        if len(indices) % chunk_size == 0:
            return split
        elif len(split) > 0:
            return split[:-1]
        else:
            return []


"""MMD LOSS"""

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        return

    def gaussian_kernel(
        self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None
    ):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [
            torch.exp(-L2_distance / bandwidth_temp)
            for bandwidth_temp in bandwidth_list
        ]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(
            source,
            target,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num,
            fix_sigma=self.fix_sigma,
        )
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss

# Assuming MMD_loss is defined elsewhere
class LossFunction:
    def __init__(self, MMD_sigma: float, kernel_num: int, matched_IO: bool = False):
        """
        Initializes the LossFunction class with required parameters.

        Args:
            MMD_sigma (float): Sigma value for MMD kernel.
            kernel_num (int): Number of kernels for MMD.
            matched_IO (bool): Whether matched input/output pairs are used.
        """
        self.MMD_sigma = MMD_sigma
        self.kernel_num = kernel_num
        self.matched_IO = matched_IO
        self.mse_loss_fn = nn.MSELoss()

    def compute_loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        x_recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        var: torch.Tensor,
        G: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the losses: MMD, MSE, KL-divergence, and L1 regularization.

        Args:
            y_hat (torch.Tensor): Predicted output.
            y (torch.Tensor): True output.
            x_recon (torch.Tensor): Reconstructed input.
            x (torch.Tensor): True input.
            mu (torch.Tensor): Latent mean.
            var (torch.Tensor): Latent variance.
            G (torch.Tensor): Optional adjacency matrix for graph regularization.

        Returns:
            Tuple: MMD loss, MSE loss, KL-divergence, L1 loss.
        """
        # Choose the appropriate matching function based on matched_IO
        matching_function_interv = (
            MMD_loss(fix_sigma=self.MMD_sigma, kernel_num=self.kernel_num)
            if not self.matched_IO
            else self.mse_loss_fn
        )

        # Reconstruction loss using MSE
        matching_function_recon = self.mse_loss_fn

        # MMD loss (or MSE if matched_IO is True)
        MMD = 0 if y_hat is None else matching_function_interv(y_hat, y)
        
        # Mean Squared Error (MSE) loss
        MSE = matching_function_recon(x_recon, x)

        # KL-Divergence (KLD) loss
        logvar = torch.log(var)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.mean(KLD_element).mul_(-0.5) / x.shape[0]

        # L1 Regularization (only if adjacency matrix G is provided)
        L1 = (
            torch.norm(torch.triu(G, diagonal=1), p=1)
            / torch.sum(torch.triu(torch.ones_like(G), diagonal=1))
            if G is not None
            else torch.tensor(0.0)
        )

        return MMD, MSE, KLD, L1