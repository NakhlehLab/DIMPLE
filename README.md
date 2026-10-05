# DIMPLE

DIMPLE infers a phylogenetic network on many taxa by dividing the problem
along a tree of blobs (TOB), inferring a small network for every subset with
PhyloNet's maximum pseudo-likelihood search, and merging the subnetworks back
onto the TOB.

The workflow uses TOB-QMC, implemented in TREE-QMC v4.0.1, PhyloNet v3.8.2,
and ASTRAL v5.7.8.

## Requirements

- Python 3.11 with `networkx`, `dendropy`, `numpy`, `pandas`
- PhyNetPy 0.6.0 (`pip install phynetpy==0.6.0`), for the pseudo-likelihood
  scoring in the merger
- Java 17 and `PhyloNet.jar` v3.8.2 (https://phylogenomics.rice.edu/html/phylonet.html)
- TREE-QMC v4.0.1 (https://github.com/molloy-lab/TREE-QMC)
- ASTRAL v5.7.8 (https://github.com/smirarab/ASTRAL)

Put `src/` on `PYTHONPATH`:

```
export PYTHONPATH=/path/to/DIMPLE/src
```

## Inputs

- `gene_trees.tre`: one rooted newick per line, all containing the outgroup
  leaf (`OUT` below)
- `astral.tre`: a binary tree on the same taxa
- `tob_rooted.tre`: the tree of blobs, rooted at the outgroup (see below)

The example commands below use `gene_trees.tre`, containing one Newick gene
tree per line on the full taxon set, and `OUT` as the outgroup label. The file
`tob_rooted.tre` denotes the inferred TOB rooted at the designated outgroup
before division. These examples illustrate the workflow; the settings used in
each experiment are described in the paper.

## Tree of blobs (TOB)

**Step 1: Infer a base tree and annotate its branches.**
First, we estimate the base tree from the gene trees, apply the quartet
hypothesis tests around each branch, and store the resulting minimum *p*-value
on that branch. The default iteration limit for the *p*-value search is two
times the number of taxa squared, while the authors of TREE-QMC recommend
setting the iteration limit to one fourth the number of taxa squared for large
data sets.

```
tree-qmc --iter_limit_blob 5000 --store_pvalue \
         -i gene_trees.tre -o base_tree_pvalues.tre
```

**Step 2: Construct and root the TOB.**
Contract the annotated branches using the stored *p*-values with the
thresholds `alpha = 1e-7` and `beta = 0.95`, and root the result at the
outgroup:

```
tree-qmc --blob --alpha 1e-7 --beta 0.95 --load_pvalue \
         --root OUT -i base_tree_pvalues.tre -o tob_rooted.tre
```

## ASTRAL

ASTRAL provides the starting tree for subnetwork inference:

```
java -jar astral.5.7.8.jar \
     -i gene_trees.tre -o astral.tre -t 4
```

## DIMPLE

The DIMPLE workflow consists of division, subnetwork inference with PhyloNet,
and merging. All three stages share one output directory chosen by the user,
named `dimple_out` in the commands below.

### Division

```
python -m dimple.divider.generate_k_divisions \
    --tob tob_rooted.tre --output_dir dimple_out \
    --size 12 --k 5 --seed 0 --outgroup OUT
```

### Subnetwork inference

Every subnetwork is inferred individually. `--list` prints each division run
with the indices of its subsets. A subnetwork can then be inferred for each
subset with the reticulation limit chosen for it; the inference results
for each tested reticulation limit are retained. Finally, `--assemble` writes
the division run's `subnets.txt` from the results, taking for each subset the
network inferred under the listed reticulation limit (one number per subset
in order).

*List the subsets in each division run.*

```
python -m dimple.phylonet.infer_subnetworks dimple_out \
    --gene-trees gene_trees.tre --base-tree astral.tre \
    --phylonet PhyloNet.jar --list
```

*Infer a subnetwork for one subset with a specified reticulation limit.*

```
python -m dimple.phylonet.infer_subnetworks dimple_out \
    --gene-trees gene_trees.tre --base-tree astral.tre \
    --phylonet PhyloNet.jar --pl 4 \
    --only blob00/run_000 --subset 1 --max-ret 1
```

*Assemble the selected subnetworks for a division run.* This example assumes
three subsets, already inferred under limits 1, 1, and 0, respectively. Repeat
inference and assembly for every division run.

```
python -m dimple.phylonet.infer_subnetworks dimple_out \
    --gene-trees gene_trees.tre --base-tree astral.tre \
    --phylonet PhyloNet.jar \
    --only blob00/run_000 --assemble 1,1,0
```

Omitting `--only` and `--subset` runs subnetwork inference for every subset
using the same `--max-ret` value.

Inside each division run, `subgenes-out/` holds:

```
subset<i>/leaf_subset.txt    the subset's taxa
subset<i>/base_tree.tre      starting tree restricted to the subset
subset<i>/tmp_mpl.nex        the NEXUS handed to PhyloNet in the latest inference
subset<i>/subnet_ret<r>.txt  inferred network under reticulation limit r, outgroup removed
subset<i>/mpl_ret<r>.log     raw PhyloNet output for that inference
subnets.txt                  line i = the network chosen for subset i
inputs.json                  which inputs every result and subnets.txt came from
mpl_runtimelog.txt           wall time and status per inference
```

### Merging

Run the merger using the assembled subnetworks:

```
python -m dimple.merger.merger_full_pip dimple_out \
    --tob tob_rooted.tre --gene-trees gene_trees.tre \
    --max-runs 10 --out-name full_merger
```

The final network is written to `dimple_out/full_merger/merged_full.nwk`; the
same folder has `summary.tsv` and `pipeline_timings.json`.

These example settings use a maximum subset size of 12 taxa, five division
attempts per megablob, and random seed 0. For subnetwork inference, the user
specifies the maximum number of reticulations allowed; the example sets this
limit to 1. For a more thorough analysis, users can repeat inference with
limits of 0, 1, 2, 3, or higher and examine how log pseudo-likelihood improves
as additional reticulations are allowed. These score profiles can inform
manual selection of the reticulation count. Each search uses the ASTRAL tree
restricted to the subset taxa as its starting tree.
