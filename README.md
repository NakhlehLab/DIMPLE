# DIMPLE

DIMPLE infers a phylogenetic network on many taxa by dividing the problem
along a tree of blobs (TOB), inferring a small network for every subset with
PhyloNet's maximum pseudo-likelihood search, and merging the subnetworks back
onto the TOB.

## Requirements

- Python 3.11 with `networkx`, `dendropy`, `numpy`, `pandas`
- `phynetpy` (pseudo-likelihood scoring in the merger)
- Java 17 and `PhyloNet.jar`, v3.8.2 (https://phylogenomics.rice.edu/html/phylonet.html)
- TREE-QMC v4.0.1 for the tree of blobs (https://github.com/molloy-lab/TREE-QMC)
- ASTRAL v5.7.8 (https://github.com/smirarab/ASTRAL) for the binary species
  tree used as the starting tree of every PhyloNet search

Put `src/` on `PYTHONPATH`:

```
export PYTHONPATH=/path/to/DIMPLE/src
```

## Inputs

- `gene_trees.tre`: one rooted newick per line, all containing the outgroup
  leaf (`OUT` below)
- `astral.tre`: a binary tree on the same taxa
- `tob_rooted.tre`: the tree of blobs, rooted at the outgroup (see below)

## Tree of blobs

One TREE-QMC run estimates the base tree from the gene trees and stores the
minimum quartet-test *p*-value on every branch; a second run contracts the
branches with the thresholds `alpha` and `beta` and roots the result at the
outgroup:

```
tree-qmc --iter_limit_blob 5000 --store_pvalue \
         -i gene_trees.tre -o base_tree_pvalues.tre

tree-qmc --blob --alpha 1e-7 --beta 0.95 --load_pvalue \
         --root OUT -i base_tree_pvalues.tre -o tob_rooted.tre
```

`1e-7` and `0.95` are TREE-QMC's defaults. `--iter_limit_blob` bounds the
*p*-value search per branch; TREE-QMC's default is twice the number of taxa
squared, and its authors recommend one fourth of the number of taxa squared
for large data sets. Do not use `--rootonly` to root the tree of blobs: it
re-resolves the polytomies that represent the blobs.

## Starting tree

ASTRAL estimates the binary tree that starts every PhyloNet search:

```
java -jar astral.5.7.8.jar -i gene_trees.tre -o astral.tre -t 4
```

## Running DIMPLE

The three stages share one directory, `dimple_out` by default; every command
below accepts another name in its place.

### 1. Divide

```
python -m dimple.divider.generate_k_divisions \
    --tob tob_rooted.tre --output_dir dimple_out \
    --size 12 --k 5 --seed 0 --outgroup OUT
```

Every blob of the TOB with more than `--size` taxa is partitioned into subsets
of at most `--size` taxa; `--k` partitions are drawn per blob (identical draws
are collapsed). Each partition is a *division run*, `dimple_out/blob00/run_000`
and so on, and each has one row per subset in
`subnetworks_output_metadata.csv`. The tree-like parts of the TOB go to
`dimple_out/non_blob/` and need no inference.

### 2. Infer the subnetworks

Every subnetwork is inferred individually. First list the division runs and
their subsets:

```
python -m dimple.phylonet.infer_subnetworks dimple_out \
    --gene-trees gene_trees.tre --base-tree astral.tre \
    --phylonet PhyloNet.jar --list
```

Then infer a subset with the number of reticulations chosen for it:

```
python -m dimple.phylonet.infer_subnetworks dimple_out \
    --gene-trees gene_trees.tre --base-tree astral.tre \
    --phylonet PhyloNet.jar --pl 4 \
    --only blob00/run_000 --subset 1 --max-ret 1
```

The gene trees and the base tree are restricted to the subset plus the
outgroup and rooted at it, PhyloNet's `InferNetwork_MPL` is run with
`--max-ret` reticulations, and the outgroup is pruned from the result. Every
bound tried for a subset is kept, so the same subset can be inferred again
with another `--max-ret`. `--subset` takes a list (`1,3`), `--pl` is the number
of PhyloNet threads, and `--force` redoes a result that already exists.

Finally choose the number of reticulations for each subset and write the
division run's `subnets.txt`:

```
python -m dimple.phylonet.infer_subnetworks dimple_out \
    --gene-trees gene_trees.tre --base-tree astral.tre \
    --phylonet PhyloNet.jar \
    --only blob00/run_000 --assemble 1,2,3,0,0,0
```

The list has one number per subset in `--list` order: subset 1 takes its
network inferred with 1 reticulation, subset 2 the one with 2, and so on. A
subset without a result at the listed bound, or a list of the wrong length, is
refused. Repeat for every division run.

To choose the bound for a subset, infer it with `--max-ret` 0, 1, 2, 3 or
higher and compare the log pseudo-likelihood PhyloNet reports in each
`subgenes-out/subset<i>/mpl_ret<r>.log`; the point at which an extra
reticulation stops improving the score is a reasonable choice. Each search
starts from the ASTRAL tree restricted to the subset's taxa.

Omitting `--only`, `--subset` and `--assemble` infers every subset of every
division run with the same `--max-ret` and writes `subnets.txt` directly;
`--parallel N` then runs N division runs at once.

Inside each division run, `subgenes-out/` holds:

```
subset<i>/leaf_subset.txt    the subset's taxa
subset<i>/base_tree.tre      starting tree restricted to the subset
subset<i>/tmp_mpl.nex        the NEXUS handed to PhyloNet in the latest inference
subset<i>/subnet_ret<r>.txt  inferred network for r reticulations, outgroup removed
subset<i>/mpl_ret<r>.log     raw PhyloNet output for that inference
subnets.txt                  line i = the network chosen for subset i
inputs.json                  which inputs every result and subnets.txt came from
mpl_runtimelog.txt           wall time and status per inference
```

A result is only reused when the gene trees, base tree, PhyloNet jar and
metadata it was made from are unchanged; `--list` marks anything else STALE.

### 3. Merge

```
python -m dimple.merger.merger_full_pip dimple_out \
    --tob tob_rooted.tre --gene-trees gene_trees.tre \
    --max-runs 10 --out-name full_merger
```

The merger reads `subnets.txt` of up to `--max-runs` division runs per blob,
merges the subnetworks of each blob, grafts the blobs and the tree-like parts
back onto the TOB, and adds the reticulations supported across division runs.
The final network is `dimple_out/full_merger/merged_full.nwk`; the same folder
has `summary.tsv` and `pipeline_timings.json`. `--subgenes-out-dir` points the
merger at a differently named stage-2 output.

