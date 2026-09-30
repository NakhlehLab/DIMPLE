#!/usr/bin/env python3
"""
DIMPLE, end to end: gene trees + tree of blobs -> phylogenetic network.

This runs the three stages of the method in one process:

  1. Division   (dimple.divider.generate_k_divisions)
                 cuts the tree of blobs into subproblems of at most --size
                 taxa, sampling --k alternative divisions per blob.
  2. Inference  (dimple.phylonet.infer_subnetworks)
                 runs PhyloNet's InferNetwork_MPL on every division, seeded
                 from the binary starting tree given by --base-tree.
  3. Merger     (dimple.merger.merger_full_pip)
                 stitches the inferred subnetworks back into one network on
                 the full taxon set, using the tree of blobs as the scaffold.

Each stage is also runnable on its own; see the module docstrings. Stage 2 is
skipped for divisions that already have their PhyloNet inference, so an
interrupted run can be resumed by re-issuing the same command (pass --force to
redo the inference from scratch).

Inputs
  --gene-trees  one newick gene tree per line, on the full taxon set,
                including the outgroup leaf named by --outgroup.
  --tob         the tree of blobs, rooted at the outgroup (TREE-QMC --blob).
  --base-tree   a BINARY starting tree on the same taxa: the ASTRAL tree
                estimated from the same gene trees, or the TREE-QMC first-pass
                tree. The blob-collapsed tree of blobs has polytomies and
                PhyloNet rejects it as a start.

Output
  <out>/dimple_network.nwk   the estimated network (extended newick)
  <out>/divisions/           divisions, per-division PhyloNet runs, and the
                             merger's working files and timings

Usage:
    python -m dimple.run_dimple \\
        --gene-trees gene_trees.tre \\
        --tob tob_rooted.tre \\
        --base-tree astral.tre \\
        --phylonet PhyloNet.jar \\
        --out dimple_out
"""
import os
import json
import time
import shutil
import argparse
import tempfile

from dimple.divider.generate_k_divisions import process_division_leafsets
from dimple.phylonet.infer_subnetworks import (
    infer_divisions, DEFAULT_JAVA_MEM, DEFAULT_SUBGENES_DIR,
)
from dimple.merger.merger_full_pip import run_full_merger

FINAL_NETWORK_NAME = 'dimple_network.nwk'
TIMINGS_NAME = 'dimple_timings.json'


def run_dimple(gene_trees, tob, base_tree, phylonet_jar, out_dir,
               outgroup='OUT', size=12, k=15, seed=0, max_ret=1,
               parallel=1, threads=1, java='java', java_mem=DEFAULT_JAVA_MEM,
               max_runs=None, force=False,
               skip_division=False, skip_inference=False):
    """Run all three DIMPLE stages. Returns the path of the final network."""
    os.makedirs(out_dir, exist_ok=True)
    final_path = os.path.join(out_dir, FINAL_NETWORK_NAME)
    timings_path = os.path.join(out_dir, TIMINGS_NAME)
    previous = [p for p in (final_path, timings_path) if os.path.lexists(p)]
    if previous:
        archive = tempfile.mkdtemp(prefix='.previous-result-', dir=out_dir)
        for path in previous:
            os.replace(path, os.path.join(archive, os.path.basename(path)))
        print(f'  Previous final result archived at {archive}', flush=True)

    for path, what in ((gene_trees, 'gene trees'), (tob, 'tree of blobs'),
                       (base_tree, 'base tree'), (phylonet_jar, 'PhyloNet jar')):
        if not os.path.exists(path):
            raise SystemExit(f'ERROR: {what} not found: {path}')

    divisions_dir = os.path.join(out_dir, 'divisions')
    stage_seconds = {}

    # ------------------------------------------------------------------
    # Stage 1 — division
    # ------------------------------------------------------------------
    print(f'\n{"="*70}\n[DIMPLE stage 1/3] division\n{"="*70}', flush=True)
    t0 = time.time()
    if skip_division and os.path.isdir(divisions_dir):
        print(f'  reusing existing divisions in {divisions_dir}', flush=True)
    else:
        with open(tob) as f:
            tob_str = f.read().strip()
        outgroup_leaves = [s.strip() for s in outgroup.split(',') if s.strip()]
        process_division_leafsets(
            tob_str, SIZE=size, output_dir=divisions_dir, k=k, seed=seed,
            outgroup_leaves=outgroup_leaves)
    stage_seconds['stage1_division'] = time.time() - t0

    # ------------------------------------------------------------------
    # Stage 2 — PhyloNet inference per division
    # ------------------------------------------------------------------
    print(f'\n{"="*70}\n[DIMPLE stage 2/3] PhyloNet inference per division'
          f'\n{"="*70}', flush=True)
    t0 = time.time()
    if skip_inference:
        print('  --skip-inference given: reusing existing subnets.txt files',
              flush=True)
    else:
        # The outgroup label is a single leaf in the gene trees and base tree.
        # A multi-taxon outgroup must be collapsed to one leaf beforehand; only
        # the first label is used here.
        outgroup_leaf = outgroup.split(',')[0].strip()
        _n_ok, n_fail = infer_divisions(
            divisions_dir, gene_trees, base_tree, phylonet_jar,
            max_ret=max_ret, subgenes_out_dir=DEFAULT_SUBGENES_DIR,
            outgroup=outgroup_leaf, parallel=parallel, threads=threads,
            java_mem=java_mem, java=java,
            force=force, max_runs=max_runs)
        if n_fail:
            raise SystemExit(
                f'ERROR: PhyloNet inference failed for {n_fail} division '
                f'directory(ies). The merger would silently drop or misalign '
                f'those subnetworks, so DIMPLE stops here. See the '
                f'mpl_runtimelog.txt files under {divisions_dir}.')
    stage_seconds['stage2_phylonet'] = time.time() - t0

    # ------------------------------------------------------------------
    # Stage 3 — merger
    # ------------------------------------------------------------------
    print(f'\n{"="*70}\n[DIMPLE stage 3/3] merger\n{"="*70}', flush=True)
    t0 = time.time()
    merged = run_full_merger(
        divisions_dir, tob=tob, gene_trees=gene_trees,
        subgenes_out_dir=DEFAULT_SUBGENES_DIR, max_runs=max_runs,
        out_name='full_merger')
    stage_seconds['stage3_merger'] = time.time() - t0
    stage_seconds['total'] = sum(stage_seconds.values())

    if not merged or not os.path.isfile(merged) or os.path.getsize(merged) == 0:
        raise SystemExit(
            f'ERROR: the merger did not produce a final network '
            f'(expected {merged}).')
    # Publish the network last: a failure copying or serializing results must
    # not expose a partial file under the advertised final output name.
    with tempfile.TemporaryDirectory(prefix='.dimple-result-', dir=out_dir) as stage:
        staged_network = os.path.join(stage, FINAL_NETWORK_NAME)
        staged_timings = os.path.join(stage, TIMINGS_NAME)
        shutil.copyfile(merged, staged_network)
        with open(staged_timings, 'w') as f:
            json.dump({**stage_seconds,
                       'note': 'wall-clock seconds per DIMPLE stage. Per-stage '
                               'merger detail is in '
                               'divisions/full_merger/pipeline_timings.json; '
                               'per-division PhyloNet times are in each '
                               'mpl_runtimelog.txt.'}, f, indent=2)
        os.replace(staged_timings, timings_path)
        try:
            os.replace(staged_network, final_path)
        except BaseException:
            os.remove(timings_path)
            raise

    print(f'\n{"="*70}\n✓ DIMPLE done.\n'
          f'  network  → {final_path}\n'
          f'  workdir  → {divisions_dir}\n'
          f'  division {stage_seconds["stage1_division"]:.1f}s | '
          f'phylonet {stage_seconds["stage2_phylonet"]:.1f}s | '
          f'merger {stage_seconds["stage3_merger"]:.1f}s | '
          f'total {stage_seconds["total"]:.1f}s\n{"="*70}', flush=True)
    return final_path


def main():
    ap = argparse.ArgumentParser(
        description='Run DIMPLE end to end: divide the tree of blobs, infer '
                    'each division with PhyloNet, merge back into one network.')
    ap.add_argument('--gene-trees', required=True,
                    help='Gene trees, one newick per line, including the '
                         'outgroup leaf.')
    ap.add_argument('--tob', required=True,
                    help='Tree of blobs, rooted at the outgroup.')
    ap.add_argument('--base-tree', required=True,
                    help='Binary starting tree for PhyloNet: the ASTRAL '
                         'tree estimated from the same gene trees, or the '
                         'TREE-QMC first-pass tree. Must be binary.')
    ap.add_argument('--phylonet', required=True, help='Path to PhyloNet.jar.')
    ap.add_argument('--out', required=True, help='Output directory.')
    ap.add_argument('--outgroup', default='OUT',
                    help='Outgroup leaf label (default: OUT). Pruned from the '
                         'tree of blobs before division, and attached to each '
                         'division for rooting during inference. A multi-taxon '
                         'outgroup must be collapsed to a single leaf first; '
                         'extra comma-separated labels are pruned from the '
                         'tree of blobs only.')
    ap.add_argument('--size', type=int, default=12,
                    help='Maximum taxa per division (default: 12).')
    ap.add_argument('--k', type=int, default=15,
                    help='Alternative divisions sampled per blob (default: 15).')
    ap.add_argument('--seed', type=int, default=0,
                    help='Random seed for the divider (default: 0).')
    ap.add_argument('--max-ret', type=int, default=1,
                    help='Reticulation bound per division (default: 1).')
    ap.add_argument('--max-runs', type=int, default=None,
                    help='Use only the first K divisions per blob (default: '
                         'all of them).')
    ap.add_argument('--parallel', type=int, default=1,
                    help='Divisions inferred concurrently (default: 1).')
    ap.add_argument('--threads', type=int, default=1,
                    help="Threads per PhyloNet search (PhyloNet's -pl).")
    ap.add_argument('--java', default='java', help='Java executable.')
    ap.add_argument('--java-mem', default=DEFAULT_JAVA_MEM,
                    help=f'-Xmx for PhyloNet (default: {DEFAULT_JAVA_MEM}).')
    ap.add_argument('--force', action='store_true',
                    help='Re-run PhyloNet on divisions that already have an '
                         'inference.')
    ap.add_argument('--skip-division', action='store_true',
                    help='Reuse the divisions already in <out>/divisions.')
    ap.add_argument('--skip-inference', action='store_true',
                    help='Reuse the PhyloNet inferences already in '
                         '<out>/divisions and go straight to the merger.')
    args = ap.parse_args()

    run_dimple(args.gene_trees, args.tob, args.base_tree, args.phylonet,
               args.out, outgroup=args.outgroup, size=args.size, k=args.k,
               seed=args.seed, max_ret=args.max_ret, parallel=args.parallel,
               threads=args.threads, java=args.java, java_mem=args.java_mem,
               max_runs=args.max_runs,
               force=args.force, skip_division=args.skip_division,
               skip_inference=args.skip_inference)


if __name__ == '__main__':
    main()
