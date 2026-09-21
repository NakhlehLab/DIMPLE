#!/usr/bin/env python3
"""
Full DIMPLE merger pipeline: per-blob inference + combine + merge into TOB.

End-to-end given a single dataset:
  1. Per-blob inference  : run DIMPLE merger (run_merger from blob_merger) on
                            each blob's PhyloNet-inferred subnets, producing
                            an inferred network per blob.
  2. Build TOB-restricted prune subnets from the cut-time-siblings metadata.
  3. Combine             : run add_pruned_subtree per blob, attaching prune
                            subnets at MRCA of cut-time sibling leaves.
  4. Merge into TOB      : graft per-blob combined networks back into the
                            iqtree TOB scaffold.

Inputs:
  divisions_dir          : <ds>/divisions_final_phylonet — has blob*/run_*/
                            subgenes-out/subnets.txt (PhyloNet inferences).
  --metadata-dir         : <ds>/divisions3_test (cut-time-siblings metadata).
  --tob                  : iqtree TOB newick.
  --gene-trees           : iqtree gene trees (default: <parent>/1k_iqtree_reroot.txt).

This script is GT-FREE and does NOT perform any evaluation. It produces
network newicks only.

Outputs (all under <divisions_dir>/full_merger/):
  blob*/merged_network.nwk           — per-blob inferred network
  blob*/inputs_full.txt              — phylonet inputs used for inference
  blob*/dt_selected_subnets.txt      — DT-selected displayed trees (1 per
                                       phylonet network, picked by PL scoring)
  blob*/compat_subnets.txt           — compatible subset of DT-selected
                                       (the actual NJMerge input)
  blob*/base_tree.nwk                — NJMerge output (pre-retic)
  blob*/timing.json                  — per-blob inference timing + step breakdown
                                       (DM, DT-select, compat, NJMerge, retics)
  combined_blobs/<blob>_combined.nwk — per-blob combined (with prunes attached)
  tob_prune_subnets.txt              — TOB-restricted prune newicks
  merged_full.nwk                    — final whole-network output
  summary.tsv                        — per-blob inference summary (no evaluation)
  pipeline_timings.json              — wall time for each of stages 1..4 + total
  stdout.txt / stderr.txt            — wrapped /usr/bin/time -v logs (whole-
                                       pipeline wall time + Maximum RSS)

Usage:
    conda run -n phylo-env python -u -m \\
        dimple.merger.merger_full_pip \\
        data/division_iqtree_tob/lvl2/n00/divisions_final_phylonet \\
        --metadata-dir data/division_iqtree_tob/lvl2/n00/divisions3_test
"""
import os
# Cap BLAS thread fan-out before any numpy/scipy/etc. import. OpenBLAS reads
# these env vars eagerly during initialization; setting them later has no
# effect. Without this, scipy.linalg can try to spawn 64+ threads and trip
# RLIMIT_NPROC on shared hosts.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import sys
import csv
import time
import json
import glob
import argparse
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# src/dimple/merger -> repo root is three levels up (four pointed at its parent)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from dimple.utils.network_util import (
    newick_to_nx, build_newick_from_graph, get_leafset,
    extract_subnetwork_by_leaves, contract_degree2_nodes,
    clean_extended_newick, strip_branch_lengths,
)
from dimple.merger.blob_merger import (
    get_blob_name, list_run_dirs, read_phylonet_subnets,
    analyze_blob, write_inputs_file, run_merger,
    _default_gene_trees,
)
from dimple.merger.merger_util import compute_dm_full
from dimple.utils.division_util import (
    load_or_build_tob_subnets as _load_non_blob_tob_subnets,
)


def _cpu_now():
    """Total CPU time consumed so far (this process + already-reaped children),
    summed across user + system. Includes subprocesses launched via
    subprocess.run/Popen.wait once they have terminated."""
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


def per_blob_infer(blob_dir, blob_out_dir, gene_trees, subnet_source,
                    verbose=True, max_runs=None, dm=None):
    """Run DIMPLE merger on one blob; write merged_network.nwk + inputs_full.txt
    + timing.json into blob_out_dir. Returns (status, info_dict).

    `dm` is the whole-dataset gene-tree distance matrix (see
    `compute_dm_full`), built once by the caller and shared across blobs. Pass
    None to have each blob build its own the old way.

    If max_runs is set, only the first `max_runs` run dirs (sorted by name —
    so run_000, run_001, …) are used."""
    os.makedirs(blob_out_dir, exist_ok=True)
    runs = list_run_dirs(blob_dir)
    if max_runs is not None and max_runs > 0:
        runs = runs[:max_runs]
    if not runs:
        return 'no_runs', None
    if len(runs) == 1:
        run_dir = os.path.join(blob_dir, runs[0])
        subnets = read_phylonet_subnets(run_dir, subnet_source)
        if not subnets:
            return 'shortcut_empty', None
        # True shortcut only when this single run has a SINGLE subnet (no
        # merging needed). If the run has multiple subnets (one per retic
        # count), fall through to the full pipeline so DT-select + NJ +
        # add_retics produce a single coherent merged_network.nwk instead
        # of the multi-line newick the shortcut used to emit.
        if len(subnets) == 1:
            run_subnets = {runs[0]: [nwk for _, nwk in subnets]}
            inputs_full = os.path.join(blob_out_dir, 'inputs_full.txt')
            write_inputs_file(inputs_full, runs, run_subnets)
            with open(os.path.join(blob_out_dir, 'merged_network.nwk'), 'w') as f:
                f.write(subnets[0][1].rstrip() + '\n')
            with open(os.path.join(blob_out_dir, 'timing.json'), 'w') as f:
                json.dump({'mode': 'shortcut', 'merger_seconds': 0.0,
                           'n_runs': 1}, f, indent=2)
            return 'ok', {'mode': 'shortcut', 'merger_seconds': 0.0,
                          'n_runs': 1, 'n_retics_added': 0}
    # Full pipeline — restrict to first max_runs by filtering analyze_blob's
    # output (analyze_blob returns ALL runs; we keep only those in our subset).
    # `groups` is a list of dicts with key 'runs' = set of run names that
    # discovered that unique retic — restrict each group's run set, drop
    # any group whose runs are now empty.
    runs_found_all, run_subnets_all, groups_all = analyze_blob(blob_dir, subnet_source)
    runs_set = set(runs)
    runs_found = [r for r in runs_found_all if r in runs_set]
    run_subnets = {r: v for r, v in run_subnets_all.items() if r in runs_set}
    groups = []
    for g in groups_all:
        kept = g.get('runs', set()) & runs_set
        if kept:
            g2 = dict(g)
            g2['runs'] = kept
            groups.append(g2)
    inputs_full = os.path.join(blob_out_dir, 'inputs_full.txt')
    write_inputs_file(inputs_full, runs_found, run_subnets)
    t0 = time.time()
    try:
        _base, merged, n_added, timings = run_merger(
            blob_out_dir, inputs_full, gene_trees, verbose=verbose,
            source_blob_dir=blob_dir, dm=dm)
    except Exception as e:
        import traceback; traceback.print_exc()
        return f'merger_err:{type(e).__name__}', None
    elapsed = time.time() - t0
    with open(os.path.join(blob_out_dir, 'merged_network.nwk'), 'w') as f:
        f.write(merged + '\n')
    with open(os.path.join(blob_out_dir, 'timing.json'), 'w') as f:
        json.dump({'mode': 'full', 'merger_seconds': elapsed,
                   'n_runs': len(runs_found),
                   'n_unique_retics': len(groups),
                   'n_retics_added': n_added,
                   'step_timings': timings}, f, indent=2)
    return 'ok', {'mode': 'full', 'merger_seconds': elapsed,
                  'n_runs': len(runs_found), 'n_retics_added': n_added}


def build_phylonet_prune_newicks(metadata_dir, out_path, subgenes_out_dir='subgenes-out'):
    """Use PhyloNet's per-prune inferences (non_blob/<subgenes_out_dir>/subnets.txt)
    as prune subnets, indexed by subnet_idx in the metadata CSV. Returns the
    number of non-empty newick lines written.

    Each pruned_subtree row's `subnet_idx` indexes into the corresponding line
    of subnets.txt. Rows without a phylonet inference (e.g. taxa < phylonet's
    minimum) leave a blank line.
    """
    meta_csv = os.path.join(metadata_dir, 'non_blob',
                            'subnetworks_output_metadata.csv')
    sn_path = os.path.join(metadata_dir, 'non_blob', subgenes_out_dir, 'subnets.txt')
    if not os.path.exists(sn_path):
        raise FileNotFoundError(f'phylonet prune source missing: {sn_path}')
    with open(sn_path) as f:
        sn_lines = [l.strip() for l in f]
    rows = list(csv.DictReader(open(meta_csv)))
    if not rows: return 0
    max_idx = max(int(r['subnet_idx']) for r in rows)
    out = [''] * (max_idx + 1)
    n = 0
    for r in rows:
        if r['type'] != 'pruned_subtree': continue
        idx = int(r['subnet_idx'])
        if idx < len(sn_lines) and sn_lines[idx]:
            out[idx] = sn_lines[idx]
            n += 1
    with open(out_path, 'w') as f:
        for line in out: f.write(line + '\n')
    return n


def build_tob_prune_newicks(tob_path, metadata_dir, out_path):
    """Materialize <metadata_dir>/non_blob/tob_subnets.txt at out_path so the
    add_pruned_subtree step (which expects a file path) can consume it. The
    v3 divider writes tob_subnets.txt directly; for older divisions we fall
    back to building it from the metadata CSV + tob_path. Returns the number
    of non-empty newick lines written."""
    nwks = _load_non_blob_tob_subnets(metadata_dir, tob_path)
    with open(out_path, 'w') as f:
        for line in nwks:
            f.write(line + '\n')
    return sum(1 for s in nwks if s)


def list_blobs_from_metadata(metadata_dir):
    """Return [(blob_dir_basename, blob_name, items_leafset)].

    Raises SystemExit if any `blob*/run_000/subnetworks_output_metadata.csv`
    is missing — the merger needs the authoritative blob graph-node names
    and leafsets from this metadata for stages 3-4 (prune lookup + TOB graft).
    Falling back to folder names and subnets.txt leafsets would silently
    produce incorrect outputs (wrong prune attachment, wrong TOB graft point).
    """
    out = []
    for bd in sorted(glob.glob(f'{metadata_dir}/blob*')):
        name = os.path.basename(bd)
        meta = f'{bd}/run_000/subnetworks_output_metadata.csv'
        if not os.path.exists(meta):
            raise SystemExit(
                f'ERROR: missing per-blob metadata: {meta}\n'
                f'  This file is the authoritative source for the blob graph-node '
                f'name and leafset used by stages 3-4 (prune attachment + TOB '
                f'graft). The merger refuses to fall back to folder-name + '
                f'subnets.txt leafset because that path silently produces '
                f'incorrect grafts. Re-run the divider on this dataset to '
                f'regenerate the metadata.')
        blob_name = None
        items = set()
        with open(meta) as f:
            for r in csv.DictReader(f):
                if r['type'] == 'blob_group':
                    blob_name = r['blob']
                    items |= set(r['all_leaves'].split(','))
        if blob_name:
            out.append((name, blob_name, items))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('divisions_dir',
                    help='Directory containing blobXX subdirs with PhyloNet runs '
                         '(e.g. <ds>/divisions_final_phylonet).')
    ap.add_argument('--metadata-dir', default=None,
                    help='Division metadata folder with cut-time-siblings. '
                         'Defaults to the divisions_dir argument (since the '
                         'metadata is merged into divisions_final_phylonet).')
    ap.add_argument('--tob', required=True,
                    help='iqtree TOB newick (required).')
    ap.add_argument('--gene-trees', required=True,
                    help='Gene trees file (required).')
    ap.add_argument('--subnet-source', default='subnets.txt',
                    help="File inside each blob*/run_*/<subgenes-out-dir>/ holding "
                         "the subnetworks (default: 'subnets.txt', PhyloNet's "
                         "inferences). The true setting reads the divider's own "
                         "networks instead, e.g. 'subnetworks_output.txt'.")
    ap.add_argument('--subgenes-out-dir', default='subgenes-out',
                    help="Directory name under each blob*/run_*/ AND under "
                         "non_blob/ that holds 'subnets.txt' (default: "
                         "'subgenes-out'). Use this to point at re-runs like "
                         "'subgenes-out-r1'.")
    ap.add_argument('--max-runs', type=int, default=None,
                    help='If set, only use the first K phylonet run dirs '
                         '(sorted by name) per blob. Default: use all runs.')
    ap.add_argument('--out-name', default='full_merger',
                    help='Output subfolder name under divisions_dir '
                         '(default: full_merger)')
    ap.add_argument('--prune-source', choices=('tob', 'phylonet'), default='tob',
                    help='Prune-subnet source for stage 2. '
                         '"tob" (default) extracts each prune leafset from the '
                         'iqtree TOB. "phylonet" uses non_blob/subgenes-out/'
                         'subnets.txt (PhyloNet-inferred prunes).')
    args = ap.parse_args()
    _run(args)


def run_full_merger(divisions_dir, tob, gene_trees, metadata_dir=None,
                    subgenes_out_dir='subgenes-out', max_runs=None,
                    out_name='full_merger', prune_source='tob',
                    subnet_source='subnets.txt'):
    """Run the merger in-process. Returns the path of the final network.

    Same pipeline as the CLI; used by dimple.run_dimple so the three DIMPLE
    stages can run under one command.
    """
    return _run(argparse.Namespace(
        divisions_dir=divisions_dir, metadata_dir=metadata_dir, tob=tob,
        gene_trees=gene_trees, subgenes_out_dir=subgenes_out_dir,
        max_runs=max_runs, out_name=out_name, prune_source=prune_source,
        subnet_source=subnet_source))


def _run(args):
    # Stage 3 runs as a subprocess with cwd=PROJECT_ROOT, so a path given relative
    # to the caller's directory would resolve somewhere else there.
    for _a in ('divisions_dir', 'metadata_dir', 'tob', 'gene_trees'):
        if getattr(args, _a):
            setattr(args, _a, os.path.abspath(getattr(args, _a)))

    if not os.path.isdir(args.divisions_dir):
        print(f'ERROR: {args.divisions_dir} not a directory'); sys.exit(1)
    if args.metadata_dir is None:
        args.metadata_dir = args.divisions_dir
    if not os.path.isdir(args.metadata_dir):
        print(f'ERROR: --metadata-dir {args.metadata_dir} not a directory'); sys.exit(1)


    out_dir = os.path.join(args.divisions_dir, args.out_name)
    os.makedirs(out_dir, exist_ok=True)
    # [F1] A result left by an earlier run must never stand in for this one -- removed
    # HERE, before any stage, so a failure in stage 1-3 cannot leave it behind either.
    _stale = os.path.join(out_dir, 'merged_full.nwk')
    if os.path.exists(_stale):
        os.remove(_stale)

    if not os.path.exists(args.tob):
        print(f'ERROR: TOB not found: {args.tob}'); sys.exit(1)
    if not os.path.exists(args.gene_trees):
        print(f'ERROR: gene trees not found: {args.gene_trees}'); sys.exit(1)

    print(f'divisions_dir → {args.divisions_dir}', flush=True)
    print(f'metadata_dir  → {args.metadata_dir}', flush=True)
    print(f'tob           → {args.tob}', flush=True)
    print(f'gene_trees    → {args.gene_trees}', flush=True)
    print(f'output_dir    → {out_dir}', flush=True)

    # Identify blobs from metadata (authoritative source for blob names + items)
    blobs = list_blobs_from_metadata(args.metadata_dir)
    if not blobs:
        # No blobs in the divider's metadata. This is an error condition, not a
        # valid result: the divider should have produced at least one blob for
        # any non-trivial network. Report and fail instead of silently copying
        # the TOB tree through as merged_full.nwk (which masks divider failures).
        raise SystemExit(
            f'ERROR: no blobs found in metadata_dir {args.metadata_dir} — '
            f'nothing to merge. The divider produced no blob groups; check that '
            f'its non_blob/blob* outputs were generated and transferred '
            f'correctly. Refusing to fall back to copying the TOB tree.')
    print(f'blobs         → {len(blobs)}', flush=True)

    # Stage timings (peak memory captured by /usr/bin/time -v wrapper for
    # the whole pipeline; per-stage RSS would require separate time-wraps).
    # CPU time uses os.times() so already-reaped subprocesses are included.
    stage_timings = {}
    pipeline_t0 = time.time(); pipeline_c0 = _cpu_now()

    # ------------------------------------------------------------------
    # Stage 0: shared gene-tree distance matrix
    # ------------------------------------------------------------------
    # Built ONCE for the dataset and handed to every blob. compute_dm builds a
    # full N x N phylogenetic_distance_matrix() per gene tree -- O(G * N^2) --
    # and each blob only reads the sub-block for its own taxa, so doing it per
    # blob repeated the same work `len(blobs)` times. Slicing a shared matrix
    # is exact (see compute_dm_full), so results are unchanged.
    print(f'\n{"="*70}\n[stage 0] shared gene-tree distance matrix\n{"="*70}',
          flush=True)
    t0s = time.time(); c0s = _cpu_now()
    shared_dm = compute_dm_full(args.gene_trees, verbose=True)
    stage_timings['stage0_distance_matrix'] = time.time() - t0s
    stage_timings['stage0_distance_matrix_cpu'] = _cpu_now() - c0s
    print(f'  {len(shared_dm.index)} taxa, reused by all {len(blobs)} blob(s): '
          f'wall={stage_timings["stage0_distance_matrix"]:.2f}s '
          f'cpu={stage_timings["stage0_distance_matrix_cpu"]:.2f}s', flush=True)

    # ------------------------------------------------------------------
    # Stage 1: per-blob inference (DIMPLE merger over phylonet runs)
    # ------------------------------------------------------------------
    print(f'\n{"="*70}\n[stage 1] per-blob DIMPLE merger inference\n{"="*70}',
          flush=True)
    t1 = time.time(); c1 = _cpu_now()
    per_blob_results = []
    for bd_name, blob_name, items in blobs:
        # Find matching blob dir under divisions_dir (may differ in name from
        # metadata_dir's blob* — match by the blob_group's `blob` field).
        target_bd = None
        for cand in sorted(glob.glob(f'{args.divisions_dir}/blob*')):
            cm = f'{cand}/run_000/subnetworks_output_metadata.csv'
            if not os.path.exists(cm): continue
            with open(cm) as f:
                for r in csv.DictReader(f):
                    if r['type'] == 'blob_group' and r['blob'] == blob_name:
                        target_bd = cand
                        break
            if target_bd: break
        if target_bd is None:
            raise SystemExit(
                f'ERROR: stage 1 — no matching blob dir under '
                f'{args.divisions_dir} for blob {blob_name}. Refusing to '
                f'continue (would silently produce a multifurcated '
                f'merged_full.nwk where this blob stays uncombined).')
        blob_out = os.path.join(out_dir, bd_name)
        print(f'  ── {bd_name} ({blob_name}) ──', flush=True)
        status, info = per_blob_infer(target_bd, blob_out, args.gene_trees,
                                       f'{args.subgenes_out_dir}/{args.subnet_source}',
                                       verbose=False,
                                       max_runs=args.max_runs,
                                       dm=shared_dm)
        info = info or {}
        info.update({'blob': blob_name, 'status': status})
        if status == 'ok':
            ms = info.get('merger_seconds', 0.0)
            print(f'    {info.get("mode","?"):<8} runs={info.get("n_runs","?")} '
                  f'retics_added={info.get("n_retics_added","?")} '
                  f'time={ms:.2f}s', flush=True)
        else:
            raise SystemExit(
                f'ERROR: stage 1 (per_blob_infer) failed for blob '
                f'{blob_name} (status={status}). Refusing to continue — '
                f'downstream stages would produce a multifurcated '
                f'merged_full.nwk where {blob_name} stays uncombined.')
        per_blob_results.append(info)
    stage_timings['stage1_per_blob_inference'] = time.time() - t1
    stage_timings['stage1_per_blob_inference_cpu'] = _cpu_now() - c1
    print(f'  total stage-1 time: wall={stage_timings["stage1_per_blob_inference"]:.2f}s '
          f'cpu={stage_timings["stage1_per_blob_inference_cpu"]:.2f}s', flush=True)

    # ------------------------------------------------------------------
    # Stage 2: build TOB-restricted prune newicks
    # ------------------------------------------------------------------
    t2 = time.time(); c2 = _cpu_now()
    nb_meta = os.path.join(args.metadata_dir, 'non_blob',
                            'subnetworks_output_metadata.csv')
    if args.prune_source == 'phylonet':
        prunes_path = os.path.join(out_dir, 'phylonet_prune_subnets.txt')
        n_pr = build_phylonet_prune_newicks(args.metadata_dir, prunes_path,
                                              args.subgenes_out_dir)
        src_label = 'PhyloNet-inferred'
    else:
        prunes_path = os.path.join(out_dir, 'tob_prune_subnets.txt')
        n_pr = build_tob_prune_newicks(args.tob, args.metadata_dir, prunes_path)
        src_label = 'TOB-restricted'
    stage_timings['stage2_build_tob_prunes'] = time.time() - t2
    stage_timings['stage2_build_tob_prunes_cpu'] = _cpu_now() - c2
    print(f'\n[stage 2] built {n_pr} {src_label} prune subnets '
          f'(wall={stage_timings["stage2_build_tob_prunes"]:.2f}s '
          f'cpu={stage_timings["stage2_build_tob_prunes_cpu"]:.2f}s) → {prunes_path}',
          flush=True)

    # ------------------------------------------------------------------
    # Stage 3: combine (add_pruned_subtree per blob)
    # ------------------------------------------------------------------
    print(f'\n{"="*70}\n[stage 3] combine: add_pruned_subtree per blob\n{"="*70}',
          flush=True)
    t3 = time.time(); c3 = _cpu_now()
    combined_dir = os.path.join(out_dir, 'combined_blobs')
    os.makedirs(combined_dir, exist_ok=True)
    for bd_name, blob_name, _items in blobs:
        blob_out = os.path.join(out_dir, bd_name)
        blob_inferred = os.path.join(blob_out, 'merged_network.nwk')
        if not os.path.exists(blob_inferred):
            raise SystemExit(
                f'ERROR: stage 3 — no merged_network.nwk for blob '
                f'{blob_name} at {blob_inferred}. This should have been '
                f'produced by stage 1; if stage 1 raised, you should not '
                f'see this. Refusing to continue.')
        out_nwk = os.path.join(combined_dir, f'{blob_name}_combined.nwk')
        print(f'  ── {blob_name} ──', flush=True)
        cmd = [sys.executable, '-u', '-m',
               'dimple.merger.add_pruned_subtree',
               '--metadata', nb_meta,
               '--non-blob-nwks', prunes_path,
               '--blob', blob_inferred,
               '--blob-name', blob_name,
               '--tob', args.tob,
               '--out', out_nwk]
        # Run with this checkout's src/ on PYTHONPATH so the child resolves
        # `dimple` regardless of how the parent was launched.
        child_env = dict(os.environ)
        src_dir = os.path.join(PROJECT_ROOT, 'src')
        child_env['PYTHONPATH'] = (
            src_dir + os.pathsep + child_env['PYTHONPATH']
            if child_env.get('PYTHONPATH') else src_dir)
        r = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, env=child_env)
        if r.returncode != 0:
            raise SystemExit(
                f'ERROR: stage 3 (add_pruned_subtree) failed for blob '
                f'{blob_name} (exit {r.returncode}). Refusing to continue '
                f'to stage 4 — would produce a multifurcated merged_full.nwk '
                f'where {blob_name} stays uncombined. Common cause: fork '
                f'exhaustion under high parallelism. Re-run this dataset\'s '
                f'merger with lower concurrency.')
    stage_timings['stage3_combine'] = time.time() - t3
    stage_timings['stage3_combine_cpu'] = _cpu_now() - c3
    print(f'  total stage-3 time: wall={stage_timings["stage3_combine"]:.2f}s '
          f'cpu={stage_timings["stage3_combine_cpu"]:.2f}s', flush=True)

    # ------------------------------------------------------------------
    # Stage 4: merge into TOB
    # ------------------------------------------------------------------
    print(f'\n{"="*70}\n[stage 4] merge into TOB\n{"="*70}', flush=True)
    t4 = time.time(); c4 = _cpu_now()
    # merge_to_tob expects ds_dir + divisions-dir-name.
    # Our combined_blobs/ lives inside out_dir, not under <divisions_dir>/combined_blobs_<mode>/.
    # Workaround: call merge_to_tob directly via merge() function.
    from dimple.merger.merge_to_tob import merge as _merge_to_tob
    # We'll point divisions_dir at out_dir's parent name and combined as subfolder.
    # Simpler path: temporarily symlink combined_blobs → combined_blobs_full_merger
    # under divisions_dir, run merge_to_tob with --mode full_merger, then clean up.
    link_name = f'combined_blobs_{args.out_name}'
    link_path = os.path.join(args.divisions_dir, link_name)
    if os.path.lexists(link_path):
        if os.path.islink(link_path): os.unlink(link_path)
        else:
            print(f'WARNING: {link_path} exists and is not a symlink — skipping merge')
            link_path = None
    created_link = False
    if link_path:
        try:
            os.symlink(os.path.abspath(combined_dir), link_path)
            created_link = True
        except Exception as e:
            print(f'WARNING: could not create symlink {link_path}: {e}')
            link_path = None
    final_path = os.path.join(out_dir, 'merged_full.nwk')
    stage4_error = None
    if link_path:
        ds_dir_for_merge = os.path.dirname(os.path.abspath(args.divisions_dir))
        try:
            _merge_to_tob(ds_dir_for_merge, mode=args.out_name,
                          out_path=final_path,
                          divisions_dir=os.path.basename(args.divisions_dir),
                          tob_path=args.tob,
                          metadata_dir=os.path.basename(args.metadata_dir)
                                        if os.path.dirname(os.path.abspath(args.metadata_dir))
                                            == ds_dir_for_merge
                                        # merge() joins this onto ds_dir as text, so an
                                        # absolute path would be glued on after it
                                        else os.path.relpath(args.metadata_dir, ds_dir_for_merge))
        except Exception as e:
            import traceback; traceback.print_exc()
            stage4_error = e
        if created_link:
            try: os.unlink(link_path)
            except Exception: pass
    # [F1] Assembly failing (or being skipped for want of the symlink) used to
    # fall through to the success message below.
    # A partly written file is non-empty, so the exception decides, not the file.
    if stage4_error is not None or not (os.path.exists(final_path)
                                        and os.path.getsize(final_path) > 0):
        if os.path.exists(final_path):
            os.remove(final_path)
        raise SystemExit(f'ERROR: stage 4 (merge_to_tob) failed: {stage4_error!r}; '
                         f'no {final_path} written')
    stage_timings['stage4_merge_to_tob'] = time.time() - t4
    stage_timings['stage4_merge_to_tob_cpu'] = _cpu_now() - c4
    print(f'  total stage-4 time: wall={stage_timings["stage4_merge_to_tob"]:.2f}s '
          f'cpu={stage_timings["stage4_merge_to_tob_cpu"]:.2f}s', flush=True)

    stage_timings['pipeline_total'] = time.time() - pipeline_t0
    stage_timings['pipeline_total_cpu'] = _cpu_now() - pipeline_c0

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    # Persist consolidated stage timings (peak memory captured by /usr/bin/time -v
    # for the whole pipeline lives in stderr.txt — see "Maximum resident set size").
    timings_path = os.path.join(out_dir, 'pipeline_timings.json')
    with open(timings_path, 'w') as f:
        json.dump({
            'stage0_distance_matrix':        stage_timings['stage0_distance_matrix'],
            'stage0_distance_matrix_cpu':    stage_timings['stage0_distance_matrix_cpu'],
            'stage1_per_blob_inference':     stage_timings['stage1_per_blob_inference'],
            'stage1_per_blob_inference_cpu': stage_timings['stage1_per_blob_inference_cpu'],
            'stage2_build_tob_prunes':       stage_timings['stage2_build_tob_prunes'],
            'stage2_build_tob_prunes_cpu':   stage_timings['stage2_build_tob_prunes_cpu'],
            'stage3_combine':                stage_timings['stage3_combine'],
            'stage3_combine_cpu':            stage_timings['stage3_combine_cpu'],
            'stage4_merge_to_tob':           stage_timings['stage4_merge_to_tob'],
            'stage4_merge_to_tob_cpu':       stage_timings['stage4_merge_to_tob_cpu'],
            'pipeline_total':                stage_timings['pipeline_total'],
            'pipeline_total_cpu':            stage_timings['pipeline_total_cpu'],
            'note': 'wall = wall-clock seconds. cpu = user+system CPU including '
                    'reaped subprocesses (os.times). Peak RSS for whole pipeline '
                    'is in stderr.txt (Maximum resident set size).',
        }, f, indent=2)
    print(f'\nstage timings → {timings_path}', flush=True)
    summary_tsv = os.path.join(out_dir, 'summary.tsv')
    cols = ['blob', 'status', 'mode', 'n_runs', 'n_retics_added',
            'merger_seconds']
    with open(summary_tsv, 'w') as f:
        f.write('\t'.join(cols) + '\n')
        for r in per_blob_results:
            f.write('\t'.join(str(r.get(c, '')) for c in cols) + '\n')

    print(f'\n{"="*70}\n✓ Done. Final network: {final_path}\n'
          f'Summary: {summary_tsv}', flush=True)
    return final_path


if __name__ == '__main__':
    main()
