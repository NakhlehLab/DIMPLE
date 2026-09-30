#!/usr/bin/env python3
"""
PhyloNet subnet → DIMPLE merger pipeline, one divisions dir at a time.

Takes a divisions layout where each blob lives in its own subdir
(e.g. divisions_final_phylonet/blob00/run_000/subgenes-out/subnets.txt).
Per-blob, it:

  1. Collects the PhyloNet subnets across all runs of that blob.
  2. Writes phylonet_inputs_full.txt inside the blob dir (all-runs view).
  3. Runs the DIMPLE merger: DT-select → compatible subset → OverlapNJ →
     orientation-aware retic addition (real MPL scorer). Single-run blobs
     skip steps 2-3 and use their one subnet directly.
  4. Saves base_tree.nwk and merged_network.nwk inside the blob dir.

Usage:
    conda run -n phylo-env python -u -m dimple.merger.blob_merger \
        data/division_iqtree_tob/lvl2/n05/divisions_final_phylonet

Gene trees default to `<parent_of_divisions_dir>/1k_iqtree_reroot.txt` (so for
the example above that's data/division_iqtree_tob/lvl2/n05/1k_iqtree_reroot.txt).
Override with --gene-trees if needed.
"""

import os
# Cap BLAS thread fan-out before any numpy/scipy import — OpenBLAS reads
# these eagerly. Without this, scipy.linalg can spawn 64+ threads and trip
# RLIMIT_NPROC on shared hosts.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import sys
import csv
import json
import time
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from dimple.utils.network_util import (
    newick_to_nx, get_leafset, strip_branch_lengths, clean_extended_newick,
)
from dimple.merger.merger_util import (
    find_reticulations as find_retics_with_sig,
    retics_are_same,
    parse_inputs_file,
    get_taxa, compute_dm, compute_dm_full,
)
from dimple.merger.overlap_njmerge import (
    run_overlap_njmerge, select_compatible_subset,
)
from dimple.merger.select_displayed_trees import select_displayed_trees
from dimple.merger.add_reticulations import add_retics_from_inputs


# ---------------------------------------------------------------------------
# Metadata + subnet discovery
# ---------------------------------------------------------------------------

def get_blob_name(blob_dir):
    """Return the blob name (first blob_group row in any run's metadata)."""
    for run_name in sorted(os.listdir(blob_dir)):
        run_path = os.path.join(blob_dir, run_name)
        meta = os.path.join(run_path, 'subnetworks_output_metadata.csv')
        if not (run_name.startswith('run_') and os.path.exists(meta)):
            continue
        with open(meta) as f:
            for r in csv.DictReader(f):
                if r.get('type') == 'blob_group' and r.get('blob'):
                    return r['blob']
    return None


def list_run_dirs(blob_dir):
    return sorted([d for d in os.listdir(blob_dir)
                   if d.startswith('run_') and
                   os.path.isdir(os.path.join(blob_dir, d))])


def read_phylonet_subnets(run_dir, subnet_source):
    """Return list of (subnet_idx, newick) for all blob_group rows in this
    run. The subnet_idx indexes into the file at `subnet_source` within the
    run dir (e.g. subgenes-out/subnets.txt)."""
    meta = os.path.join(run_dir, 'subnetworks_output_metadata.csv')
    subnets_file = os.path.join(run_dir, subnet_source)
    if not (os.path.exists(meta) and os.path.exists(subnets_file)):
        return []
    with open(subnets_file) as f:
        # [F5] subnet_idx is a PHYSICAL line number: dropping blank lines here
        # paired every later tree with the previous row's metadata.
        lines = [l.strip() for l in f]
    out = []
    with open(meta) as f:
        for r in csv.DictReader(f):
            if r.get('type') != 'blob_group':
                continue
            idx = int(r['subnet_idx'])
            if idx < len(lines) and lines[idx]:
                out.append((idx, lines[idx]))
    return out


# ---------------------------------------------------------------------------
# Retic coverage (mirrors summarize_reticulations, scoped to one blob)
# ---------------------------------------------------------------------------

def load_sibling_leaves(blob_dir):
    """Read any run_*/subnetworks_output_metadata.csv in `blob_dir` and return
    the union of TOB-external `sibling_leaves` listed in source_item JSON of
    `blob_group` rows. Empty set for root mega-blobs (no external siblings)."""
    siblings = set()
    for run in list_run_dirs(blob_dir):
        meta = os.path.join(blob_dir, run, 'subnetworks_output_metadata.csv')
        if not os.path.exists(meta):
            continue
        try:
            with open(meta) as f:
                for row in csv.DictReader(f):
                    if row.get('type') != 'blob_group':
                        continue
                    src = row.get('source_item', '')
                    if not src.startswith('{'):
                        continue
                    try:
                        siblings.update(json.loads(src).get('sibling_leaves', []))
                    except Exception:
                        pass
        except Exception:
            continue
        if siblings:
            break
    return siblings


def compute_outgroup_distances(dm, blob_taxa, sibling_leaves,
                               outlier_multiplier=2.0):
    """Per-blob-taxon distance to the synthetic OUT outgroup, computed as
    `outlier_multiplier × mean(distance from t to each TOB-external sibling)`.

    The multiplier (default 2.0) keeps OUT reliably outlying — raw mean
    distances to sibling leaves can be comparable to intra-blob distances,
    which lets NJ slip OUT between blob leaves rather than strictly outside.
    Scaling preserves the per-taxon bias direction (closer to TOB-parent)
    while ensuring the absolute magnitude is large enough."""
    out = {}
    sibs = [s for s in sibling_leaves if s in dm.index]
    if not sibs:
        return out
    for t in blob_taxa:
        if t not in dm.index:
            continue
        vals = [float(dm.at[t, s]) for s in sibs]
        if vals:
            out[t] = outlier_multiplier * (sum(vals) / len(vals))
    return out


def analyze_blob(blob_dir, subnet_source):
    """For one blob, analyze which runs cover which unique reticulations.

    Returns:
      runs: list of run names (e.g. ['run_000', 'run_001', ...])
      run_subnets: {run_name: [newick, ...]}  (phylonet subnets for this blob)
      retic_groups: list of group dicts — one per unique reticulation with
        keys: signature, retic_leaves, parent_leaves, runs (set)
    """
    runs = list_run_dirs(blob_dir)
    run_subnets = {}
    raw_retics = []  # (retic_info, run_name)

    for run_name in runs:
        run_dir = os.path.join(blob_dir, run_name)
        subnets = read_phylonet_subnets(run_dir, subnet_source)
        run_subnets[run_name] = [nwk for _, nwk in subnets]
        for _idx, nwk in subnets:
            try:
                G = newick_to_nx(nwk)
            except Exception:
                continue
            for r in find_retics_with_sig(G):
                raw_retics.append((r, run_name))

    # Merge retics by signature (retics_are_same compares parent-sibling leaves)
    groups = []
    for retic, run_name in raw_retics:
        matched = None
        for i, g in enumerate(groups):
            if retics_are_same(retic['signature'], g['signature']):
                matched = i
                break
        if matched is None:
            groups.append({
                'signature': retic['signature'],
                'retic_leaves': retic['retic_leaves'],
                'parent_leaves': retic['parent_leaves'],
                'runs': {run_name},
            })
        else:
            groups[matched]['runs'].add(run_name)

    return runs, run_subnets, groups


# ---------------------------------------------------------------------------
# Input file writers
# ---------------------------------------------------------------------------

def _count_retics_in_newick(nwk):
    # Each reticulation appears as #Hnum twice in the newick (primary + ref),
    # so count unique #H labels.
    import re
    return len(set(re.findall(r'#H\d+', nwk)))


def write_inputs_file(out_path, run_names, run_subnets):
    """Write the merger-format inputs file with '# run=... , reticulations=N' headers."""
    n_written = 0
    with open(out_path, 'w') as f:
        for run in run_names:
            subnets = run_subnets.get(run, [])
            if not subnets:
                continue
            n_ret = max((_count_retics_in_newick(s) for s in subnets), default=0)
            f.write(f'# run={run}, reticulations={n_ret}\n')
            for nwk in subnets:
                f.write(nwk + '\n')
            n_written += 1
    return n_written


# ---------------------------------------------------------------------------
# Merger
# ---------------------------------------------------------------------------

def run_merger(blob_dir, inputs_full_path, gene_trees, verbose=True,
               source_blob_dir=None, dm=None):
    """Run the full DIMPLE merger on a blob's phylonet_inputs_full.txt.
    Writes base_tree.nwk and merged_network.nwk into blob_dir.
    Returns (base_nwk, merged_nwk, n_retics_added).

    `source_blob_dir`, if provided, is the original blob_dir under
    `divisions_dir/blob*/` (which holds run_*/subnetworks_output_metadata.csv).
    Used to pull TOB-external `sibling_leaves` for biasing the NJMerge OUT
    rooting toward the TOB-parent direction. Defaults to `blob_dir`.

    `dm`, if provided, is a distance matrix covering (at least) this blob's
    taxa plus its sibling leaves -- normally the whole-dataset matrix from
    `compute_dm_full`, built once and shared across every blob. Downstream
    consumers already slice by label (`run_overlap_njmerge` takes
    `dm.loc[avail, avail]`, `compute_outgroup_distances` indexes by taxon), so
    a superset matrix is used as-is. Pass None to compute a blob-local matrix
    the old way.
    """
    if source_blob_dir is None:
        source_blob_dir = blob_dir
    runs = parse_inputs_file(inputs_full_path)
    all_subnets = [s for r in runs.values() for s in r['subnets']]
    if not all_subnets:
        raise RuntimeError(f'No subnets parsed from {inputs_full_path}')

    all_taxa = set()
    for s in all_subnets:
        all_taxa |= get_taxa(s)
    all_taxa = {t for t in all_taxa if not t.startswith('#')}
    if verbose:
        print(f'  {len(runs)} runs, {len(all_subnets)} subnets, '
              f'{len(all_taxa)} taxa', flush=True)

    timings = {'dm': 0.0, 'dt_select': 0.0, 'compat': 0.0,
               'njmerge': 0.0, 'retics': 0.0}

    # 1. Distance matrix — extend with TOB-external sibling leaves so we can
    #    compute real gene-tree distances to a virtual OUT outgroup (= mean
    #    distance to the sibling group). NJMerge uses this to root the blob
    #    toward the TOB-parent direction instead of midpoint-rooting.
    sibling_leaves = load_sibling_leaves(source_blob_dir)
    if verbose and sibling_leaves:
        print(f'  TOB-external sibling leaves: {len(sibling_leaves)} '
              f'(used to bias NJMerge OUT placement)', flush=True)
    t0 = time.time()
    if dm is None:
        if verbose:
            print(f'  Computing distance matrix...', flush=True)
        dm = compute_dm(gene_trees, all_taxa | set(sibling_leaves))
    else:
        if verbose:
            print(f'  Reusing shared distance matrix ({len(dm.index)} taxa)', flush=True)
        # [F8] compute_dm_full leaves NaN where two taxa never share a gene tree.
        need = [t for t in all_taxa | set(sibling_leaves) if t in dm.index]
        if dm.loc[need, need].isna().values.any():
            raise ValueError('this blob needs a distance between taxa that never '
                             'co-occur in any gene tree')
    outgroup_distances = compute_outgroup_distances(dm, all_taxa, sibling_leaves)
    timings['dm'] = time.time() - t0

    # 2. DT-select
    t0 = time.time()
    try:
        sel, _ = select_displayed_trees(
            all_subnets, gene_tree_file=gene_trees)
        if verbose:
            print(f'  DT-select: {len(sel)} trees from {len(all_subnets)} subnets',
                  flush=True)
    except Exception as e:
        # [F2] A crash here is not the result "no reticulate input": fail unless
        # the tree-only fallback was asked for, and always say why.
        print(f'  DT-select failed ({e})', flush=True)
        if not os.environ.get('DIMPLE_ALLOW_TREE_FALLBACK') == '1':
            raise
        sel = [clean_extended_newick(s) for s in all_subnets if '#H' not in s]
    timings['dt_select'] = time.time() - t0

    # 3. Compatible subset
    t0 = time.time()
    compat = select_compatible_subset(sel)
    timings['compat'] = time.time() - t0

    # Save the selected newicks for diagnostics — `dt_selected_subnets.txt`
    # are the displayed trees DT-select chose (one per phylonet network),
    # `compat_subnets.txt` is the compatible subset that actually feeds NJMerge.
    try:
        with open(os.path.join(blob_dir, 'dt_selected_subnets.txt'), 'w') as f:
            for s in sel: f.write(s.rstrip() + '\n')
        with open(os.path.join(blob_dir, 'compat_subnets.txt'), 'w') as f:
            for s in compat: f.write(s.rstrip() + '\n')
    except Exception as _e:
        if verbose: print(f'  warning: could not save selected subnets: {_e}', flush=True)

    # 4. OverlapNJ merge
    if verbose:
        print(f'  Running NJMerge on {len(compat)} compatible trees...', flush=True)
    t0 = time.time()
    base_nwk = run_overlap_njmerge(compat, dm,
                                    outgroup_distances=outgroup_distances or None,
                                    required_taxa=all_taxa)
    timings['njmerge'] = time.time() - t0
    with open(os.path.join(blob_dir, 'base_tree.nwk'), 'w') as f:
        f.write(base_nwk + '\n')

    # 5. Orientation-aware retic addition (real PL)
    t0 = time.time()
    try:
        final_nwk, n_added = add_retics_from_inputs(
            base_nwk, inputs_full_path, gene_trees,
            verbose=verbose)
        if verbose:
            print(f'  Retics added: {n_added}', flush=True)
    except Exception as e:
        # [F2] Otherwise indistinguishable from "no reticulation was accepted".
        print(f'  add_retics failed: {e}', flush=True)
        if not os.environ.get('DIMPLE_ALLOW_TREE_FALLBACK') == '1':
            raise
        final_nwk = base_nwk
        n_added = 0
    timings['retics'] = time.time() - t0

    if verbose:
        print(f'  Step timings: '
              f"DM={timings['dm']:.2f}s "
              f"DT-select={timings['dt_select']:.2f}s "
              f"compat={timings['compat']:.2f}s "
              f"NJMerge={timings['njmerge']:.2f}s "
              f"retics={timings['retics']:.2f}s", flush=True)

    return base_nwk, final_nwk, n_added, timings


# ---------------------------------------------------------------------------
# Per-blob orchestration
# ---------------------------------------------------------------------------

def process_blob(blob_dir, gene_trees, subnet_source, verbose=True, dm=None):
    """Full pipeline for one blob subdir. Writes:
      phylonet_inputs_full.txt
      base_tree.nwk (only in multi-run case)
      merged_network.nwk
    """
    blob_name = get_blob_name(blob_dir)
    if blob_name is None:
        return None, 'no_blob_name'

    runs = list_run_dirs(blob_dir)
    if not runs:
        return None, 'no_runs'

    # Shortcut: single-run blob — use its one phylonet subnet directly
    if len(runs) == 1:
        run_dir = os.path.join(blob_dir, runs[0])
        subnets = read_phylonet_subnets(run_dir, subnet_source)
        if not subnets:
            return None, 'shortcut_empty'
        # Write a minimal inputs_full for completeness
        run_subnets = {runs[0]: [nwk for _, nwk in subnets]}
        write_inputs_file(os.path.join(blob_dir, 'phylonet_inputs_full.txt'),
                          runs, run_subnets)
        # The single subnet becomes the merged network (if multiple blob_group
        # rows in this one run, concat them — that's the complete blob model)
        t0 = time.time()
        merged = '\n'.join(nwk for _, nwk in subnets) + '\n'
        with open(os.path.join(blob_dir, 'merged_network.nwk'), 'w') as f:
            f.write(merged)
        merger_elapsed = time.time() - t0
        if verbose:
            print(f'  Merger time (shortcut): {merger_elapsed:.3f}s', flush=True)
        return {
            'blob': blob_name, 'n_runs': 1, 'mode': 'shortcut',
            'n_retics_added': 0, 'merger_seconds': merger_elapsed,
        }, 'ok'

    # Full pipeline
    runs_found, run_subnets, groups = analyze_blob(blob_dir, subnet_source)

    inputs_full = os.path.join(blob_dir, 'phylonet_inputs_full.txt')
    n_full = write_inputs_file(inputs_full, runs_found, run_subnets)
    if verbose:
        print(f'  Wrote {n_full} runs to phylonet_inputs_full.txt '
              f'({len(groups)} unique retics)', flush=True)

    # Run merger (always on inputs_full.txt so we get full context).
    # Time ONLY the merger step (NJMerge + retic addition), excluding
    # input-file writing and metadata discovery.
    merger_t0 = time.time()
    try:
        _base, merged, n_added, step_timings = run_merger(
            blob_dir, inputs_full, gene_trees, verbose=verbose, dm=dm)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f'merger_err:{type(e).__name__}:{e}'
    merger_elapsed = time.time() - merger_t0

    with open(os.path.join(blob_dir, 'merged_network.nwk'), 'w') as f:
        f.write(merged + '\n')

    if verbose:
        print(f'  Merger time: {merger_elapsed:.2f}s', flush=True)

    return {
        'blob': blob_name, 'n_runs': len(runs_found),
        'n_unique_retics': len(groups),
        'n_retics_added': n_added,
        'mode': 'full',
        'merger_seconds': merger_elapsed,
        'step_timings': step_timings,
    }, 'ok'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _default_gene_trees(divisions_dir):
    """Auto-detect 1k_iqtree_reroot.txt in the dataset dir (parent of
    divisions_dir)."""
    return os.path.join(os.path.dirname(os.path.abspath(divisions_dir)),
                        '1k_iqtree_reroot.txt')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('divisions_dir',
                    help='Dir containing blobXX subdirs (e.g. '
                         'data/.../n05/divisions_final_phylonet)')
    ap.add_argument('--gene-trees', default=None,
                    help="Gene trees file for PL scoring / DM. Defaults to "
                         "'<parent_of_divisions_dir>/1k_iqtree_reroot.txt'.")
    ap.add_argument('--subnet-source', default='subgenes-out/subnets.txt',
                    help='Subpath within each run dir to read subnets from')
    ap.add_argument('--blob-prefix', default='blob',
                    help="Subdir prefix to treat as blobs (default 'blob')")
    args = ap.parse_args()

    if not os.path.isdir(args.divisions_dir):
        print(f'ERROR: {args.divisions_dir} is not a directory')
        sys.exit(1)

    if args.gene_trees is None:
        args.gene_trees = _default_gene_trees(args.divisions_dir)
    if not os.path.exists(args.gene_trees):
        print(f'ERROR: gene trees file not found: {args.gene_trees}')
        sys.exit(1)

    print(f'gene trees   → {args.gene_trees}', flush=True)
    wall_t0 = time.time()

    blob_dirs = sorted([os.path.join(args.divisions_dir, d)
                        for d in os.listdir(args.divisions_dir)
                        if d.startswith(args.blob_prefix) and
                        os.path.isdir(os.path.join(args.divisions_dir, d))])
    print(f'Found {len(blob_dirs)} blob dirs', flush=True)

    # Build the gene-tree distance matrix ONCE for the whole dataset. Every
    # blob reads a sub-block of the same matrix, so computing it per blob
    # repeated O(G * N^2) work `len(blob_dirs)` times for no gain.
    dm_t0 = time.time()
    shared_dm = compute_dm_full(args.gene_trees, verbose=True)
    dm_build_seconds = time.time() - dm_t0
    print(f'Shared distance matrix built in {dm_build_seconds:.2f}s '
          f'({len(shared_dm.index)} taxa, reused by all {len(blob_dirs)} blobs)',
          flush=True)

    n_ok = n_err = 0
    total_merger_seconds = 0.0
    step_totals = {'dm': 0.0, 'dt_select': 0.0, 'compat': 0.0,
                   'njmerge': 0.0, 'retics': 0.0}
    per_blob_times = []
    for bd in blob_dirs:
        print(f'\n{"="*70}', flush=True)
        print(f'Processing {bd}', flush=True)
        print(f'{"="*70}', flush=True)
        result, status = process_blob(bd, args.gene_trees, args.subnet_source,
                                      dm=shared_dm)
        if result is None:
            print(f'  FAILED: {status}', flush=True)
            n_err += 1
        else:
            print(f'  OK: {result}', flush=True)
            n_ok += 1
            t = result.get('merger_seconds', 0.0)
            total_merger_seconds += t
            per_blob_times.append((os.path.basename(bd), result['mode'], t,
                                    result.get('step_timings')))
            st = result.get('step_timings')
            if st:
                for k, v in st.items():
                    step_totals[k] += v

    wall_elapsed = time.time() - wall_t0
    print(f'\n=== DONE: {n_ok} ok, {n_err} failed ===', flush=True)
    print(f'\n--- Merger-only timing (excludes input-file writing / discovery) ---',
          flush=True)
    for name, mode, t, st in per_blob_times:
        if st:
            print(f'  {name} [{mode}]: {t:.2f}s  '
                  f"(DM={st['dm']:.2f}, DT-select={st['dt_select']:.2f}, "
                  f"compat={st['compat']:.2f}, NJMerge={st['njmerge']:.2f}, "
                  f"retics={st['retics']:.2f})", flush=True)
        else:
            print(f'  {name} [{mode}]: {t:.2f}s', flush=True)
    print(f'  TOTAL merger time: {total_merger_seconds:.2f}s '
          f'({total_merger_seconds/60:.2f} min)', flush=True)
    print(f'  (+ shared distance matrix, built once: '
          f'{dm_build_seconds:.2f}s)', flush=True)
    print(f'\n--- Per-step totals (sum across all blobs) ---', flush=True)
    st_total = sum(step_totals.values())
    for k in ['dm', 'dt_select', 'compat', 'njmerge', 'retics']:
        pct = 100 * step_totals[k] / st_total if st_total > 0 else 0
        print(f'  {k:<10}: {step_totals[k]:7.2f}s ({pct:5.1f}%)', flush=True)
    print(f'\nWall-clock total: {wall_elapsed:.2f}s ({wall_elapsed/60:.2f} min)',
          flush=True)
    print(f'(Peak memory is reported by /usr/bin/time -v in stderr.txt.)',
          flush=True)


if __name__ == '__main__':
    main()
