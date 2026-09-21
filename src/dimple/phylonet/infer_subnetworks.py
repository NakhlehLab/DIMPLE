#!/usr/bin/env python3
"""
Stage 2 of DIMPLE: infer a network for every division with PhyloNet.

Given a divisions directory produced by `dimple.divider.generate_k_divisions`,
this walks every `blob*/run_*/` (and optionally `non_blob/`) and, for each
division listed in that directory's `subnetworks_output_metadata.csv`:

  1. Restricts the gene trees to the division's leafset (plus the outgroup)
     and reroots each restricted tree at the outgroup.
  2. Restricts the binary starting tree to the same leaf space. PhyloNet's
     rearrangement moves reject a non-binary start, which is why the
     blob-collapsed tree of blobs cannot be used here — pass the ASTRAL tree
     estimated from the same gene trees (what the manuscript's experiments
     used), or the binary tree from the first TREE-QMC pass, as
     `--base-tree`.
  3. Runs PhyloNet's maximum pseudo-likelihood search (InferNetwork_MPL) on
     that subproblem with the reticulation bound `--max-ret`.
  4. Collects "Inferred Network #1" from each PhyloNet output, prunes the
     outgroup back off, and writes one newick per division to
     `<run_dir>/<subgenes-out-dir>/subnets.txt` — the file the merger reads.

Per-division files inside `<run_dir>/<subgenes-out-dir>/`:
  subgeneset_<i>_ret<r>.txt   restricted gene trees for division i
  subbase_<i>_ret<r>.tree     restricted starting tree for division i
  tmp_phylonet_<i>.nex        the NEXUS block handed to PhyloNet
  phylonet_out_<i>.txt        raw PhyloNet output
  subnets.txt                 one inferred network per division (outgroup removed)
  phylonet-runtimelog.txt     per-division wall time + exit status

Divisions coming from the non-blob part of the tree of blobs (metadata rows of
type `non_blob_set` / `pruned_subtree`) are tree-like by construction and are
inferred with the reticulation bound fixed to 0. They are skipped entirely
unless `--include-non-blob` is given, because the default merger takes its
non-blob subnetworks straight off the tree of blobs.

Usage:
    python -m dimple.phylonet.infer_subnetworks \\
        --divisions dimple_out/divisions \\
        --gene-trees gene_trees.tre \\
        --base-tree astral.tre \\
        --phylonet PhyloNet.jar \\
        --max-ret 1 --parallel 5 --pl 4
"""
import os
import re
import sys
import csv
import time
import json
import argparse
import subprocess
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

import dendropy

from dimple.utils.network_util import (
    newick_to_nx,
    build_newick_from_graph,
    contract_degree2_nodes,
    clean_extended_newick,
    get_leafset,
)

METADATA_FILENAME = 'subnetworks_output_metadata.csv'
TREE_LIKE_ROW_TYPES = ('non_blob_set', 'pruned_subtree')
DEFAULT_SUBGENES_DIR = 'subgenes-out'
DEFAULT_JAVA_MEM = '16000M'


# ---------------------------------------------------------------------------
# Division leafsets
# ---------------------------------------------------------------------------

def read_division_leafsets(metadata_csv, max_ret):
    """Return {1: (leafset, r), 2: (...), ...} for one run directory.

    Indices are 1-based and follow the CSV row order, which is the order the
    merger expects `subnets.txt` lines in. Rows whose type marks them as
    tree-like (non-blob / pruned subtree) get r = 0; every other row gets
    r = max_ret. Rows with fewer than two leaves are skipped.
    """
    info = {}
    idx = 1
    with open(metadata_csv, newline='') as fin:
        for row in csv.DictReader(fin):
            leaves_field = (row.get('all_leaves') or '').strip()
            if not leaves_field:
                continue
            leaves = [t.strip() for t in leaves_field.split(',') if t.strip()]
            if len(leaves) < 2:
                continue
            row_type = (row.get('type') or '').strip()
            r = 0 if row_type in TREE_LIKE_ROW_TYPES else max_ret
            info[idx] = (set(leaves), r)
            idx += 1
    return info


# ---------------------------------------------------------------------------
# Restriction of gene trees / starting tree to a division
# ---------------------------------------------------------------------------

def _load_trees(path):
    with open(path) as f:
        data = f.read()
    return dendropy.TreeList.get(
        data=data, schema='newick',
        taxon_namespace=dendropy.TaxonNamespace(),
        rooting='default-rooted',
    )


def _restrict_and_reroot(tree, leaf_set, outgroup):
    """Restrict `tree` to `leaf_set` and reroot it at the outgroup leaf."""
    sub_t = tree.extract_tree_with_taxa_labels(leaf_set)
    out_node = sub_t.find_node_with_taxon_label(outgroup)
    if out_node is not None and out_node.parent_node is not None:
        sub_t.reroot_at_edge(out_node.edge, update_bipartitions=False)
    return sub_t.as_string(schema='newick', suppress_rooting=True).strip()


def extract_subgene_trees(gene_trees_path, divisions, out_dir, outgroup='OUT'):
    """Write subgeneset_<i>_ret<r>.txt for every division."""
    os.makedirs(out_dir, exist_ok=True)
    trees = _load_trees(gene_trees_path)
    for idx, (leaf_set, retic) in divisions.items():
        wanted = set(leaf_set) | {outgroup}
        out_path = os.path.join(out_dir, f'subgeneset_{idx}_ret{retic}.txt')
        with open(out_path, 'w') as fout:
            for t in trees:
                try:
                    fout.write(_restrict_and_reroot(t, wanted, outgroup) + '\n')
                except Exception as e:
                    print(f'    warning: skipped a gene tree for division '
                          f'{idx}: {e}', flush=True)


def extract_subbase_trees(base_tree_path, divisions, out_dir, outgroup='OUT'):
    """Write subbase_<i>_ret<r>.tree (the PhyloNet starting tree) per division."""
    os.makedirs(out_dir, exist_ok=True)
    base_trees = _load_trees(base_tree_path)
    if len(base_trees) == 0:
        raise ValueError(f'No tree found in base tree file {base_tree_path}')
    base_tree = base_trees[0]
    for idx, (leaf_set, retic) in divisions.items():
        wanted = set(leaf_set) | {outgroup}
        out_path = os.path.join(out_dir, f'subbase_{idx}_ret{retic}.tree')
        try:
            nwk = clean_extended_newick(
                _restrict_and_reroot(base_tree, wanted, outgroup))
        except Exception as e:
            raise ValueError(
                f'failed to restrict the starting tree to division {idx} '
                f'({len(wanted)} leaves): {e}')
        with open(out_path, 'w') as fout:
            fout.write(nwk + '\n')


# ---------------------------------------------------------------------------
# PhyloNet
# ---------------------------------------------------------------------------

def write_nexus(gene_trees, start_tree, max_ret, nexus_path, threads=1):
    """Write the NEXUS block for one InferNetwork_MPL search."""
    if not start_tree.endswith(';'):
        start_tree += ';'
    pl_flag = f' -pl {threads}' if threads and threads > 1 else ''
    with open(nexus_path, 'w') as fout:
        fout.write('#NEXUS\n\nBEGIN TREES;\n')
        for i, tree in enumerate(gene_trees, 1):
            fout.write(f'  Tree gt{i} = {tree}\n')
        fout.write('END;\n\nBEGIN NETWORKS;\n')
        fout.write(f'  Network net = {start_tree}\n')
        fout.write('END;\n\nBEGIN PHYLONET;\n')
        fout.write(f'  InferNetwork_MPL (all) {max(max_ret, 0)} '
                   f'-s net{pl_flag};\n')
        fout.write('END;\n')


def run_phylonet_one(subgene_path, subbase_path, out_path, max_ret, jar,
                     threads=1, java_mem=DEFAULT_JAVA_MEM, java='java'):
    """Run one InferNetwork_MPL search. Returns True on success."""
    if not os.path.exists(jar):
        raise FileNotFoundError(f'PhyloNet jar not found: {jar}')
    if not os.path.exists(subgene_path):
        print(f'    warning: missing {subgene_path}, skipping', flush=True)
        return False
    if not os.path.exists(subbase_path):
        print(f'    warning: missing {subbase_path}, skipping', flush=True)
        return False

    with open(subbase_path) as f:
        start_tree = clean_extended_newick(f.read().strip())
    with open(subgene_path) as f:
        gene_trees = [ln.strip() for ln in f if ln.strip()]
    if len(gene_trees) < 2:
        print(f'    warning: {os.path.basename(subgene_path)} has only '
              f'{len(gene_trees)} gene tree(s), skipping', flush=True)
        return False

    nexus_path = os.path.join(
        os.path.dirname(out_path),
        'tmp_' + os.path.basename(out_path).replace('phylonet_out_', 'phylonet_')
        .replace('.txt', '.nex'))
    write_nexus(gene_trees, start_tree, max_ret, nexus_path, threads)

    with open(out_path, 'w') as fout:
        proc = subprocess.run([java, f'-Xmx{java_mem}', '-jar', jar, nexus_path],
                              stdout=fout, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f'    warning: PhyloNet exited {proc.returncode} for '
              f'{os.path.basename(subgene_path)} (see {out_path})', flush=True)
        return False
    return True


# ---------------------------------------------------------------------------
# Collecting the inferred networks
# ---------------------------------------------------------------------------

def parse_inferred_network(path):
    """Return the newick after 'Inferred Network #1:' in a PhyloNet output."""
    with open(path) as f:
        content = f.read()
    match = re.search(r'Inferred Network #1:\s*([\s\S]*?);', content)
    if not match:
        return None
    return re.sub(r'\s+', ' ', match.group(1).strip() + ';')


def remove_outgroup(newick, outgroup='OUT'):
    """Prune the outgroup leaf back off an inferred subnetwork."""
    G = newick_to_nx(newick)
    leaves = get_leafset(G)
    if outgroup not in leaves:
        return newick
    G.remove_node(outgroup)
    kept = leaves - {outgroup}
    changed = True
    while changed:
        changed = False
        for n in list(G.nodes()):
            if n == 'seed':
                continue
            if G.out_degree(n) == 0 and n not in kept:
                G.remove_node(n)
                changed = True
    return build_newick_from_graph(contract_degree2_nodes(G))


def combine_subnets(subgenes_dir, n_divisions, outgroup='OUT'):
    """Write subnets.txt from phylonet_out_*.txt. Returns the newick list.

    Every division must have produced an inferred network: the merger indexes
    subnets.txt by division order, so a missing line would silently shift every
    later division onto the wrong leafset. A gap therefore aborts and leaves no
    subnets.txt behind.
    """
    out_path = os.path.join(subgenes_dir, 'subnets.txt')
    networks = []
    for idx in range(1, n_divisions + 1):
        fpath = os.path.join(subgenes_dir, f'phylonet_out_{idx}.txt')
        if not os.path.exists(fpath):
            raise RuntimeError(
                f'no PhyloNet output for division {idx} in {subgenes_dir} '
                f'({os.path.basename(fpath)} missing) — refusing to write a '
                f'subnets.txt whose lines no longer line up with the divisions')
        newick = parse_inferred_network(fpath)
        if newick is None:
            raise RuntimeError(
                f'no "Inferred Network #1" in {fpath} — PhyloNet did not '
                f'finish this division; refusing to write a misaligned '
                f'subnets.txt')
        networks.append(remove_outgroup(newick, outgroup))
    with open(out_path, 'w') as fout:
        for nwk in networks:
            fout.write(nwk + '\n')
    return networks


# ---------------------------------------------------------------------------
# Driving one run directory / a whole divisions tree
# ---------------------------------------------------------------------------

def infer_run(run_dir, gene_trees, base_tree, jar, max_ret=1,
              subgenes_out_dir=DEFAULT_SUBGENES_DIR, outgroup='OUT',
              threads=1, java_mem=DEFAULT_JAVA_MEM, java='java',
              force=False, label=None):
    """Infer every division in one `run_*` (or `non_blob`) directory."""
    label = label or os.path.basename(run_dir)
    metadata_csv = os.path.join(run_dir, METADATA_FILENAME)
    if not os.path.exists(metadata_csv):
        print(f'  [{label}] no {METADATA_FILENAME}, skipping', flush=True)
        return False

    subgenes_dir = os.path.join(run_dir, subgenes_out_dir)
    done_marker = os.path.join(subgenes_dir, 'subnets.txt')
    if (not force and os.path.isfile(done_marker)
            and os.path.getsize(done_marker) > 0):
        print(f'  [{label}] already done, skipping (use --force to redo)',
              flush=True)
        return True

    divisions = read_division_leafsets(metadata_csv, max_ret)
    if not divisions:
        print(f'  [{label}] no usable divisions in metadata, skipping',
              flush=True)
        return False

    os.makedirs(subgenes_dir, exist_ok=True)
    runtime_log = os.path.join(subgenes_dir, 'phylonet-runtimelog.txt')
    log = open(runtime_log, 'w')
    log.write(f'{label}: {len(divisions)} division(s), max_ret={max_ret}\n')
    log.flush()

    print(f'  [{label}] {len(divisions)} division(s), max_ret={max_ret}',
          flush=True)
    t_extract = time.time()
    extract_subgene_trees(gene_trees, divisions, subgenes_dir, outgroup)
    extract_subbase_trees(base_tree, divisions, subgenes_dir, outgroup)
    log.write(f'extract: {time.time() - t_extract:.2f}s\n')
    log.flush()

    for idx, (leaf_set, retic) in divisions.items():
        subgene = os.path.join(subgenes_dir, f'subgeneset_{idx}_ret{retic}.txt')
        subbase = os.path.join(subgenes_dir, f'subbase_{idx}_ret{retic}.tree')
        out_path = os.path.join(subgenes_dir, f'phylonet_out_{idx}.txt')
        t0 = time.time()
        ok = run_phylonet_one(subgene, subbase, out_path, retic, jar,
                              threads=threads, java_mem=java_mem, java=java)
        elapsed = time.time() - t0
        log.write(f'division {idx}: {len(leaf_set)} taxa, r={retic}, '
                  f'{elapsed:.2f}s, {"ok" if ok else "FAILED"}\n')
        log.flush()
        print(f'    division {idx}/{len(divisions)}: {len(leaf_set)} taxa, '
              f'r={retic}, {elapsed:.1f}s'
              f'{"" if ok else "  [FAILED]"}', flush=True)

    try:
        networks = combine_subnets(subgenes_dir, len(divisions), outgroup)
    except RuntimeError as e:
        log.write(f'combine FAILED: {e}\n')
        log.close()
        if os.path.exists(done_marker):
            os.remove(done_marker)
        print(f'  [{label}] ERROR: {e}', flush=True)
        return False
    log.write(f'wrote {len(networks)} networks to subnets.txt\n')
    log.close()
    print(f'  [{label}] wrote {len(networks)} network(s) → '
          f'{os.path.join(subgenes_dir, "subnets.txt")}', flush=True)
    return True


def list_run_dirs(divisions_dir, include_non_blob=False):
    """Return [(label, run_dir)] for every division directory to infer."""
    jobs = []
    for blob_dir in sorted(d for d in glob(os.path.join(divisions_dir, 'blob*'))
                           if os.path.isdir(d)):
        blob = os.path.basename(blob_dir)
        runs = sorted(d for d in glob(os.path.join(blob_dir, 'run_*'))
                      if os.path.isdir(d))
        for run_dir in runs:
            jobs.append((f'{blob}/{os.path.basename(run_dir)}', run_dir))
    if include_non_blob:
        nb = os.path.join(divisions_dir, 'non_blob')
        if os.path.isfile(os.path.join(nb, METADATA_FILENAME)):
            jobs.append(('non_blob', nb))
    return jobs


def infer_divisions(divisions_dir, gene_trees, base_tree, jar, max_ret=1,
                    subgenes_out_dir=DEFAULT_SUBGENES_DIR, outgroup='OUT',
                    parallel=1, threads=1, java_mem=DEFAULT_JAVA_MEM,
                    java='java', include_non_blob=False, force=False,
                    max_runs=None):
    """Infer every division under `divisions_dir`. Returns (n_ok, n_failed)."""
    for path, what in ((divisions_dir, 'divisions dir'),
                       (gene_trees, 'gene trees'),
                       (base_tree, 'base tree'),
                       (jar, 'PhyloNet jar')):
        if not os.path.exists(path):
            raise SystemExit(f'ERROR: {what} not found: {path}')

    jobs = list_run_dirs(divisions_dir, include_non_blob)
    if max_runs is not None and max_runs > 0:
        keep, seen = [], {}
        for label, run_dir in jobs:
            blob = label.split('/')[0]
            seen[blob] = seen.get(blob, 0) + 1
            if blob == 'non_blob' or seen[blob] <= max_runs:
                keep.append((label, run_dir))
        jobs = keep
    if not jobs:
        raise SystemExit(
            f'ERROR: no blob*/run_*/ directories under {divisions_dir}. '
            f'Run dimple.divider.generate_k_divisions first.')

    print(f'PhyloNet stage: {len(jobs)} division directory(ies), '
          f'{parallel} at a time, -pl {threads}', flush=True)
    n_ok = n_fail = 0
    with ThreadPoolExecutor(max_workers=max(parallel, 1)) as pool:
        futures = {
            pool.submit(infer_run, run_dir, gene_trees, base_tree, jar,
                        max_ret, subgenes_out_dir, outgroup, threads,
                        java_mem, java, force, label): label
            for label, run_dir in jobs
        }
        for fut in as_completed(futures):
            label = futures[fut]
            try:
                ok = fut.result()
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f'  [{label}] ERROR: {e}', flush=True)
                ok = False
            n_ok += bool(ok)
            n_fail += (not ok)
    print(f'PhyloNet stage done: {n_ok} ok, {n_fail} failed', flush=True)
    return n_ok, n_fail


def main():
    ap = argparse.ArgumentParser(
        description='Infer a network per division with PhyloNet '
                    '(stage 2 of DIMPLE).')
    ap.add_argument('--divisions', required=True,
                    help='Divisions directory written by '
                         'dimple.divider.generate_k_divisions.')
    ap.add_argument('--gene-trees', required=True,
                    help='Gene trees, one newick per line, including the '
                         'outgroup leaf.')
    ap.add_argument('--base-tree', required=True,
                    help='Binary starting tree on the full taxon set: the '
                         'ASTRAL tree estimated from the same gene trees, or '
                         'the TREE-QMC first-pass tree. The blob-collapsed '
                         'tree of blobs will NOT work: PhyloNet rejects a '
                         'non-binary start.')
    ap.add_argument('--phylonet', required=True, help='Path to PhyloNet.jar.')
    ap.add_argument('--max-ret', type=int, default=1,
                    help='Reticulation bound handed to InferNetwork_MPL for '
                         'each division (default: 1).')
    ap.add_argument('--subgenes-out-dir', default=DEFAULT_SUBGENES_DIR,
                    help='Output subdirectory name inside each run dir '
                         f'(default: {DEFAULT_SUBGENES_DIR}). Use a distinct '
                         'name per --max-ret to keep sweeps side by side.')
    ap.add_argument('--outgroup', default='OUT',
                    help='Outgroup leaf label present in the gene trees and '
                         'the base tree (default: OUT). Each division is '
                         'inferred with the outgroup attached and rooted at '
                         'it; it is pruned back off afterwards.')
    ap.add_argument('--parallel', type=int, default=1,
                    help='Division directories to process concurrently.')
    ap.add_argument('--pl', type=int, default=1,
                    help="Threads per PhyloNet search (PhyloNet's -pl).")
    ap.add_argument('--max-runs', type=int, default=None,
                    help='Only infer the first K run_* dirs per blob.')
    ap.add_argument('--java', default='java', help='Java executable.')
    ap.add_argument('--java-mem', default=DEFAULT_JAVA_MEM,
                    help=f'-Xmx for PhyloNet (default: {DEFAULT_JAVA_MEM}).')
    ap.add_argument('--include-non-blob', action='store_true',
                    help='Also infer the non_blob divisions. Not needed for '
                         'the default merger, which reads its non-blob '
                         'subnetworks off the tree of blobs.')
    ap.add_argument('--force', action='store_true',
                    help='Re-infer run dirs that already have subnets.txt.')
    args = ap.parse_args()

    n_ok, n_fail = infer_divisions(
        args.divisions, args.gene_trees, args.base_tree, args.phylonet,
        max_ret=args.max_ret, subgenes_out_dir=args.subgenes_out_dir,
        outgroup=args.outgroup, parallel=args.parallel, threads=args.pl,
        java_mem=args.java_mem, java=args.java,
        include_non_blob=args.include_non_blob, force=args.force,
        max_runs=args.max_runs)
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
