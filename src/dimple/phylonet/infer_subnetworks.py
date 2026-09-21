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
  inputs.json                 what subnets.txt was made from; a run is only
                              skipped as done when this still matches

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
        --num-ret 1 --parallel 5 --pl 4
"""
import os
import re
import sys
import csv
import time
import json
import hashlib
import argparse
import subprocess
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

import dendropy

from dimple.utils.network_util import (
    newick_to_nx,
    build_newick_from_graph,
    contract_degree2_nodes,
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
    """Return (n_rows, {i: (leafset, r)}) for one run directory.

    i is 1-based and equals the metadata row position + 1, i.e. line i of
    `subnets.txt`. The merger looks a division up by `subnet_idx` as a
    PHYSICAL line number, so rows that cannot be inferred (no leaves, or fewer
    than two) are left out of the dict but still counted in n_rows: they get a
    blank line rather than shifting every later division up. Rows whose type
    marks them as tree-like (non-blob / pruned subtree) get r = 0; every other
    row gets r = max_ret.
    """
    info = {}
    n_rows = 0
    with open(metadata_csv, newline='') as fin:
        for pos, row in enumerate(csv.DictReader(fin)):
            n_rows = pos + 1
            sid = (row.get('subnet_idx') or '').strip()
            if sid and int(sid) != pos:
                raise ValueError(
                    f'{metadata_csv}: subnet_idx {sid} on row {pos}; the '
                    f'merger reads subnets.txt by row position')
            leaves = [t.strip() for t in (row.get('all_leaves') or '').split(',')
                      if t.strip()]
            if len(leaves) < 2:
                continue
            row_type = (row.get('type') or '').strip()
            r = 0 if row_type in TREE_LIKE_ROW_TYPES else max_ret
            info[pos + 1] = (set(leaves), r)
    return n_rows, info


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
        preserve_underscores=True,      # as the divider and merger do
    )


def _restrict_and_reroot(tree, leaf_set, outgroup, topology_only=False):
    """Restrict `tree` to `leaf_set` and reroot it at the outgroup leaf.
    Returns (newick, restricted dendropy tree)."""
    sub_t = tree.extract_tree_with_taxa_labels(leaf_set)
    out_node = sub_t.find_node_with_taxon_label(outgroup)
    if out_node is not None and out_node.parent_node is not None:
        sub_t.reroot_at_edge(out_node.edge, update_bipartitions=False)
    kw = dict(suppress_edge_lengths=True, suppress_internal_node_labels=True,
              suppress_annotations=True) if topology_only else {}
    nwk = sub_t.as_string(schema='newick', suppress_rooting=True,
                          unquoted_underscores=True, **kw).strip()
    return nwk, sub_t


def extract_subgene_trees(gene_trees_path, divisions, out_dir, outgroup='OUT'):
    """Write subgeneset_<i>_ret<r>.txt for every division."""
    os.makedirs(out_dir, exist_ok=True)
    trees = _load_trees(gene_trees_path)
    labels = [{lf.taxon.label for lf in t.leaf_node_iter()} for t in trees]
    skipped = {}
    for idx, (leaf_set, retic) in divisions.items():
        wanted = set(leaf_set) | {outgroup}
        out_path = os.path.join(out_dir, f'subgeneset_{idx}_ret{retic}.txt')
        skipped[idx] = 0
        with open(out_path, 'w') as fout:
            for t, have in zip(trees, labels):
                # a gene tree may miss taxa, but without the outgroup it cannot
                # be rooted, and below 2 ingroup taxa it carries no rooted triple
                if outgroup not in have or len(have & set(leaf_set)) < 2:
                    skipped[idx] += 1
                    continue
                fout.write(_restrict_and_reroot(t, wanted, outgroup)[0] + '\n')
        if skipped[idx]:
            print(f'    division {idx}: {skipped[idx]}/{len(trees)} gene trees '
                  f'skipped (no {outgroup}, or < 2 division taxa)', flush=True)
    return skipped


def extract_subbase_trees(base_tree_path, divisions, out_dir, outgroup='OUT'):
    """Write subbase_<i>_ret<r>.tree (the PhyloNet starting tree) per division."""
    os.makedirs(out_dir, exist_ok=True)
    base_trees = _load_trees(base_tree_path)
    if len(base_trees) == 0:
        raise ValueError(f'No tree found in base tree file {base_tree_path}')
    base_tree = base_trees[0]
    have = {lf.taxon.label for lf in base_tree.leaf_node_iter()}
    for idx, (leaf_set, retic) in divisions.items():
        wanted = set(leaf_set) | {outgroup}
        out_path = os.path.join(out_dir, f'subbase_{idx}_ret{retic}.tree')
        if wanted - have:
            raise ValueError(
                f'base tree lacks {sorted(wanted - have)[:5]} needed by '
                f'division {idx}')
        nwk, sub_t = _restrict_and_reroot(base_tree, wanted, outgroup,
                                          topology_only=True)
        if any(len(nd.child_nodes()) != 2 for nd in sub_t.preorder_internal_node_iter()):
            raise ValueError(
                f'base tree restricted to division {idx} is not binary; '
                f'PhyloNet rejects a non-binary start')
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
    if os.path.exists(out_path):
        os.remove(out_path)
    if not os.path.exists(subgene_path):
        print(f'    warning: missing {subgene_path}, skipping', flush=True)
        return False
    if not os.path.exists(subbase_path):
        print(f'    warning: missing {subbase_path}, skipping', flush=True)
        return False

    with open(subbase_path) as f:
        start_tree = f.read().strip()
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


def combine_subnets(subgenes_dir, n_rows, divisions, outgroup='OUT'):
    """Write subnets.txt from phylonet_out_*.txt. Returns the newick list.

    One line per metadata row; rows not in `divisions` (too few leaves) are
    written blank so line numbers stay equal to subnet_idx.

    Every division must have produced an inferred network: the merger indexes
    subnets.txt by division order, so a missing line would silently shift every
    later division onto the wrong leafset. A gap therefore aborts and leaves no
    subnets.txt behind.
    """
    out_path = os.path.join(subgenes_dir, 'subnets.txt')
    networks = []
    for idx in range(1, n_rows + 1):
        if idx not in divisions:
            networks.append('')
            continue
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
        pruned = remove_outgroup(newick, outgroup)
        got = get_leafset(newick_to_nx(pruned))
        if got != set(divisions[idx][0]):
            raise RuntimeError(
                f'division {idx}: inferred network has {len(got)} leaves, the '
                f'division has {len(divisions[idx][0])} '
                f'(differs on {sorted(got ^ set(divisions[idx][0]))[:5]})')
        networks.append(pruned)
    with open(out_path + '.tmp', 'w') as fout:
        for nwk in networks:
            fout.write(nwk + '\n')
    os.replace(out_path + '.tmp', out_path)
    return networks


STAMP_FILENAME = 'inputs.json'


def _file_id(path):
    """Read a fresh content hash, including edits that preserve size and mtime."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _shared_input_ids(gene_trees, base_tree, jar):
    """Hash shared inputs once per inference invocation, before starting workers."""
    return {'gene_trees': _file_id(gene_trees), 'base_tree': _file_id(base_tree),
            'phylonet_jar': _file_id(jar)}


def _input_stamp(metadata_csv, input_ids, max_ret, outgroup):
    """What a finished subnets.txt was made from; resume only on an exact match."""
    with open(metadata_csv, 'rb') as f:
        meta_sha = hashlib.sha256(f.read()).hexdigest()
    return {'metadata_sha256': meta_sha, **input_ids, 'max_ret': max_ret,
            'outgroup': outgroup}


def _read_stamp(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Driving one run directory / a whole divisions tree
# ---------------------------------------------------------------------------

def infer_run(run_dir, gene_trees, base_tree, jar, max_ret=1,
              subgenes_out_dir=DEFAULT_SUBGENES_DIR, outgroup='OUT',
              threads=1, java_mem=DEFAULT_JAVA_MEM, java='java',
              force=False, label=None, *, _input_ids=None):
    """Infer every division in one `run_*` (or `non_blob`) directory."""
    label = label or os.path.basename(run_dir)
    metadata_csv = os.path.join(run_dir, METADATA_FILENAME)
    if not os.path.exists(metadata_csv):
        print(f'  [{label}] no {METADATA_FILENAME}, skipping', flush=True)
        return False

    subgenes_dir = os.path.join(run_dir, subgenes_out_dir)
    done_marker = os.path.join(subgenes_dir, 'subnets.txt')
    stamp_path = os.path.join(subgenes_dir, STAMP_FILENAME)
    input_ids = (_shared_input_ids(gene_trees, base_tree, jar)
                 if _input_ids is None else _input_ids)
    stamp = _input_stamp(metadata_csv, input_ids, max_ret, outgroup)
    if not force and os.path.isfile(done_marker) and _read_stamp(stamp_path) == stamp:
        print(f'  [{label}] already done, skipping (use --force to redo)',
              flush=True)
        return True

    n_rows, divisions = read_division_leafsets(metadata_csv, max_ret)
    if not divisions:
        print(f'  [{label}] no usable divisions in metadata, skipping',
              flush=True)
        return False

    # Start clean: nothing from an earlier run (other inputs, other divisions)
    # may be picked up by the collection step below.
    os.makedirs(subgenes_dir, exist_ok=True)
    for pat in ('subnets.txt', 'subnets.txt.tmp', STAMP_FILENAME,
                'phylonet_out_*.txt', 'subgeneset_*.txt', 'subbase_*.tree',
                'tmp_phylonet_*.nex'):
        for f in glob(os.path.join(subgenes_dir, pat)):
            os.remove(f)

    print(f'  [{label}] {len(divisions)} division(s), max_ret={max_ret}',
          flush=True)
    with open(os.path.join(subgenes_dir, 'phylonet-runtimelog.txt'), 'w') as log:
        def note(msg):
            log.write(msg + '\n')
            log.flush()
        note(f'{label}: {len(divisions)} division(s), max_ret={max_ret}')
        t_extract = time.time()
        skipped = extract_subgene_trees(gene_trees, divisions, subgenes_dir, outgroup)
        extract_subbase_trees(base_tree, divisions, subgenes_dir, outgroup)
        note(f'extract: {time.time() - t_extract:.2f}s')

        failed = []
        for idx, (leaf_set, retic) in divisions.items():
            subgene = os.path.join(subgenes_dir, f'subgeneset_{idx}_ret{retic}.txt')
            subbase = os.path.join(subgenes_dir, f'subbase_{idx}_ret{retic}.tree')
            out_path = os.path.join(subgenes_dir, f'phylonet_out_{idx}.txt')
            t0 = time.time()
            ok = run_phylonet_one(subgene, subbase, out_path, retic, jar,
                                  threads=threads, java_mem=java_mem, java=java)
            elapsed = time.time() - t0
            if not ok:
                failed.append(idx)
            note(f'division {idx}: {len(leaf_set)} taxa, r={retic}, '
                 f'{skipped[idx]} gene trees skipped, {elapsed:.2f}s, '
                 f'{"ok" if ok else "FAILED"}')
            print(f'    division {idx} (row {idx - 1}): {len(leaf_set)} taxa, '
                  f'r={retic}, {elapsed:.1f}s'
                  f'{"" if ok else "  [FAILED]"}', flush=True)

        try:
            if failed:
                raise RuntimeError(f'PhyloNet failed on division(s) {failed}')
            networks = combine_subnets(subgenes_dir, n_rows, divisions, outgroup)
        except RuntimeError as e:
            note(f'combine FAILED: {e}')
            print(f'  [{label}] ERROR: {e}', flush=True)
            return False
        with open(stamp_path, 'w') as f:
            json.dump(stamp, f, indent=1)
        note(f'wrote {len(networks)} lines to subnets.txt')
    print(f'  [{label}] wrote {len(divisions)} network(s) to '
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

    # infer_run clears this directory inside every run dir before it starts,
    # so it has to be a private name there, never a path shared between runs.
    if os.path.basename(subgenes_out_dir) != subgenes_out_dir or subgenes_out_dir in ('', '.', '..'):
        raise SystemExit(f'ERROR: --subgenes-out-dir must be a plain directory '
                         f'name, not a path: {subgenes_out_dir!r}')

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
    input_ids = _shared_input_ids(gene_trees, base_tree, jar)
    n_ok = n_fail = 0
    with ThreadPoolExecutor(max_workers=max(parallel, 1)) as pool:
        futures = {
            pool.submit(infer_run, run_dir, gene_trees, base_tree, jar,
                        max_ret, subgenes_out_dir, outgroup, threads,
                        java_mem, java, force, label, _input_ids=input_ids): label
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
    ap.add_argument('--max-ret', '--num-ret', dest='max_ret', type=int, default=1,
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
