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

Files inside `<run_dir>/<subgenes-out-dir>/`:
  subset<i>/leaf_subset.txt   the taxa of subset i (metadata row i-1)
  subset<i>/base_tree.tre     starting tree restricted to the subset, rooted at the outgroup
  subset<i>/tmp_mpl.nex       the NEXUS handed to PhyloNet in the latest inference of the subset
  subset<i>/subnet_ret<r>.txt inferred network for bound r, outgroup removed (every bound is kept)
  subset<i>/mpl_ret<r>.log    raw PhyloNet output for bound r
  subnets.txt                 line i = the network chosen for subset i (the merger's input)
  inputs.json                 what every subnet_ret<r>.txt and subnets.txt were made from
                              (gene trees, base tree, jar, metadata hashes; the bound per
                              subset); a run is only skipped as done when this still matches
  mpl_runtimelog.txt          per-subset wall time + status, appended across calls

Divisions coming from the non-blob part of the tree of blobs (metadata rows of
type `non_blob_set` / `pruned_subtree`) are tree-like by construction and are
inferred with the reticulation bound fixed to 0. They are skipped entirely
unless `--include-non-blob` is given, because the default merger takes its
non-blob subnetworks straight off the tree of blobs.

Usage:
    python -m dimple.phylonet.infer_subnetworks \\
        --gene-trees gene_trees.tre --base-tree astral.tre \\
        --phylonet PhyloNet.jar --max-ret 1 --parallel 5 --pl 4

Per-subnetwork reticulation numbers: a division run (blob*/run_*) is one
partition of a blob into subsets. List them, infer subsets with whatever
bounds you want to compare (every bound's result is kept), then assemble
the run's subnets.txt from the bound chosen for each subset.
    python -m dimple.phylonet.infer_subnetworks ... --list
    python -m dimple.phylonet.infer_subnetworks ... --only blob00/run_000 --subset 1 --max-ret 2
    python -m dimple.phylonet.infer_subnetworks ... --only blob00/run_000 --subset 1 --max-ret 1
    python -m dimple.phylonet.infer_subnetworks ... --only blob00/run_000 --subset 2 --max-ret 0
    python -m dimple.phylonet.infer_subnetworks ... --only blob00/run_000 --assemble 1,0
        # subnets.txt from subset 1's max_ret=1 network and subset 2's max_ret=0 network
"""
import os
import re
import sys
import csv
import time
import json
import hashlib
import argparse
import fcntl
import shutil
import subprocess
from contextlib import contextmanager
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


def extract_subgene_trees(gene_trees_path, divisions, outgroup='OUT'):
    """Restrict the gene trees to every subset. Returns {i: (newicks, n_skipped)}."""
    trees = _load_trees(gene_trees_path)
    labels = [{lf.taxon.label for lf in t.leaf_node_iter()} for t in trees]
    out = {}
    for idx, (leaf_set, _) in divisions.items():
        wanted = set(leaf_set) | {outgroup}
        kept, skipped = [], 0
        for t, have in zip(trees, labels):
            # a gene tree may miss taxa, but without the outgroup it cannot
            # be rooted, and below 2 ingroup taxa it carries no rooted triple
            if outgroup not in have or len(have & set(leaf_set)) < 2:
                skipped += 1
                continue
            kept.append(_restrict_and_reroot(t, wanted, outgroup)[0])
        if skipped:
            print(f'    subset {idx}: {skipped}/{len(trees)} gene trees '
                  f'skipped (no {outgroup}, or < 2 subset taxa)', flush=True)
        out[idx] = (kept, skipped)
    return out


def subset_dir(subgenes_dir, idx):
    return os.path.join(subgenes_dir, f'subset{idx}')


def extract_subbase_trees(base_tree_path, divisions, subgenes_dir, outgroup='OUT'):
    """Write subset<i>/base_tree.tre (the PhyloNet starting tree) and
    subset<i>/leaf_subset.txt for every subset."""
    base_trees = _load_trees(base_tree_path)
    if len(base_trees) == 0:
        raise ValueError(f'No tree found in base tree file {base_tree_path}')
    base_tree = base_trees[0]
    have = {lf.taxon.label for lf in base_tree.leaf_node_iter()}
    for idx, (leaf_set, _) in divisions.items():
        wanted = set(leaf_set) | {outgroup}
        if wanted - have:
            raise ValueError(
                f'base tree lacks {sorted(wanted - have)[:5]} needed by '
                f'subset {idx}')
        nwk, sub_t = _restrict_and_reroot(base_tree, wanted, outgroup,
                                          topology_only=True)
        if any(len(nd.child_nodes()) != 2 for nd in sub_t.preorder_internal_node_iter()):
            raise ValueError(
                f'base tree restricted to subset {idx} is not binary; '
                f'PhyloNet rejects a non-binary start')
        d = subset_dir(subgenes_dir, idx)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'base_tree.tre'), 'w') as fout:
            fout.write(nwk + '\n')
        with open(os.path.join(d, 'leaf_subset.txt'), 'w') as fout:
            fout.write('\n'.join(sorted(leaf_set)) + '\n')


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


def run_phylonet_one(nexus_path, subbase_path, out_path, max_ret, jar,
                     threads=1, java_mem=DEFAULT_JAVA_MEM, java='java'):
    """Run PhyloNet on an already written NEXUS file, raw output to
    `out_path`. Returns True when the JVM exits 0."""
    if not os.path.exists(jar):
        raise FileNotFoundError(f'PhyloNet jar not found: {jar}')
    if os.path.exists(out_path):
        os.remove(out_path)
    for p, what in ((nexus_path, 'NEXUS'), (subbase_path, 'starting tree')):
        if not os.path.exists(p):
            print(f'    warning: missing {what} {p}, skipping', flush=True)
            return False
    with open(out_path, 'w') as fout:
        proc = subprocess.run([java, f'-Xmx{java_mem}', '-jar', jar, nexus_path],
                              stdout=fout, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f'    warning: PhyloNet exited {proc.returncode} for '
              f'{nexus_path} (see {out_path})', flush=True)
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


def _check_leaves(newick, leaf_set, what):
    got = get_leafset(newick_to_nx(newick))
    if got != set(leaf_set):
        raise RuntimeError(
            f'{what}: network has {len(got)} leaves, the subset has '
            f'{len(leaf_set)} (differs on {sorted(got ^ set(leaf_set))[:5]})')
    return newick


def _check_inferred(out_path, leaf_set, outgroup='OUT'):
    """Return the outgroup-pruned network in the raw PhyloNet output
    `out_path`; RuntimeError if it has no 'Inferred Network #1' or its
    leaves are not `leaf_set`."""
    newick = parse_inferred_network(out_path)
    if newick is None:
        raise RuntimeError(f'no "Inferred Network #1" in {out_path}: PhyloNet '
                           f'did not finish this subset')
    return _check_leaves(remove_outgroup(newick, outgroup), leaf_set,
                         os.path.basename(out_path))


def combine_subnets(subgenes_dir, n_rows, divisions, outgroup='OUT'):
    """Write subnets.txt from subset<i>/subnet_ret<r>.txt, r taken from
    `divisions[i]`. Returns the newick list.

    One line per metadata row; rows not in `divisions` (too few leaves) are
    written blank so line numbers stay equal to subnet_idx.

    Every subset must have its network: the merger indexes subnets.txt by
    row, so a missing line would silently shift every later subset onto the
    wrong leafset. A gap therefore aborts and leaves no subnets.txt behind.
    """
    out_path = os.path.join(subgenes_dir, 'subnets.txt')
    networks = []
    for idx in range(1, n_rows + 1):
        if idx not in divisions:
            networks.append('')
            continue
        leaves, retic = divisions[idx]
        fpath = os.path.join(subset_dir(subgenes_dir, idx), f'subnet_ret{retic}.txt')
        if not os.path.exists(fpath):
            raise RuntimeError(
                f'no network for subset {idx} with max_ret={retic} '
                f'({fpath} missing); refusing to write a subnets.txt whose '
                f'lines no longer line up')
        with open(fpath) as f:
            nwk = f.read().strip()
        networks.append(_check_leaves(nwk, leaves, os.path.relpath(fpath, subgenes_dir)))
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


def _metadata_sha(metadata_csv):
    with open(metadata_csv, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def _input_stamp(meta_sha, input_ids, bounds, outgroup):
    """What a finished subnets.txt was made from; resume only on an exact match.
    `bounds` maps subset index -> reticulation bound, so a run assembled
    from per-subset inferences with different bounds is recorded as such."""
    return {'metadata_sha256': meta_sha, **input_ids,
            'max_ret': {str(i): r for i, r in sorted(bounds.items())},
            'outgroup': outgroup}


def _read_stamp(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


@contextmanager
def _locked(subgenes_dir):
    """Serialise read-modify-write of inputs.json across invocations that
    work on the same run at the same time (e.g. per-subset cluster jobs)."""
    os.makedirs(subgenes_dir, exist_ok=True)
    with open(os.path.join(subgenes_dir, STAMP_FILENAME + '.lock'), 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def _load_inputs(subgenes_dir):
    """inputs.json: {'results': {subset: {bound: inputs the network came from}},
    'assembled': what subnets.txt was made from (absent until assembled)}."""
    d = _read_stamp(os.path.join(subgenes_dir, STAMP_FILENAME))
    return d if isinstance(d, dict) and 'results' in d else {'results': {}}


def _save_inputs(subgenes_dir, d):
    with open(os.path.join(subgenes_dir, STAMP_FILENAME), 'w') as f:
        json.dump(d, f, indent=1)


def _stamps(subgenes_dir, idx):
    """{bound: inputs subset<idx>/subnet_ret<bound>.txt came from}."""
    return _load_inputs(subgenes_dir)['results'].get(str(idx), {})


def _record_result(subgenes_dir, idx, retic, stamp):
    with _locked(subgenes_dir):
        d = _load_inputs(subgenes_dir)
        d['results'].setdefault(str(idx), {})[str(retic)] = stamp
        _save_inputs(subgenes_dir, d)


def _forget_result(subgenes_dir, idx, retic):
    with _locked(subgenes_dir):
        d = _load_inputs(subgenes_dir)
        d['results'].get(str(idx), {}).pop(str(retic), None)
        _save_inputs(subgenes_dir, d)


def _inferred_bounds(subgenes_dir, idx, current):
    """{bound: True if made from the current inputs else False} for one subset."""
    d = subset_dir(subgenes_dir, idx)
    return {int(r): st == current for r, st in _stamps(subgenes_dir, idx).items()
            if os.path.isfile(os.path.join(d, f'subnet_ret{r}.txt'))}


def _drop_result(subgenes_dir, idx, retic):
    """Remove one (subset, bound) result and its stamp."""
    d = subset_dir(subgenes_dir, idx)
    for f in (f'subnet_ret{retic}.txt', f'mpl_ret{retic}.log'):
        if os.path.exists(os.path.join(d, f)):
            os.remove(os.path.join(d, f))
    _forget_result(subgenes_dir, idx, retic)


# ---------------------------------------------------------------------------
# Driving one run directory / a whole divisions tree
# ---------------------------------------------------------------------------

def infer_run(run_dir, gene_trees, base_tree, jar, max_ret=1,
              subgenes_out_dir=DEFAULT_SUBGENES_DIR, outgroup='OUT',
              threads=1, java_mem=DEFAULT_JAVA_MEM, java='java',
              force=False, label=None, only=None, assemble=None, *,
              _input_ids=None):
    """Infer the subsets of one division run (`blob*/run_*` or `non_blob`).

    only=None, assemble=None: infer every subset with `max_ret` from a clean
        output directory and write subnets.txt.
    only={i, ...}: infer just those subsets (1-based, see --list) with
        `max_ret`. Results of other subsets and other bounds are kept, and
        subnets.txt is NOT written: assemble it with the bounds you choose.
    assemble={i: r, ...}: run no PhyloNet; write subnets.txt from the existing
        subset<i>/subnet_ret<r>.txt of every subset, refusing results made
        from other inputs.
    """
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
    meta_sha = _metadata_sha(metadata_csv)
    current = {'metadata_sha256': meta_sha, **input_ids, 'outgroup': outgroup}

    n_rows, divisions = read_division_leafsets(metadata_csv, max_ret if max_ret is not None else 1)
    if not divisions:
        print(f'  [{label}] no usable subsets in metadata, skipping', flush=True)
        return False
    os.makedirs(subgenes_dir, exist_ok=True)

    def _assemble(bounds, note):
        """subnets.txt from subset<i>/subnet_ret<bounds[i]>.txt for every subset."""
        problems = []
        for i in divisions:
            have = _inferred_bounds(subgenes_dir, i, current)
            if bounds[i] not in have:
                problems.append(f'subset {i}: no result for max_ret={bounds[i]}'
                                f' (have {sorted(r for r in have if have[r]) or "none"})')
            elif not have[bounds[i]]:
                problems.append(f'subset {i}: result for max_ret={bounds[i]} was made '
                                f'from other inputs (STALE); re-infer it')
        if problems:
            note('not assembled: ' + '; '.join(problems))
            print(f'  [{label}] subnets.txt not written:\n    ' + '\n    '.join(problems), flush=True)
            return False
        networks = combine_subnets(
            subgenes_dir, n_rows, {i: (ls, bounds[i]) for i, (ls, _) in divisions.items()}, outgroup)
        with _locked(subgenes_dir):
            d = _load_inputs(subgenes_dir)
            d['assembled'] = _input_stamp(meta_sha, input_ids, bounds, outgroup)
            _save_inputs(subgenes_dir, d)
        note(f'wrote {len(networks)} lines to subnets.txt (bounds {bounds})')
        print(f'  [{label}] wrote {len(divisions)} network(s) to {done_marker} '
              f'with bounds {bounds}', flush=True)
        return True

    if assemble is not None:
        unknown = sorted(set(assemble) - set(divisions))
        if unknown:
            raise SystemExit(f'ERROR: [{label}] no inferable subset {unknown}; '
                             f'this run has {sorted(divisions)} (see --list)')
        with open(os.path.join(subgenes_dir, 'mpl_runtimelog.txt'), 'a') as log:
            return _assemble(dict(assemble), lambda m: (log.write(m + '\n'), log.flush()))

    if only is None:
        selected = divisions
        stamp = _input_stamp(meta_sha, input_ids,
                             {i: r for i, (_, r) in divisions.items()}, outgroup)
        if (not force and os.path.isfile(done_marker)
                and _load_inputs(subgenes_dir).get('assembled') == stamp):
            print(f'  [{label}] already done, skipping (use --force to redo)',
                  flush=True)
            return True
        # Start clean: nothing from an earlier run (other inputs, other
        # bounds) may be picked up by the assembly below.
        for f in (done_marker, done_marker + '.tmp', stamp_path):
            if os.path.exists(f):
                os.remove(f)
        for d in glob(os.path.join(subgenes_dir, 'subset*')):
            if os.path.isdir(d):
                shutil.rmtree(d)
    else:
        unknown = sorted(set(only) - set(divisions))
        if unknown:
            raise SystemExit(f'ERROR: [{label}] no inferable subset {unknown}; '
                             f'this run has {sorted(divisions)} (see --list)')
        selected = {i: divisions[i] for i in sorted(only)}
        if not force:
            for i in list(selected):
                if _inferred_bounds(subgenes_dir, i, current).get(selected[i][1]):
                    print(f'  [{label}] subset {i} already inferred with '
                          f'max_ret={selected[i][1]}, skipping (use --force to redo)',
                          flush=True)
                    del selected[i]
        for i, (_, r) in selected.items():          # only this (subset, bound)
            if os.path.isdir(subset_dir(subgenes_dir, i)):
                _drop_result(subgenes_dir, i, r)

    if only is None or selected:
        print(f'  [{label}] {len(selected)} subset(s) to infer, max_ret={max_ret}',
              flush=True)
    with open(os.path.join(subgenes_dir, 'mpl_runtimelog.txt'),
              'w' if only is None else 'a') as log:
        def note(msg):
            log.write(msg + '\n')
            log.flush()
        note(f'{label}: {len(selected)} subset(s), max_ret={max_ret}')
        gts = {}
        if selected:
            t_extract = time.time()
            gts = extract_subgene_trees(gene_trees, selected, outgroup)
            extract_subbase_trees(base_tree, selected, subgenes_dir, outgroup)
            note(f'extract: {time.time() - t_extract:.2f}s')

        failed = []
        for idx, (leaf_set, retic) in selected.items():
            d = subset_dir(subgenes_dir, idx)
            trees, skipped = gts[idx]
            # PhyloNet reads a NEXUS private to this invocation, so a second
            # invocation on the same subset (another bound, in parallel) cannot
            # swap it under the JVM; it is kept as tmp_mpl.nex afterwards.
            nexus = os.path.join(d, f'tmp_mpl.{os.getpid()}.nex')
            subbase = os.path.join(d, 'base_tree.tre')
            raw = os.path.join(d, f'mpl_ret{retic}.log')
            t0 = time.time()
            ok = len(trees) >= 2
            if not ok:
                print(f'    subset {idx}: only {len(trees)} usable gene tree(s)', flush=True)
            else:
                with open(subbase) as f:
                    write_nexus(trees, f.read().strip(), retic, nexus, threads)
                ok = run_phylonet_one(nexus, subbase, raw, retic, jar,
                                      threads=threads, java_mem=java_mem, java=java)
                os.replace(nexus, os.path.join(d, 'tmp_mpl.nex'))
            if ok:
                # only a parseable network on exactly the subset's taxa counts
                # as done; otherwise every later retry would skip it
                try:
                    pruned = _check_inferred(raw, leaf_set, outgroup)
                    with open(os.path.join(d, f'subnet_ret{retic}.txt'), 'w') as f:
                        f.write(pruned + '\n')
                    _record_result(subgenes_dir, idx, retic, current)
                except RuntimeError as e:
                    print(f'    {e}', flush=True)
                    ok = False
            elapsed = time.time() - t0
            if not ok:
                failed.append(idx)
            note(f'subset {idx}: {len(leaf_set)} taxa, r={retic}, '
                 f'{skipped} gene trees skipped, {elapsed:.2f}s, '
                 f'{"ok" if ok else "FAILED"}')
            print(f'    subset {idx} (row {idx - 1}): {len(leaf_set)} taxa, '
                  f'r={retic}, {elapsed:.1f}s'
                  f'{"" if ok else "  [FAILED]"}', flush=True)
        if failed:
            note(f'FAILED on subset(s) {failed}')
            print(f'  [{label}] ERROR: PhyloNet failed on subset(s) {failed}', flush=True)
            return False
        if only is not None:
            print(f'  [{label}] done; choose the bounds and run --assemble to '
                  f'write subnets.txt', flush=True)
            return True
        return _assemble({i: r for i, (_, r) in divisions.items()}, note)


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
                    max_runs=None, only_runs=None, only_divisions=None,
                    list_only=False, assemble=None):
    """Infer every division under `divisions_dir`. Returns (n_ok, n_failed).

    `only_runs` restricts to division runs by label ('blob00/run_001');
    `only_divisions` (1-based subset indices, see `list_only`) restricts to
    subsets within them, inferred with `max_ret` and assembled with what the
    other subsets already have. `assemble` = [bound of subset 1, of subset 2,
    ...] runs no PhyloNet and writes subnets.txt from those results; it needs
    exactly one run dir (`only_runs`).
    """
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
    if only_runs:
        known = {label for label, _ in jobs}
        bad = sorted(set(only_runs) - known)
        if bad:
            raise SystemExit(f'ERROR: unknown run dir(s) {bad}; known: {sorted(known)}')
        jobs = [(label, d) for label, d in jobs if label in only_runs]
    if list_only:
        input_ids = _shared_input_ids(gene_trees, base_tree, jar)
        print(f'{"run":18s} {"subset":>6s} {"taxa":>5s}  status')
        for label, run_dir in jobs:
            meta = os.path.join(run_dir, METADATA_FILENAME)
            n_rows, divisions = read_division_leafsets(meta, max_ret)
            sub = os.path.join(run_dir, subgenes_out_dir)
            current = {'metadata_sha256': _metadata_sha(meta), **input_ids, 'outgroup': outgroup}
            for i, (leaves, _) in divisions.items():
                have = _inferred_bounds(sub, i, current)
                good = sorted(r for r in have if have[r]); stale = sorted(r for r in have if not have[r])
                status = ('inferred max_ret=' + ','.join(map(str, good)) if good else 'not inferred') + \
                         (f'  STALE max_ret={",".join(map(str, stale))} (inputs changed)' if stale else '')
                print(f'{label:18s} {i:6d} {len(leaves):5d}  {status}   '
                      f'{",".join(sorted(leaves)[:6])}{",..." if len(leaves) > 6 else ""}')
            st = _load_inputs(sub).get('assembled')
            print(f'{label:18s}  subnets.txt: ' + (
                f'present, bounds {st["max_ret"]}' if st and os.path.isfile(os.path.join(sub, 'subnets.txt'))
                else 'absent'))
        return 0, 0
    if assemble is not None:
        if only_divisions:
            raise SystemExit('ERROR: --assemble lists the bound of every subset; it takes no --subset')
        if len(jobs) != 1:
            raise SystemExit('ERROR: --assemble needs exactly one division run; add '
                             f'--only <run> (candidates: {[l for l, _ in jobs]})')
    if only_divisions and len(jobs) != 1:
        raise SystemExit('ERROR: --subset needs exactly one division run; add '
                         f'--only <run> (candidates: {[l for l, _ in jobs]})')

    print(f'PhyloNet stage: {len(jobs)} division run(s), '
          f'{parallel} at a time, -pl {threads}', flush=True)
    input_ids = _shared_input_ids(gene_trees, base_tree, jar)

    def _bounds_for(run_dir):
        """--assemble: the listed bounds mapped onto this run's subsets, or exit."""
        _, divisions = read_division_leafsets(os.path.join(run_dir, METADATA_FILENAME), 0)
        if len(assemble) != len(divisions):
            raise SystemExit(f'ERROR: --assemble lists {len(assemble)} bound(s) but '
                             f'{run_dir} has {len(divisions)} subsets '
                             f'({sorted(divisions)}; see --list)')
        return dict(zip(sorted(divisions), assemble))

    n_ok = n_fail = 0
    with ThreadPoolExecutor(max_workers=max(parallel, 1)) as pool:
        futures = {
            pool.submit(infer_run, run_dir, gene_trees, base_tree, jar,
                        max_ret, subgenes_out_dir, outgroup, threads,
                        java_mem, java, force, label,
                        set(only_divisions) if only_divisions else None,
                        _bounds_for(run_dir) if assemble is not None else None,
                        _input_ids=input_ids): label
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
    ap.add_argument('dir', nargs='?', default='dimple_out',
                    help='Directory written by dimple.divider.generate_k_divisions '
                         '(default: dimple_out).')
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
                         'each subset (default: 1).')
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
    ap.add_argument('--only', metavar='RUN', action='append', default=None,
                    help="Only this run dir, e.g. 'blob00/run_001' "
                         '(repeatable, or comma-separated).')
    ap.add_argument('--subset', metavar='N', action='append', default=None,
                    help='Only this subset of the --only division run (1-based '
                         'index as printed by --list; repeatable or '
                         'comma-separated). It is inferred with --max-ret and '
                         'assembled with the subsets already inferred, so one '
                         'call per subset sets a per-subnetwork bound.')
    ap.add_argument('--list', action='store_true',
                    help='List every division run and its subsets, with what '
                         'is inferred so far, then exit.')
    ap.add_argument('--assemble', metavar='R1,R2,...', default=None,
                    help='Run no PhyloNet: write subnets.txt of the --only run '
                         'from the results already inferred, taking for subset '
                         '1 its max_ret=R1 network, for subset 2 its max_ret=R2 '
                         'network, and so on (one number per subset, in --list '
                         'order). This is where the number of reticulations per '
                         'subnetwork is decided.')
    args = ap.parse_args()
    split = lambda xs: [v.strip() for x in (xs or []) for v in x.split(',') if v.strip()]
    only_runs = split(args.only) or None
    try:
        only_divisions = [int(v) for v in split(args.subset)] or None
        assemble = ([int(v) for v in split([args.assemble])]
                    if args.assemble is not None else None)
    except ValueError:
        ap.error('--subset and --assemble take integers')
    if assemble is not None and not assemble:
        ap.error('--assemble needs one bound per subset, e.g. --assemble 1,0,2')

    n_ok, n_fail = infer_divisions(
        args.dir, args.gene_trees, args.base_tree, args.phylonet,
        max_ret=args.max_ret, subgenes_out_dir=args.subgenes_out_dir,
        outgroup=args.outgroup, parallel=args.parallel, threads=args.pl,
        java_mem=args.java_mem, java=args.java,
        include_non_blob=args.include_non_blob, force=args.force,
        max_runs=args.max_runs, only_runs=only_runs,
        only_divisions=only_divisions, list_only=args.list,
        assemble=assemble)
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
