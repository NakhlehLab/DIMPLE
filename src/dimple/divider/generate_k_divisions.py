'''
Division algorithm v3 — driver (per-blob output layout).

Same algorithm as v2 (isolated mega-blob + orphan-internals cleanup). Output
format splits by (mega-)blob and only runs the allocator k times for blobs
where the item total exceeds SIZE.

Layout:
  <output_dir>/
  ├── non_blob/
  │   ├── subnetworks_output_metadata.csv   # pruned_subtree + non_blob_set rows
  │   └── subnetworks_output.txt            # corresponding newicks
  ├── blob00/                               # first (mega-)blob
  │   ├── run_000/
  │   │   ├── subnetworks_output_metadata.csv   # this blob's blob_group rows
  │   │   └── subnetworks_output.txt
  │   ├── run_001/                          # present only if blob needs > 1 group
  │   │   └── ...
  │   └── ... up to k_actual runs (fewer if dedup)
  ├── blob01/
  │   └── ...

Per-blob `k_actual`:
  - If sum(item sizes) <= SIZE, exactly 1 run (everything in one group).
  - Else, up to k runs with random bin-packing; duplicates (same fingerprint
    against prior runs for the same blob) are skipped.

Blob indexing (blob00, blob01, ...) is by sorted mega_blob_name.
'''

import os
import re
import random
import csv
import json
import argparse
import ast
import time
import resource
import hashlib
import tempfile
from pathlib import Path
import networkx as nx
from collections import defaultdict

from dimple.utils.network_util import (
    newick_to_nx,
    clean_extended_newick,
    build_newick_from_graph,
    get_leafset,
    extract_subnetwork_by_leaves,
    get_blob_nodes,
    count_reticulations,
)
from dimple.utils.division_util import (
    find_direct_child_blobs_dict,
    compute_forbidden_edges,
    global_prune_pass,
    build_mega_blobs,
    mega_blob_root,
    mega_blob_items,
    mega_blob_leafset,
    mega_blob_name,
    isolate_mega_blobs,
    final_cut_edges as _v1_final_cut_edges,
    greedy_recursive_cuts as _v1_greedy_recursive_cuts,
    extract_from_tob_graph as _extract_non_blob_tob_subnets,
    NON_BLOB_TOB_FILE_NAME as _NON_BLOB_TOB_FILE,
)


# ---------------------------------------------------------------------------
# Helpers reused verbatim from v1
# ---------------------------------------------------------------------------

def group_indices_by_value(s):
    groups = defaultdict(list)
    for i, w in enumerate(s):
        groups[w].append(i)
    return dict(groups)


def group_leafsets_by_index_groups(items, index_groups):
    grouped = {}
    item_mapping = {}
    for group_id, indices in index_groups.items():
        merged = set()
        items_in_group = []
        for i in indices:
            merged |= items[i][1]
            items_in_group.append(items[i])
        grouped[group_id] = merged
        item_mapping[group_id] = items_in_group
    return grouped, item_mapping


def bin_pack_items(item_indices, weights, max_size, min_size=3):
    shuffled = list(item_indices)
    random.shuffle(shuffled)
    print(f"  DEBUG bin_pack: n_items={len(shuffled)}, order_sizes={[weights[i] for i in shuffled]}")

    bins = []
    bin_sizes = []
    for i in shuffled:
        placed = False
        for b in range(len(bins)):
            if bin_sizes[b] + weights[i] <= max_size:
                bins[b].append(i)
                bin_sizes[b] += weights[i]
                placed = True
                break
        if not placed:
            bins.append([i])
            bin_sizes.append(weights[i])

    if len(bins) > 1 and bin_sizes[-1] < min_size:
        last_bin = bins[-1]
        last_size = bin_sizes[-1]
        merged = False
        for b in range(len(bins) - 1):
            if bin_sizes[b] + last_size <= max_size:
                bins[b].extend(last_bin)
                bin_sizes[b] += last_size
                bins.pop()
                bin_sizes.pop()
                print(f"  Merged undersized bin ({last_size}) into bin {b} (now {bin_sizes[b]})")
                merged = True
                break
        if not merged:
            needed = min_size - last_size
            while needed > 0:
                candidates = []
                for b in range(len(bins) - 1):
                    if bin_sizes[b] > min_size:
                        for i in bins[b]:
                            if bin_sizes[b] - weights[i] >= min_size and last_size + weights[i] <= max_size:
                                candidates.append((b, i))
                if not candidates:
                    break
                b, i = random.choice(candidates)
                bins[b].remove(i)
                bin_sizes[b] -= weights[i]
                last_bin.append(i)
                last_size += weights[i]
                bin_sizes[-1] = last_size
                needed = min_size - last_size
                print(f"  Moved item {i} (size {weights[i]}) from bin {b} to last bin")
            if last_size < min_size:
                print(f"  WARNING: Last bin size ({last_size}) < {min_size}, could not fix")

    return bins


def enumerate_unique_partitions(weights, max_size, min_size=3, max_partitions=None):
    """
    Enumerate all unique partitions of items [0..N) such that each bin:
      - sum(weights) <= max_size (hard)
      - sum(weights) >= min_size (hard; if no such partition exists caller can
        relax by passing min_size=1)

    Uses restricted growth canonical form: item 0 always in bin 0; each later
    item either joins an existing bin (by order of creation) or starts a new
    bin. This guarantees each set partition is visited exactly once, so we get
    unique outputs without post-hoc deduplication.

    If max_partitions is provided, stops early after finding that many.

    Returns list of partitions (each a list of bins, each bin a list of item indices).
    """
    N = len(weights)
    if N == 0:
        return []
    results = []
    bins = []  # list of [sum, [item_indices]]

    def backtrack(item_idx):
        if max_partitions is not None and len(results) >= max_partitions:
            return True
        if item_idx == N:
            if all(b[0] >= min_size for b in bins):
                results.append([list(b[1]) for b in bins])
            return False
        w = weights[item_idx]
        # Try joining each existing bin (in creation order — canonical)
        for b in bins:
            if b[0] + w <= max_size:
                b[0] += w
                b[1].append(item_idx)
                if backtrack(item_idx + 1):
                    b[0] -= w
                    b[1].pop()
                    return True
                b[0] -= w
                b[1].pop()
        # Start a new bin
        bins.append([w, [item_idx]])
        stop = backtrack(item_idx + 1)
        bins.pop()
        return stop

    backtrack(0)
    return results


def partition_to_division(bins, out_leafsets):
    """Convert a list of bins (item indices) to (division_dict, item_mapping)."""
    division_dict = {}
    item_mapping = {}
    for gid, bin_items in enumerate(bins):
        merged = set()
        items_in = []
        for i in bin_items:
            merged |= out_leafsets[i][1]
            items_in.append(out_leafsets[i])
        division_dict[gid] = merged
        item_mapping[gid] = items_in
    return division_dict, item_mapping


def heuristic_allocation(weights, out_leafsets, max_size=12, min_size=3, strategy="random"):
    N = len(weights)
    total_size = sum(weights)
    if total_size <= max_size:
        if total_size < min_size:
            print(f"  WARNING: Total size ({total_size}) < {min_size}, group will be undersized")
        print(f"  Total size ({total_size}) <= {max_size}, keeping all items together")
        all_items = set()
        for _, leafset in out_leafsets:
            all_items |= leafset
        return {0: all_items}, {0: out_leafsets}

    if strategy == "smallest_first":
        print(f"  Total size ({total_size}) > {max_size}, smallest-first ...")
        sorted_indices = sorted(range(N), key=lambda i: weights[i])
        first_bin, first_bin_size, rest = [], 0, []
        for i in sorted_indices:
            if first_bin_size + weights[i] <= max_size:
                first_bin.append(i)
                first_bin_size += weights[i]
            else:
                rest.append(i)
        print(f"  First bin: {len(first_bin)} items (size {first_bin_size}), rest: {len(rest)} items")
        if first_bin_size < min_size:
            bins = bin_pack_items(list(range(N)), weights, max_size, min_size)
        else:
            bins = [first_bin] if first_bin else []
            if rest:
                bins += bin_pack_items(rest, weights, max_size, min_size)
    else:
        print(f"  Total size ({total_size}) > {max_size}, bin packing {N} items randomly...")
        bins = bin_pack_items(list(range(N)), weights, max_size, min_size)

    assign = [-1] * N
    for current_group, bin_items in enumerate(bins):
        bin_size = sum(weights[i] for i in bin_items)
        for i in bin_items:
            assign[i] = current_group
        print(f"    Group {current_group}: {len(bin_items)} items (total size {bin_size})")

    indice_dict = group_indices_by_value(assign)
    division_dict, item_mapping = group_leafsets_by_index_groups(out_leafsets, indice_dict)
    return division_dict, item_mapping


# ---------------------------------------------------------------------------
# Oversize-subtree splitting (for pruned subtrees > SIZE)
# ---------------------------------------------------------------------------

def leafsets_after_cuts(G, cut_edges):
    H = G.copy()
    H.remove_edges_from(cut_edges)
    subnets = list(nx.weakly_connected_components(H))
    leafset_dict = {}
    for i, comp in enumerate(subnets, start=1):
        subH = H.subgraph(comp).copy()
        leaves = get_leafset(subH)
        leafset_dict[f"subnet_{i}"] = leaves
    return leafset_dict


def cut_large_group(T, leafset, SIZE=12, min_size=3):
    """Split a leafset > SIZE into pieces <= SIZE using the pruned subtree's topology."""
    if len(leafset) <= SIZE:
        return {"subnet_1": leafset}

    subtree = extract_subnetwork_by_leaves(T, leafset)
    num_cuts = max(1, len(leafset) // SIZE)
    all_edges = list(subtree.edges())
    # relaxed=True: drop the out_degree<=2 filter (the pruned subtree has no blobs anyway)
    refined = _v1_final_cut_edges(subtree, all_edges, relaxed=True)

    if refined and num_cuts > 0:
        final_cut = _v1_greedy_recursive_cuts(subtree, refined, num_cuts)
        result = leafsets_after_cuts(subtree, final_cut)
        out = {}
        for key, sub in result.items():
            if len(sub) > SIZE:
                sub_result = cut_large_group(T, sub, SIZE, min_size)
                for sk, sl in sub_result.items():
                    out[f"{key}_{sk}"] = sl
            else:
                out[key] = sub
        return out
    else:
        print(f"  No valid cut edges for {len(leafset)}-leaf group — falling back to leaf bin-packing.")
        leaves = sorted(leafset)
        random.shuffle(leaves)
        bins, current = [], []
        for leaf in leaves:
            if len(current) >= SIZE:
                bins.append(current)
                current = []
            current.append(leaf)
        if current:
            bins.append(current)
        return {f"heuristic_bin_{i+1}": set(b) for i, b in enumerate(bins)}


# ---------------------------------------------------------------------------
# Source-item lookup for pruned subtrees (for the source_item CSV column)
# ---------------------------------------------------------------------------

def find_source_item_for_pruned_subtree(T, origin_mega, cut_edge,
                                         allocation_results, pruned_actual_leaves=None,
                                         cut_siblings=None):
    """Return the RAW source_item for a pruned subtree.

    Prefers `cut_siblings[cut_edge]` (sibling leaves measured in residual H
    at the moment of THIS cut) — avoids stale references to leaves that were
    cut earlier by other prunes.

    Falls back to original-TOB calculation: leafset(parent in TOB) - cut content.

    Raises ValueError if sibling_leaves is empty.

    Returns (sibling_leaves, source_type, origin_mega) where source_type is:
      - 'blob_item'  — all sibling leaves are blob items of origin_mega
      - 'prune'      — all sibling leaves belong to other pruned subtrees
      - 'mixed'      — some blob items and some pruned leaves
    """
    parent_node, child_node = cut_edge
    if cut_siblings is not None and cut_edge in cut_siblings:
        sibling_leaves = set(cut_siblings[cut_edge])
    else:
        parent_leaves = get_leafset(T, parent_node)
        if pruned_actual_leaves is not None:
            pruned_leaves = set(pruned_actual_leaves)
        else:
            pruned_leaves = get_leafset(T, child_node)
        sibling_leaves = parent_leaves - pruned_leaves
    if not sibling_leaves:
        raise ValueError(
            f"Empty sibling_leaves for cut {cut_edge} in mega-blob "
            f"{origin_mega}: parent_leaves={len(parent_leaves)}, "
            f"pruned_leaves={len(pruned_leaves)}"
        )

    # Classify sibling leaves
    result = allocation_results.get(origin_mega, {}) if origin_mega else {}
    mappings = result.get('item_mappings', {}) if result else {}
    all_item_leaves = set()
    for group_name, items in mappings.items():
        for item_node, item_leafset in items:
            all_item_leaves |= item_leafset

    in_blob = sibling_leaves & all_item_leaves
    if in_blob == sibling_leaves:
        source_type = 'blob_item'
    elif not in_blob:
        source_type = 'prune'
    else:
        source_type = 'mixed'

    return sorted(list(sibling_leaves)), source_type, origin_mega


# ---------------------------------------------------------------------------
# Main division pipeline (per-blob folder layout, per-blob k dedup)
# ---------------------------------------------------------------------------

CSV_FIELDS = ['subnet_idx', 'type', 'blob', 'parent_blob', 'child_blobs',
              'group', 'cut_edge', 'source_item', 'all_leaves', 'items']


def _build_row(subnet_idx, key, leafset, info, allocation_results=None, T=None,
                cut_siblings=None):
    """Build one CSV row dict from a subnet entry."""
    row = {
        'subnet_idx': subnet_idx,
        'type': info['type'] or '',
        'blob': info['blob'] or '',
        'parent_blob': '',
        'child_blobs': '',
        'group': '',
        'cut_edge': '',
        'source_item': '',
        'all_leaves': ','.join(sorted(list(leafset))),
        'items': ''
    }
    if info['type'] == 'blob_group':
        row['group'] = info['group']
        mega_name = info['blob']
        mega_members = info.get('mega_members') or []
        try:
            gid_int = int(info['group'].split('_')[-1])
        except (ValueError, IndexError):
            gid_int = info['group']
        items_dict = {}
        if allocation_results and mega_name in allocation_results:
            mappings = allocation_results[mega_name].get('item_mappings', {})
            if gid_int in mappings:
                for item_node, item_leafset in mappings[gid_int]:
                    if item_node in mega_members and len(mega_members) > 1:
                        items_dict[f"member_blob_{item_node}"] = sorted(list(item_leafset))
                    else:
                        items_dict[str(item_node)] = sorted(list(item_leafset))
        if len(mega_members) > 1:
            items_dict["__members__"] = list(mega_members)
        row['items'] = json.dumps(items_dict)
        # Record blob's source_item (attachment anchor) — same across all runs
        # of the same blob. Pulled from info['blob_source_item'] if present.
        bsi = info.get('blob_source_item')
        if bsi:
            row['source_item'] = json.dumps({
                'parent_in_tob': bsi.get('parent'),
                'sibling_leaves': bsi.get('sibling_leaves'),
            })

    elif info['type'] == 'pruned_subtree':
        row['cut_edge'] = info['cut_edge']
        items_dict = {key: sorted(list(leafset))}
        row['items'] = json.dumps(items_dict)
        if allocation_results is not None and T is not None:
            try:
                cut_edge_tuple = ast.literal_eval(info['cut_edge'])
                src_leaves, src_type, src_mega = find_source_item_for_pruned_subtree(
                    T, info['origin_mega'], cut_edge_tuple, allocation_results,
                    cut_siblings=cut_siblings,
                )
                row['source_item'] = json.dumps({
                    "leaves": src_leaves or [],
                    "source_type": src_type,
                    "mega_blob": src_mega,
                })
            except Exception as e:
                print(f"    Could not resolve source_item for {key}: {e}")

    elif info['type'] == 'non_blob_set':
        items_dict = {key: sorted(list(leafset))}
        row['items'] = json.dumps(items_dict)
    return row


def _write_rows_to_folder(folder, rows, allocation_results=None, T=None,
                           cut_siblings=None):
    """Write rows (list of (key, leafset, info)) to folder/subnetworks_output_metadata.csv.
    Leafset-only variant: no GT network is required; only metadata is written.
    """
    os.makedirs(folder, exist_ok=True)
    csv_path = os.path.join(folder, 'subnetworks_output_metadata.csv')
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for subnet_idx, (key, leafset, info) in enumerate(rows):
            try:
                row = _build_row(subnet_idx, key, leafset, info, allocation_results, T,
                                  cut_siblings=cut_siblings)
                writer.writerow(row)
            except Exception as e:
                print(f"  Error writing {key}: {e}")
                writer.writerow({'subnet_idx': subnet_idx, 'type': 'error', 'blob': '',
                                 'parent_blob': '', 'child_blobs': '', 'group': '',
                                 'cut_edge': '', 'source_item': '', 'all_leaves': '',
                                 'items': json.dumps({'error': str(e)})})


def _prune_outgroup_from_newick(tob_tree_str, outgroup_leaves):
    """Prune candidate outgroup leaves from a newick string.

    `outgroup_leaves` is a list of candidate labels. Any candidate not present
    in the tree is silently skipped. Returns (new_newick, present, missing).
    """
    if not outgroup_leaves:
        return tob_tree_str, [], []
    import dendropy
    tree = dendropy.Tree.get(data=tob_tree_str, schema="newick",
                             preserve_underscores=True)
    label_to_taxon = {t.label: t for t in tree.taxon_namespace}
    present = [lbl for lbl in outgroup_leaves if lbl in label_to_taxon]
    missing = [lbl for lbl in outgroup_leaves if lbl not in label_to_taxon]
    if not present:
        return tob_tree_str, [], missing
    tree.prune_taxa([label_to_taxon[lbl] for lbl in present])
    tree.suppress_unifurcations()
    new_str = tree.as_string(schema="newick", suppress_rooting=True,
                             unquoted_underscores=True).strip()
    return new_str, present, missing


DIVISION_STATE_FILE = 'division_inputs.json'


def _division_layout(folder):
    """Record blob/run directories, including unexpected empty directories."""
    root = Path(folder)
    blobs = [p for p in root.glob('blob*') if p.is_dir()]
    return sorted(str(p.relative_to(root)) for p in
                  blobs + [r for b in blobs for r in b.glob('run_*') if r.is_dir()])


def _matching_division_summary(folder, inputs):
    """Reuse only a complete generation with the same inputs and directory layout."""
    root = Path(folder)
    try:
        state = json.loads((root / DIVISION_STATE_FILE).read_text())
        if state['inputs'] != inputs or state['layout'] != _division_layout(root):
            return None
        for relative_path, digest in state['files'].items():
            if hashlib.sha256((root / relative_path).read_bytes()).hexdigest() != digest:
                return None
        return state['summary']
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def process_division_leafsets(tob_tree_str, SIZE=12, output_dir="division_output",
                               k=15, seed=0, outgroup_leaves=("OUT",)):
    """Generate a complete division layout, preserving matching inference runs.

    Changed inputs are generated in a fresh directory and published only after
    successful completion. The previous directory is archived beside it, so
    obsolete blob/run directories cannot leak into inference or merging.
    """
    if SIZE < 1 or k < 1:
        raise ValueError('subset size and number of division attempts must be positive')
    output = Path(os.path.abspath(output_dir))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise ValueError(f'division output must not be a symlink: {output}')
    outgroups = list(outgroup_leaves) if outgroup_leaves else []
    inputs = {'format_version': 1,
              'tob_sha256': hashlib.sha256(tob_tree_str.encode()).hexdigest(),
              'size': SIZE, 'k': k, 'seed': seed, 'outgroups': outgroups}
    summary = _matching_division_summary(output, inputs)
    if summary is not None:
        print(f'  Reusing matching divisions in {output}', flush=True)
        return summary

    with tempfile.TemporaryDirectory(prefix=f'.{output.name}.building-',
                                     dir=output.parent) as stage_dir:
        stage = Path(stage_dir)
        summary = _generate_division_leafsets(
            tob_tree_str, SIZE=SIZE, output_dir=str(stage), k=k, seed=seed,
            outgroup_leaves=outgroups)
        files = {str(p.relative_to(stage)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in stage.rglob('*') if p.is_file()}
        (stage / DIVISION_STATE_FILE).write_text(json.dumps({
            'inputs': inputs, 'summary': summary, 'files': files,
            'layout': _division_layout(stage)}, indent=2) + '\n')
        backup = None
        if output.exists():
            backup = Path(tempfile.mkdtemp(prefix=f'.{output.name}.previous-',
                                           dir=output.parent))
            backup.rmdir()
            os.replace(output, backup)
        try:
            os.replace(stage, output)
        except BaseException:
            if backup is not None:
                os.replace(backup, output)
            raise
        if backup is not None:
            print(f'  Previous divisions archived at {backup}', flush=True)
    return summary


def _generate_division_leafsets(tob_tree_str, SIZE=12, output_dir="division_output",
                                k=15, seed=0, outgroup_leaves=("OUT",)):
    """
    v3 process_division: per-blob output layout.

    Layout:
      output_dir/
        non_blob/            (written once)
        blob00/run_000/
        blob00/run_001/      (only if blob needs > 1 allocation)
        ...
        blob01/run_000/
        ...

    Returns dict with 'blobs' (list of per-blob summaries) and 'non_blob' stats.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Phase timing: leafset formation ---
    # No GT network is consumed; this variant produces only per-blob leafsets.
    _t0 = time.perf_counter()
    _ru0 = resource.getrusage(resource.RUSAGE_SELF)
    pending_writes = []  # list of (folder, rows, allocation_results, T_for_lookup, cut_siblings)

    # --- Outgroup handling ---
    # Always physically PRUNE the candidate outgroup leaves from the TOB before
    # division. Lenient — any candidate not in the tree is silently skipped.
    # Single label and multi-label clades go through the same code path.
    outgroup_list = list(outgroup_leaves) if outgroup_leaves else []
    if outgroup_list:
        tob_tree_str, present, missing = _prune_outgroup_from_newick(
            tob_tree_str, outgroup_list)
        if present:
            print(f"Pruned {len(present)} outgroup leaves from TOB: {present}")
        if missing:
            print(f"  (skipped — not in tree: {missing})")

    # --- Setup ---
    tob_clean = clean_extended_newick(tob_tree_str)
    T = newick_to_nx(tob_clean)
    all_tob_leaves = get_leafset(T)
    outgroup_set = frozenset(outgroup_list)

    all_blobs = get_blob_nodes(T)
    all_blobs_set = set(all_blobs)
    print(f"\n{'='*60}\nBlob identification\n{'='*60}")
    print(f"Total blobs: {len(all_blobs)}: {all_blobs}")

    blob_relationships = find_direct_child_blobs_dict(all_blobs, T)
    forbidden = compute_forbidden_edges(T, all_blobs)
    mega_blobs = build_mega_blobs(all_blobs, blob_relationships)

    print(f"\n{'='*60}\nIsolate mega-blobs\n{'='*60}")
    iso_cuts, mega_pieces, outside = isolate_mega_blobs(
        T, all_blobs, real_taxa=all_tob_leaves, outgroup_set=outgroup_set)

    # Record each mega-blob's source_item: where the blob attaches in the
    # surrounding structure. For isolation cut (parent, mega_root) in TOB, the
    # blob's attachment anchor is identified by its siblings in TOB — i.e.,
    # the OTHER descendants of `parent` that remain after the cut. At merge
    # time, we find the MRCA of those sibling leaves in the in-progress
    # assembled network and attach the blob there.
    blob_source_item = {}
    for (parent_node, mega_root) in iso_cuts:
        try:
            parent_leaves = get_leafset(T, parent_node)
            mega_leaves = get_leafset(T, mega_root)
        except Exception:
            continue
        sibling_leaves = parent_leaves - mega_leaves
        M = next(MM for MM in mega_blobs if mega_root in MM)
        name = mega_blob_name(M)
        blob_source_item[name] = {
            'parent': parent_node,
            'sibling_leaves': sorted(list(sibling_leaves)),
        }

    # --- Prune each mega-blob piece to collect deterministic pruned_subtrees
    # and per-blob items ready for allocation. ---
    print(f"\n{'='*60}\nPer-mega-blob prune + items\n{'='*60}")
    pruned_subtrees = {}
    pruned_origin = {}  # (u,v) -> mega_blob name that produced this prune
    pruned_cut_siblings = {}  # (u,v) -> set of sibling leaves at cut time
    blob_specs = []  # list of (name, M, items) sorted by name

    for root, piece in mega_pieces.items():
        M = next(MM for MM in mega_blobs if root in MM)
        name = mega_blob_name(M)

        members_present = M & set(piece.nodes())
        local_pruned, piece_residual, local_cut_siblings = global_prune_pass(
            piece, forbidden, SIZE, list(members_present),
            real_taxa=all_tob_leaves, verbose=True, outgroup_set=outgroup_set
        )
        pruned_subtrees.update(local_pruned)
        pruned_cut_siblings.update(local_cut_siblings)
        for cut_edge in local_pruned:
            pruned_origin[cut_edge] = name

        items = mega_blob_items(piece_residual, M, all_blobs_set)
        split_items = []
        for item_node, item_leaves in items:
            if len(item_leaves) <= SIZE:
                split_items.append((item_node, item_leaves))
                continue
            print(f"    Item {item_node} ({len(item_leaves)}) > SIZE, splitting ...")
            pieces_split = cut_large_group(piece_residual, item_leaves, SIZE)
            for i, (_, p) in enumerate(pieces_split.items()):
                sub_name = f"{item_node}_piece{i}" if i > 0 else item_node
                split_items.append((sub_name, p))
        if not split_items:
            print(f"    WARNING: {name} has no items after pruning, skipping")
            continue
        blob_specs.append((name, M, split_items))

    # --- Outside leaves (for non_blob_set rows) ---
    outside_leaves = set()
    if outside is not None:
        outside_leaves = set(outside.nodes()) & all_tob_leaves

    # --- Build non_blob rows (deterministic) ---
    non_blob_rows = []
    pruned_row_meta = {}
    for (u, v), leafset in pruned_subtrees.items():
        origin = pruned_origin.get((u, v))
        if len(leafset) <= SIZE:
            key = str((u, v))
            non_blob_rows.append((key, set(leafset),
                {'type': 'pruned_subtree', 'blob': 'N/A', 'cut_edge': str((u, v)),
                 'group': None, 'mega_members': None, 'origin_mega': origin}))
            pruned_row_meta[key] = str((u, v))
        else:
            pieces_p = cut_large_group(T, leafset, SIZE)
            for i, (_, piece) in enumerate(pieces_p.items(), start=1):
                key = f"{str((u, v))}_cut{i}"
                non_blob_rows.append((key, set(piece),
                    {'type': 'pruned_subtree', 'blob': 'N/A', 'cut_edge': str((u, v)),
                     'group': None, 'mega_members': None, 'origin_mega': origin}))
                pruned_row_meta[key] = str((u, v))

    if outside_leaves:
        if len(outside_leaves) <= SIZE:
            non_blob_rows.append(('remaining_subnet_1', set(outside_leaves),
                {'type': 'non_blob_set', 'blob': 'N/A', 'cut_edge': None,
                 'group': None, 'mega_members': None}))
        else:
            pieces_o = cut_large_group(T, outside_leaves, SIZE)
            for i, (_, p) in enumerate(pieces_o.items(), start=1):
                non_blob_rows.append((f'remaining_subnet_{i}', set(p),
                    {'type': 'non_blob_set', 'blob': 'N/A', 'cut_edge': None,
                     'group': None, 'mega_members': None}))

    # --- Per-blob allocation loop (k runs with per-blob dedup) ---
    print(f"\n{'='*60}\nPer-blob allocation\n{'='*60}")
    blob_allocations_first_run = {}  # for source_item resolution when writing non_blob
    blob_summaries = []

    # Enumeration cutoff: for small item counts we enumerate all unique valid
    # partitions (guaranteed no duplicates) instead of random-and-reject.
    # N=12 is a practical ceiling; beyond this the enumeration space grows fast.
    ENUM_MAX_ITEMS = 12

    for blob_idx, (name, M, items) in enumerate(blob_specs):
        blob_dir_name = f"blob{blob_idx:02d}"
        blob_dir = os.path.join(output_dir, blob_dir_name)
        os.makedirs(blob_dir, exist_ok=True)

        weights = [len(ls) for _, ls in items]
        total_size = sum(weights)
        N = len(items)

        # Trivial case: everything fits in one group
        if total_size <= SIZE:
            print(f"\n  [{blob_dir_name}] {name}: {N} items, total={total_size} <= SIZE, single run")
            division, item_mapping = heuristic_allocation(
                weights, items, max_size=SIZE, min_size=3, strategy="smallest_first"
            )
            rows = []
            for gid, leafset in division.items():
                info = {'type': 'blob_group', 'blob': name, 'group': f'group_{gid}',
                        'cut_edge': None, 'mega_members': sorted(M),
                        'blob_source_item': blob_source_item.get(name)}
                key = f"{name}_group_{gid}"
                rows.append((key, set(leafset), info))
            alloc_res = {name: {'groups': division, 'item_mappings': item_mapping,
                                'members': sorted(M)}}
            run_dir = os.path.join(blob_dir, "run_000")
            pending_writes.append((run_dir, rows, alloc_res, None, None))
            blob_allocations_first_run.setdefault(name, alloc_res[name])
            blob_summaries.append({'name': name, 'blob_dir': blob_dir_name,
                                   'n_items': N, 'total_size': total_size, 'n_runs': 1,
                                   'mode': 'single'})
            continue

        # Small enough: use v2-style strategy (smallest_first + random+reject) FIRST,
        # then fill remaining slots with enumeration to guarantee v3 >= v2.
        if N <= ENUM_MAX_ITEMS:
            print(f"\n  [{blob_dir_name}] {name}: {N} items, total={total_size}, "
                  f"hybrid v2 strategy + enumeration fill")
            seen_fps = set()
            allocations = []  # list of (division, item_mapping) in intended run order

            # --- Phase 1: v2 strategy (smallest_first + random) ---
            for i in range(k):
                random.seed(seed + i)
                strat = "smallest_first" if i == 0 else "random"
                division, item_mapping = heuristic_allocation(
                    weights, items, max_size=SIZE, min_size=3, strategy=strat
                )
                fp = frozenset(frozenset(v) for v in division.values())
                if fp in seen_fps:
                    continue
                seen_fps.add(fp)
                allocations.append((division, item_mapping))

            n_v2 = len(allocations)
            print(f"    phase 1 (v2 strategy): {n_v2} distinct allocations")

            # --- Phase 2: enumerate remaining space, add any unseen partitions ---
            if len(allocations) < k:
                partitions = enumerate_unique_partitions(
                    weights, max_size=SIZE, min_size=3, max_partitions=None
                )
                if not partitions:
                    partitions = enumerate_unique_partitions(
                        weights, max_size=SIZE, min_size=1, max_partitions=None
                    )
                # Shuffle enumeration deterministically so we don't always pick canonical head
                random.seed(seed)
                random.shuffle(partitions)
                for bins in partitions:
                    if len(allocations) >= k:
                        break
                    division, item_mapping = partition_to_division(bins, items)
                    fp = frozenset(frozenset(v) for v in division.values())
                    if fp in seen_fps:
                        continue
                    seen_fps.add(fp)
                    allocations.append((division, item_mapping))
                n_enum = len(allocations) - n_v2
                print(f"    phase 2 (enumeration fill): +{n_enum} allocations "
                      f"(total enum space: {len(partitions)})")

            accepted = 0
            for division, item_mapping in allocations:
                run_dir = os.path.join(blob_dir, f"run_{accepted:03d}")
                rows = []
                for gid, leafset in division.items():
                    info = {'type': 'blob_group', 'blob': name, 'group': f'group_{gid}',
                            'cut_edge': None, 'mega_members': sorted(M),
                            'blob_source_item': blob_source_item.get(name)}
                    key = f"{name}_group_{gid}"
                    rows.append((key, set(leafset), info))
                alloc_res = {name: {'groups': division, 'item_mappings': item_mapping,
                                    'members': sorted(M)}}
                pending_writes.append((run_dir, rows, alloc_res, None, None))
                blob_allocations_first_run.setdefault(name, alloc_res[name])
                accepted += 1
            print(f"    => {accepted} distinct run(s) (v2:{n_v2} + enum-fill:{accepted-n_v2})")
            blob_summaries.append({'name': name, 'blob_dir': blob_dir_name,
                                   'n_items': N, 'total_size': total_size, 'n_runs': accepted,
                                   'mode': 'hybrid'})
            continue

        # Large case: random + reject with early termination on consecutive duplicates
        print(f"\n  [{blob_dir_name}] {name}: {N} items > {ENUM_MAX_ITEMS}, "
              f"using random+reject with early-stop")
        seen_fingerprints = set()
        accepted = 0
        # Mirror v2 exactly: run k attempts, dedupe by fingerprint. No early-stop.
        for i in range(k):
            run_seed = seed + i
            random.seed(run_seed)
            strategy = "smallest_first" if i == 0 else "random"
            division, item_mapping = heuristic_allocation(
                weights, items, max_size=SIZE, min_size=3, strategy=strategy
            )
            fp = frozenset(frozenset(v) for v in division.values())
            if fp in seen_fingerprints:
                continue
            seen_fingerprints.add(fp)

            run_dir = os.path.join(blob_dir, f"run_{accepted:03d}")
            rows = []
            for gid, leafset in division.items():
                info = {'type': 'blob_group', 'blob': name, 'group': f'group_{gid}',
                        'cut_edge': None, 'mega_members': sorted(M),
                        'blob_source_item': blob_source_item.get(name)}
                key = f"{name}_group_{gid}"
                rows.append((key, set(leafset), info))
            alloc_res = {name: {'groups': division, 'item_mappings': item_mapping,
                                'members': sorted(M)}}
            pending_writes.append((run_dir, rows, alloc_res, None, None))
            blob_allocations_first_run.setdefault(name, alloc_res[name])
            accepted += 1

        print(f"    => {accepted} distinct run(s) from random+reject")
        blob_summaries.append({'name': name, 'blob_dir': blob_dir_name,
                               'n_items': len(items), 'total_size': total_size,
                               'n_runs': accepted})

    # --- Defer non_blob write (source_item lookup uses first-run allocations) ---
    non_blob_dir = os.path.join(output_dir, 'non_blob')
    pending_writes.append((non_blob_dir, non_blob_rows,
                           blob_allocations_first_run, T, pruned_cut_siblings))

    # --- Leafset-formation phase ends here. Snapshot wall + CPU + peak RSS. ---
    _leafset_wall = time.perf_counter() - _t0
    _ru1 = resource.getrusage(resource.RUSAGE_SELF)
    _leafset_user_cpu = _ru1.ru_utime - _ru0.ru_utime
    _leafset_sys_cpu = _ru1.ru_stime - _ru0.ru_stime
    _leafset_cpu = _leafset_user_cpu + _leafset_sys_cpu
    _leafset_max_rss_kb = _ru1.ru_maxrss

    rt_path = os.path.join(output_dir, 'runtimelog.txt')
    mins, secs = divmod(_leafset_wall, 60)
    with open(rt_path, 'w') as _rtf:
        _rtf.write("Phase: leafset_formation (excludes subnetwork extraction)\n")
        _rtf.write(f"Elapsed (wall clock) seconds: {_leafset_wall:.4f}\n")
        _rtf.write(f"Elapsed (wall clock) time (h:mm:ss or m:ss): "
                   f"{int(mins)}:{secs:05.2f}\n")
        _rtf.write(f"User time (seconds): {_leafset_user_cpu:.4f}\n")
        _rtf.write(f"System time (seconds): {_leafset_sys_cpu:.4f}\n")
        _rtf.write(f"CPU time (seconds): {_leafset_cpu:.4f}\n")
        _rtf.write(f"Maximum resident set size (kbytes): {_leafset_max_rss_kb}\n")
    print(f"\n  Leafset phase: wall={_leafset_wall:.3f}s  cpu={_leafset_cpu:.3f}s  "
          f"maxRSS={_leafset_max_rss_kb}KB -> {rt_path}")

    # --- Flush deferred writes (metadata only — no GT network needed). ---
    for folder, rows, alloc_res, T_for_lookup, cut_siblings in pending_writes:
        _write_rows_to_folder(folder, rows,
                              allocation_results=alloc_res,
                              T=T_for_lookup,
                              cut_siblings=cut_siblings)
    print(f"\n  Wrote {len(non_blob_rows)} non_blob rows to {non_blob_dir}")

    # --- TOB-induced non_blob subtrees (consumed by inphynet + full_merger). ---
    tob_subnets = _extract_non_blob_tob_subnets(non_blob_rows, T.copy())
    tob_path = os.path.join(non_blob_dir, _NON_BLOB_TOB_FILE)
    with open(tob_path, 'w') as f:
        for line in tob_subnets:
            f.write(line + '\n')
    print(f"  Wrote {sum(1 for s in tob_subnets if s)} TOB-induced "
          f"non_blob subtrees to {tob_path}")

    # --- Validation (use first run of every blob as representative) ---
    all_covered = set(outside_leaves)
    for (_, ls, _) in non_blob_rows:
        all_covered |= ls
    # Use first run's blob_group leaves per blob
    for name, alloc in blob_allocations_first_run.items():
        for gid, ls in alloc['groups'].items():
            all_covered |= ls
    missing = all_tob_leaves - all_covered
    extra = all_covered - all_tob_leaves
    if missing or extra:
        print(f"  VALIDATION WARNING: missing={len(missing)} extra={len(extra)}")

    print(f"\n{'='*60}\nSummary\n{'='*60}")
    print(f"  Total blobs processed: {len(blob_summaries)}")
    print(f"  Runs per blob:")
    for b in blob_summaries:
        print(f"    {b['blob_dir']} ({b['name']}): {b['n_runs']} run(s)")
    print(f"  non_blob rows: {len(non_blob_rows)}")

    return {'blobs': blob_summaries, 'non_blob': {'n_rows': len(non_blob_rows)}}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate k divisions (v3, leafset-only — no GT network required)')
    parser.add_argument('--tob', type=str, required=True,
                        help='Path to TOB tree newick (rerooted with the outgroup as root).')
    parser.add_argument('--output_dir', type=str, default='dimple_out',
                        help='Where to write the division runs (default: dimple_out).')
    parser.add_argument('--size', type=int, default=12)
    parser.add_argument('--k', type=int, default=15)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--outgroup', type=str, default='OUT',
                        help='Comma-separated outgroup leaf labels to PRUNE from '
                             'the TOB before division. Lenient — any label not in '
                             'the tree is silently skipped. Default: OUT '
                             '(works for simulated datasets). For a clade, list '
                             'all candidates, e.g. WRSL,QWRA,XDLL,PRIQ,UTRE.')
    args = parser.parse_args()

    with open(args.tob, 'r') as f:
        tob_tree_str = f.read().strip()

    outgroup_list = [s.strip() for s in args.outgroup.split(',') if s.strip()]
    print(f"leafset divider: output_dir={args.output_dir}, SIZE={args.size}, "
          f"k={args.k}, seed={args.seed}, outgroup={outgroup_list}\n")

    try:
        summary = process_division_leafsets(
            tob_tree_str,
            SIZE=args.size, output_dir=args.output_dir,
            k=args.k, seed=args.seed, outgroup_leaves=outgroup_list,
        )
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise

    print(f"\nBlob summary:")
    for b in summary['blobs']:
        print(f"  {b['blob_dir']} ({b['name']}): {b['n_runs']} run(s), "
              f"{b['n_items']} items, total_size={b['total_size']}")
