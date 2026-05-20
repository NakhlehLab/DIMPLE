"""
Division algorithm v2 helpers.

New flow:
  1. Identify all blobs (multifurcating nodes in the TOB, out_degree > 2).
  2. Compute a global forbidden-edge set = direct out-edges of every blob in
     the ORIGINAL TOB. Forbidden edges are never cut, preserving each blob's
     multifurcation.
  3. Global prune pass: iteratively cut the largest candidate subtree (not in
     forbidden set, leafset size in [min_size, SIZE], leaves both sides >=3)
     until every remaining blob has "resolvable size" <= SIZE.
  4. Build mega-blobs: union blobs connected by non-blob, non-leaf internal
     nodes in the residual tree.
  5. Hand each mega-blob to heuristic_allocation (single shot, no recursion).

Fixes Bug A (forbidden-edge set is computed on the original TOB, not the
residual), Bug B/C/D (no re-extraction of sub-subnetwork; no handoff of a
child's leafset entry to its parent).
"""

import os
import csv

import networkx as nx

from dimple.utils.network_util import (
    get_leafset,
    get_blob_nodes,
    contract_degree2_nodes,
    newick_to_nx,
    build_newick_from_graph,
    extract_subnetwork_by_leaves,
    clean_extended_newick,
)


# ---------------------------------------------------------------------------
# Cleanup: remove orphaned internal nodes introduced by cuts
# ---------------------------------------------------------------------------

def prune_orphan_internals(H, real_taxa, outgroup_set=frozenset({'OUT'})):
    """
    Iteratively remove nodes with out_degree==0 that are not real taxa (and not
    the seed/outgroup markers). A cut can leave an internal node childless;
    without cleanup, get_leafset() treats it as a fake leaf.

    `outgroup_set` is a set of leaf labels that act as rooting markers and
    should not be treated as orphans. Defaults to {'OUT'} for backwards
    compatibility with simulated datasets.

    After removing orphans, contracts any degree-2 internals that resulted.
    Repeats until stable.
    """
    changed = True
    while changed:
        changed = False
        to_remove = [n for n in H.nodes()
                     if H.out_degree(n) == 0
                     and n not in real_taxa
                     and n != 'seed' and n not in outgroup_set]
        if to_remove:
            H.remove_nodes_from(to_remove)
            changed = True
    H = contract_degree2_nodes(H)
    return H


# ---------------------------------------------------------------------------
# Blob relationships (reused from v1 verbatim)
# ---------------------------------------------------------------------------

def find_direct_child_blobs_dict(blob_list, tob_tree):
    """Build {blob: {parents, children}} by walking direct edges in the TOB."""
    blob_relationships = {}
    for blob in blob_list:
        blob_relationships[blob] = {'parents': set(), 'children': set()}

    for blob1 in blob_list:
        for blob2 in blob_list:
            if blob1 != blob2 and tob_tree.has_edge(blob1, blob2):
                blob_relationships[blob1]['children'].add(blob2)
                blob_relationships[blob2]['parents'].add(blob1)
    return blob_relationships


# ---------------------------------------------------------------------------
# Forbidden edges (fixes Bug A)
# ---------------------------------------------------------------------------

def compute_forbidden_edges(T, all_blobs):
    """Direct out-edges of every blob in the ORIGINAL TOB. Never cut these."""
    forbidden = set()
    for B in all_blobs:
        for c in T.successors(B):
            forbidden.add((B, c))
    return forbidden


# ---------------------------------------------------------------------------
# Global prune pass
# ---------------------------------------------------------------------------

def _resolvable_size(H, blob):
    """Sum of direct child subtree sizes at a blob in H."""
    if blob not in H:
        return 0
    return sum(len(get_leafset(H, c)) for c in H.successors(blob))


def _candidate_cut_edges(H, forbidden_edges, all_blobs_set=None, min_side=3):
    """
    Return (u, v) edges in H that:
      - are not in forbidden_edges,
      - the source u is NOT a blob (any out-edge of a blob in the residual
        tree still encodes the blob's multifurcation and must never be cut),
      - the subtree below v contains NO blob node (cutting would drop the
        entire blob from the residual tree, losing its allocation),
      - produce a cut where the subtree below v has >= min_side leaves,
      - and the rest of H has >= min_side leaves.
    "Seed" edges are skipped (cutting the root's synthetic edge is meaningless).
    """
    all_leaves = get_leafset(H)
    candidates = []
    for u, v in H.edges():
        if u == 'seed':
            continue
        if (u, v) in forbidden_edges:
            continue
        if all_blobs_set is not None and u in all_blobs_set:
            continue
        # Subtree below v: include v itself + descendants
        descendants_v = nx.descendants(H, v) | {v}
        if all_blobs_set is not None and descendants_v & all_blobs_set:
            # cut would orphan a blob — skip
            continue
        leaves_v = get_leafset(H, v)
        if len(leaves_v) < min_side:
            continue
        if len(all_leaves) - len(leaves_v) < min_side:
            continue
        candidates.append((u, v, leaves_v))
    return candidates


def global_prune_pass(T, forbidden_edges, SIZE, all_blobs, real_taxa=None,
                      verbose=True, outgroup_set=frozenset({'OUT'})):
    """
    Global greedy pruning. Cuts subtrees outside forbidden edges, largest first,
    preferring candidates whose leafset fits within SIZE.

    Stops when every remaining blob's resolvable_size <= SIZE.

    Returns:
        pruned_subtrees: dict[(u, v) -> set(leaves)]   (each <= SIZE; larger
            residual subtrees remain on the residual tree for the caller to
            split via cut_large_group if needed).
        T_residual: nx.DiGraph with cut edges removed and degree-2 nodes
            suppressed. Blob nodes retain all their direct out-edges.
        cut_siblings: dict[(u, v) -> set(leaves)]
            For each cut, the leaves that were siblings AT CUT TIME (i.e.,
            leafset(parent_in_residual_H_just_before_cut) - cut_leaves).
            Use this for source_item.leaves to avoid stale references to
            leaves cut earlier.
    """
    H = T.copy()
    pruned = {}
    cut_siblings = {}
    all_blobs_set = set(all_blobs)
    if real_taxa is None:
        real_taxa = {n for n in H.nodes()
                     if H.out_degree(n) == 0 and n != 'seed' and n not in outgroup_set}
    # Clean up any orphans inherited from the input (e.g. post-isolate pieces)
    H = prune_orphan_internals(H, real_taxa, outgroup_set=outgroup_set)

    while True:
        # Stop condition: all blobs resolvable size <= SIZE
        needs_more = [B for B in all_blobs if B in H and _resolvable_size(H, B) > SIZE]
        if not needs_more:
            if verbose:
                print(f"  All blobs <= SIZE; stopping prune pass.")
            break

        cands = _candidate_cut_edges(H, forbidden_edges, all_blobs_set=all_blobs_set)
        if not cands:
            if verbose:
                print(f"  No more candidate cut edges (blobs still > SIZE: {needs_more}).")
            break

        # Prefer largest leafset that still fits within SIZE. Fall back to
        # largest overall (will be split later by cut_large_group).
        fitting = [c for c in cands if len(c[2]) <= SIZE]
        if fitting:
            best = max(fitting, key=lambda c: len(c[2]))
        else:
            best = max(cands, key=lambda c: len(c[2]))

        u, v, leaves_v = best

        # Before committing, make sure the cut helps SOMEBODY (i.e. it lies
        # under a blob whose resolvable size currently exceeds SIZE). If no
        # blob needs this cut, bail out.
        blob_under_cut = None
        for B in needs_more:
            if B in H and (v == B or nx.has_path(H, B, v) if v in H else False):
                blob_under_cut = B
                break
        if blob_under_cut is None:
            if verbose:
                print(f"  No over-size blob upstream of {(u, v)}; stopping.")
            break

        pruned[(u, v)] = set(leaves_v)
        # Record cut-time sibling leaves: parent's leafset in current H
        # minus the leaves being cut. Sequential cuts mean earlier cuts
        # have already been removed from H, so this is the residual sibling.
        parent_lv = set(get_leafset(H, u)) - {'seed'} - outgroup_set
        cut_siblings[(u, v)] = parent_lv - set(leaves_v)
        if verbose:
            print(f"  Prune {(u, v)}: {len(leaves_v)} leaves (under blob {blob_under_cut})"
                  f", siblings={len(cut_siblings[(u, v)])}lv")

        # Remove edge, drop the orphaned component below v
        H.remove_edge(u, v)
        descendants_v = set(nx.descendants(H, v)) | {v}
        H.remove_nodes_from(descendants_v)
        H = prune_orphan_internals(H, real_taxa)

    return pruned, H, cut_siblings


# ---------------------------------------------------------------------------
# Mega-blob construction
# ---------------------------------------------------------------------------

def build_mega_blobs(all_blobs, blob_relationships):
    """
    Merge blobs into mega-blobs based on direct parent-child relationships in
    blob_relationships. A parent->child->grandchild chain becomes one mega-blob.
    Blobs that are tree-descendants but not in blob-parent-child relation stay
    separate (e.g. node_24 nested inside node_3's subtree via non-blob nodes).

    Returns a list of mega-blobs, each a frozenset of blob node names.
    """
    parent = {B: B for B in all_blobs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for B, rels in blob_relationships.items():
        for child in rels['children']:
            union(B, child)

    groups = {}
    for B in all_blobs:
        r = find(B)
        groups.setdefault(r, set()).add(B)

    return [frozenset(g) for g in groups.values()]


def mega_blob_root(T_residual, M, blob_relationships):
    """
    Earliest member of M in the blob hierarchy (has no blob-parent in M).
    """
    for B in M:
        parents_in_M = blob_relationships[B]['parents'] & M
        if not parents_in_M:
            return B
    return next(iter(M))


def mega_blob_items(T_residual, M, all_blobs_set):
    """
    Walk from mega-blob root down through member blobs only. At non-member
    nodes, emit the node as one item — but SUBTRACT any leaves of non-member
    blobs that sit deeper inside (those non-member blobs have their own
    mega-blob allocation).

    For a non-member node whose subtree contains a non-member blob B', the
    item's leafset = subtree_leaves(node) - subtree_leaves(B') for each such B'.
    If the residual item has no leaves, it's skipped.
    """
    # Find the actual member root in T_residual. For a member root (from
    # blob_relationships), T_residual may have suppressed it if it has out_degree 1
    # after pruning — but blobs have out_degree >=3 (forbidden-edges invariant)
    # so they survive.
    root = None
    # Prefer a member of M that is in T_residual and has no other member of M
    # as ancestor via T_residual paths.
    for B in M:
        if B not in T_residual:
            continue
        has_member_ancestor = False
        for B2 in M:
            if B2 != B and B2 in T_residual and nx.has_path(T_residual, B2, B):
                has_member_ancestor = True
                break
        if not has_member_ancestor:
            root = B
            break
    if root is None:
        # Fallback: first member present in T_residual
        for B in M:
            if B in T_residual:
                root = B
                break
    if root is None:
        return []

    items = []
    other_blobs_in_M_set = all_blobs_set - M   # non-member blobs (may sit inside items)

    def walk(node):
        if node in M:
            # Member blob: recurse into its direct successors
            for c in T_residual.successors(node):
                walk(c)
        else:
            # Non-member: this is one item. Compute its leafset, subtract
            # any non-member blobs that sit inside its subtree (those are
            # allocated in their own mega-blobs).
            leaves = set(get_leafset(T_residual, node))
            for B_other in other_blobs_in_M_set:
                if B_other == node:
                    # This IS a non-member blob; the entire subtree belongs
                    # to B_other's mega-blob, so emit nothing here.
                    leaves = set()
                    break
                if B_other in T_residual and nx.has_path(T_residual, node, B_other):
                    leaves -= set(get_leafset(T_residual, B_other))
            if leaves:
                items.append((node, leaves))

    walk(root)
    return items


def mega_blob_leafset(T_residual, M, all_blobs_set):
    """Union of all items' leafsets (the total leaves this mega-blob covers)."""
    items = mega_blob_items(T_residual, M, all_blobs_set)
    out = set()
    for _, ls in items:
        out |= ls
    return out


def mega_blob_name(M):
    """Stable name for a mega-blob: member names joined by '+' sorted."""
    return "+".join(sorted(M))


# ---------------------------------------------------------------------------
# Isolate mega-blobs: cut every edge (parent, B) where B is a blob and
# parent is NOT a blob. This separates each mega-blob (parent-child blob
# chains stay together) into its own component.
# ---------------------------------------------------------------------------

def isolate_mega_blobs(T, all_blobs, real_taxa=None,
                        outgroup_set=frozenset({'OUT'})):
    """
    For every blob B in the original TOB, if B's predecessor is NOT a blob,
    cut the edge (parent, B). This isolates each mega-blob's subtree
    (with all its descendant leaves) from the rest of the tree.

    Connected parent-child blobs stay together because, for the child blob,
    its predecessor IS a blob (the parent), so we don't cut.

    Returns:
      cuts:        list of (u, v) edges cut
      mega_pieces: dict[blob_root -> nx.DiGraph]
                   one piece per mega-blob, rooted at the topmost member
      outside:     nx.DiGraph (the "rest" of the tree, no blob descendants)
    """
    blobs_set = set(all_blobs)
    H = T.copy()
    cuts = []
    cut_targets = []  # the v in each (u, v) cut — these become roots of mega-blob subtrees

    for B in all_blobs:
        preds = list(H.predecessors(B))
        if not preds:
            continue
        parent = preds[0]
        if parent == 'seed':
            continue
        if parent in blobs_set:
            continue  # parent-child blobs: don't cut
        cuts.append((parent, B))
        cut_targets.append(B)

    for u, v in cuts:
        H.remove_edge(u, v)

    # Clean up orphaned internal nodes left by the cuts.
    if real_taxa is None:
        real_taxa = {n for n in T.nodes()
                     if T.out_degree(n) == 0 and n != 'seed' and n not in outgroup_set}
    H = prune_orphan_internals(H, real_taxa, outgroup_set=outgroup_set)

    # Now H has |cuts|+1 weakly connected components: one per mega-blob root,
    # plus the "outside" containing the rest of the tree.
    mega_pieces = {}
    outside = None
    for comp in nx.weakly_connected_components(H):
        compH = H.subgraph(comp).copy()
        # Find the root of this component (in-degree 0 within compH, excluding seed)
        roots = [n for n in compH.nodes if compH.in_degree(n) == 0]
        # A blob mega-blob root is the cut_target that lives in this component
        is_mega = False
        for B in cut_targets:
            if B in compH.nodes:
                mega_pieces[B] = compH
                is_mega = True
                break
        if not is_mega:
            outside = compH
    return cuts, mega_pieces, outside


# ---------------------------------------------------------------------------
# v1 helpers (used by generate_k_divisions for oversize-subtree
# splitting on already-pruned subtrees with no blobs)
# ---------------------------------------------------------------------------

def final_cut_edges(g, cuttable_edges_list, relaxed=False):
    valid_cuts = []
    if not relaxed:
        cuttable_edges_list = [
            (u, v) for (u, v) in cuttable_edges_list
            if g.out_degree(u) <= 2
        ]

    roots = [n for n in g.nodes if g.in_degree(n) == 0]
    all_leaves = get_leafset(g)

    if len(roots) == 1 and "seed" in str(roots[0]).lower():
        seed_root = roots[0]
        my_root = list(g.successors(seed_root))
        assert len(my_root) == 1, "G must be a rooted tree with seed leading to single child"
        roots.append(my_root[0])
    elif len(roots) == 1:
        my_root = roots[0]
    else:
        raise ValueError("No or multiple root found in graph!")

    edge_leafsizes = {e: len(get_leafset(g, e[1])) for e in cuttable_edges_list}
    sorted_edges = sorted(cuttable_edges_list, key=lambda e: edge_leafsizes[e], reverse=True)

    parent_cut_tracker = set()
    for (u, v) in sorted_edges:
        if u in parent_cut_tracker:
            continue
        leaves_v = get_leafset(g, v)
        other_leaves = all_leaves - leaves_v
        if len(leaves_v) >= 3 and len(other_leaves) >= 3:
            valid_cuts.append((u, v))
            parent_cut_tracker.add(u)

    return valid_cuts


def greedy_recursive_cuts(G, refined_cuttable_edges_list, num_cuts):
    chosen = []
    H = G.copy()
    candidates = list(refined_cuttable_edges_list)

    for _ in range(num_cuts):
        if not candidates:
            break
        leafsets = {e: get_leafset(H, e[1]) for e in candidates}
        best_edge = max(leafsets, key=lambda e: len(leafsets[e]))
        chosen.append(best_edge)
        H.remove_edge(*best_edge)
        new_candidates = []
        for component in nx.weakly_connected_components(H):
            subH = H.subgraph(component).copy()
            comp_edges = [e for e in candidates if e[0] in subH and e[1] in subH]
            refined = final_cut_edges(subH, comp_edges)
            new_candidates.extend(refined)
        candidates = new_candidates

    return chosen


# ---------------------------------------------------------------------------
# TOB-induced subtrees for non_blob rows
#
# The v3 divider writes <div>/non_blob/tob_subnets.txt — one newick per row in
# subnet_idx order, where each newick is the iqtree TOB restricted to that row's
# `all_leaves` (with degree-2 nodes contracted). The merger reads this file
# instead of re-deriving it from metadata + TOB.
#
# Producer: generate_k_divisions.py calls extract_from_tob_graph(...)
# Consumer: merger_full_pip.py calls load_or_build(...)
# ---------------------------------------------------------------------------

NON_BLOB_TOB_FILE_NAME = 'tob_subnets.txt'


def _extract_tob_subnet(T, leafset):
    return build_newick_from_graph(
        contract_degree2_nodes(extract_subnetwork_by_leaves(T, leafset)))


def extract_from_tob_graph(non_blob_rows, T):
    """Return list of newicks in non_blob_rows order; '' for empty leafsets.
    `T` may be mutated; caller should pass a copy if reuse is needed.
    Strips OUT if present so it never leaks into extracted subtrees."""
    if 'OUT' in T.nodes:
        T.remove_node('OUT')
    out = []
    for (_key, leafset, _info) in non_blob_rows:
        out.append(_extract_tob_subnet(T, leafset) if leafset else '')
    return out


def load_or_build_tob_subnets(div_dir, tob_path=None):
    """Return list of TOB-induced non_blob newicks (one per metadata row, in
    subnet_idx order). Tries <div_dir>/non_blob/tob_subnets.txt first; if
    missing, builds from <div_dir>/non_blob/subnetworks_output_metadata.csv
    and tob_path, then writes the file for next time. Returns [] if neither
    source is available."""
    nb = os.path.join(div_dir, 'non_blob')
    fp = os.path.join(nb, NON_BLOB_TOB_FILE_NAME)
    if os.path.exists(fp):
        with open(fp) as f:
            return [ln.rstrip('\n') for ln in f]
    meta = os.path.join(nb, 'subnetworks_output_metadata.csv')
    if not os.path.exists(meta) or not tob_path or not os.path.exists(tob_path):
        return []
    T = newick_to_nx(clean_extended_newick(open(tob_path).read().strip()))
    if 'OUT' in T.nodes:
        T.remove_node('OUT')
    with open(meta) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    max_idx = max(int(r['subnet_idx']) for r in rows)
    out = [''] * (max_idx + 1)
    for r in rows:
        leaves = set(r['all_leaves'].split(',')) if r.get('all_leaves') else set()
        if not leaves:
            continue
        out[int(r['subnet_idx'])] = _extract_tob_subnet(T, leaves)
    try:
        os.makedirs(nb, exist_ok=True)
        with open(fp, 'w') as f:
            for line in out:
                f.write(line + '\n')
    except Exception:
        pass
    return out
