"""
Add reticulations to a merged base tree.

This file has two layers:

1. Core placement algorithm (private + 3 public fns:
   find_reticulations, add_retics_greedily, add_all_reticulations).
   For each input network the major displayed tree was extracted (keeping the
   higher-probability parent per reticulation). The functions below identify
   the dropped (minor) reticulation edges and find all valid placement
   positions in the merged tree.

2. Pipeline entry point at the bottom (add_retics_from_inputs +
   collect_unique_retics): parses inputs_full.txt, groups retics across
   networks by signature, picks representatives, builds a PL scorer, and
   calls add_retics_greedily.

For a reticulation with child subtree C:
  - MAJOR parent (kept): already in the merged tree
  - MINOR parent (dropped): had sibling subtree S in the original network

The minor edge should be added from some edge in the merged tree to the edge
above C. The candidate source edges are constrained: the minor parent's
position in the network tells us which taxa were "above" it, so the new
reticulation edge should originate from an edge whose descendant leaves
include the minor parent's sibling taxa.
"""

import re
import networkx as nx

from dimple.utils.network_util import (
    newick_to_nx,
    build_newick_from_graph,
    strip_branch_lengths,
    contract_degree2_nodes,
    get_leafset,
)
from dimple.merger.merger_util import (
    MPL_score,
    find_reticulations as find_retics_with_sig,
    identify_unique_reticulations,
    parse_inputs_file,
)


def find_reticulations(network, used_major=True):
    """
    Find reticulations in a network and identify kept/dropped parents.

    If used_major=True (default), the major (highest prob) parent was kept
    in the displayed tree, so we add back the minor side.
    If used_major=False, the minor parent was kept, so we add back the major side.

    Returns list of dicts with:
        retic_node, retic_leaves, major_parent, major_prob,
        minor_parent, minor_prob, minor_sibling_leaves,
        kept_sibling_leaves — sibling on the side that was kept in the displayed
                              tree (so we can detect orientation at placement time)
    Where minor_parent/minor_sibling_leaves always refers to the DROPPED side
    (the one we need to add back) assuming used_major is correct.
    """

    retics = []
    for n in network.nodes():
        if network.in_degree(n) <= 1:
            continue

        parents = list(network.predecessors(n))
        retic_leaves = get_leafset(network, n)

        if len(parents) != 2:
            continue

        # Find major (highest prob) and minor (lowest prob) parent
        probs = [(network.edges[p, n].get('prob', 0.5), p) for p in parents]
        probs.sort(reverse=True)
        major_parent_orig = probs[0][1]
        major_prob_orig = probs[0][0]
        minor_parent_orig = probs[1][1]
        minor_prob_orig = probs[1][0]

        # Always compute sibling leaves on BOTH sides (before any swap)
        def sibling_leaves(parent_node):
            siblings = [c for c in network.successors(parent_node) if c != n]
            out = set()
            for s in siblings:
                out |= get_leafset(network, s)
            return out

        major_sibling_leaves_orig = sibling_leaves(major_parent_orig)
        minor_sibling_leaves_orig = sibling_leaves(minor_parent_orig)

        if used_major:
            # Major was kept -> drop minor -> add back minor side
            major_parent = major_parent_orig
            major_prob = major_prob_orig
            minor_parent = minor_parent_orig
            minor_prob = minor_prob_orig
            dropped_sibling_leaves = minor_sibling_leaves_orig
            kept_sibling_leaves = major_sibling_leaves_orig
        else:
            # Minor was kept -> drop major -> add back major side
            # Swap: the "dropped" side is now the major
            major_parent = minor_parent_orig
            major_prob = minor_prob_orig
            minor_parent = major_parent_orig
            minor_prob = major_prob_orig
            dropped_sibling_leaves = major_sibling_leaves_orig
            kept_sibling_leaves = minor_sibling_leaves_orig

        retics.append({
            'retic_node': n,
            'retic_leaves': retic_leaves,
            'major_parent': major_parent,
            'major_prob': major_prob,
            'minor_parent': minor_parent,
            'minor_prob': minor_prob,
            'minor_sibling_leaves': dropped_sibling_leaves,
            'kept_sibling_leaves': kept_sibling_leaves,
        })

    return retics


def find_node_for_leaves(tree, target_leaves):
    """
    Find the smallest node in tree whose descendant leaves contain all target_leaves.
    Returns (node, node_leaves).
    """
    best_node = None
    best_size = float('inf')
    for n in tree.nodes():
        leaves = get_leafset(tree, n)
        if target_leaves <= leaves and len(leaves) < best_size:
            best_size = len(leaves)
            best_node = n
    return best_node


def find_candidate_edges(tree, retic_info, network_taxa, return_exact_flag=False):
    """
    Find all edges in the merged tree where the added (reticulation) edge
    could originate from.

    The reticulation in the subnet has two parents — one whose subtree was
    kept in the base tree's displayed form (kept_sibling_leaves) and one that
    needs to be added back (minor_sibling_leaves).

    Because the base tree comes from DT-selection that picks major or minor
    per retic independently, we detect orientation here: check retic_child's
    current sibling in the base tree against kept_sibling_leaves and
    minor_sibling_leaves — whichever side matches was "kept", so the other
    side is what we add back. Without this, we'd try to add the edge at
    retic_child's own subtree, creating a degenerate retic.

    Constraint: v's leaves restricted to network_taxa must be a subset of
    the target sibling leaves. Maximize coverage.

    Returns exact matches first. If none, returns partial matches sorted
    by coverage. If return_exact_flag is True, returns (candidates, had_exact).
    """
    retic_leaves = retic_info['retic_leaves']
    minor_sibling_leaves = retic_info['minor_sibling_leaves']
    kept_sibling_leaves = retic_info.get('kept_sibling_leaves', set())

    # Find the node in the merged tree corresponding to the retic child
    retic_child = find_node_for_leaves(tree, retic_leaves)
    if retic_child is None:
        return ([], False) if return_exact_flag else []

    # Orientation check: does retic_child's current sibling match kept side
    # (expected) or minor side (base tree is actually the other DT)?
    target_sibling_leaves = minor_sibling_leaves
    parents = list(tree.predecessors(retic_child))
    if parents and kept_sibling_leaves:
        cur_parent = parents[0]
        cur_siblings = [c for c in tree.successors(cur_parent) if c != retic_child]
        cur_sib_leaves = set()
        for s in cur_siblings:
            cur_sib_leaves |= get_leafset(tree, s)
        cur_in_net = cur_sib_leaves & network_taxa
        # Exact match first; fall back to best-overlap side
        if cur_in_net and cur_in_net <= minor_sibling_leaves and not (
                cur_in_net <= kept_sibling_leaves):
            # Base tree kept the MINOR side — swap: add the MAJOR/kept side back
            target_sibling_leaves = kept_sibling_leaves
        elif cur_in_net and cur_in_net <= kept_sibling_leaves and not (
                cur_in_net <= minor_sibling_leaves):
            # Standard orientation — add back minor side
            target_sibling_leaves = minor_sibling_leaves
        elif cur_in_net:
            # Ambiguous: pick the side with higher Jaccard overlap to sibling
            j_kept = (len(cur_in_net & kept_sibling_leaves) /
                      max(len(cur_in_net | kept_sibling_leaves), 1))
            j_minor = (len(cur_in_net & minor_sibling_leaves) /
                       max(len(cur_in_net | minor_sibling_leaves), 1))
            target_sibling_leaves = (minor_sibling_leaves if j_kept >= j_minor
                                     else kept_sibling_leaves)

    # Exclude edges where v is the retic_child or a descendant (would create cycle).
    descendants = nx.descendants(tree, retic_child)
    exclude_v = descendants | {retic_child}

    exact = []
    partial = []  # (coverage_count, (u, v))
    for u, v in tree.edges():
        if u == 'seed' or v == 'seed':
            continue
        if v in exclude_v:
            continue

        v_leaves = get_leafset(tree, v)
        v_leaves_in_network = v_leaves & network_taxa
        # Strictly no extra network taxa
        if not v_leaves_in_network or not (v_leaves_in_network <= target_sibling_leaves):
            continue
        if v_leaves_in_network == target_sibling_leaves:
            exact.append((u, v))
        else:
            partial.append((len(v_leaves_in_network), (u, v)))

    had_exact = bool(exact)
    if exact:
        result = exact
    else:
        # Partial matches sorted by most coverage first
        partial.sort(key=lambda x: x[0], reverse=True)
        result = [edge for _, edge in partial]

    if return_exact_flag:
        return result, had_exact
    return result


def add_reticulation(tree, retic_info, source_edge, retic_counter):
    """
    Add a reticulation edge to the tree.

    Inserts a minor parent node on source_edge (u, v) and a reticulation
    node on the edge above the retic child, then connects them.

    Before:
        u → v               major_parent → retic_child
    After:
        u → minor_p → v     major_parent → retic_node → retic_child
                minor_p → retic_node

    Args:
        tree: nx.DiGraph to modify in place
        retic_info: dict from find_reticulations
        source_edge: (u, v) edge to split for the minor parent
        retic_counter: int for unique naming

    Returns:
        (retic_node_name, minor_parent_name)
    """
    u, v = source_edge
    retic_leaves = retic_info['retic_leaves']
    major_prob = retic_info['major_prob']
    minor_prob = retic_info['minor_prob']

    # Find the retic child node and its current parent (major parent side)
    retic_child = find_node_for_leaves(tree, retic_leaves)
    major_parent = list(tree.predecessors(retic_child))[0]

    # Create new node names
    retic_node = f'#H{retic_counter}'
    minor_parent = f'minor_p_{retic_counter}'

    # 1. Insert minor_parent on edge (u, v)
    #    u → v  becomes  u → minor_p → v
    edge_attrs_uv = dict(tree.edges[u, v])
    tree.remove_edge(u, v)
    tree.add_node(minor_parent)
    tree.add_edge(u, minor_parent, **edge_attrs_uv)
    tree.add_edge(minor_parent, v)

    # 2. Insert retic_node on edge (major_parent, retic_child)
    #    major_parent → retic_child  becomes  major_parent → retic_node → retic_child
    edge_attrs_major = dict(tree.edges[major_parent, retic_child])
    tree.remove_edge(major_parent, retic_child)
    tree.add_node(retic_node)
    tree.add_edge(major_parent, retic_node, prob=major_prob, **{k: val for k, val in edge_attrs_major.items() if k != 'prob'})
    tree.add_edge(retic_node, retic_child)

    # 3. Add minor parent edge: minor_p → retic_node
    tree.add_edge(minor_parent, retic_node, prob=minor_prob)

    return retic_node, minor_parent


def undo_reticulation(tree, retic_node, minor_parent):
    """
    Undo add_reticulation: remove the reticulation node and minor parent,
    restoring the original tree edges.
    """
    # retic_node has 2 parents (major_parent, minor_parent) and 1 child (retic_child)
    retic_parents = list(tree.predecessors(retic_node))
    major_parent = [p for p in retic_parents if p != minor_parent][0]
    retic_child = list(tree.successors(retic_node))[0]

    # minor_parent has 1 parent (u) and 2 children (v, retic_node)
    u = list(tree.predecessors(minor_parent))[0]
    minor_children = list(tree.successors(minor_parent))
    v = [c for c in minor_children if c != retic_node][0]

    # Save edge attrs
    edge_attrs_u_mp = dict(tree.edges[u, minor_parent])
    edge_attrs_mp_rn = dict(tree.edges[major_parent, retic_node])

    # Remove retic_node and minor_parent
    tree.remove_node(retic_node)
    tree.remove_node(minor_parent)

    # Restore original edges
    # u → v (was u → minor_parent → v)
    tree.add_edge(u, v, **edge_attrs_u_mp)
    # major_parent → retic_child (was major_parent → retic_node → retic_child)
    tree.add_edge(major_parent, retic_child, **{k: val for k, val in edge_attrs_mp_rn.items() if k != 'prob'})


def _collect_and_sort_retics(networks, used_major_list):
    """
    Collect all reticulations from each network, then sort in dependency order.

    Dependency: if retic A's child leaves appear in retic B's minor sibling
    leaves, A must be added before B (A creates the cluster B's placement needs).

    Returns sorted list of (net_idx, network_taxa, retic_info).
    """
    all_retics = []
    for net_idx, network in enumerate(networks):
        network_taxa = {n for n in network.nodes() if network.out_degree(n) == 0 and n != 'seed'}
        retics = find_reticulations(network, used_major=used_major_list[net_idx])
        for retic in retics:
            all_retics.append((net_idx, network_taxa, retic))

    n = len(all_retics)
    deps = [set() for _ in range(n)]
    for i in range(n):
        child_i = all_retics[i][2]['retic_leaves']
        for j in range(n):
            if i == j:
                continue
            sibling_j = all_retics[j][2]['minor_sibling_leaves']
            if child_i & sibling_j:
                deps[j].add(i)

    in_deg = [len(d) for d in deps]
    queue = [i for i in range(n) if in_deg[i] == 0]
    order = []
    while queue:
        idx = queue.pop(0)
        order.append(idx)
        for j in range(n):
            if idx in deps[j]:
                deps[j].discard(idx)
                in_deg[j] -= 1
                if in_deg[j] == 0:
                    queue.append(j)
    for i in range(n):
        if i not in order:
            order.append(i)

    return [all_retics[i] for i in order]


def _edge_distance(tree, node_a, node_b, max_dist=3):
    """BFS shortest undirected path length between two nodes, up to max_dist.
    Returns max_dist+1 if not reachable within max_dist."""
    if node_a == node_b:
        return 0
    if node_a not in tree or node_b not in tree:
        return max_dist + 1
    visited = {node_a}
    frontier = [node_a]
    for dist in range(1, max_dist + 1):
        next_frontier = []
        for n in frontier:
            # undirected neighbors: parents + children
            neighbors = list(tree.successors(n)) + list(tree.predecessors(n))
            for nb in neighbors:
                if nb == node_b:
                    return dist
                if nb not in visited:
                    visited.add(nb)
                    next_frontier.append(nb)
        frontier = next_frontier
    return max_dist + 1


def _is_near_existing_retic(tree, retic, best_edge, added_retics, max_edge_dist=2):
    """Check if this retic is similar to an already-added one:
    - retic children overlap (subset or superset)
    - minor parent placement is within max_edge_dist edges"""
    retic_leaves = set(retic['retic_leaves'])
    u, v = best_edge

    for _, existing_rn, existing_info in added_retics:
        existing_leaves = set(existing_info['retic_leaves'])

        # Check child overlap: one is subset/superset of the other,
        # or they share any taxa
        if not (retic_leaves & existing_leaves):
            continue

        # Check placement proximity: is the minor parent edge near the
        # existing retic node?
        dist = _edge_distance(tree, v, existing_rn, max_edge_dist)
        if dist <= max_edge_dist:
            return True, existing_rn, dist

    return False, None, None


def add_retics_greedily(tree, networks, score_fn, used_major_list=None,
                        min_improvement=0.0, redundancy_threshold=0.02):
    """
    Greedily add reticulations. Candidate edges come from subnet-consistent
    placement (see `find_candidate_edges`); PL scoring picks among candidates
    and decides whether to accept. Stricter redundancy_threshold applies when
    a retic is near an already-added one.

    Args:
        tree:           nx.DiGraph (modified in place)
        networks:       list of representative NX network graphs
        score_fn:       callable(G) -> float, lower = better (e.g. PL score)
        used_major_list: list of bools, one per network
        min_improvement: base min rel improvement (default 0.0)
        redundancy_threshold: stricter threshold for near-redundant retics.

    Returns list of (net_idx, retic_node, retic_info) for each added reticulation.
    """
    if used_major_list is None:
        used_major_list = [True] * len(networks)

    existing_ids = [int(n[2:]) for n in tree.nodes()
                    if isinstance(n, str) and re.match(r'^#H\d+$', n)]
    retic_counter = max(existing_ids, default=0) + 1

    all_retics = _collect_and_sort_retics(networks, used_major_list)

    current_score = score_fn(tree)
    print(f"  Greedy retic addition: base score={current_score:.6f}, "
          f"{len(all_retics)} candidates, "
          f"redundancy_threshold={redundancy_threshold}", flush=True)

    added = []
    n_skipped = 0

    for net_idx, network_taxa, retic in all_retics:
        candidates = find_candidate_edges(tree, retic, network_taxa)
        if not candidates:
            print(f"  Network {net_idx+1}, {retic['retic_node']}: "
                  f"no valid candidates — skipping", flush=True)
            n_skipped += 1
            continue

        # Find the candidate placement with lowest score
        best_score = float('inf')
        best_edge = None
        for edge in candidates:
            rn, mp = add_reticulation(tree, retic, edge, retic_counter)
            s = score_fn(tree)
            undo_reticulation(tree, rn, mp)
            if s < best_score:
                best_score = s
                best_edge = edge

        improvement = current_score - best_score
        rel_improvement = improvement / current_score if current_score > 0 else 0

        # Determine threshold: stricter if near-redundant with existing retic
        is_redundant, near_rn, near_dist = _is_near_existing_retic(
            tree, retic, best_edge, added)
        threshold = redundancy_threshold if is_redundant else min_improvement
        redundant_tag = (f" [REDUNDANT near {near_rn}, dist={near_dist}, "
                         f"threshold={threshold}]") if is_redundant else ""

        if rel_improvement >= threshold and best_score < current_score:
            rn, mp = add_reticulation(tree, retic, best_edge, retic_counter)
            u, v = best_edge
            v_leaves = sorted(get_leafset(tree, v))
            print(f"  Network {net_idx+1}, {retic['retic_node']}: added {rn} "
                  f"(score {current_score:.6f} → {best_score:.6f}, "
                  f"rel_improvement={rel_improvement:.6f}){redundant_tag}",
                  flush=True)
            print(f"    Retic child: {sorted(retic['retic_leaves'])}", flush=True)
            print(f"    Minor edge: {mp} → {rn} "
                  f"(from edge {u}→{v}[{','.join(v_leaves[:5])}])", flush=True)
            print(f"    Probs: major={retic['major_prob']:.4f}, "
                  f"minor={retic['minor_prob']:.4f}", flush=True)
            added.append((net_idx, rn, retic))
            retic_counter += 1
            current_score = best_score
        else:
            print(f"  Network {net_idx+1}, {retic['retic_node']}: skipped "
                  f"(best={best_score:.6f}, rel_improvement={rel_improvement:.6f} "
                  f"< threshold={threshold}){redundant_tag}", flush=True)
            n_skipped += 1

    print(f"  Greedy result: {len(added)} added, {n_skipped} skipped, "
          f"final score={current_score:.6f}", flush=True)
    return added


def add_all_reticulations(tree, networks, score_fn=None, used_major_list=None):
    """
    Add back all missing reticulation edges from input networks.

    used_major_list: optional list of bools, one per network. True means
    the major displayed tree was used (add back minor side), False means
    minor was used (add back major side). Defaults to all True.

    Returns list of (net_idx, retic_node, retic_info) for each added reticulation.
    """
    added = []
    existing_ids = [int(n[2:]) for n in tree.nodes() if isinstance(n, str) and re.match(r'^#H\d+$', n)]
    retic_counter = max(existing_ids, default=0) + 1

    if used_major_list is None:
        used_major_list = [True] * len(networks)

    all_retics = _collect_and_sort_retics(networks, used_major_list)

    for net_idx, network_taxa, retic in all_retics:
        # Find candidates on the CURRENT tree
        candidates = find_candidate_edges(tree, retic, network_taxa)

        if not candidates:
            print(f"  WARNING: Network {net_idx+1}, {retic['retic_node']}: "
                  f"no valid placement found (sibling leaves: {sorted(retic['minor_sibling_leaves'])})")
            continue

        if len(candidates) == 1 or score_fn is None:
            # Single candidate or no scoring — just use first
            source_edge = candidates[0]
        else:
            # Try each candidate, score, pick best
            print(f"  Network {net_idx+1}, {retic['retic_node']}: evaluating {len(candidates)} candidates...")
            best_score = float('inf')
            best_edge = None

            for edge in candidates:
                retic_node, minor_parent = add_reticulation(tree, retic, edge, retic_counter)
                score = score_fn(tree)
                undo_reticulation(tree, retic_node, minor_parent)

                u, v = edge
                v_leaves = sorted(get_leafset(tree, v))
                leaf_str = ','.join(v_leaves[:5])
                if len(v_leaves) > 5:
                    leaf_str += f',...({len(v_leaves)} total)'
                print(f"    ({u} → {v}[{leaf_str}]) => score={score}")

                if score is not None and score < best_score:
                    best_score = score
                    best_edge = edge

            source_edge = best_edge if best_edge else candidates[0]
            print(f"    Best: {source_edge[0]} → {source_edge[1]} (score={best_score})")

        # Apply the chosen placement
        retic_node, minor_parent = add_reticulation(tree, retic, source_edge, retic_counter)

        u, v = source_edge
        v_leaves = sorted(get_leafset(tree, v))
        print(f"  Network {net_idx+1}, {retic['retic_node']}: added {retic_node}")
        print(f"    Retic child: {sorted(retic['retic_leaves'])}")
        print(f"    Minor edge: {minor_parent} → {retic_node} (from edge {u}→{v}[{','.join(v_leaves[:5])}])")
        print(f"    Probs: major={retic['major_prob']:.4f}, minor={retic['minor_prob']:.4f}")

        added.append((net_idx, retic_node, retic))
        retic_counter += 1

    return added




# ===========================================================================
# Pipeline entry point
# ---------------------------------------------------------------------------
# Identifies unique reticulations across input networks, then calls the
# greedy retic-addition above to place each unique one onto the base tree.
# ===========================================================================

def collect_unique_retics(network_newicks):
    """
    Find all unique reticulations across a list of network newicks.

    Uses retics_are_same (via identify_unique_reticulations) to group
    duplicates — two reticulations are the same if both parent-side sibling
    leaf sets overlap.

    For each unique reticulation, returns the network newick whose retic_leaves
    set is the largest (most informative placement target).

    Returns:
        list of representative newicks — one per unique retic.
    """
    all_retics = []  # list of (retic_dict_with_sig, newick)
    for nwk in network_newicks:
        if '#H' not in nwk:
            continue
        try:
            G = newick_to_nx(nwk)
            retics = find_retics_with_sig(G)
            for r in retics:
                all_retics.append((r, nwk))
        except Exception as e:
            print(f"  Warning: could not parse network: {e}")

    if not all_retics:
        return []

    retic_dicts = [r for r, _ in all_retics]
    groups = identify_unique_reticulations(retic_dicts)

    print(f"  Found {len(all_retics)} retics across {len(network_newicks)} networks "
          f"→ {len(groups)} unique", flush=True)

    representatives = []
    for group in groups:
        best = max(
            ((r, nwk) for r, nwk in all_retics if r in group),
            key=lambda x: len(x[0]['retic_leaves']),
        )
        best_nwk = best[1]
        representatives.append(best_nwk)
        leaves = sorted(best[0]['retic_leaves'])[:5]
        print(f"    Retic group: {leaves}{'...' if len(best[0]['retic_leaves']) > 5 else ''}",
              flush=True)

    return representatives


def add_retics_from_inputs(base_nwk, inputs_file, gene_trees_file, verbose=True,
                           min_improvement=0.0,
                           redundancy_threshold=0.02):
    """
    Pipeline entry point: identify unique retics from inputs_file, add to
    base_nwk via greedy MPL-scored placement.

    Args:
        base_nwk:        Newick string of the base tree (output of NJMerge)
        inputs_file:     Path to node_XX_inputs_full.txt
        gene_trees_file: Path to gene trees file for MPL scoring
        verbose:         Print progress

    Returns:
        (network_newick, n_retics_added)
    """
    runs = parse_inputs_file(inputs_file)
    all_subnets = [s for rdata in runs.values() for s in rdata['subnets']]
    network_newicks = [s for s in all_subnets if '#H' in s]

    if verbose:
        print(f"  Input: {len(runs)} runs, {len(all_subnets)} subnets, "
              f"{len(network_newicks)} are networks", flush=True)

    if not network_newicks:
        if verbose:
            print("  No networks found — returning base tree unchanged")
        return base_nwk, 0

    if verbose:
        print("  Identifying unique reticulations...", flush=True)
    representative_newicks = collect_unique_retics(network_newicks)

    if not representative_newicks:
        if verbose:
            print("  No reticulations identified")
        return base_nwk, 0

    all_taxa = get_leafset(newick_to_nx(base_nwk)) - {'seed'}
    if verbose:
        print(f"  Building MPL scorer on {len(all_taxa)} taxa...", flush=True)
    retic_scorer = MPL_score(gene_trees_file, all_taxa)

    tree_G = newick_to_nx(base_nwk)
    rep_Gs = []
    used_major_list = []
    for nwk in representative_newicks:
        try:
            rep_Gs.append(newick_to_nx(nwk))
            used_major_list.append(True)
        except Exception as e:
            print(f"  Warning: could not parse representative: {e}")

    if not rep_Gs:
        return base_nwk, 0

    if verbose:
        print(f"  Greedily adding up to {len(rep_Gs)} unique reticulation(s)...", flush=True)
    added = add_retics_greedily(tree_G, rep_Gs, score_fn=retic_scorer,
                                used_major_list=used_major_list,
                                min_improvement=min_improvement,
                                redundancy_threshold=redundancy_threshold)

    if added:
        contract_degree2_nodes(tree_G)
        if verbose:
            print(f"  Successfully added {len(added)} reticulation(s)", flush=True)

    result_nwk = strip_branch_lengths(build_newick_from_graph(tree_G))
    return result_nwk, len(added)
