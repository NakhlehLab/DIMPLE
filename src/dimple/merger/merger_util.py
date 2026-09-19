"""Consolidated utilities needed by blob_merger, merger_full_pip, and the
overlapNJ / combine-blob modules.

"""

import pickle
from itertools import combinations
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx
import dendropy
from dendropy import TreeList

from dimple.utils.network_util import (
    get_leafset, enumerate_displayed_trees, clean_extended_newick,
)


# ---------------------------------------------------------------------------
# Shared helper used by add_pruned_subtree.py and merge_to_tob.py
# (was duplicated identically in both files in DIMPLE2)
# ---------------------------------------------------------------------------

def smallest_containing(T, leaves):
    """Return the smallest node in T whose leafset contains `leaves`."""
    best, best_sz = None, float('inf')
    for n in T.nodes():
        if n == 'seed':
            continue
        nl = get_leafset(T, n)
        if leaves <= nl and len(nl) < best_sz:
            best_sz, best = len(nl), n
    return best


# ---------------------------------------------------------------------------
# From dimple.merger.nonoverlap_merger.merger_blob_util
# Used by: add_pruned_subtree.py, merge_to_tob.py
# ---------------------------------------------------------------------------

def find_root(g):
    """Find the root node (in-degree 0) of a rooted network."""
    G_roots = [n for n in g.nodes if g.in_degree(n) == 0]
    assert len(G_roots) == 1, "G must be a rooted tree"
    return G_roots[0]


def copy_with_unique_names(subgraph, target_graph, root, used_names):
    """
    Copy a subgraph with renamed nodes to avoid conflicts with target graph.
    Preserves leaf names and renames internal nodes with 'm_' prefix.
    """
    used_names = used_names.copy()
    graft = subgraph.copy()
    mapping = {}

    for node in graft.nodes:
        # Leaves (out_degree == 0) are real taxon labels — keep them as-is.
        if graft.out_degree(node) == 0 and not str(node).startswith("#H"):
            new_name = node
            used_names.add(new_name)
        elif node.startswith("#H"):
            new_name = node
            i = 1
            while (new_name in used_names or
                   new_name in target_graph.nodes or
                   new_name in graft.nodes):
                i += 1
                new_name = f"#H{i}"
            used_names.add(new_name)
        else:
            base = node
            safe_base = base.replace("#", "r") if "#" in base else base
            counter = 1
            new_name = f"m_{safe_base}_{counter}"

            while (new_name in used_names or
                   new_name in target_graph.nodes or
                   new_name in mapping.values()):
                counter += 1
                new_name = f"m_{safe_base}_{counter}"
            used_names.add(new_name)

        mapping[node] = new_name

    graft = nx.relabel_nodes(graft, mapping, copy=True)
    new_root_name = mapping[root]
    used_names.update(graft.nodes)

    return graft, new_root_name, used_names


# ---------------------------------------------------------------------------
# From dimple.merger.nonoverlap_merger.nonoverlap_merger
# Used by: blob_merger.py, add_retics_from_inputs.py
# ---------------------------------------------------------------------------

def parse_inputs_file(filepath):
    """
    Parse a node_XX_inputs.txt file into runs.

    Returns:
        dict: {run_name: {'reticulations': int, 'subnets': [newick_str, ...]}}
    """
    runs = {}
    current_run = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                parts = line.lstrip('#').strip()
                run_name = None
                retics = 0
                for token in parts.split(','):
                    token = token.strip()
                    if token.startswith('run='):
                        run_name = token.split('=', 1)[1].strip()
                    elif token.startswith('reticulations='):
                        retics = int(token.split('=', 1)[1].strip())
                if run_name:
                    current_run = run_name
                    runs[current_run] = {'reticulations': retics, 'subnets': []}
            else:
                if current_run is not None:
                    runs[current_run]['subnets'].append(line)

    return runs


# ---------------------------------------------------------------------------
# From dimple.select_networks.identify_reticulations
# Used by: blob_merger.py (find_reticulations, retics_are_same),
#          add_retics_from_inputs.py (find_reticulations, identify_unique_reticulations)
# ---------------------------------------------------------------------------

def get_direct_branches(G, node):
    """
    Get the direct child branches of a node, each represented by its full leaf set.
    Returns list of frozensets.
    """
    if G.out_degree(node) == 0:
        return [frozenset({node})]
    branches = []
    for child in G.successors(node):
        leaves = get_leafset(G, child)
        if leaves:
            branches.append(frozenset(leaves))
    return branches


def _branches_overlap(branches_a, branches_b):
    """Check if any branch from A shares any leaf with any branch from B."""
    for ba in branches_a:
        for bb in branches_b:
            if ba & bb:
                return True
    return False


def retic_identity_signature(G, retic_node):
    """
    Compute the identity signature of a reticulation node in graph G.

    Returns:
        (child_branches, parent_branches) where each is a list of
        lists of frozensets (leaf sets of direct child branches).
    """
    children = list(G.successors(retic_node))
    if children:
        child_branches = get_direct_branches(G, children[0])
    else:
        child_branches = [frozenset({retic_node})]

    parents = list(G.predecessors(retic_node))
    parent_branches = []
    for p in parents:
        siblings = [c for c in G.successors(p) if c != retic_node]
        if siblings:
            branches = []
            for s in siblings:
                branches.extend(get_direct_branches(G, s))
            parent_branches.append(branches)
        else:
            parent_branches.append([])

    return child_branches, parent_branches


def find_reticulations(G):
    """
    Find all reticulations in a network.

    Returns list of dicts:
        retic_node: node name
        retic_leaves: frozenset of leaves below the reticulation
        parent_leaves: list of frozensets, one per parent (sibling leaves on that side)
        signature: (child_branches, parent_branches) for matching
    """
    retics = []
    for n in G.nodes():
        if G.in_degree(n) <= 1:
            continue

        retic_leaves = frozenset(get_leafset(G, n))
        parents = list(G.predecessors(n))

        if len(parents) != 2:
            continue

        # Compute parent sibling leaves
        parent_leaves_list = []
        for p in parents:
            siblings = [c for c in G.successors(p) if c != n]
            sibling_leaves = set()
            for s in siblings:
                sibling_leaves |= get_leafset(G, s)
            parent_leaves_list.append(frozenset(sibling_leaves))

        retics.append({
            'retic_node': n,
            'retic_leaves': retic_leaves,
            'parent_leaves': parent_leaves_list,
            'signature': retic_identity_signature(G, n),
        })

    return retics


def retics_are_same(sig_a, sig_b):
    """
    Two reticulations are the same if both parents match (order-independent,
    branch overlap). Each parent's sibling branches must share at least one
    leaf with the corresponding parent on the other side.
    """
    if isinstance(sig_a, dict):
        sig_a = sig_a['signature']
    if isinstance(sig_b, dict):
        sig_b = sig_b['signature']

    _, parents_a = sig_a
    _, parents_b = sig_b

    if len(parents_a) != 2 or len(parents_b) != 2:
        return False

    straight = (_branches_overlap(parents_a[0], parents_b[0]) and
                _branches_overlap(parents_a[1], parents_b[1]))
    crossed = (_branches_overlap(parents_a[0], parents_b[1]) and
               _branches_overlap(parents_a[1], parents_b[0]))
    return straight or crossed


def identify_unique_reticulations(retics_list):
    """
    Given a flat list of reticulation dicts (possibly from multiple networks),
    group them into unique reticulations by signature matching.

    Returns list of groups, where each group is a list of retic dicts
    that are considered the same reticulation.
    """
    groups = []
    for retic in retics_list:
        matched = None
        for i, group in enumerate(groups):
            if retics_are_same(retic['signature'], group[0]['signature']):
                matched = i
                break
        if matched is not None:
            groups[matched].append(retic)
        else:
            groups.append([retic])
    return groups


# ---------------------------------------------------------------------------
# From dimple.merger.overlapmerge.pl_scorer
# Pseudolikelihood scorer: precomputes gene-tree triple frequencies once,
# then scores each candidate tree by counting how many observed triples it
# displays. For a tree without branch lengths, each triple has probability
# 1 if displayed or 0 if not. Score = fraction NOT displayed (lower = better).
#
# Used by: select_displayed_trees.py, add_retics_from_inputs.py
# ---------------------------------------------------------------------------

def _extract_triples_from_dendropy(tree_str, leaf_set):
    """Extract rooted triples (a,b|c) from a newick string, pruned to leaf_set."""
    tree = dendropy.Tree.get(data=tree_str, schema='newick')
    tree.retain_taxa_with_labels(leaf_set)
    tree.suppress_unifurcations()

    triples = set()
    all_leaves = set(l.taxon.label for l in tree.leaf_node_iter())

    for node in tree.preorder_node_iter():
        if node.is_leaf() or node.parent_node is None:
            continue
        desc = set(l.taxon.label for l in node.leaf_iter())
        if not (1 < len(desc) < len(all_leaves)):
            continue
        outside = all_leaves - desc
        for a, b in combinations(sorted(desc), 2):
            for c in outside:
                triples.add((a, b, c))

    return triples


def _extract_triples_from_nx(tree):
    """Extract rooted triples (a,b|c) from an nx.DiGraph tree."""
    all_leaves = set(n for n in tree.nodes if tree.out_degree(n) == 0 and n != 'seed')
    triples = set()

    desc_cache = {}
    for node in tree.nodes:
        if tree.out_degree(node) == 0 or node == 'seed':
            continue
        if node not in desc_cache:
            desc = set()
            for n in nx.descendants(tree, node):
                if tree.out_degree(n) == 0 and n != 'seed':
                    desc.add(n)
            desc_cache[node] = desc
        else:
            desc = desc_cache[node]

        if not (1 < len(desc) < len(all_leaves)):
            continue
        outside = all_leaves - desc
        for a, b in combinations(sorted(desc), 2):
            for c in outside:
                triples.add((a, b, c))

    return triples


def precompute_gene_tree_triples(gene_tree_file, leaf_set):
    """
    Parse gene trees and count observed triple frequencies.

    Returns Counter: (a, b, c) -> count
    where (a, b, c) means a,b cluster together excluding c.
    """
    with open(gene_tree_file) as f:
        raw_trees = [line.strip() for line in f if line.strip()]

    print(f"  Precomputing triples from {len(raw_trees)} gene trees...")
    counts = Counter()

    for i, gt in enumerate(raw_trees):
        gt_triples = _extract_triples_from_dendropy(gt, leaf_set)
        counts.update(gt_triples)
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(raw_trees)} gene trees processed")

    print(f"  {len(counts)} unique triples observed")
    return counts


def load_triple_cache(cache_path, leaf_set=None):
    """
    Load precomputed triple counts from a pickle file.
    If leaf_set is provided, filter to only triples where all 3 taxa are in leaf_set.
    """
    with open(cache_path, 'rb') as f:
        raw = pickle.load(f)

    if leaf_set is None:
        counts = Counter(raw)
        print(f"  Loaded triple cache: {len(counts)} triples")
        return counts

    counts = Counter()
    for (a, b, c), count in raw.items():
        if a in leaf_set and b in leaf_set and c in leaf_set:
            counts[(a, b, c)] = count
    print(f"  Loaded triple cache: {len(counts)} triples "
          f"(filtered from {len(raw)} to {len(leaf_set)} taxa)")
    return counts




# ---------------------------------------------------------------------------
# From dimple.merger.overlapmerge.PL_score_real
# Real pseudolikelihood scorer using phynetpy's MPL (multispecies network
# coalescent). Score = negated log pseudo-likelihood. Lower = better.
#
# Used by: add_retics_from_inputs.py
# ---------------------------------------------------------------------------

from phynetpy.MPL import MPL
from phynetpy.IO import read_newick_file, convert_newick
from phynetpy.Network import Network

from dimple.utils.network_util import build_newick_from_graph


DEFAULT_BRANCH_LENGTH = 1.0


def _nx_to_phynetpy(G, default_branch_length=DEFAULT_BRANCH_LENGTH):
    """
    Convert a NetworkX DiGraph (DIMPLE format) to a phynetpy Network.

    Uses build_newick_from_graph (PhyloNet convention with ::gamma),
    then convert_newick to PhyNetPy convention ([&gamma=...]),
    then Network.from_newick to parse.

    If any edge has length=None, ALL edge lengths are set to
    default_branch_length before conversion.
    """
    has_missing = any(G[u][v].get('length') is None for u, v in G.edges())
    if has_missing:
        G = G.copy()
        for u, v in G.edges():
            G[u][v]['length'] = default_branch_length

    phylonet_nwk = build_newick_from_graph(G)
    phynetpy_nwk = convert_newick(phylonet_nwk, standard="PhyNetPy")
    return Network.from_newick(phynetpy_nwk)


def precompute_gt_triplets(gene_tree_file, taxa):
    """
    Precompute gene tree triplet frequencies using phynetpy's MPL.

    Returns (gt_triplets, mapping).
    """
    taxa = sorted(set(taxa))
    mapping = {t: [t] for t in taxa}

    gts = read_newick_file(gene_tree_file, return_type="genetrees",
                           species_gene_mapping=mapping)

    print(f"  Computing MPL triplets from gene trees for {len(taxa)} taxa...")
    gt_triplets = MPL.compute_gene_tree_triplets(gts, mapping,
                                                  species_labels=taxa)
    print(f"  Computed rho for {len(gt_triplets.triplets)} species triplets")

    return gt_triplets, mapping


def MPL_score(gene_tree_file, leaf_set, gt_triplets=None,
                        default_branch_length=DEFAULT_BRANCH_LENGTH):
    """
    Create a real pseudolikelihood scoring function using phynetpy's MPL.

    The score is the negated log pseudo-likelihood under the multispecies
    network coalescent model. Lower = better.

    If gt_triplets is provided (from precompute_gt_triplets), skips the
    expensive triplet computation.
    """
    if gt_triplets is None:
        gt_triplets, mapping = precompute_gt_triplets(gene_tree_file, leaf_set)
    else:
        mapping = {t: [t] for t in sorted(set(leaf_set))}

    def score_fn(G):
        try:
            net = _nx_to_phynetpy(G, default_branch_length)
            result = MPL.score_species_network_triplets(net, gt_triplets)
            return -result.log_pseudo_likelihood
        except Exception:
            return float('inf')

    score_fn.gt_triplets = gt_triplets
    score_fn.mapping = mapping
    return score_fn


# ---------------------------------------------------------------------------
# General newick / gene-tree helpers (used by blob_merger and overlap_njmerge)
# ---------------------------------------------------------------------------

def get_taxa(newick_str):
    """Extract leaf labels from a newick string (works on extended newick)."""
    t = dendropy.Tree.get(data=clean_extended_newick(newick_str), schema='newick',
                          preserve_underscores=True)
    return set(l.taxon.label for l in t.leaf_nodes())


def tree_to_newick(t):
    """Serialize a dendropy.Tree to a newick string (strips the dendropy prefix)."""
    s = t.as_string(schema='newick').strip()
    i = s.find('(')
    return s[i:] if i >= 0 else s


def compute_dm(gene_trees_file, taxa_set):
    """Average pairwise distance matrix across a file of gene trees.

    Returns a pandas DataFrame indexed by taxon labels (sorted), with cell
    [i, j] = mean distance(i, j) across all gene trees where both are present.
    """
    shared_ns = dendropy.TaxonNamespace()
    trees = TreeList.get(data=open(gene_trees_file).read(), schema='newick',
                         taxon_namespace=shared_ns, rooting='default-rooted',
                         preserve_underscores=True)
    labels = sorted(t for t in taxa_set if shared_ns.get_taxon(t))
    tmap = {t.label: t for t in shared_ns}
    n = len(labels)
    S = np.zeros((n, n))
    cnt = 0
    for tree in trees:
        tt = set(l.taxon.label for l in tree.leaf_nodes())
        pdm = tree.phylogenetic_distance_matrix()
        for i, a in enumerate(labels):
            if a not in tt:
                continue
            for j, b in enumerate(labels):
                if b not in tt or i == j:
                    continue
                S[i, j] += pdm(tmap[a], tmap[b])
        cnt += 1
    return pd.DataFrame(S / max(cnt, 1), index=labels, columns=labels)


def compute_dm_full(gene_trees_file, verbose=False):
    """Average gene-tree distance matrix over EVERY taxon in `gene_trees_file`.

    Identical values to `compute_dm` on any sub-block: entry (a, b) is the sum
    of d_tree(a, b) over all gene trees divided by the TOTAL tree count, which
    depends only on the pair (a, b) -- never on which other taxa happen to be
    in the label set. So `compute_dm_full(f).loc[sub, sub]` equals
    `compute_dm(f, sub)` exactly; slicing is not an approximation.

    Computing this once per dataset instead of once per blob removes the
    merger's dominant cost. `compute_dm` builds a full N x N
    phylogenetic_distance_matrix() for every gene tree -- O(G * N^2) -- and
    that work was previously repeated for each blob even though every blob
    reads a sub-block of the same matrix.
    """
    shared_ns = dendropy.TaxonNamespace()
    trees = TreeList.get(data=open(gene_trees_file).read(), schema='newick',
                         taxon_namespace=shared_ns, rooting='default-rooted',
                         preserve_underscores=True)
    labels = sorted(t.label for t in shared_ns)
    idx = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    S = np.zeros((n, n))
    cnt = 0
    for tree in trees:
        # Iterate the taxa actually present in this tree rather than testing
        # every label for membership -- same result, far fewer lookups.
        present = [l.taxon for l in tree.leaf_nodes() if l.taxon is not None]
        pdm = tree.phylogenetic_distance_matrix()
        for a in present:
            i = idx.get(a.label)
            if i is None:
                continue
            for b in present:
                j = idx.get(b.label)
                if j is None or i == j:
                    continue
                S[i, j] += pdm(a, b)
        cnt += 1
    if verbose:
        print(f'  Shared DM: {n} taxa from {cnt} gene trees', flush=True)
    return pd.DataFrame(S / max(cnt, 1), index=labels, columns=labels)

