"""
NJMerge for overlapping constraint trees.

Two layers:

1. Core in-place NJMerge algorithm (mirrors InPhyNet.jl): for each constraint
   tree we maintain a leaf-pair sibling set. Validity for joining (a, b)
   reduces to checking sibling membership in each tree where both leaves
   are present. No deepcopy in the inner loop — constraint trees are mutated
   in place.

   Public API:
     matrix_to_dendropy_pdm(dmat, taxa)            -> dendropy PDM
     are_two_trees_incompatible(tree1, tree2)      -> bool
     merge_trees_via_nj(pdm, trees, verbose=False) -> dendropy.Tree

2. High-level wrappers used by blob_merger:
     run_overlap_njmerge(newick_list, dm)  -> merged newick string
     select_compatible_subset(newicks)     -> max-clique compatible subset

Based on NJMerge from:
    Molloy, E.K., Warnow, T. (2018). NJMerge: A generic technique for
    scaling phylogeny estimation methods and its application to
    species trees.
"""

import os
import sys
from copy import deepcopy

import numpy as np
import dendropy
from dendropy.calculate.treecompare import false_positives_and_negatives
import networkx as nx

from dimple.utils.network_util import clean_extended_newick
from dimple.merger.merger_util import tree_to_newick

sys.setrecursionlimit(200000)


# =============================================================================
# Utility functions (kept from original overlap_njmerge.py)
# =============================================================================

def matrix_to_dendropy_pdm(dmat, taxa):
    """Convert numpy distance matrix to dendropy PhylogeneticDistanceMatrix."""
    pdm = dendropy.PhylogeneticDistanceMatrix()
    pdm.taxon_namespace = dendropy.TaxonNamespace()
    pdm._mapped_taxa = set()

    for i, si in enumerate(taxa):
        for j, sj in enumerate(taxa):
            dij = dmat[i, j]

            xi = pdm.taxon_namespace.get_taxon(si)
            if not xi:
                xi = dendropy.Taxon(si)
                pdm.taxon_namespace.add_taxon(xi)
                pdm._mapped_taxa.add(xi)
                pdm._taxon_phylogenetic_distances[xi] = {}

            xj = pdm.taxon_namespace.get_taxon(sj)
            if not xj:
                xj = dendropy.Taxon(sj)
                pdm.taxon_namespace.add_taxon(xj)
                pdm._mapped_taxa.add(xj)
                pdm._taxon_phylogenetic_distances[xj] = {}

            pdm._taxon_phylogenetic_distances[xi][xj] = float(dij)
    return pdm


def get_leafset_dendropy(subtree):
    """Return set of leaf labels from a dendropy.Tree (cf. utils.get_leafset
    which operates on an nx.DiGraph)."""
    return set([l.taxon.label for l in subtree.leaf_nodes()])


def are_two_trees_incompatible(tree1, tree2):
    """Check if two unrooted trees are incompatible on their shared taxon set."""
    leaves1 = get_leafset_dendropy(tree1)
    leaves2 = get_leafset_dendropy(tree2)
    shared = list(leaves1.intersection(leaves2))

    taxa = dendropy.TaxonNamespace(shared)

    if len(shared) < 4:
        return False

    tree1.retain_taxa_with_labels(shared)
    tree1.migrate_taxon_namespace(taxa)
    tree1.is_rooted = False
    tree1.collapse_basal_bifurcation()
    tree1.update_bipartitions()

    tree2.retain_taxa_with_labels(shared)
    tree2.migrate_taxon_namespace(taxa)
    tree2.is_rooted = False
    tree2.collapse_basal_bifurcation()
    tree2.update_bipartitions()

    [fp, fn] = false_positives_and_negatives(tree1, tree2)
    return fp > 0 or fn > 0


# =============================================================================
# Sibling-pair maintenance (in-place strategy, no deepcopy in inner loop)
# =============================================================================

def _leaf_label_set(tree):
    return {leaf.taxon.label for leaf in tree.leaf_node_iter()
            if leaf.taxon is not None}


def _is_real_leaf(node):
    return node.is_leaf() and node.taxon is not None


def _unrooted_adjacency(tree):
    """Undirected adjacency of `tree` with every internal node of degree <= 2
    spliced out.

    Neighbor-joining is unrooted and the constraint trees carry no meaningful
    root: the stored ones are written with a unifurcating seed node, and
    `_contract_siblings` leaves the seed node with one or two children
    depending on the order its two labels were passed in. Dropping the
    parent/child direction makes all of that irrelevant -- the seed node is
    suppressed like any other degree-2 node, so eligibility depends only on
    the topology.

    Returns (adj, leaf_labels), adj mapping node -> set of neighbour nodes.
    """
    adj = {}
    leaf_labels = {}
    for node in tree.preorder_node_iter():
        adj.setdefault(node, set())
        if _is_real_leaf(node):
            leaf_labels[node] = node.taxon.label
        for c in node.child_node_iter():
            adj.setdefault(c, set())
            adj[node].add(c)
            adj[c].add(node)

    stack = [n for n in adj if n not in leaf_labels]
    while stack:
        node = stack.pop()
        if node not in adj or node in leaf_labels:
            continue
        nb = adj[node]
        if len(nb) == 2:
            u, v = tuple(nb)
            adj[u].discard(node)
            adj[v].discard(node)
            adj[u].add(v)
            adj[v].add(u)
            del adj[node]
            stack.extend(x for x in (u, v) if x not in leaf_labels)
        elif len(nb) == 1:
            (u,) = tuple(nb)
            adj[u].discard(node)
            del adj[node]
            if u not in leaf_labels:
                stack.append(u)
        elif len(nb) == 0:
            del adj[node]
    return adj, leaf_labels


def _build_sibling_pairs(tree, force_unrooted=True):
    """Return frozenset({a_label, b_label}) for every cherry of `tree`.

    A cherry is two leaves sharing a neighbour in the *unrooted* topology,
    which is the relation neighbor-joining can act on. force_unrooted=False
    keeps the strictly rooted reading (siblings share an immediate parent);
    nothing in the pipeline uses it.
    """
    pairs = set()
    if not force_unrooted:
        for inner in tree.preorder_node_iter():
            leaf_kids = [c for c in inner.child_node_iter() if _is_real_leaf(c)]
            for i in range(len(leaf_kids)):
                for j in range(i + 1, len(leaf_kids)):
                    pairs.add(frozenset((leaf_kids[i].taxon.label,
                                         leaf_kids[j].taxon.label)))
        return pairs

    if tree.seed_node is None:
        return pairs

    adj, leaf_labels = _unrooted_adjacency(tree)

    # Two labels left: suppression collapses the tree to a single edge between
    # them. The leaf branch below already emits this pair, but a constraint can
    # shrink to two labels while the overall merge still has many components,
    # and joining them must stay permitted -- so make the invariant explicit.
    if len(leaf_labels) == 2:
        return {frozenset(leaf_labels.values())}

    for node, nb in adj.items():
        near = [leaf_labels[x] for x in nb if x in leaf_labels]
        if node in leaf_labels:
            # Only on a two-taxon tree, where the two leaves become directly
            # adjacent once the node between them is suppressed.
            for lab in near:
                pairs.add(frozenset((leaf_labels[node], lab)))
        else:
            for i in range(len(near)):
                for j in range(i + 1, len(near)):
                    pairs.add(frozenset((near[i], near[j])))
    return pairs


def _cleanup_degenerate(node):
    """Walk up from `node` and prune empty internals or suppress degree-1
    internals so the tree doesn't accumulate stale structural noise."""
    cur = node
    while cur is not None and cur.parent_node is not None:
        children = list(cur.child_node_iter())
        if len(children) == 0 and cur.taxon is None:
            parent = cur.parent_node
            parent.remove_child(cur)
            cur = parent
            continue
        if len(children) == 1 and cur.taxon is None:
            parent = cur.parent_node
            child = children[0]
            cur.remove_child(child)
            parent.remove_child(cur)
            parent.add_child(child)
            cur = parent
            continue
        break


def _contract_siblings(tree, a_label, b_label, new_label, taxon_namespace):
    """Merge sibling leaves a and b in `tree` into a single leaf `new_label`.
    Pre-condition: a and b share an immediate parent in tree."""
    leaf_a = None
    leaf_b = None
    for leaf in tree.leaf_node_iter():
        if leaf.taxon.label == a_label:
            leaf_a = leaf
        elif leaf.taxon.label == b_label:
            leaf_b = leaf
    if leaf_a is None or leaf_b is None:
        raise ValueError(f"contract: leaf missing in tree (a={a_label}, b={b_label})")

    parent_a = leaf_a.parent_node
    parent_b = leaf_b.parent_node
    if parent_a is parent_b:
        parent = parent_a
        parent.remove_child(leaf_a)
        parent.remove_child(leaf_b)
        new_taxon = taxon_namespace.get_taxon(new_label) or taxon_namespace.new_taxon(new_label)
        new_leaf = tree.node_factory()
        new_leaf.taxon = new_taxon
        parent.add_child(new_leaf)
        _cleanup_degenerate(parent)
    else:
        # Force-unrooted case: a and b are in different children of the root.
        parent_a.remove_child(leaf_a)
        parent_b.remove_child(leaf_b)
        new_taxon = taxon_namespace.get_taxon(new_label) or taxon_namespace.new_taxon(new_label)
        new_leaf = tree.node_factory()
        new_leaf.taxon = new_taxon
        if parent_a.child_nodes():
            parent_a.add_child(new_leaf)
        else:
            parent_b.add_child(new_leaf)
        _cleanup_degenerate(parent_a)
        _cleanup_degenerate(parent_b)


def _rename_leaf(tree, old_label, new_label, taxon_namespace):
    """Rename a single leaf in `tree` from old_label to new_label."""
    for leaf in tree.leaf_node_iter():
        if leaf.taxon.label == old_label:
            new_taxon = taxon_namespace.get_taxon(new_label) or taxon_namespace.new_taxon(new_label)
            leaf.taxon = new_taxon
            break


# =============================================================================
# Main NJMerge driver
# =============================================================================

class _NJState:
    """Holds per-merge state (Q matrix updates handled outside)."""
    def __init__(self, pdm, trees):
        self.taxa = list(pdm.taxon_namespace)
        self.taxa_labels = [t.label for t in self.taxa]
        n = len(self.taxa)
        self.n = n

        D = np.zeros((n, n))
        for i, ti in enumerate(self.taxa):
            for j, tj in enumerate(self.taxa):
                if i == j: continue
                D[i, j] = pdm.distance(ti, tj)
        self.D = D
        self.labels = list(self.taxa_labels)
        self.subnet = {label: label for label in self.labels}
        self.trees = trees
        self.namespace = trees[0].taxon_namespace if trees else dendropy.TaxonNamespace()
        self.tree_leaves = [set(_leaf_label_set(t)) for t in trees]
        self.tree_sibs = [_build_sibling_pairs(t) for t in trees]

    def both_in(self, tree_idx, a, b):
        return a in self.tree_leaves[tree_idx] and b in self.tree_leaves[tree_idx]

    def either_in(self, tree_idx, a, b):
        return a in self.tree_leaves[tree_idx] or b in self.tree_leaves[tree_idx]

    def violates(self, a, b):
        for i, leaves in enumerate(self.tree_leaves):
            if a in leaves and b in leaves:
                if frozenset((a, b)) not in self.tree_sibs[i]:
                    return True
        return False

    def join(self, a, b, new_label):
        """Apply the join (a,b → new_label) to all constraint trees.
        Pre-condition: not self.violates(a, b)."""
        for i, leaves in enumerate(self.tree_leaves):
            in_a = a in leaves; in_b = b in leaves
            if in_a and in_b:
                _contract_siblings(self.trees[i], a, b, new_label, self.namespace)
                leaves.discard(a); leaves.discard(b); leaves.add(new_label)
                self.tree_sibs[i] = _build_sibling_pairs(self.trees[i])
            elif in_a:
                _rename_leaf(self.trees[i], a, new_label, self.namespace)
                leaves.discard(a); leaves.add(new_label)
                self.tree_sibs[i] = {
                    frozenset((new_label if x == a else x for x in s))
                    for s in self.tree_sibs[i]}
            elif in_b:
                _rename_leaf(self.trees[i], b, new_label, self.namespace)
                leaves.discard(b); leaves.add(new_label)
                self.tree_sibs[i] = {
                    frozenset((new_label if x == b else x for x in s))
                    for s in self.tree_sibs[i]}


def _newick_join(left_subnet_nwk, right_subnet_nwk):
    """Combine two subnet newicks into one with an unspecified internal node."""
    return f'({left_subnet_nwk},{right_subnet_nwk})'


class NoValidJoin(RuntimeError):
    """Raised instead of relaxing, when a state sets fail_on_no_join."""


def merge_trees_via_nj(pdm, trees, verbose=False, state_cls=None):
    """Overlap-aware NJ merge of constraint trees.

    Strategy: InPhyNet-style sibling-pair lookup; constraint trees mutated in
    place (no deepcopy in the inner loop). If validity fails for all candidate
    pairs at some step, relax constraints and pick the lowest-Q pair.

    state_cls lets a variant supply its own _NJState (e.g. one that adds a
    between-constraint conflict check). Default behaviour is unchanged.
    """
    state = (state_cls or _NJState)(pdm, trees)
    n = state.n
    n_relax = 0
    join_counter = 0

    while n > 1:
        D = state.D
        x_sub = D.sum(axis=1)
        Q_pairs = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                qv = (n - 2) * D[i, j] - x_sub[i] - x_sub[j]
                Q_pairs.append((qv, i, j))
        Q_pairs.sort()

        chosen = None
        for (qv, i, j) in Q_pairs:
            a = state.labels[i]; b = state.labels[j]
            if not state.violates(a, b):
                chosen = (i, j); break
        if chosen is None:
            if getattr(state, 'fail_on_no_join', False):
                raise NoValidJoin(
                    f'no join satisfies the constraints with {n} taxa left')
            # Relax constraints — pick the lowest-Q pair regardless
            n_relax += 1
            qv, i, j = Q_pairs[0]
            chosen = (i, j)
            # A state may implement relax_for() to retire only the constraints
            # that actually blocked this pair, instead of letting join() force
            # a contraction through every tree. No-op when absent.
            relax_for = getattr(state, 'relax_for', None)
            if relax_for is not None:
                relax_for(state.labels[i], state.labels[j])

        i, j = chosen
        a = state.labels[i]; b = state.labels[j]
        new_label = f'__joined_{join_counter}__'
        join_counter += 1

        left = state.subnet[a]; right = state.subnet[b]
        merged = _newick_join(left, right)

        state.join(a, b, new_label)

        D_new = np.zeros((n - 1, n - 1))
        new_dist = np.zeros(n)
        for k in range(n):
            if k == i or k == j: continue
            new_dist[k] = (D[i, k] + D[j, k] - D[i, j]) / 2

        keep = [k for k in range(n) if k != j]
        for new_a, old_a in enumerate(keep):
            for new_b, old_b in enumerate(keep):
                if new_a == new_b: continue
                if old_a == i:
                    D_new[new_a, new_b] = new_dist[old_b]
                elif old_b == i:
                    D_new[new_a, new_b] = new_dist[old_a]
                else:
                    D_new[new_a, new_b] = D[old_a, old_b]

        state.D = D_new
        new_labels = list(state.labels)
        new_labels[i] = new_label
        del new_labels[j]
        state.labels = new_labels
        del state.subnet[a]; del state.subnet[b]
        state.subnet[new_label] = merged

        n -= 1

    final_label = state.labels[0]
    final_newick = state.subnet[final_label] + ';'
    final_tree = dendropy.Tree.get(data=final_newick, schema='newick',
                                   preserve_underscores=True)
    state.n_relax = n_relax
    if verbose:
        print(f'  merge_trees_via_nj: relaxed {n_relax} times', flush=True)
    return final_tree


# ===========================================================================
# High-level wrappers (used by blob_merger)
# ===========================================================================

def run_overlap_njmerge(newick_list, dm, outgroup_distances=None,
                        required_taxa=None):
    """Clean inputs, add an OUT outgroup taxon, run NJMerge, then reroot.

    required_taxa : iterable[str] or None
        The blob's COMPLETE taxon set, independent of `newick_list`. The
        compatibility filter discards whole constraint trees, and a discarded
        tree can hold taxa that appear in no surviving one; deriving the taxon
        set from `newick_list` alone then drops those taxa from the backbone
        even though the distance matrix has them. Pass the full set to keep
        them. A taxon with no surviving constraint is simply unconstrained in
        NJ, which is the correct semantics -- no constraint, no restriction.

    outgroup_distances : dict[str, float]
        REQUIRED. Per-real-taxon distance to the synthetic OUT outgroup.
        Used to bias NJ to attach OUT toward the TOB-parent direction so the
        resulting blob backbone is rooted consistent with the TOB. Every real
        taxon in `newick_list` ∩ `dm.index` must be present. Passing None
        raises ValueError — the legacy `max(dmat)*2` fallback was removed
        because it produced TOB-inconsistent rooting.
    """
    if outgroup_distances is None:
        raise ValueError(
            "run_overlap_njmerge requires outgroup_distances; the legacy "
            "'max(dmat)*2 for all taxa' default has been removed because it "
            "produced TOB-inconsistent rooting. Pass per-taxon distances "
            "(e.g. mean DM distance to TOB-external sibling leaves)."
        )
    trees = []
    all_taxa = set()
    for nwk in newick_list:
        t = dendropy.Tree.get(data=clean_extended_newick(nwk), schema='newick',
                              preserve_underscores=True)
        all_taxa |= set(l.taxon.label for l in t.leaf_nodes())
        trees.append(t)

    if required_taxa is not None:
        all_taxa |= {t for t in required_taxa if t and not t.startswith('#')}

    # Silently dropping a label that is not in the matrix hides two real
    # failures: a taxon lost to the compatibility filter, and a label that
    # parsed differently from the one the matrix is keyed on (dendropy
    # rewrites unquoted underscores to spaces unless preserve_underscores).
    # Both used to surface far downstream as an AttributeError inside
    # reroot_at_edge. Fail here instead, naming the labels.
    absent = sorted(t for t in all_taxa if t not in dm.index)
    if absent:
        sample = sorted(dm.index)[:3]
        raise ValueError(
            f"{len(absent)} taxon label(s) absent from the distance matrix: "
            f"{absent[:8]}{'...' if len(absent) > 8 else ''}. The matrix is "
            f"indexed by {len(dm.index)} labels such as {sample}. If the two "
            f"differ only by underscores vs spaces, a newick was parsed "
            f"without preserve_underscores=True."
        )

    avail = sorted(all_taxa)
    dmat = dm.loc[avail, avail].to_numpy()
    taxa = list(dm.loc[avail, avail].index)
    n = len(taxa)

    missing = [t for t in taxa if t not in outgroup_distances]
    if missing:
        raise ValueError(
            f"outgroup_distances missing values for {len(missing)} taxa: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    new_d = np.zeros((n + 1, n + 1))
    new_d[:n, :n] = dmat
    out_row = np.array([float(outgroup_distances[t]) for t in taxa],
                       dtype=float)
    for i in range(n):
        new_d[i, n] = out_row[i]
        new_d[n, i] = out_row[i]

    # OUT is carried in the DISTANCE MATRIX only -- it is deliberately NOT
    # attached to any constraint tree.
    #
    # The two rooting signals do different jobs. `outgroup_distances` is soft:
    # it shifts the Q-matrix so NJ prefers to join OUT toward the TOB-parent
    # direction, and the data can overrule it. Grafting OUT under
    # `trees[0].seed_node` was hard: it became a topological constraint
    # enforced through `violates()` like any other cherry, derived from that
    # one tree's STORED root. Since the constraint roots are arbitrary (the
    # stored trees are unifurcating, and `_contract_siblings` leaves one or
    # two children depending on argument order), that constraint had no
    # biological justification, and it made the output depend on which
    # rerooting of an identical unrooted constraint happened to be on disk:
    #
    #     ((a,b),(c,d));    ->  ((a,b),(c,d));
    #     (a,(b,(c,d)));    ->  (a,(b,(c,d)));      same unrooted tree
    #
    # OUT is in `pdm` but in no tree's leaf set, so `violates()` never
    # constrains it and `join()` treats it as a no-op tree-side; it joins
    # purely on distance. The tree is then rooted on OUT's edge below.
    pdm = matrix_to_dendropy_pdm(new_d, taxa + ['OUT'])
    merged = merge_trees_via_nj(pdm, trees)
    out_nd = merged.find_node_with_taxon_label('OUT')
    if out_nd:
        merged.reroot_at_edge(out_nd.edge, update_bipartitions=False)
        merged.prune_taxa_with_labels(labels=['OUT'])
    return tree_to_newick(merged)


def _compat_greedy(newick_list):
    """First-fit greedy compatibility filter (the older logic)."""
    selected_nwks = []
    selected_trees = []
    for nwk in newick_list:
        try:
            candidate = dendropy.Tree.get(data=clean_extended_newick(nwk),
                                          schema='newick',
                                          preserve_underscores=True)
        except Exception:
            continue
        conflicts = False
        for sel_tree in selected_trees:
            t1 = deepcopy(candidate); t2 = deepcopy(sel_tree)
            if are_two_trees_incompatible(t1, t2):
                conflicts = True
                break
        if not conflicts:
            selected_nwks.append(nwk)
            selected_trees.append(candidate)
    return selected_nwks


def select_compatible_subset(newick_list, strategy=None):
    """Pick a set of pairwise-compatible trees from newick_list.

    Strategy: 'maxclique' (default) chooses the LARGEST clique in the
    compatibility graph. 'greedy' uses first-fit. Both use the same
    `are_two_trees_incompatible` bipartition check.

    Default can be overridden via env var DIMPLE_COMPAT_FILTER (greedy|maxclique).
    """
    if strategy is None:
        strategy = os.environ.get('DIMPLE_COMPAT_FILTER', 'maxclique')
    if strategy not in ('greedy', 'maxclique'):
        raise ValueError(f"strategy must be 'greedy' or 'maxclique', got {strategy!r}")

    if len(newick_list) <= 1:
        return newick_list

    if strategy == 'greedy':
        out = _compat_greedy(newick_list)
        print(f"  Compatible subset (greedy): {len(out)}/{len(newick_list)} trees kept",
              flush=True)
        return out

    trees = []
    for nwk in newick_list:
        try:
            trees.append(dendropy.Tree.get(
                data=clean_extended_newick(nwk), schema='newick',
                preserve_underscores=True))
        except Exception:
            trees.append(None)
    n = len(trees)

    G = nx.Graph()
    G.add_nodes_from(i for i in range(n) if trees[i] is not None)
    for i in range(n):
        if trees[i] is None: continue
        for j in range(i + 1, n):
            if trees[j] is None: continue
            t1 = deepcopy(trees[i])
            t2 = deepcopy(trees[j])
            if not are_two_trees_incompatible(t1, t2):
                G.add_edge(i, j)

    best = max(nx.find_cliques(G), key=len, default=[])
    selected_idx = sorted(best)
    selected_nwks = [newick_list[i] for i in selected_idx]
    print(f"  Compatible subset (max clique): {len(selected_nwks)}/"
          f"{len(newick_list)} trees kept", flush=True)
    return selected_nwks
