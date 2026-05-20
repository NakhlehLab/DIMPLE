"""
Pre-processing step for overlap NJMerge: select displayed trees from network inputs.

For each network input, enumerates its displayed trees and picks the one
most compatible with the other (tree) inputs on shared taxa. 

pick DT with best pseudolikelihood score (gene tree triple support)
   on the taxa shared with other inputs

"""

import dendropy
from dendropy.calculate.treecompare import false_positives_and_negatives

from dimple.utils.network_util import (
    newick_to_nx, build_newick_from_graph, get_leafset,
    enumerate_displayed_trees, clean_extended_newick,
    contract_degree2_nodes,
)
from dimple.merger.merger_util import (
    precompute_gene_tree_triples, _extract_triples_from_nx,
    load_triple_cache,
)


def is_network(newick_str):
    """Check if a newick string represents a network (has reticulations)."""
    G = newick_to_nx(newick_str)
    return any(G.in_degree(n) > 1 for n in G.nodes())


def get_displayed_tree_newicks(newick_str):
    """Enumerate displayed trees from a network newick, return as newick strings."""
    G = newick_to_nx(newick_str)
    retics = [n for n in G.nodes() if G.in_degree(n) > 1]
    if not retics:
        return [newick_str]

    displayed = enumerate_displayed_trees(G)
    newicks = []
    for dt in displayed:
        nwk = build_newick_from_graph(dt)
        newicks.append(nwk)
    return newicks


def _score_dt_pl(dt_nwk, shared_taxa, observed_counts):
    """Score a displayed tree on shared_taxa using PL (triple match).
    Returns fraction NOT matched (lower = better)."""
    G = newick_to_nx(clean_extended_newick(dt_nwk))

    # Restrict to shared taxa
    g_leaves = get_leafset(G) - {'seed', 'OUT'}
    keep = g_leaves & shared_taxa
    if len(keep) < 3:
        return 1.0

    remove = g_leaves - keep
    if remove:
        for leaf in list(remove):
            if leaf in G:
                G.remove_node(leaf)
        G = contract_degree2_nodes(G)

    tree_triples = _extract_triples_from_nx(G)

    matched = 0
    total = 0
    for (a, b, c), count in observed_counts.items():
        if a in keep and b in keep and c in keep:
            total += count
            if (a, b, c) in tree_triples:
                matched += count

    if total == 0:
        return 1.0
    return 1.0 - (matched / total)


def select_displayed_trees(newick_list, gene_tree_file=None, triple_cache=None):
    """
    Given a list of newick strings (some may be networks), replace each
    network with its best displayed tree (PL-scored against gene trees).

    gene_tree_file is REQUIRED — there is no compatibility-based fallback.

    Returns:
        list of newick strings (all trees, no networks)
        list of dicts with selection info
    """
    if not gene_tree_file:
        raise ValueError(
            "select_displayed_trees requires gene_tree_file; "
            "no compatibility fallback is supported.")

    n = len(newick_list)
    is_net = [is_network(nwk) for nwk in newick_list]
    candidates = []

    for i, nwk in enumerate(newick_list):
        if is_net[i]:
            dts = get_displayed_tree_newicks(nwk)
            candidates.append(dts)
            print(f"  Input {i}: NETWORK with {len(dts)} displayed trees")
        else:
            candidates.append([nwk])
            taxa = _get_taxa_quick(nwk)
            print(f"  Input {i}: tree ({len(taxa)} taxa)")

    # Precompute PL triples
    all_taxa = set()
    for nwk in newick_list:
        all_taxa |= _get_taxa_quick(nwk)
    all_taxa = {t for t in all_taxa if not t.startswith('#')}

    if triple_cache and __import__('os').path.exists(triple_cache):
        observed_counts = load_triple_cache(triple_cache, all_taxa)
    else:
        observed_counts = precompute_gene_tree_triples(gene_tree_file, all_taxa)
    print(f"  PL scoring: {len(observed_counts)} triples")

    # Collect shared taxa between each network and all other inputs
    selected = list(newick_list)
    info = []

    for i in range(n):
        if not is_net[i] or len(candidates[i]) <= 1:
            info.append({'type': 'tree' if not is_net[i] else 'network_1dt',
                         'selected': 0, 'n_candidates': len(candidates[i])})
            selected[i] = candidates[i][0]
            continue

        # Collect all taxa from other inputs that overlap with this network
        net_taxa = _get_taxa_quick(newick_list[i])
        other_shared = set()
        for j in range(n):
            if j == i:
                continue
            other_taxa = _get_taxa_quick(
                candidates[j][0] if not is_net[j] else selected[j])
            other_shared |= (net_taxa & other_taxa)

        # Score each displayed tree by PL on shared taxa
        dt_scores = []
        for dt_idx, dt_nwk in enumerate(candidates[i]):
            score = _score_dt_pl(dt_nwk, other_shared, observed_counts)
            dt_scores.append({'idx': dt_idx, 'score': score, 'type': 'pl'})

        dt_scores.sort(key=lambda x: x['score'])
        best = dt_scores[0]
        selected[i] = candidates[i][best['idx']]

        print(f"  Input {i}: selected DT {best['idx']} "
              f"(pl={best['score']:.4f})")
        for s in dt_scores:
            marker = " <--" if s['idx'] == best['idx'] else ""
            print(f"    DT {s['idx']}: pl={s['score']:.4f}{marker}")

        info.append({
            'type': 'network',
            'selected': best['idx'],
            'n_candidates': len(candidates[i]),
            'scores': dt_scores,
        })

    return selected, info


def _get_taxa_quick(newick_str):
    """Quick taxa extraction."""
    t = dendropy.Tree.get(data=clean_extended_newick(newick_str), schema="newick")
    return set(l.taxon.label for l in t.leaf_nodes())
