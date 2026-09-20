"""Regression tests for the adversarial-review findings F2-F8 (commit d9b71e1).

Each case is the reviewer's reproduction, asserting the INTENDED behaviour.
Run:  PYTHONPATH=src python tests/test_review_fixes.py      (or under pytest)
"""
import os, csv, tempfile
import numpy as np
from dimple.utils.network_util import newick_to_nx, build_newick_from_graph, strip_branch_lengths
from dimple.merger import add_reticulations as AR


def _snap(G):
    return sorted((u, v, tuple(sorted(d.items()))) for u, v, d in G.edges(data=True))


def _gt(text):
    p = os.path.join(tempfile.mkdtemp(), 'g.txt')
    open(p, 'w').write(text)
    return p


def test_f3_undo_restores_every_edge_attribute():
    G = newick_to_nx('((a,(b)#H1:1::0.8),(#H1:1::0.2,c));')
    before = _snap(G)
    info = {'retic_leaves': {'b'}, 'major_prob': 0.6, 'minor_prob': 0.4}
    child = AR.find_node_for_leaves(G, {'b'})          # picks the hybrid node itself
    src = next((u, v) for u, v in G.edges() if v == 'c')
    rn, mp = AR.add_reticulation(G, info, src, 9)
    assert sorted(d['prob'] for _, _, d in G.in_edges(child, data=True)) == [0.2, 0.8]
    AR.undo_reticulation(G, rn, mp)
    assert _snap(G) == before


def test_f4_probabilities_follow_the_orientation():
    N = newick_to_nx('((a,(b)#H1:1::0.9),(#H1:1::0.1,c));')
    retic = AR.find_reticulations(N, used_major=True)[0]
    T = newick_to_nx('((b,c),a);')                     # base tree kept the MINOR side
    edge = AR.find_candidate_edges(T, retic, {'a', 'b', 'c'})[0]
    rn, mp = AR.add_reticulation(T, retic, edge, 1)
    kept = [d['prob'] for u, _, d in T.in_edges(rn, data=True) if u != mp][0]
    assert (T.edges[mp, rn]['prob'], kept) == (0.9, 0.1)          # the behaviour, not the flag


def test_f4_no_swap_in_the_standard_orientation():
    N = newick_to_nx('((a,(b)#H1:1::0.9),(#H1:1::0.1,c));')
    retic = AR.find_reticulations(N, used_major=True)[0]
    T = newick_to_nx('((a,b),c);')                     # base tree kept the MAJOR side
    edge = AR.find_candidate_edges(T, retic, {'a', 'b', 'c'})[0]
    rn, mp = AR.add_reticulation(T, retic, edge, 1)
    assert T.edges[mp, rn]['prob'] == 0.1


def test_f7_literal_duplicates_are_scored_once():
    # one network holding two events is returned as the representative of BOTH
    two = '(((a,(b)#H1:1::0.7),(#H1:1::0.3,c)),((d,(e)#H2:1::0.6),(#H2:1::0.4,f)));'
    reps = AR.collect_unique_retics([two])
    assert len(reps) == 2 and reps[0] == reps[1]
    Gs = [newick_to_nx(r) for r in reps]
    got = AR._collect_and_sort_retics(Gs, [True, True])
    assert sorted(sorted(r['retic_leaves']) for _, _, r in got) == [['b'], ['e']]     # was 4 entries


def test_f7_distinct_variants_of_one_event_still_compete():
    # same hybrid child, different inheritance probabilities -> not a literal repeat
    v1 = '((a,(b)#H1:1::0.7),(#H1:1::0.3,c));'
    v2 = '((a,(b)#H1:1::0.6),(#H1:1::0.4,c));'
    got = AR._collect_and_sort_retics([newick_to_nx(v1), newick_to_nx(v2)], [True, True])
    assert len(got) == 2


def test_f2_broken_scorer_is_not_no_reticulation():
    N = newick_to_nx('((a,(b)#H1:1::0.9),(#H1:1::0.1,c));')
    try:
        AR.add_retics_greedily(newick_to_nx('((b,c),a);'), [N], score_fn=lambda g: float('inf'))
    except RuntimeError:
        return
    raise AssertionError('non-finite base score was accepted')


def test_f5_subnet_idx_is_a_physical_line_number():
    from dimple.merger.blob_merger import read_phylonet_subnets
    d = tempfile.mkdtemp()
    open(os.path.join(d, 'subnets.txt'), 'w').write('(a,b);\n\n(e,f);\n')
    with open(os.path.join(d, 'subnetworks_output_metadata.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['type', 'subnet_idx'])
        for i in range(3):
            w.writerow(['blob_group', i])
    assert read_phylonet_subnets(d, 'subnets.txt') == [(0, '(a,b);'), (2, '(e,f);')]


def _gammas(G):
    return sorted(d['prob'] for n in G if G.in_degree(n) > 1 for _, _, d in G.in_edges(n, data=True))


def test_f6_gamma_survives_the_stage3_clean():
    from dimple.merger.add_pruned_subtree import _clean_keep_gamma
    G = newick_to_nx(_clean_keep_gamma('((a:1,(b:1)#H1:1::0.9):2,(#H1:1::0.1,c:1):3);'))
    assert _gammas(G) == [0.1, 0.9]
    assert '::1' not in build_newick_from_graph(G).replace('::1.', '')


def test_f6_support_labels_do_not_merge_clades():
    # newick_to_nx uses an internal label as the node's identity: two clades both labelled 95
    # would become ONE node and the reticulation between them would vanish
    from dimple.merger.add_pruned_subtree import _clean_keep_gamma
    G = newick_to_nx(_clean_keep_gamma('((a,(b)#H1:1::0.9)95:1,(#H1:1::0.1,c)95:1);'))
    hyb = [n for n in G if G.in_degree(n) > 1]
    assert len(hyb) == 1 and len(set(G.predecessors(hyb[0]))) == 2 and _gammas(G) == [0.1, 0.9]
    T = newick_to_nx(_clean_keep_gamma('((a,b)95,(c,d)95);'))
    assert sum(1 for n in T if T.out_degree(n) == 2 and n != 'seed') == 3      # root + two clades


def test_f6_triple_colon_keeps_its_values():
    assert strip_branch_lengths('(a,#H1:::0.447);') == '(a,#H1:0::0.447);'
    assert _gammas(newick_to_nx(strip_branch_lengths('((a,(b)#H1:::0.9),(#H1:::0.1,c));'))) == [0.1, 0.9]


def test_f3_accepted_insertion_is_the_one_that_was_scored():
    # undo re-adds an edge, which reorders networkx predecessors; when the retic child is itself
    # a hybrid node the accepted insertion used to land above a different parent than the trial
    tree = newick_to_nx('((a,(b)#H1:1::0.8),(#H1:1::0.2,c));')
    cand = newick_to_nx('((a,(b)#H1:1::0.9),(#H1:1::0.1,c));')
    seen = []
    def scorer(G):
        hyb = [n for n in G if G.in_degree(n) > 1]
        if len(hyb) == 1:
            return 100.0
        new = [h for h in hyb if any(str(p).startswith('minor_p_') for p in G.predecessors(h))][0]
        above = [p for p in G.predecessors(new) if not str(p).startswith('minor_p_')][0]
        import networkx as nx
        sc = 10.0 if 'a' in nx.descendants(G, above) else 200.0
        seen.append(sc)
        return sc
    AR.add_retics_greedily(tree, [cand], score_fn=scorer)
    trials = seen[:]
    assert scorer(tree) == min(trials + [100.0]), (scorer(tree), trials)


def test_f2_add_all_reticulations_refuses_a_broken_scorer():
    N = newick_to_nx('((a,(b)#H1:1::0.9),(#H1:1::0.1,c));')
    try:
        AR.add_all_reticulations(newick_to_nx('((a,b),c);'), [N], score_fn=lambda g: float('inf'))
    except RuntimeError:
        return
    raise AssertionError('non-finite base score was accepted')


def test_f8_mean_over_trees_where_the_pair_cooccurs():
    from dimple.merger.merger_util import compute_dm, compute_dm_full
    try:
        compute_dm(_gt('(a:1,b:1);\n(a:1,c:1);\n'), {'a', 'b', 'c'})
        raise AssertionError('b,c never co-occur but no error')
    except ValueError:
        pass
    dm = compute_dm(_gt('(a:1,b:1);\n(a:1,c:1);\n(b:2,c:2);\n'), {'a', 'b', 'c'})
    assert dm.loc['a', 'b'] == 2 and dm.loc['b', 'c'] == 4                  # was 2/3 and 4/3
    # the whole-dataset matrix must not fail for pairs a blob never needs
    part = compute_dm_full(_gt('(a:1,b:1,c:1);\n(x:1,y:1);\n'))
    assert (part.loc[['a', 'b', 'c'], ['a', 'b', 'c']].values[~np.eye(3, dtype=bool)] == 2).all()
    assert np.isnan(part.loc['a', 'x'])
    full = _gt('(a:1,(b:1,c:1):1);\n(a:2,(b:1,c:1):1);\n')                   # complete coverage: unchanged
    assert compute_dm(full, {'a', 'b', 'c'}).loc['a', 'b'] == 3.5 == compute_dm_full(full).loc['a', 'b']


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t(); print('PASS ', t.__name__)
    print(f'\n{len(tests)} passed')
