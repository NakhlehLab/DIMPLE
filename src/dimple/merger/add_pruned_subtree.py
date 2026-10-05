"""Add pruned subtrees back to a blob network.

For a given blob's output newick, look up all pruned_subtree rows in the
non_blob metadata whose source_item.mega_blob matches the blob name, and
insert each subtree as sibling of the MRCA of its source_item.leaves in the
blob network.

When multiple pruned subtrees share the same source_item (collision), the
insertion order uses TOB-LCA proximity: the cut structurally CLOSER to the
source_item's MRCA in TOB gets processed LAST so it lands at the INNER
(deeper) sibling position.

Reticulation rule: when the MRCA is a reticulation node, attach BELOW the
retic (as sibling of the retic's single child) so the pruned subtree lands
INSIDE the retic clade — never orphaned outside it.

Usage (from project root):
    conda run -n phylo-env python -u -m dimple.merger.add_pruned_subtree \\
        --metadata data/division_iqtree_tob/lvl2/n02/divisions3/non_blob/subnetworks_output_metadata.csv \\
        --non-blob-nwks data/division_iqtree_tob/lvl2/n02/divisions3/non_blob/subnetworks_output.txt \\
        --blob data/division_iqtree_tob/lvl2/n02/divisions3/gt_subnets/node_1+node_12+node_4_ground_truth.nwk \\
        --blob-name node_1+node_12+node_4 \\
        --tob data/division_iqtree_tob/lvl2/n02/tob_iqtree/tob_iqtree_reroot.tre \\
        --out /tmp/n02_node1_with_pruned.nwk
"""
import os
import sys
import csv
import json
import ast
import argparse
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from dimple.utils.network_util import (
    newick_to_nx, build_newick_from_graph, get_leafset, contract_degree2_nodes,
    clean_extended_newick, strip_branch_lengths,
)


def _clean_keep_gamma(nwk):
    """clean_extended_newick, but keeping the inheritance probabilities.

    Numeric support labels are still dropped: newick_to_nx uses an internal
    label as the node's identity, so two clades with support '95' would merge.
    """
    s = re.sub(r"'\[pp\d=[^]]+\]'", '', nwk)          # ASTRAL-style annotations
    s = strip_branch_lengths(re.sub(r'\s+', '', s))     # drops lengths, keeps ':0::gamma'
    return re.sub(r'\)[-+]?\d[0-9.eE+-]*', ')', s)      # numeric support labels


from dimple.merger.merger_util import (
    find_root, copy_with_unique_names, smallest_containing,
)


def load_pruned_rows(metadata_csv, nwks_file, blob_name):
    """Return list of dicts {idx, cut_edge, source_item, leaves, G} filtered
    to rows where source_item.mega_blob == blob_name. Every pruned subtree is
    one row: the divider only cuts subtrees that fit within its size limit.
    """
    with open(nwks_file) as f:
        # subnet_idx is a physical line number; a missing estimate is an empty
        # line that keeps its slot, so blank lines must not be dropped.
        newicks = [l.strip() for l in f]

    matched_rows = []
    with open(metadata_csv) as f:
        for row in csv.DictReader(f):
            if row['type'] != 'pruned_subtree':
                continue
            si = None
            if row['source_item']:
                try:
                    si = json.loads(row['source_item'])
                except Exception:
                    si = None
            if not si or si.get('mega_blob') != blob_name:
                continue
            idx = int(row['subnet_idx'])
            if idx >= len(newicks) or not newicks[idx]:
                continue
            ce = ast.literal_eval(row['cut_edge']) if row['cut_edge'] else None
            matched_rows.append({
                'idx': idx, 'cut_edge': ce, 'source_item': si,
                'leaves': set(row['all_leaves'].split(',')),
                'nwk': newicks[idx],
            })

    out = []
    for r in matched_rows:
        G = newick_to_nx(_clean_keep_gamma(r['nwk']))        # pruned piece may be a network
        out.append({'idx': r['idx'], 'cut_edge': r['cut_edge'],
                    'source_item': r['source_item'],
                    'leaves': r['leaves'], 'G': G})
    return out


def adjust_for_retic(T, mrca):
    """If attaching at `mrca` would place the subtree AMBIGUOUSLY above or
    below a reticulation, always prefer BELOW. Specifically:
      - If mrca is a retic node (in_degree>1), step DOWN to its single child
        so the new subtree becomes sibling INSIDE the retic clade.
      - Recurse in case multiple retic nodes stack.
    Returns the adjusted attachment target.
    """
    cur = mrca
    while T.in_degree(cur) > 1:
        succs = list(T.successors(cur))
        if len(succs) != 1:
            break  # unusual: retic with multiple children — stop
        cur = succs[0]
    return cur


def attach_as_sibling(T, subG, mrca, used_names, label_id):
    """Insert a new internal node above mrca and attach subG as its sibling.

    If mrca is a reticulation, attach BELOW the retic (as sibling of the
    retic's child) instead of above it — so pruned content goes INSIDE the
    retic clade, never orphaned outside it.
    """
    anchor = adjust_for_retic(T, mrca)

    preds = list(T.predecessors(anchor))
    if not preds:
        return T, used_names, label_id, False
    # Pick a non-retic parent edge for the insertion (any parent works, but
    # for retic children we should pick exactly one of their parent edges to
    # detach-and-rewire).
    parent = preds[0]
    new_name = f'reatt_{label_id}'
    while new_name in used_names:
        label_id += 1
        new_name = f'reatt_{label_id}'
    used_names.add(new_name)
    label_id += 1
    attrs = T[parent][anchor]
    T.remove_edge(parent, anchor)
    T.add_edge(parent, new_name, **attrs)
    T.add_edge(new_name, anchor, **attrs)

    sub = subG.copy()
    r = find_root(sub)
    if r == 'seed':
        r = list(sub.successors('seed'))[0]
        sub.remove_node('seed')
    gr, new_r, used_names = copy_with_unique_names(sub, T, r, used_names)
    T.add_nodes_from(gr.nodes(data=True))
    for u, v in gr.edges():
        T.add_edge(u, v, **gr[u][v])
    T.add_edge(new_name, new_r)
    return T, used_names, label_id, True


def node_depth(T, n):
    """Depth of n in T (root = 0)."""
    d = 0
    cur = n
    while cur in T:
        preds = list(T.predecessors(cur))
        if not preds:
            break
        d += 1
        cur = preds[0]
    return d


def lca(T, a, b):
    """Lowest common ancestor of a, b in T (directed tree). None if not found."""
    if a not in T or b not in T:
        return None
    anc_a = set()
    cur = a
    while cur in T:
        anc_a.add(cur)
        preds = list(T.predecessors(cur))
        if not preds:
            break
        cur = preds[0]
    cur = b
    while cur in T:
        if cur in anc_a:
            return cur
        preds = list(T.predecessors(cur))
        if not preds:
            break
        cur = preds[0]
    return None


def collision_sort_key(p, T_orig):
    """Sort key: cuts STRUCTURALLY CLOSER to their source_item.leaves in TOB
    get processed LAST (so they land at the INNER / deeper sibling position).

    Primary:   depth(LCA(cut_edge.parent, source_mrca_in_TOB)) — higher is closer.
    Secondary: depth(cut_edge.parent) — deeper parent wins ties.
    Tertiary:  idx — stable tiebreaker (deterministic).
    """
    if T_orig is None:
        return (0, 0, p['idx'])
    ce = p['cut_edge']
    si = p['source_item'] or {}
    src = set(si.get('leaves', []))
    if not ce or not src:
        return (0, 0, p['idx'])
    parent_node, _ = ce
    if parent_node not in T_orig:
        return (0, 0, p['idx'])
    src_present = src & (get_leafset(T_orig) - {'seed'})
    if not src_present:
        return (0, 0, p['idx'])
    s_mrca = smallest_containing(T_orig, src_present)
    if s_mrca is None:
        return (0, 0, p['idx'])
    anc = lca(T_orig, parent_node, s_mrca)
    lca_d = node_depth(T_orig, anc) if anc is not None else 0
    parent_d = node_depth(T_orig, parent_node)
    return (lca_d, parent_d, p['idx'])


def add_pruned(blob_nwk, pruned_rows, T_orig=None):
    """Return (T, stats) after inserting each pruned subtree into blob network.

    T_orig (TOB, recommended) is used to:
      1. Order colliding source_item attachments by TOB proximity.
      2. Decide INSIDE vs OUTSIDE: if the cut was made outside the blob's TOB
         clade, attach the pruned subtree at the BLOB ROOT (as sibling of the
         entire blob expansion), not at source_item's MRCA inside the blob.
    """
    G = newick_to_nx(_clean_keep_gamma(blob_nwk))
    if 'OUT' in G.nodes:
        G.remove_node('OUT')
        G = contract_degree2_nodes(G)

    # Remove blob leaves that will be re-attached via prunes. The phylonet-
    # inferred blob may have been built from an earlier division layout that
    # included leaves now reassigned to a prune by the current metadata.
    # Without this step those leaves end up duplicated in the combined output.
    prune_leaves_total = set()
    for p in pruned_rows:
        prune_leaves_total |= p.get('leaves', set())
    initial_lv = get_leafset(G) - {'seed'}
    overlap = initial_lv & prune_leaves_total
    if overlap:
        for l in overlap:
            G.remove_node(l)
        G = contract_degree2_nodes(G)
        print(f'  Removed {len(overlap)} blob leaves overlapping with prune leaves: '
              f'{sorted(overlap)[:5]}{"..." if len(overlap) > 5 else ""}')

    used_names = set(G.nodes())
    label_id = 1

    blob_leaves = get_leafset(G) - {'seed'}

    # Sort ascending: smaller key processed FIRST (becomes OUTER);
    # larger key processed LAST (becomes INNER).
    pruned_rows = sorted(pruned_rows, key=lambda p: collision_sort_key(p, T_orig))

    log = []
    ok = skipped = 0

    # Identify all blob-item leaves (they're already in blob_leaves at start)
    initial_blob_leaves = set(blob_leaves)

    # Topological placement:
    #  - blob_item / mixed cut: attach at MRCA of (target ∩ blob leaves) —
    #    doesn't need to wait for other prunes.
    #  - prune cut: source_item leaves are other pruned subtrees; wait for
    #    ALL of them to be placed, then attach at MRCA of full target.
    placed = {}
    remaining = list(pruned_rows)

    while remaining:
        progress = False
        still_waiting = []
        for p in remaining:
            si = p['source_item'] or {}
            target = set(si.get('leaves', []))
            source_type = si.get('source_type', 'blob_item')
            if not target:
                raise ValueError(
                    f"pruned_{p['idx']} has empty source_item.leaves — division "
                    f"must always return a source (cut_edge={p['cut_edge']})."
                )

            if source_type in ('blob_item', 'mixed'):
                # Use blob-item subset of target (these are already in G via initial blob)
                effective = target & initial_blob_leaves
                if not effective:
                    raise RuntimeError(
                        f"pruned_{p['idx']} (source_type={source_type}, "
                        f"cut={p['cut_edge']}): target {sorted(target)[:5]} "
                        f"has no overlap with blob items — cannot attach. "
                        f"No fallbacks allowed."
                    )
                else:
                    mrca = smallest_containing(G, effective)
                    if mrca is None:
                        still_waiting.append(p); continue
                    G, used_names, label_id, attached = attach_as_sibling(
                        G, p['G'], mrca, used_names, label_id)
                    if attached:
                        ok += 1
                        placed[p['idx']] = p['leaves']
                        progress = True
                        log.append(
                            f"  REATTACH pruned_{p['idx']} ({p['cut_edge']}): "
                            f"{len(p['leaves'])} leaves → sibling of MRCA"
                            f"({sorted(effective)[:3]}{'...' if len(effective)>3 else ''}) = {mrca}"
                        )
                    else:
                        skipped += 1
                        log.append(f"  SKIP pruned_{p['idx']}: attach failed")
                    continue

            # source_type == 'prune': need all target leaves in G
            cur_leaves_G = get_leafset(G) - {'seed'}
            present = target & cur_leaves_G
            if present != target:
                still_waiting.append(p); continue
            mrca = smallest_containing(G, present)
            if mrca is None:
                still_waiting.append(p); continue
            G, used_names, label_id, attached = attach_as_sibling(
                G, p['G'], mrca, used_names, label_id)
            if attached:
                ok += 1
                placed[p['idx']] = p['leaves']
                progress = True
                log.append(
                    f"  REATTACH pruned_{p['idx']} ({p['cut_edge']}) [prune-chain]: "
                    f"{len(p['leaves'])} leaves → MRCA({sorted(present)[:3]}...) = {mrca}"
                )
            else:
                skipped += 1

        if not progress:
            # Generalized host search: for each stuck prune, the host can be
            # G OR any other not-yet-placed prune's subnet. Pick the host
            # containing the MOST of p's target leaves; attach p there at
            # MRCA of that subset. Place one prune per stuck-iteration so
            # the others can re-evaluate after p's content joins a host.
            def host_avail(p, host_G):
                t = set(p['source_item'].get('leaves', []))
                return t & (set(get_leafset(host_G)) - {'seed'})

            best_p = None
            best_host_idx = None  # None = G; otherwise idx of the host prune
            best_count = -1
            for p in still_waiting:
                # candidate hosts: G + every other still-waiting prune
                cands = [(None, G)] + [(q['idx'], q['G'])
                                         for q in still_waiting
                                         if q['idx'] != p['idx']]
                for h_idx, h_G in cands:
                    cnt = len(host_avail(p, h_G))
                    if cnt > best_count:
                        best_count = cnt
                        best_p = p
                        best_host_idx = h_idx
            if best_p is None or best_count == 0:
                stuck = []
                for p in still_waiting:
                    t = set(p['source_item'].get('leaves', []))
                    stuck.append(f"    idx={p['idx']} target_size={len(t)} "
                                  f"cut={p['cut_edge']}")
                raise RuntimeError(
                    f"add_pruned_subtree: {len(still_waiting)} prune(s) stuck — "
                    f"no host (G or another prune) contains any target leaf. "
                    f"Stuck prunes:\n" + "\n".join(stuck)
                )

            # Attach best_p inside its chosen host at MRCA of available subset
            target = set(best_p['source_item'].get('leaves', []))
            if best_host_idx is None:
                host_G = G
                host_label = 'G'
            else:
                host_q = next(q for q in still_waiting
                               if q['idx'] == best_host_idx)
                host_G = host_q['G']
                host_label = f"prune_{best_host_idx}"
            avail = target & (set(get_leafset(host_G)) - {'seed'})
            mrca = smallest_containing(host_G, avail)
            if mrca is None:
                raise RuntimeError(
                    f"pruned_{best_p['idx']}: smallest_containing failed for "
                    f"avail={sorted(avail)[:4]}... in host {host_label}."
                )
            host_G, used_names, label_id, attached = attach_as_sibling(
                host_G, best_p['G'], mrca, used_names, label_id)
            if not attached:
                raise RuntimeError(
                    f"pruned_{best_p['idx']}: attach_as_sibling failed at "
                    f"MRCA={mrca} in host {host_label}."
                )
            if best_host_idx is None:
                G = host_G
                tag = '[host=G]'
            else:
                host_q['G'] = host_G
                tag = f'[host=prune_{best_host_idx}]'
            ok += 1
            placed[best_p['idx']] = best_p['leaves']
            log.append(
                f"  REATTACH pruned_{best_p['idx']} ({best_p['cut_edge']}) "
                f"{tag} {len(avail)}/{len(target)} target leaves: "
                f"{len(best_p['leaves'])} leaves → MRCA"
                f"({sorted(avail)[:3]}{'...' if len(avail)>3 else ''}) = {mrca}"
            )
            remaining = [p for p in still_waiting if p['idx'] != best_p['idx']]
            continue
        remaining = still_waiting

    stats = {'reattached': ok, 'skipped': skipped, 'total': len(pruned_rows), 'log': log}
    return G, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metadata', required=True,
                    help='non_blob subnetworks_output_metadata.csv')
    ap.add_argument('--non-blob-nwks', required=True,
                    help='non_blob subnetworks_output.txt')
    ap.add_argument('--blob', required=True,
                    help='Blob network newick file')
    ap.add_argument('--blob-name', required=True,
                    help='Blob name (e.g. node_1+node_12+node_4)')
    ap.add_argument('--tob', default=None,
                    help='TOB newick (optional, used for depth ordering)')
    ap.add_argument('--out', required=True, help='Output newick path')
    args = ap.parse_args()

    blob_nwk = open(args.blob).read().strip()
    print(f'Blob: {args.blob_name}')

    T_orig = None
    if args.tob and os.path.exists(args.tob):
        T_orig = newick_to_nx(clean_extended_newick(
            open(args.tob).read().strip()))
        if 'OUT' in T_orig.nodes:
            T_orig.remove_node('OUT')
        T_orig = contract_degree2_nodes(T_orig)

    pruned = load_pruned_rows(args.metadata, args.non_blob_nwks, args.blob_name)
    print(f'Found {len(pruned)} pruned subtrees for mega_blob={args.blob_name}')

    G, stats = add_pruned(blob_nwk, pruned, T_orig)
    for line in stats['log']:
        print(line)
    print(f"\nreattached={stats['reattached']}/{stats['total']}, "
          f"skipped={stats['skipped']}")

    out_nwk = strip_branch_lengths(build_newick_from_graph(G))
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(out_nwk + '\n')
    print(f'\nOutput → {args.out}')
    print(f'Output leaves: {len(get_leafset(G) - {"seed"})}')


if __name__ == '__main__':
    main()
