"""Merge combined blobs back into TOB.

Strategy:
  1. Scaffold = full TOB (keep OUT).
  2. Order blobs DEEPEST first (innermost TOB position first).
  3. For each blob:
     a. Find blob's TOB MRCA node (by blob's leafset in current T).
     b. Detach any already-placed inner blob grafts nested inside that MRCA
        subtree — save them.
     c. Remove MRCA subtree.
     d. Graft blob's combined network at the former parent.
     e. Re-attach saved inner grafts by finding their source_item anchor in
        the newly grafted combined content.
"""
import os
import csv
import json

import networkx as nx
from dimple.utils.network_util import (
    newick_to_nx, build_newick_from_graph, get_leafset, contract_degree2_nodes,
    clean_extended_newick, strip_branch_lengths,
)
from dimple.merger.merger_util import (
    find_root, copy_with_unique_names, smallest_containing,
)


def graft_as_sibling_of(T, sub_G, anchor, used_names):
    """Insert sub_G as sibling of anchor."""
    preds = list(T.predecessors(anchor))
    if not preds:
        return None, used_names
    parent = preds[0]
    new_int = 'm_root'
    k = 1
    while new_int in used_names:
        new_int = f'm_root_{k}'; k += 1
    used_names.add(new_int)
    attrs = T[parent][anchor]
    T.remove_edge(parent, anchor)
    T.add_edge(parent, new_int, **attrs)
    T.add_edge(new_int, anchor, **attrs)
    sub = sub_G.copy()
    r = find_root(sub)
    if r == 'seed':
        r = list(sub.successors('seed'))[0]
        sub.remove_node('seed')
    gr, new_r, used_names = copy_with_unique_names(sub, T, r, used_names)
    T.add_nodes_from(gr.nodes(data=True))
    for u, v in gr.edges():
        T.add_edge(u, v, **gr[u][v])
    T.add_edge(new_int, new_r)
    return new_r, used_names


def graft_at_parent(T, sub_G, parent, used_names):
    """Add sub_G as a new child of parent."""
    sub = sub_G.copy()
    r = find_root(sub)
    if r == 'seed':
        r = list(sub.successors('seed'))[0]
        sub.remove_node('seed')
    gr, new_r, used_names = copy_with_unique_names(sub, T, r, used_names)
    T.add_nodes_from(gr.nodes(data=True))
    for u, v in gr.edges():
        T.add_edge(u, v, **gr[u][v])
    T.add_edge(parent, new_r)
    return new_r, used_names


def load_blob_source_items(ds_dir, divisions_dir='divisions3'):
    src_items = {}
    div3 = f'{ds_dir}/{divisions_dir}'
    for d in sorted(os.listdir(div3)):
        bdir = f'{div3}/{d}'
        if not d.startswith('blob') or not os.path.isdir(bdir):
            continue
        meta = f'{bdir}/run_000/subnetworks_output_metadata.csv'
        if not os.path.exists(meta): continue
        with open(meta) as f:
            for row in csv.DictReader(f):
                if row['type'] != 'blob_group': continue
                si = row.get('source_item', '')
                if not si: continue
                try: js = json.loads(si)
                except: continue
                blob = row['blob']
                if blob not in src_items:
                    src_items[blob] = js
    return src_items


def find_blob_tob_node(T, blob_name, blob_tob_leaves):
    """Find a node in T whose leafset equals blob_tob_leaves (the original TOB
    blob leaves, which may be a subset of blob's combined leafset). Falls back
    to smallest-containing."""
    # Try exact leafset match first
    for n in T.nodes():
        if n == 'seed': continue
        nl = get_leafset(T, n)
        if nl == blob_tob_leaves:
            return n
    # Fallback: smallest containing
    return smallest_containing(T, blob_tob_leaves)


def load_blob_tob_leaves(ds_dir, divisions_dir='divisions3'):
    """Return blob_name -> the blob's ORIGINAL TOB subtree leafset (from
    blob_group rows' all_leaves field across all item_mappings)."""
    result = {}
    div3 = f'{ds_dir}/{divisions_dir}'
    for d in sorted(os.listdir(div3)):
        bdir = f'{div3}/{d}'
        if not d.startswith('blob') or not os.path.isdir(bdir): continue
        meta = f'{bdir}/run_000/subnetworks_output_metadata.csv'
        if not os.path.exists(meta): continue
        blob_leaves = set()
        blob_name = None
        with open(meta) as f:
            for row in csv.DictReader(f):
                if row['type'] != 'blob_group': continue
                blob_name = row['blob']
                blob_leaves |= set(row['all_leaves'].split(','))
        if blob_name:
            result[blob_name] = blob_leaves
    return result


def merge(ds_dir, mode='gt', out_path=None, divisions_dir='divisions3',
          tob_path=None, metadata_dir=None):
    if tob_path is None:
        tob_path = f'{ds_dir}/tob_iqtree/tob_iqtree_reroot.tre'
    combined_dir = f'{ds_dir}/{divisions_dir}/combined_blobs_{mode}'
    if not os.path.isdir(combined_dir):
        print(f'ERROR: {combined_dir} not found'); return None
    # If --metadata-dir is set, use it for blob source_item + tob_leaves;
    # otherwise default to divisions_dir (back-compat).
    meta_dir = metadata_dir if metadata_dir else divisions_dir

    T = newick_to_nx(clean_extended_newick(open(tob_path).read().strip()))
    T = contract_degree2_nodes(T)
    used_names = set(T.nodes())

    src_items = load_blob_source_items(ds_dir, meta_dir)
    blob_tob_leaves = load_blob_tob_leaves(ds_dir, meta_dir)
    blobs = []
    for fn in sorted(os.listdir(combined_dir)):
        if not fn.endswith('_combined.nwk'): continue
        blob_name = fn.replace('_combined.nwk', '')
        nwk = open(f'{combined_dir}/{fn}').read().strip()
        G = newick_to_nx(nwk)
        if 'OUT' in G.nodes:
            G.remove_node('OUT')
            G = contract_degree2_nodes(G)
        leafset = get_leafset(G) - {'seed'}
        blobs.append({'name': blob_name, 'leafset': leafset, 'G': G,
                      'source_item': src_items.get(blob_name),
                      'tob_leaves': blob_tob_leaves.get(blob_name, set())})
    # Sort DEEPEST first by TOB depth: blobs nested inside another blob's TOB
    # subtree must be processed FIRST so they don't get wiped out when the
    # outer blob's subtree is replaced.
    def tob_depth(b):
        # Load TOB once
        T0 = newick_to_nx(clean_extended_newick(open(tob_path).read().strip()))
        T0 = contract_degree2_nodes(T0)
        # Find the smallest TOB node containing blob's leaves
        mrca = smallest_containing(T0, b['tob_leaves']) if b['tob_leaves'] else None
        if mrca is None:
            return 0
        d = 0
        cur = mrca
        while cur in T0:
            preds = list(T0.predecessors(cur))
            if not preds: break
            d += 1
            cur = preds[0]
        return d
    # Higher depth (deeper) → process first, so sort descending.
    blobs.sort(key=lambda b: -tob_depth(b))

    print(f'  scaffold: full TOB with {len(get_leafset(T) - {"seed"})} leaves')
    for b in blobs:
        print(f'    {b["name"]}: TOB_leaves={len(b["tob_leaves"])}, '
              f'combined_leaves={len(b["leafset"])}')

    # Track placed blob roots (so we can preserve them when outer blob replaces)
    placed = []  # list of (blob_name, root_node_in_T, leafset, source_item)

    for b in blobs:
        # 1. Find blob's TOB node (leafset match) in current T
        tob_node = find_blob_tob_node(T, b['name'], b['tob_leaves'])
        if tob_node is None:
            # Try using combined leafset
            tob_node = smallest_containing(T, b['leafset'])
        if tob_node is None:
            print(f'  SKIP {b["name"]}: no TOB node found')
            continue

        # 2. Detach any already-placed inner blob grafts nested inside this subtree.
        # Only collect TOPMOST nested blobs — if pb_A's root is already inside
        # pb_B's saved subgraph, skip pb_A (its content is carried inside pb_B's
        # subG and will be re-attached together). Otherwise we'd duplicate
        # pb_A's leaves once via pb_B's subG and once via its own.
        cand = [pb for pb in placed
                if pb['root'] in T.nodes and nx.has_path(T, tob_node, pb['root'])]
        # Sort: process outermost (largest) first so we mark inner ones as nested
        cand.sort(key=lambda pb: -len(set(nx.descendants(T, pb['root'])) | {pb['root']}))
        inside_nested = []
        claimed = set()
        for pb in cand:
            sub_nodes = set(nx.descendants(T, pb['root'])) | {pb['root']}
            if pb['root'] in claimed:
                continue   # nested inside another already-collected pb
            inside_nested.append({'name': pb['name'], 'root': pb['root'],
                                  'subG': T.subgraph(sub_nodes).copy(),
                                  'source_item': pb['source_item'],
                                  'leafset': pb['leafset']})
            claimed |= sub_nodes
        for n in inside_nested:
            sub_nodes = set(nx.descendants(T, n['root'])) | {n['root']}
            T.remove_nodes_from(sub_nodes)

        # 3. Remove blob's TOB subtree — but FIRST graft combined at a new
        # node under tob_node's parent, so parent doesn't get contracted.
        preds = list(T.predecessors(tob_node))
        if not preds:
            print(f'  SKIP {b["name"]}: TOB node is scaffold root')
            continue
        parent = preds[0]

        # 3a. Graft combined at parent BEFORE removing anything
        new_r, used_names = graft_at_parent(T, b['G'], parent, used_names)

        # 3b. Now remove the old TOB subtree
        desc = set(nx.descendants(T, tob_node)) | {tob_node}
        # Don't remove anything in the just-grafted subgraph:
        graft_desc = set(nx.descendants(T, new_r)) | {new_r}
        to_remove = desc - graft_desc
        T.remove_nodes_from(to_remove)

        # 3c. Remove any OTHER leaves in T (not in the graft) that are in the
        # combined leafset (pruned leaves that existed at non-blob positions
        # in TOB and would otherwise duplicate with the combined network).
        cur_leaves_T = (get_leafset(T) - {'seed'})
        graft_leaves = get_leafset(T, new_r)
        overlap = (b['leafset'] & cur_leaves_T) - graft_leaves
        if overlap:
            T.remove_nodes_from(overlap)
        T = contract_degree2_nodes(T)
        # Re-identify the grafted root by leafset — the original new_r may
        # have been contracted away by contract_degree2_nodes if its root
        # had out_degree=1. smallest_containing returns the current node in T
        # whose leafset contains the blob's combined leafset.
        canonical_root = smallest_containing(T, b['leafset']) or new_r
        b_root = canonical_root
        print(f'  ✓ {b["name"]}: grafted at parent={parent}, new root={canonical_root}')
        placed.append({'name': b['name'], 'root': canonical_root, 'leafset': b['leafset'],
                       'source_item': b['source_item']})

        # 5. Re-attach any detached nested grafts — keep binary by using
        # graft_as_sibling_of throughout. When source leaves are unavailable,
        # fall back to attaching as sibling of an arbitrary existing descendant
        # of new_r, not as a 3rd child of new_r.
        def fallback_sibling_of_graft(T, subG, new_r, used_names, name):
            # Pick any existing descendant of new_r as the sibling anchor.
            succs = list(T.successors(new_r))
            if succs:
                anchor = succs[0]
                return graft_as_sibling_of(T, subG, anchor, used_names)
            return graft_at_parent(T, subG, new_r, used_names)

        # Iterative re-attach with smart ordering: at each step, re-attach
        # the blob with the MOST available siblings in the current T (since
        # other detached blobs may contain each other's siblings, attaching
        # one expands the available-sibling set for the rest).
        remaining_nested = list(inside_nested)
        while remaining_nested:
            # Score each: how many of its sibling_leaves are currently in T?
            cur_leaves_now = get_leafset(T) - {'seed'}
            def score(n):
                si = n.get('source_item') or {}
                sl = set(si.get('sibling_leaves', []))
                return len(sl & cur_leaves_now)
            remaining_nested.sort(key=score, reverse=True)
            n = remaining_nested.pop(0)
            si = n['source_item']
            if not si:
                print(f'    WARN: nested {n["name"]} has no source_item; sibling of graft root')
                n_root, used_names = fallback_sibling_of_graft(T, n['subG'], canonical_root, used_names, n['name'])
                for pb in placed:
                    if pb['name'] == n['name']:
                        pb['root'] = n_root
                        break
                continue
            sibling_leaves = set(si.get('sibling_leaves', []))
            avail = sibling_leaves & cur_leaves_now
            if not avail:
                print(f'    WARN: nested {n["name"]} source leaves missing; sibling of graft root')
                n_root, used_names = fallback_sibling_of_graft(T, n['subG'], canonical_root, used_names, n['name'])
                for pb in placed:
                    if pb['name'] == n['name']:
                        pb['root'] = n_root
                        break
                continue
            anchor = smallest_containing(T, avail)
            if anchor is None:
                n_root, used_names = fallback_sibling_of_graft(T, n['subG'], canonical_root, used_names, n['name'])
                for pb in placed:
                    if pb['name'] == n['name']:
                        pb['root'] = n_root
                        break
                continue
            n_root, used_names = graft_as_sibling_of(T, n['subG'], anchor, used_names)
            print(f'    ↻ re-attached nested {n["name"]} at MRCA({sorted(avail)[:3]}) = {anchor}')
            for pb in placed:
                if pb['name'] == n['name']:
                    pb['root'] = n_root
                    break

    T = contract_degree2_nodes(T)

    nwk_out = strip_branch_lengths(build_newick_from_graph(T))
    if out_path is None:
        out_path = f'{ds_dir}/{divisions_dir}/merged_full_{mode}.nwk'
    with open(out_path, 'w') as f:
        f.write(nwk_out + '\n')
    final_leaves = get_leafset(T) - {'seed'}
    print(f'  → {out_path}: {len(final_leaves)} total leaves')
    return out_path

