from copy import deepcopy
import itertools
import re

import networkx as nx

def get_leafset(net, node=None):
    leaves = {n for n in net.nodes if net.out_degree(n) == 0}
    if node is None:
        return leaves
    if net.out_degree(node) == 0:
        return {node} & leaves
    return nx.descendants(net, node) & leaves

def get_blob_nodes(T):
    return [v for v in T.nodes if T.out_degree(v) > 2]

def parse_node_info(inp_str, G, idx, parent_id, is_astral=False):
    node_info_str = inp_str.rsplit(')', maxsplit=1)[-1]
    branch_length = 1.
    branch_support = 1.
    branch_prob = 1.
    if node_info_str == "":
        node_name = f"node_{idx}"
        idx += 1
    else:
        num_colons = node_info_str.count(':')
        if num_colons == 1 and is_astral:
            supp, blen = node_info_str.split(":")
            if blen != '':
                branch_length = float(blen)
            if supp != '':
                branch_support = float(supp)
            node_name = f"node_{idx}"
            idx += 1
        else:
            if num_colons == 3:
                node_name, blen, supp, prob = node_info_str.split(":")
                if blen != '':
                    branch_length = float(blen)
                if supp != '':
                    branch_support = float(supp)
                if prob != '':
                    branch_prob = float(prob)
            elif num_colons == 2:
                node_name, blen, supp = node_info_str.split(":")
                if blen != '':
                    branch_length = float(blen)
                if supp != '':
                    branch_support = float(supp)
            elif num_colons == 1:
                node_name, blen = node_info_str.split(":")
                if blen != '':
                    branch_length = float(blen)
            else:
                node_name = node_info_str
        if node_name == "":
            node_name = f"node_{idx}"
            idx += 1
    if '#' in node_name:
        node_name = "#"+node_name.split("#", maxsplit=1)[1]

    G.add_node(node_name)
    G.add_edge(parent_id, node_name,
               length=branch_length,
               support=branch_support,
               prob=branch_prob)

    return (G, idx, node_name)

def find_splits(inp_str):
    idxs = []
    paren_lvl = 0
    for idx in range(len(inp_str)):
        if inp_str[idx] == '(':
            paren_lvl += 1
        elif inp_str[idx] == ')':
            paren_lvl -= 1
        elif inp_str[idx] == ',' and paren_lvl == 0:
            idxs.append(idx)

    return idxs

def parse_children(inp_str, net, node_id, node_name, is_astral=False):
    rem_str = inp_str.rsplit(')', maxsplit=1)[0][1:]
    split_idxs = find_splits(rem_str)
    if split_idxs == [] and rem_str == inp_str[1:]:
        return [], node_id
    new_nodes = []
    start = 0
    for idx in split_idxs:
        net, node_id, new_node_name = parse_node_info(rem_str[start:idx], net, node_id, node_name, is_astral)
        new_nodes.append((new_node_name, rem_str[start:idx]))
        start = idx+1
    net, node_id, new_node_name = parse_node_info(rem_str[start:], net, node_id, node_name, is_astral)
    new_nodes.append((new_node_name, rem_str[start:]))

    return new_nodes, node_id

def newick_to_nx(inp_str, is_astral=False):
    net = nx.DiGraph()
    net.add_node("seed")

    # Rename any node named exactly "seed" to avoid conflict with parser root
    inp_str = re.sub(r'\bseed\b', '_seed_', inp_str)

    newick_str = inp_str[:-1]
    net, node_id, node_name = parse_node_info(newick_str, net, 0, "seed")
    new_nodes, node_id = parse_children(newick_str, net, node_id, node_name, is_astral)

    while len(new_nodes) > 0:
        next_gen_nodes = []
        for node in new_nodes:
            __nodes, node_id = parse_children(node[1], net, node_id, node[0], is_astral)
            next_gen_nodes += __nodes
        new_nodes = deepcopy(next_gen_nodes)

    return net

def contract_degree2_nodes(subG):
    leaves = {n for n in subG.nodes if subG.out_degree(n) == 0}
    to_contract = [n for n in subG.nodes()
                if subG.in_degree(n) == 1
                and subG.out_degree(n) == 1
                and n not in leaves
                and n not in {'seed', 'node_0'}]
    for n in to_contract:
        parent = next(subG.predecessors(n))
        child = next(subG.successors(n))

        attrs_pn = subG[parent][n]
        attrs_nc = subG[n][child]

        new_attrs = {}
        all_keys = set(attrs_pn.keys()) | set(attrs_nc.keys())
        for k in all_keys:
            if k == 'length':
                new_attrs['length'] = attrs_pn.get('length', 0) + attrs_nc.get('length', 0)
            elif k in ('support', 'prob'):
                # keep the minimum if both exist, otherwise whichever is present
                if k in attrs_pn and k in attrs_nc:
                    new_attrs[k] = min(attrs_pn[k], attrs_nc[k])
                else:
                    new_attrs[k] = attrs_pn.get(k, attrs_nc.get(k))
            else:
                new_attrs[k] = attrs_nc.get(k, attrs_pn.get(k))

        subG.remove_edge(parent, n)
        subG.remove_edge(n, child)
        subG.remove_node(n)
        subG.add_edge(parent, child, **new_attrs)
    return subG

def prune_non_leaf_deadends(G, original_leaves):
    to_remove = [n for n in G.nodes if G.out_degree(n) == 0 and n not in original_leaves]
    G.remove_nodes_from(to_remove)
    return G

def process_degree2_seed(subG):
    leaves = {n for n in subG.nodes if subG.out_degree(n) == 0}
    to_contract = [n for n in subG.nodes()
                if subG.in_degree(n) == 1
                and subG.out_degree(n) == 1
                and n not in leaves]
    for n in to_contract:
        parent = next(subG.predecessors(n))
        child = next(subG.successors(n))

        attrs_pn = subG[parent][n]
        attrs_nc = subG[n][child]

        new_attrs = {}
        all_keys = set(attrs_pn.keys()) | set(attrs_nc.keys())
        for k in all_keys:
            if k == 'length':
                new_attrs['length'] = attrs_pn.get('length', 0) + attrs_nc.get('length', 0)
            elif k in ('support', 'prob'):
                # keep the minimum if both exist, otherwise whichever is present
                if k in attrs_pn and k in attrs_nc:
                    new_attrs[k] = min(attrs_pn[k], attrs_nc[k])
                else:
                    new_attrs[k] = attrs_pn.get(k, attrs_nc.get(k))
            else:
                new_attrs[k] = attrs_nc.get(k, attrs_pn.get(k))

        subG.remove_edge(parent, n)
        subG.remove_edge(n, child)
        subG.remove_node(n)
        subG.add_edge(parent, child, **new_attrs)
    return subG

def collapse_single_exit_blobs(g):
    g = g.copy()
    g_undirected = g.to_undirected()

    for comp in nx.biconnected_components(g_undirected):
        if len(comp) <= 2:
            continue
        # --- find all outgoing edges from this blob to outside nodes ---
        outgoing_edges = [
            (u, v) for u in comp for v in g.successors(u) if v not in comp
        ]
        outgoing_nodes = {v for _, v in outgoing_edges}
        # --- if only one distinct outgoing node, we collapse the blob ---
        if len(outgoing_nodes) == 1:
            out_target = next(iter(outgoing_nodes))
            smallest = sorted(map(str, comp))[0]
            blob_name = f"blob_{re.sub('[^A-Za-z0-9_]', '_', smallest)}_{len(comp)}"

            # --- find incoming edges from outside to blob ---
            incoming_edges = [
                (u, v) for v in comp for u in g.predecessors(v) if u not in comp
            ]

            g.remove_nodes_from(comp)
            g.add_node(blob_name)

            for u, _ in incoming_edges:
                g.add_edge(u, blob_name)
            g.add_edge(blob_name, out_target)

    return g

def remove_ancestral_parent_edges(g):
    g = g.copy()

    changed = True
    while changed:
        changed = False
        for h in list(g.nodes()):
            preds = list(g.predecessors(h))
            if len(preds) != 2:
                continue  # only consider reticulations

            p1, p2 = preds

            # Check if one parent is ancestral to the other
            # (i.e., there is a directed path p1 → ... → p2)
            if nx.has_path(g, p1, p2):
                g.remove_edge(p1, h)
                changed = True
            elif nx.has_path(g, p2, p1):
                g.remove_edge(p2, h)
                changed = True

        g = collapse_single_exit_blobs(g)
        g = contract_degree2_nodes(g)

    return g

def extract_subnetwork_by_leaves(G, leaf_set):
    subG = G.copy()
    leaf_set = set(leaf_set)
    # Validate provided leaves
    actual_leaves = {n for n in G.nodes if G.out_degree(n) == 0}
    invalid = leaf_set - actual_leaves
    if invalid:
        raise ValueError(f"Nodes not leaves in G: {invalid}")

    # Track original edges TO reticulation nodes (nodes with in-degree > 1)
    # These are the only valid parent-reticulation relationships
    original_retic_parents = {}  # reticulation_node -> set of original parents
    for node in G.nodes():
        if G.in_degree(node) > 1:  # reticulation node
            original_retic_parents[node] = set(G.predecessors(node))

    to_remove = []
    for node in subG.nodes():
        if len(set(get_leafset(subG, node)) & set(leaf_set)) == 0:
            to_remove.append(node)

    subG.remove_nodes_from(to_remove)

    # Contract degree-2 internal nodes (indegree=1, outdegree=1)
    subG = contract_degree2_nodes(subG)
    subG = collapse_single_exit_blobs(subG)
    changed = True
    while changed:
        changed = False
        for retic, orig_parents in original_retic_parents.items():
            if retic not in subG.nodes():
                continue
            current_parents = list(subG.predecessors(retic))
            for p in current_parents:
                if p not in orig_parents:
                    # Only remove if this is not the last incoming edge
                    if len(current_parents) > 1:
                        # This edge was created by contraction, not in original
                        subG.remove_edge(p, retic)
                        subG = contract_degree2_nodes(subG)
                        changed = True  # Restart the whole check
                        break
            if changed:
                break
    subG = prune_non_leaf_deadends(subG, leaf_set)
    subG = remove_ancestral_parent_edges(subG)
    subG = process_degree2_seed(subG)
    return subG

def get_biconnected_component_nodes(network):
    undirected = network.to_undirected()
    loop_nodes = set()
    for component in nx.biconnected_components(undirected):
        if len(component) > 2:
            loop_nodes.update(component)
    return loop_nodes

def clean_extended_newick(newick_str):
    # Step 1: Remove [pp1=...;pp2=...;pp3=...] annotations
    cleaned = re.sub(r"'\[pp\d=[^]]+\]'", "", newick_str)
    # Step 2: Remove inheritance probabilities (::number)
    cleaned = re.sub(r'::[-+]?[0-9.eE]+', '', cleaned)
    # Step 3: Remove branch lengths (:number)
    cleaned = re.sub(r':[-+]?[0-9.eE]+', '', cleaned)
    # Step 4: Remove edge labels after closing parentheses (e.g., ")0.99")
    cleaned = re.sub(r'\)[-+]?[0-9.eE]*', ')', cleaned)
    # Step 5: Remove standalone numbers (leftovers)
    cleaned = re.sub(r'\b[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\b', '', cleaned)
    # Step 6: Remove dangling colons
    cleaned = re.sub(r':+', '', cleaned)
    # Optional: Remove whitespace
    cleaned = re.sub(r'\s+', '', cleaned)
    return cleaned

def clean_seed_wrapper(newick_str):
    newick_str = newick_str.strip()

    # If it ends with `)seed` or `)seed;`, we strip it
    if re.search(r'\)seed;?$', newick_str):
        # Remove the seed label and final paren
        newick_str = re.sub(r'\)seed;?$', '', newick_str)

        # Also remove the first '(' to balance
        first_paren = newick_str.find('(')
        if first_paren != -1:
            newick_str = newick_str[:first_paren] + newick_str[first_paren + 1:]

    return newick_str

def format_number(val):
    try:
        f = float(val)
    except:
        return str(val)
    return str(int(f)) if f.is_integer() else str(f)

def build_newick_from_graph(G):
    # Find root
    roots = [n for n, d in G.in_degree() if d == 0]
    if not roots:
        raise ValueError("No root found in graph!")
    root = roots[0]

    # Track which reticulation nodes have been rendered
    rendered_retics = set()

    def recurse(node, parent=None):
        # Get children
        children = list(G.successors(node))

        # Get edge attributes if not root
        length = None
        prob = None
        if parent is not None and G.has_edge(parent, node):
            attrs = G[parent][node]
            length = attrs.get("length", None)
            prob = attrs.get("prob", None)

        # Leaf
        if not children:
            s = node
            if length is not None:
                s += f":{format_number(length)}"
            return s

        # Internal or retic node
        parts = []
        for c in children:
            # If child is a reticulation and we've seen it, only add label, not recurse
            if G.in_degree(c) > 1:
                if c in rendered_retics:
                    # Still add the reticulation label + edge length + prob
                    edge = G[node][c]
                    label = c
                    clen = edge.get("length", None)
                    cprob = edge.get("prob", None)
                    label += f":{format_number(clen)}" if clen is not None else ""
                    label += f"::{format_number(cprob)}" if cprob is not None else ""
                    parts.append(label)
                    continue
                else:
                    rendered_retics.add(c)
            parts.append(recurse(c, node))

        subtree = "(" + ",".join(parts) + ")"
        label = node
        if length is not None:
            label += f":{format_number(length)}"
        if G.in_degree(node) > 1 and prob is not None:
            label += f"::{format_number(prob)}"
        return subtree + label

    return clean_seed_wrapper(recurse(root)) + ";"

def count_reticulations(subG):
    return sum(1 for n in subG.nodes if subG.in_degree(n) == 2)

def enumerate_displayed_trees(g):
    # Identify reticulation nodes: nodes with >1 parent
    original_leaves = [n for n in g.nodes if g.out_degree(n) == 0]
    retics = [n for n in g.nodes if g.in_degree(n) > 1]
    retic_parent_options = {
        r: list(g.in_edges(r)) for r in retics
    }
    all_choices = list(itertools.product(*[
        options for options in retic_parent_options.values()
    ]))

    tree_list = []
    for choice in all_choices:
        G_copy = g.copy()
        for r,selected_edge in zip(retics,choice):
            for edge in list(G_copy.in_edges(r)):
                if edge != selected_edge:
                    G_copy.remove_edge(*edge)

        if not nx.is_weakly_connected(G_copy):
            continue
        G_copy = contract_degree2_nodes(G_copy)
        G_copy = prune_non_leaf_deadends(G_copy,original_leaves)
        if all(G_copy.in_degree(n) <= 1 for n in G_copy.nodes):
            tree_list.append(G_copy)

    return tree_list

def strip_branch_lengths(newick):
    """Remove branch lengths from a newick string, keeping probabilities (::) for networks.

    Outputs :0::prob for reticulation edges so that newick_to_nx parses
    the probability into the 'prob' field (not 'support').
    """
    # ':::prob' (empty length, empty support, prob -- InPhyNet and PhyloNetworks
    # write this) must be caught FIRST: the ::prob rule below would leave a stray
    # ':' and produce '::0::prob', a 4-field tag the parser mis-reads, corrupting
    # every reticulation in the network.
    s = re.sub(r':::([\d.Ee+\-]+)', r'§§\1', newick)
    # Replace :length::prob with placeholder
    s = re.sub(r':[0-9Ee+\-\.]+::([\d.Ee+\-]+)', r'§§\1', s)
    # Replace bare ::prob (no branch length before it) with placeholder
    s = re.sub(r'::([\d.Ee+\-]+)', r'§§\1', s)
    # Remove remaining :length
    s = re.sub(r':[0-9Ee+\-\.]+', '', s)
    # Restore ::prob as :0::prob so parser reads it correctly
    s = re.sub(r'§§([\d.Ee+\-]+)', r':0::\1', s)
    return s
