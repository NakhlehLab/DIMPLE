"""Stage 2 (dimple.phylonet.infer_subnetworks): subnets.txt must stay aligned
with the metadata rows, because the merger reads line `subnet_idx`."""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
from collections import Counter
from dimple.phylonet import infer_subnetworks as I

META = ('subnet_idx,type,blob,all_leaves\n'
        '0,blob_group,blob00,"a,b,c"\n'
        '1,blob_group,blob00,a\n'            # too small to infer
        '2,pruned_subtree,N/A,"d,e,f"\n')


def _leaves(nwk):
    return I.get_leafset(I.newick_to_nx(nwk))


def _subnet(sub, i, r, net):
    d = sub / f'subset{i}'
    d.mkdir(exist_ok=True)
    (d / f'subnet_ret{r}.txt').write_text(net + '\n')


def _write(tmp_path):
    p = tmp_path / I.METADATA_FILENAME
    p.write_text(META)
    return str(p)


def test_rows_keep_their_line_number(tmp_path=None):
    tmp_path = Path(tempfile.mkdtemp())
    n_rows, div = I.read_division_leafsets(_write(tmp_path), max_ret=2)
    assert n_rows == 3
    assert sorted(div) == [1, 3]                 # row 1 (one leaf) is skipped, not renumbered
    assert div[1] == ({'a', 'b', 'c'}, 2)
    assert div[3] == ({'d', 'e', 'f'}, 0)        # tree-like rows get r = 0


def test_skipped_row_becomes_blank_line(tmp_path=None):
    tmp_path = Path(tempfile.mkdtemp())
    n_rows, div = I.read_division_leafsets(_write(tmp_path), max_ret=1)
    for i, net in ((1, '((a,b),c);'), (3, '((d,e),f);')):
        _subnet(tmp_path, i, div[i][1], net)
    I.combine_subnets(str(tmp_path), n_rows, div)
    lines = (tmp_path / 'subnets.txt').read_text().split('\n')
    assert len(lines) == 4 and lines[1] == '' and lines[3] == ''     # 3 rows + final newline
    assert _leaves(lines[0]) == {'a', 'b', 'c'} and _leaves(lines[2]) == {'d', 'e', 'f'}


def test_missing_phylonet_output_writes_nothing(tmp_path=None):
    tmp_path = Path(tempfile.mkdtemp())
    n_rows, div = I.read_division_leafsets(_write(tmp_path), max_ret=1)
    _subnet(tmp_path, 1, 1, '((a,b),c);')
    (tmp_path / 'subnets.txt').write_text('stale\n')
    try:
        I.combine_subnets(str(tmp_path), n_rows, div)
        raise AssertionError('a missing division must abort')
    except RuntimeError:
        pass
    assert (tmp_path / 'subnets.txt').read_text() == 'stale\n'   # untouched; infer_run clears it up front


def test_subnet_idx_must_equal_row_position(tmp_path=None):
    tmp_path = Path(tempfile.mkdtemp())
    p = tmp_path / I.METADATA_FILENAME
    p.write_text('subnet_idx,type,blob,all_leaves\n5,blob_group,blob00,"a,b,c"\n')
    try:
        I.read_division_leafsets(str(p), max_ret=1)
        raise AssertionError('misnumbered metadata must be refused')
    except ValueError:
        pass



def test_wrong_leafset_is_refused(tmp_path=None):
    tmp_path = Path(tempfile.mkdtemp())
    n_rows, div = I.read_division_leafsets(_write(tmp_path), max_ret=1)
    _subnet(tmp_path, 1, 1, '((a,b),c);')
    _subnet(tmp_path, 3, 0, '((a,b),c);')
    try:
        I.combine_subnets(str(tmp_path), n_rows, div)
        raise AssertionError('network on the wrong taxa must be refused')
    except RuntimeError:
        pass


# ---- infer_run, with PhyloNet replaced by "return the starting tree" -------

GENES = '((((t_1,2),c),(d,e)),OUT);\n' * 3 + '((t_1,2),c);\n'       # last one: no OUT
BASE = '((((t_1:1,2:1)0.9:1,c:1):1,(d:1,e:1):1):1,OUT:1);\n'


def _setup():
    run = Path(tempfile.mkdtemp())
    (run / I.METADATA_FILENAME).write_text(
        'subnet_idx,type,blob,all_leaves\n0,blob_group,blob00,"t_1,2,c,d"\n')
    (run / 'genes.tre').write_text(GENES)
    (run / 'base.tre').write_text(BASE)
    (run / 'jar').write_text('')
    return run


def _fake_phylonet(calls, ok=True):
    def fake(nexus, subbase, out_path, max_ret, jar, **kw):
        calls.append(out_path)
        # the NEXUS handed over must ask PhyloNet for exactly this bound
        assert f'InferNetwork_MPL (all) {max_ret} ' in open(nexus).read(), (nexus, max_ret)
        if os.path.exists(out_path):
            os.remove(out_path)
        if ok:
            with open(out_path, 'w') as f:
                f.write('Inferred Network #1:\n' + open(subbase).read())
        return ok
    return fake


def _run(run, **kw):
    return I.infer_run(str(run), str(run / 'genes.tre'), str(run / 'base.tre'),
                       str(run / 'jar'), **kw)


def test_infer_run_labels_resume_and_stale_outputs():
    run, calls, real = _setup(), [], I.run_phylonet_one
    try:
        I.run_phylonet_one = _fake_phylonet(calls)
        assert _run(run) is True
        sub = run / 'subgenes-out'
        # underscore and all-digit taxon names survive; the OUT-less gene tree is dropped
        assert _leaves((sub / 'subnets.txt').read_text().strip()) == {'t_1', '2', 'c', 'd'}
        assert open(sub / 'subset1' / 'base_tree.tre').read().strip() == '(OUT,(((t_1,2),c),d));'
        assert (sub / 'subset1' / 'leaf_subset.txt').read_text().split() == ['2', 'c', 'd', 't_1']
        assert (sub / 'subset1' / 'tmp_mpl.nex').read_text().count('Tree gt') == 3
        assert _run(run) is True and len(calls) == 1              # same inputs: resumed
        assert _run(run, max_ret=2) is True and len(calls) == 2   # other bound: redone
        # a failing rerun must not fall back on the previous phylonet_out / subnets.txt
        I.run_phylonet_one = _fake_phylonet(calls, ok=False)
        assert _run(run, force=True) is False
        assert not (sub / 'subnets.txt').exists()
        I.run_phylonet_one = _fake_phylonet(calls)
        assert _run(run, max_ret=2) is True and len(calls) == 4   # so this is not "already done"
    finally:
        I.run_phylonet_one = real


def test_same_size_edit_of_an_input_is_not_resumed():
    run, calls, real = _setup(), [], I.run_phylonet_one
    try:
        I.run_phylonet_one = _fake_phylonet(calls)
        assert _run(run) is True
        st = os.stat(run / 'genes.tre')
        (run / 'genes.tre').write_text(GENES.replace('((t_1,2),c)', '((t_1,c),2)'))
        os.utime(run / 'genes.tre', (st.st_atime, st.st_mtime))
        assert _run(run) is True and len(calls) == 2
    finally:
        I.run_phylonet_one = real


def test_changed_starting_tree_invalidates_resume_in_same_process():
    run, calls = _setup(), []
    with patch.object(I, 'run_phylonet_one', _fake_phylonet(calls)):
        assert _run(run) is True
        base = run / 'base.tre'
        st = base.stat()
        base.write_text(BASE.replace('(t_1:1,2:1)0.9:1,c:1',
                                     '(t_1:1,c:1)0.9:1,2:1'))
        os.utime(base, (st.st_atime, st.st_mtime))
        assert _run(run) is True and len(calls) == 2
        assert '(t_1,c)' in (run / 'subgenes-out/subset1/base_tree.tre').read_text()


def test_changed_jar_invalidates_resume():
    run, calls = _setup(), []
    with patch.object(I, 'run_phylonet_one', _fake_phylonet(calls)):
        assert _run(run) is True
        (run / 'jar').write_text('a different PhyloNet build')
        assert _run(run) is True and len(calls) == 2
        assert _run(run) is True and len(calls) == 2


def test_shared_inputs_are_hashed_once_per_invocation():
    run, calls = _setup(), []
    for name in ('run_000', 'run_001'):
        dest = run / 'blob00' / name
        dest.mkdir(parents=True)
        (dest / I.METADATA_FILENAME).write_text((run / I.METADATA_FILENAME).read_text())
    paths = [str(run / f) for f in ('genes.tre', 'base.tre', 'jar')]
    with patch.object(I, 'run_phylonet_one', _fake_phylonet(calls)), \
         patch.object(I, '_file_id', wraps=I._file_id) as hashing:
        assert I.infer_divisions(str(run), *paths, parallel=2) == (2, 0)
        assert Counter(c.args[0] for c in hashing.call_args_list) == Counter(paths)
        assert len(calls) == 2
        hashing.reset_mock()
        (run / 'jar').write_text('updated JAR')
        assert I.infer_divisions(str(run), *paths, parallel=2) == (2, 0)
        assert Counter(c.args[0] for c in hashing.call_args_list) == Counter(paths)
        assert len(calls) == 4


def test_two_taxon_division_keeps_its_gene_trees():
    out = Path(tempfile.mkdtemp())
    (out / 'g.tre').write_text('(OUT,(a,b));\n(OUT,(a,(b,c)));\n(OUT,(a,c));\n')
    trees, skipped = I.extract_subgene_trees(str(out / 'g.tre'), {1: ({'a', 'b'}, 0)})[1]
    assert skipped == 1 and len(trees) == 2


def test_output_dir_must_be_a_plain_name():
    run = _setup()
    try:
        I.infer_divisions(str(run), str(run / 'genes.tre'), str(run / 'base.tre'),
                          str(run / 'jar'), subgenes_out_dir=str(run))
        raise AssertionError('a shared output path must be refused')
    except SystemExit:
        pass


def test_base_tree_missing_a_taxon_or_not_binary_is_an_error():
    for base in ('(((t_1,2),d),OUT);\n', '((t_1,2,c,d),OUT);\n'):
        run = _setup()
        (run / 'base.tre').write_text(base)
        try:
            _run(run)
            raise AssertionError('bad base tree accepted: ' + base)
        except ValueError:
            pass



# ---- per-subset inference (--only / --subset) and --assemble -----------------

def _setup_two():
    run = _setup()
    (run / I.METADATA_FILENAME).write_text(
        'subnet_idx,type,blob,all_leaves\n0,blob_group,blob00,"t_1,2,c"\n1,blob_group,blob00,"c,d,e"\n')
    return run


def test_every_bound_is_kept_and_assembly_picks_one_per_subset():
    run, calls = _setup_two(), []
    sub = run / 'subgenes-out'
    with patch.object(I, 'run_phylonet_one', _fake_phylonet(calls)):
        assert _run(run, only={1}, max_ret=2) is True
        assert _run(run, only={1}, max_ret=1) is True          # a second bound: both results kept
        assert _run(run, only={2}, max_ret=0) is True
        assert not (sub / 'subnets.txt').exists()               # per-subset runs never assemble
        assert sorted(p.name for p in (sub / 'subset1').glob('subnet_ret*')) == ['subnet_ret1.txt', 'subnet_ret2.txt']
        assert sorted(p.name for p in (sub / 'subset2').glob('subnet_ret*')) == ['subnet_ret0.txt']
        assert sorted(p.name for p in (sub / 'subset1').iterdir()) == \
            ['base_tree.tre', 'leaf_subset.txt', 'mpl_ret1.log', 'mpl_ret2.log',
             'subnet_ret1.txt', 'subnet_ret2.txt', 'tmp_mpl.nex']
        assert _run(run, only={1}, max_ret=2) is True and len(calls) == 3   # same (subset, bound): skipped
        # a bound with no result is refused, naming what exists
        assert _run(run, assemble={1: 3, 2: 0}) is False and not (sub / 'subnets.txt').exists()
        assert _run(run, assemble={1: 2, 2: 0}) is True and len(calls) == 3
        lines = (sub / 'subnets.txt').read_text().split('\n')
        assert _leaves(lines[0]) == {'t_1', '2', 'c'} and _leaves(lines[1]) == {'c', 'd', 'e'}
        assert I._load_inputs(str(sub))['assembled']['max_ret'] == {'1': 2, '2': 0}
        assert _run(run, assemble={1: 1, 2: 0}) is True           # re-assembled from the other result
        assert I._load_inputs(str(sub))['assembled']['max_ret'] == {'1': 1, '2': 0}
        # a full run afterwards with one bound wipes the mix and redoes all
        assert _run(run, max_ret=1) is True and len(calls) == 5
        assert I._load_inputs(str(sub))['assembled']['max_ret'] == {'1': 1, '2': 1}
        assert [sorted(p.name for p in (sub / f'subset{i}').glob('subnet_ret*')) for i in (1, 2)] == \
            [['subnet_ret1.txt'], ['subnet_ret1.txt']]


def test_unknown_subset_is_refused():
    run = _setup_two()
    with patch.object(I, 'run_phylonet_one', _fake_phylonet([])):
        for kw in ({'only': {7}}, {'assemble': {7: 1}}):
            try:
                _run(run, **kw)
                raise AssertionError(f'unknown subset accepted: {kw}')
            except SystemExit:
                pass


def test_stale_result_from_other_inputs_is_not_assembled():
    run, calls = _setup_two(), []
    sub = run / 'subgenes-out'
    with patch.object(I, 'run_phylonet_one', _fake_phylonet(calls)):
        assert _run(run, only={1}) is True
        (run / 'genes.tre').write_text(GENES.replace('(d,e)', '(e,d)'))      # inputs change
        assert _run(run, only={2}) is True
        assert _run(run, assemble={1: 1, 2: 1}) is False                    # subset 1 is stale
        assert not (sub / 'subnets.txt').exists()
        assert _run(run, only={1}) is True and len(calls) == 3              # stale: re-inferred, not skipped
        assert _run(run, assemble={1: 1, 2: 1}) is True


def test_subset_needs_a_single_run_dir_and_list_mode_is_read_only():
    run = _setup()
    for name in ('run_000', 'run_001'):
        dest = run / 'blob00' / name
        dest.mkdir(parents=True)
        (dest / I.METADATA_FILENAME).write_text((run / I.METADATA_FILENAME).read_text())
    paths = [str(run / f) for f in ('genes.tre', 'base.tre', 'jar')]
    for kw in ({'only_divisions': [1]}, {'assemble': [1, 2]}, {'only_runs': ['blob00/run_009']}):
        try:
            I.infer_divisions(str(run), *paths, **kw)
            raise AssertionError(f'accepted: {kw}')
        except SystemExit:
            pass
    assert I.infer_divisions(str(run), *paths, list_only=True) == (0, 0)
    assert not list(run.rglob('subgenes-out'))


def test_invalid_phylonet_output_is_not_stamped_as_done():
    run, calls = _setup_two(), []
    sub = run / 'subgenes-out'

    def bad(subgene, subbase, out_path, *a, **kw):
        calls.append(out_path)
        Path(out_path).write_text('PhyloNet crashed before printing a network\n')
        return True                                     # exit code 0, no network

    with patch.object(I, 'run_phylonet_one', bad):
        assert _run(run, only={1}) is False
        assert not (sub / 'subset1' / 'subnet_ret1.txt').exists() and I._stamps(str(sub), 1) == {}
    with patch.object(I, 'run_phylonet_one', _fake_phylonet(calls)):
        assert _run(run, only={1}) is True and len(calls) == 2   # retried, not skipped


def test_assemble_via_infer_divisions_maps_the_list_onto_the_subsets():
    run, calls = _setup_two(), []
    paths = [str(run / f) for f in ('genes.tre', 'base.tre', 'jar')]
    rd = run / 'blob00' / 'run_000'; rd.mkdir(parents=True)
    (rd / I.METADATA_FILENAME).write_text((run / I.METADATA_FILENAME).read_text())
    sub = rd / 'subgenes-out'
    with patch.object(I, 'run_phylonet_one', _fake_phylonet(calls)):
        assert I.infer_divisions(str(run), *paths, only_runs=['blob00/run_000'], only_divisions=[1], max_ret=2) == (1, 0)
        assert I.infer_divisions(str(run), *paths, only_runs=['blob00/run_000'], only_divisions=[2], max_ret=0) == (1, 0)
        for bad in ([2], [2, 0, 1]):                          # wrong length: refused
            try:
                I.infer_divisions(str(run), *paths, assemble=bad, only_runs=['blob00/run_000'])
                raise AssertionError(f'assembled with {len(bad)} bounds for 2 subsets')
            except SystemExit:
                pass
        assert I.infer_divisions(str(run), *paths, assemble=[2, 0], only_runs=['blob00/run_000']) == (1, 0)
        assert I._load_inputs(str(sub))['assembled']['max_ret'] == {'1': 2, '2': 0}
        assert I.infer_divisions(str(run), *paths, assemble=[1, 0], only_runs=['blob00/run_000']) == (0, 1)  # subset 1 has no bound-1 result
        assert len(calls) == 2



def test_concurrent_subset_invocations_keep_both_records():
    from concurrent.futures import ThreadPoolExecutor
    run, calls = _setup_two(), []
    sub = run / 'subgenes-out'
    with patch.object(I, 'run_phylonet_one', _fake_phylonet(calls)):
        with ThreadPoolExecutor(2) as pool:
            r = list(pool.map(lambda i: _run(run, only={i}), [1, 2]))
        assert r == [True, True]
        assert I._stamps(str(sub), 1) and I._stamps(str(sub), 2)      # neither record was lost
        assert _run(run, assemble={1: 1, 2: 1}) is True
        assert (sub / 'subset1' / 'tmp_mpl.nex').exists()
        assert not list((sub / 'subset1').glob('tmp_mpl.*.nex'))


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t(); print('PASS ', t.__name__)
    print(f'\n{len(tests)} passed')
