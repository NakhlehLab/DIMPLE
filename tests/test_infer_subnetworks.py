"""Stage 2 (dimple.phylonet.infer_subnetworks): subnets.txt must stay aligned
with the metadata rows, because the merger reads line `subnet_idx`."""
import os
import tempfile
from pathlib import Path
from dimple.phylonet import infer_subnetworks as I

META = ('subnet_idx,type,blob,all_leaves\n'
        '0,blob_group,blob00,"a,b,c"\n'
        '1,blob_group,blob00,a\n'            # too small to infer
        '2,pruned_subtree,N/A,"d,e,f"\n')


def _leaves(nwk):
    return I.get_leafset(I.newick_to_nx(nwk))


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
    for i, net in ((1, '((a,b),(c,OUT));'), (3, '((d,e),(f,OUT));')):
        (tmp_path / f'phylonet_out_{i}.txt').write_text(f'Inferred Network #1:\n{net}\n')
    I.combine_subnets(str(tmp_path), n_rows, div)
    lines = (tmp_path / 'subnets.txt').read_text().split('\n')
    assert len(lines) == 4 and lines[1] == '' and lines[3] == ''     # 3 rows + final newline
    assert _leaves(lines[0]) == {'a', 'b', 'c'} and _leaves(lines[2]) == {'d', 'e', 'f'}


def test_missing_phylonet_output_writes_nothing(tmp_path=None):
    tmp_path = Path(tempfile.mkdtemp())
    n_rows, div = I.read_division_leafsets(_write(tmp_path), max_ret=1)
    (tmp_path / 'phylonet_out_1.txt').write_text('Inferred Network #1:\n((a,b),(c,OUT));\n')
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
    (tmp_path / 'phylonet_out_1.txt').write_text('Inferred Network #1:\n((a,b),(c,OUT));\n')
    (tmp_path / 'phylonet_out_3.txt').write_text('Inferred Network #1:\n((a,b),(c,OUT));\n')
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
    def fake(subgene, subbase, out_path, max_ret, jar, **kw):
        calls.append(out_path)
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
        assert open(sub / 'subbase_1_ret1.tree').read().strip() == '(OUT,(((t_1,2),c),d));'
        assert len((sub / 'subgeneset_1_ret1.txt').read_text().split()) == 3
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
        (run / 'genes.tre').write_text(GENES.replace('(d,e)', '(e,d)'))
        os.utime(run / 'genes.tre', (st.st_atime, st.st_mtime))
        I._file_id.cache_clear()
        assert _run(run) is True and len(calls) == 2
    finally:
        I.run_phylonet_one = real


def test_two_taxon_division_keeps_its_gene_trees():
    out = Path(tempfile.mkdtemp())
    (out / 'g.tre').write_text('(OUT,(a,b));\n(OUT,(a,(b,c)));\n(OUT,(a,c));\n')
    skipped = I.extract_subgene_trees(str(out / 'g.tre'), {1: ({'a', 'b'}, 0)}, str(out))
    assert skipped == {1: 1}
    assert len((out / 'subgeneset_1_ret0.txt').read_text().split()) == 2


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


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t(); print('PASS ', t.__name__)
    print(f'\n{len(tests)} passed')
