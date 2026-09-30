"""Rerun isolation, safe result publication, and root-megablob regressions.

The integration test replaces only the Java inference search with a return of
its starting tree. Division, extraction, inference bookkeeping, and all four
merger stages (including the stage-3 subprocess) execute normally.
"""
import contextlib
import csv
import io
import json
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import pytest

from dimple import run_dimple as R
from dimple.divider import generate_k_divisions as D
from dimple.phylonet import infer_subnetworks as I
from dimple.utils.division_util import isolate_mega_blobs
from dimple.utils.network_util import get_blob_nodes, get_leafset, newick_to_nx


TOB = '(OUT,(z,(a,b,c,d,e,f)));'


def divide(out, k=3, tob=TOB):
    with contextlib.redirect_stdout(io.StringIO()):
        return D.process_division_leafsets(tob, SIZE=4, output_dir=str(out), k=k)


def test_fewer_runs_cannot_reuse_old_partitions(tmp_path):
    out = tmp_path / 'divisions'
    assert divide(out, k=3)['blobs'][0]['n_runs'] == 3
    old = out / 'blob00/run_002/subgenes-out'
    old.mkdir()
    (old / 'subnets.txt').write_text('old inference\n')
    assert divide(out, k=1)['blobs'][0]['n_runs'] == 1
    assert [label for label, _ in I.list_run_dirs(str(out))] == ['blob00/run_000']
    archived = list(tmp_path.glob('.divisions.previous-*'))
    assert len(archived) == 1
    assert (archived[0] / 'blob00/run_002/subgenes-out/subnets.txt').read_text() == 'old inference\n'


def test_matching_division_keeps_existing_inference(tmp_path):
    out = tmp_path / 'divisions'
    first = divide(out)
    cache = out / 'blob00/run_000/subgenes-out'
    cache.mkdir()
    (cache / 'subnets.txt').write_text('cached inference\n')
    with patch.object(D, '_generate_division_leafsets', side_effect=AssertionError('must resume')):
        assert divide(out) == first
    assert (cache / 'subnets.txt').read_text() == 'cached inference\n'
    assert not list(tmp_path.glob('.divisions.previous-*'))


@pytest.mark.parametrize('damage', ['metadata', 'extra_run', 'extra_blob'])
def test_changed_or_incomplete_division_is_regenerated(tmp_path, damage):
    out = tmp_path / 'divisions'
    divide(out, k=1)
    if damage == 'metadata':
        (out / 'blob00/run_000/subnetworks_output_metadata.csv').write_text('truncated')
    elif damage == 'extra_run':
        (out / 'blob00/run_999').mkdir()
    else:
        (out / 'blob99').mkdir()
    divide(out, k=1)
    assert [l for l, _ in I.list_run_dirs(str(out))] == ['blob00/run_000']
    assert list(csv.DictReader((out / 'blob00/run_000/subnetworks_output_metadata.csv').open()))
    assert not (out / 'blob99').exists()


def test_changed_tob_removes_obsolete_blob_directories(tmp_path):
    out = tmp_path / 'divisions'
    divide(out, k=1, tob='(OUT,((a,b,c),(d,e,f)));')
    assert len(I.list_run_dirs(str(out))) == 2
    divide(out, k=1)
    assert len(I.list_run_dirs(str(out))) == 1
    assert not (out / 'blob01').exists()


def test_failed_division_generation_preserves_previous_layout(tmp_path):
    out = tmp_path / 'divisions'
    divide(out, k=3)
    state = (out / D.DIVISION_STATE_FILE).read_bytes()
    with patch.object(D, '_generate_division_leafsets', side_effect=RuntimeError('generation failed')):
        with pytest.raises(RuntimeError, match='generation failed'):
            divide(out, k=1)
    assert (out / D.DIVISION_STATE_FILE).read_bytes() == state
    assert len(I.list_run_dirs(str(out))) == 3
    assert not list(tmp_path.glob('.divisions.building-*'))


def test_failed_division_publication_restores_previous_layout(tmp_path):
    out = tmp_path / 'divisions'
    divide(out, k=3)
    replace = D.os.replace

    def fail_publication(src, dest):
        if '.building-' in Path(src).name and Path(dest) == out:
            raise OSError('publication failed')
        return replace(src, dest)

    with patch.object(D.os, 'replace', side_effect=fail_publication):
        with pytest.raises(OSError, match='publication failed'):
            divide(out, k=1)
    assert len(I.list_run_dirs(str(out))) == 3


@pytest.mark.parametrize('with_seed', [True, False])
def test_root_blob_is_isolated(with_seed):
    tree = newick_to_nx('(a,b,c,d);')
    if not with_seed:
        tree.remove_node('seed')
    blobs = get_blob_nodes(tree)
    _, pieces, outside = isolate_mega_blobs(tree, blobs, real_taxa=set('abcd'))
    assert set(pieces) == set(blobs)
    assert get_leafset(pieces[blobs[0]]) == set('abcd')
    assert outside is None or not (set(outside) & set('abcd'))


def test_adjacent_blobs_at_root_are_one_megablob(tmp_path):
    out = tmp_path / 'divisions'
    summary = divide(out, k=1, tob='(OUT,(a,b,(c,d,e)));')
    assert len(summary['blobs']) == 1
    rows = list(csv.DictReader((out / 'blob00/run_000/subnetworks_output_metadata.csv').open()))
    assert set.union(*(set(r['all_leaves'].split(',')) for r in rows)) == set('abcde')


def setup_pipeline(tmp_path):
    genes, tob, base, jar = [tmp_path / s for s in ('genes.tre', 'tob.tre', 'base.tre', 'phylonet.jar')]
    genes.write_text('(OUT,(a,(b,c)));\n' * 3)
    tob.write_text('(OUT,(a,b,c));\n')
    base.write_text('(OUT,(a,(b,c)));\n')
    jar.write_text('test JAR placeholder')
    out = tmp_path / 'out'
    (out / 'divisions').mkdir(parents=True)
    (out / R.FINAL_NETWORK_NAME).write_text('previous network\n')
    (out / R.TIMINGS_NAME).write_text('{"previous": true}')
    return [str(p) for p in (genes, tob, base, jar, out)]


@pytest.mark.parametrize('stage', ['validation', 'division', 'inference', 'merger'])
def test_failure_archives_previous_final_result(tmp_path, stage):
    args = setup_pipeline(tmp_path)
    out = Path(args[-1])
    if stage == 'validation':
        Path(args[0]).unlink()
    with patch.object(R, 'process_division_leafsets', side_effect=RuntimeError('division failed') if stage == 'division' else None), \
         patch.object(R, 'infer_divisions', return_value=(0, 1) if stage == 'inference' else (1, 0)), \
         patch.object(R, 'run_full_merger', side_effect=RuntimeError('merger failed')):
        with pytest.raises((SystemExit, RuntimeError)):
            R.run_dimple(*args)
    assert not (out / R.FINAL_NETWORK_NAME).exists()
    assert not (out / R.TIMINGS_NAME).exists()
    archives = list(out.glob('.previous-result-*'))
    assert len(archives) == 1
    assert (archives[0] / R.FINAL_NETWORK_NAME).read_text() == 'previous network\n'
    assert json.loads((archives[0] / R.TIMINGS_NAME).read_text()) == {'previous': True}


def test_failed_copy_does_not_publish_partial_final(tmp_path):
    args = setup_pipeline(tmp_path)
    out = Path(args[-1])
    merged = tmp_path / 'merged.nwk'
    merged.write_text('(a,b,c);\n')

    def fail_copy(src, dest):
        Path(dest).write_text('partial')
        raise OSError('copy failed')

    with patch.object(R, 'infer_divisions', return_value=(1, 0)), \
         patch.object(R, 'run_full_merger', return_value=str(merged)), \
         patch.object(R.shutil, 'copyfile', side_effect=fail_copy):
        with pytest.raises(OSError, match='copy failed'):
            R.run_dimple(*args, skip_division=True)
    assert not (out / R.FINAL_NETWORK_NAME).exists()
    assert not (out / R.TIMINGS_NAME).exists()
    assert not list(out.glob('.dimple-result-*'))


@pytest.mark.parametrize('tob,base,size,taxa,searches', [
    ('(OUT,(a,b,c));', '(OUT,(a,(b,c)));', 12, set('abc'), 1),
    ('(OUT,(((a,b),(c,(d,e))),f,g,h));',
     '(OUT,((((a,b),(c,(d,e))),(f,g)),h));', 5, set('abcdefgh'), 1),
    ('(OUT,(a,b,(z,(c,d,e))));',
     '(OUT,((a,b),(z,(c,(d,e)))));', 12, set('abcdez'), 2),
])
def test_root_blob_pipeline_and_resume(tmp_path, tob, base, size, taxa, searches):
    args = setup_pipeline(tmp_path)
    Path(args[0]).write_text((base + '\n') * 3)
    Path(args[1]).write_text(tob + '\n')
    Path(args[2]).write_text(base + '\n')
    calls = []

    def return_start(subgene, subbase, out_path, *args, **kwargs):
        calls.append(out_path)
        Path(out_path).write_text('Inferred Network #1:\n' + Path(subbase).read_text())
        return True

    with patch.object(I, 'run_phylonet_one', side_effect=return_start):
        final = R.run_dimple(*args, size=size, k=1)
        network = newick_to_nx(Path(final).read_text().strip())
        assert get_leafset(network) == taxa | {'OUT'}
        assert nx.is_directed_acyclic_graph(network)
        assert nx.is_weakly_connected(network)
        # The root polytomy was actually resolved, not returned as the TOB.
        assert all(network.out_degree(n) <= 2 for n in network)
        if searches == 2:
            assert any(get_leafset(network, n) == set('cde') for n in network)
        assert len(calls) == searches
        again = R.run_dimple(*args, size=size, k=1)
        assert again == final and len(calls) == searches
        assert json.loads((Path(args[-1]) / R.TIMINGS_NAME).read_text())['total'] >= 0
