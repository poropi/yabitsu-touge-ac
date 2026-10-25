import configparser
from pathlib import Path

from yabitsu import ext_config, treemodels


def test_autumn_targets_broadleaf_without_replacing_tree_lists(tmp_path):
    from yabitsu.environment_seasons import write_config
    ext_config.write_part(tmp_path, 'forest', '[TREES]\nACTIVE=1\nLIST_0=trees/forest.txt\n')
    write_config(tmp_path)
    ini = configparser.ConfigParser(strict=False, interpolation=None)
    ini.read(ext_config.assemble(tmp_path))
    assert ini['TREES']['list_0'] == 'trees/forest.txt'
    assert ini['TREES']['season_autumn_0'] == 'YABITSU_AUTUMN'
    assert ini['TREES']['season_autumn_1'] == 'sugi?,hinoki?,shrub?,fern?,0'
    assert set(treemodels.SPECIES)-{'broadleaf','sapling'} == {'sugi','hinoki','shrub','fern'}   # 若木（sapling）も広葉樹なので色づく
    assert 'INPUT=YEAR_PROGRESS' in (tmp_path/'extension/parts/seasons.ini').read_text()
