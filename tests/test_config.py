import pytest
from yabitsu import config


def test_protected_file_rejected():
    with pytest.raises(config.ProtectedPathError):
        config.assert_writable(config.AC_TRACKS / "nanohanadai" / "data" / "surfaces.ini")


def test_protected_dir_itself_rejected():
    with pytest.raises(config.ProtectedPathError):
        config.assert_writable(config.AC_TRACKS / "nanohanadai")


def test_protected_dir_case_insensitive():
    with pytest.raises(config.ProtectedPathError):
        config.assert_writable(config.AC_TRACKS / "NANOHANADAI" / "x.ini")


def test_new_track_allowed():
    p = config.TRACK_DIR / "s1_up" / "data" / "surfaces.ini"
    assert config.assert_writable(p) == p


def test_similar_name_allowed():
    p = config.AC_TRACKS / "nanohanadai_copy" / "x.ini"
    assert config.assert_writable(p) == p
