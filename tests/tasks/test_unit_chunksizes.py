import pytest

from fractal_tasks_core.tasks.io_models import ChunkSizes


def test_valid_chunksize_default():
    """Test valid chunksize_default with no conflicts, but not all defaults
    set."""
    chunk_sizes = ChunkSizes(t=5, c=2)
    chunksize_default = {"t": 10, "c": 1, "y": 2160, "x": 2560}
    result = chunk_sizes.get_chunksize(chunksize_default)
    # z = 10 is a ChunkSizes default that wasn't changed
    assert result == (5, 2, 10, 2160, 2560)


def test_chunksize_default_with_overrides():
    """Test chunksize_default where some keys are overridden by ChunkSizes."""
    chunk_sizes = ChunkSizes(t=5, c=None, z=20)
    chunksize_default = {"t": 10, "c": 1, "z": 15, "y": 2160, "x": 2560}
    result = chunk_sizes.get_chunksize(chunksize_default)
    assert result == (5, 1, 20, 2160, 2560)


def test_chunksize_default_with_extra_keys():
    """Test chunksize_default containing invalid keys."""
    chunk_sizes = ChunkSizes(t=5, c=2)
    chunksize_default = {"a": 100, "c": 1, "x": 2560}
    with pytest.raises(
        ValueError, match="Invalid keys in chunksize_default: {'a'}"
    ):
        chunk_sizes.get_chunksize(chunksize_default)


def test_chunksize_empty_default():
    """Test when chunksize_default is None."""
    chunk_sizes = ChunkSizes(t=5, c=2)
    result = chunk_sizes.get_chunksize()
    assert result == (5, 2, 10)


def test_chunksize_empty_chunksizes():
    """Test when no values are set in ChunkSizes, but chunksize_default is
    valid."""
    chunk_sizes = ChunkSizes(c=None, z=None)
    chunksize_default = {"c": 1, "z": 64}
    result = chunk_sizes.get_chunksize(chunksize_default)
    assert result == (1, 64)


def test_chunksize_default_with_empty_chunksize():
    """Test empty chunksize_default with all ChunkSizes as None."""
    chunk_sizes = ChunkSizes(c=None, z=None)
    result = chunk_sizes.get_chunksize()
    assert result == ()


def test_partial_chunksize_default():
    """Test partial chunksize_default with some keys missing."""
    chunk_sizes = ChunkSizes(t=5, c=None)
    chunksize_default = {"z": 10, "y": 2160}
    result = chunk_sizes.get_chunksize(chunksize_default)
    assert result == (5, 10, 2160)
