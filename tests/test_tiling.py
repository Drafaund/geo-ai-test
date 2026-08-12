from app.services.tiling import _tile_offsets


def test_single_tile_when_image_smaller_than_tile():
    assert _tile_offsets(img_size=400, tile_size=640) == [0]


def test_last_tile_flush_with_far_edge():
    offsets = _tile_offsets(img_size=1000, tile_size=640)
    assert offsets[0] == 0
    assert offsets[-1] == 1000 - 640


def test_offsets_cover_the_whole_image_without_gaps():
    offsets = _tile_offsets(img_size=3657, tile_size=640)
    assert offsets == sorted(offsets)
    assert offsets[0] == 0
    assert offsets[-1] + 640 >= 3657
    assert all(b - a <= 640 for a, b in zip(offsets, offsets[1:]))
