def test_rejects_non_tif_extension(client):
    resp = client.post(
        "/api/v1/detect",
        files={"file": ("not_a_tif.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 400
    assert "tif" in resp.json()["detail"].lower()


def test_rejects_corrupt_tif(client):
    resp = client.post(
        "/api/v1/detect",
        files={"file": ("broken.tif", b"not a real tiff", "image/tiff")},
    )
    assert resp.status_code == 400
