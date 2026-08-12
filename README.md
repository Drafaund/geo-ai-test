# Palm Counting API

Mendeteksi pohon sawit pada citra orthophoto bergeoreferensi (GeoTIFF)
menggunakan model YOLO, dan mengembalikan koordinat asli (real-world) tiap
deteksi. Hasil refactor dari `script.py` (monolitik) menjadi service FastAPI.

## Perubahan dibanding `script.py`

`script.py` tetap dibiarkan apa adanya sebagai referensi. Pipeline-nya
dipecah menjadi:

```
app/
├── main.py                 # routing FastAPI, validasi request, lapisan HTTP
├── config.py                # konfigurasi (path model/label, default threshold), bisa di-override via env
├── schemas.py                # model request/response (pydantic)
└── services/
    ├── tiling.py             # raster -> tile bergeoreferensi dengan overlap
    ├── detection.py           # load model ONNX + inferensi + penggabungan deteksi duplikat
    └── pipeline.py             # orkestrasi tiling -> deteksi -> georeferencing
```

Selain dipecah, ada beberapa perbaikan/penyederhanaan:

- **Model dimuat sekali saat startup**, bukan setiap kali dijalankan — API
  menyimpan model ONNX di memori dan memakainya ulang untuk setiap request
  (diamankan dengan lock, karena `cv2.dnn.Net.forward()` tidak aman dipanggil
  bersamaan dari beberapa thread).
- **Tiling** sekarang memakai windowed read dari `rasterio` (dengan
  `boundless=True`) alih-alih crop PIL + perhitungan bounds manual. Hasilnya
  georeferencing lebih benar untuk kasus umum (termasuk raster yang lebih
  kecil dari satu tile) dengan kode yang lebih ringkas.
- **Penggabungan deteksi duplikat** (`filter_overlap` pada kode asli)
  membutuhkan package `rtree`, yang tidak terpasang dan butuh binary hasil
  kompilasi di Windows. Diimplementasikan ulang memakai
  `shapely.strtree.STRtree` (sudah satu paket dengan `shapely>=2.0`, yang
  memang jadi dependency) dengan query jarak sungguhan (`predicate="dwithin"`)
  alih-alih pendekatan bounding-box.
- **Perbaikan threshold confidence**: kode asli membandingkan objectness
  dengan `conf_threshold`, tapi class score dibandingkan dengan angka
  hardcoded `0.25`, tidak peduli argumen `conf_threshold` yang dikirim. API
  ini memakai satu skor gabungan (`objectness * class_score`) secara
  konsisten untuk thresholding, NMS, maupun confidence yang dilaporkan.
- **`data.yaml` tidak disertakan** dalam materi ujian. Uji inferensi dummy
  mengonfirmasi output ONNX berbentuk `(1, 25200, 6)` — yaitu 4 koordinat
  box + 1 objectness + 1 class score, artinya model ini single-class.
  `data.yaml` dibuat dengan `names: {0: sawit}` (mengikuti nama raster
  contoh). Bisa di-override lewat env var `PALM_API_LABELS_PATH` kalau label
  aslinya berbeda.

## Instalasi

Daftar dependency ada di `requirements.txt`. Kalau memakai environment
conda `DataAndAI` yang dipakai saat pengembangan, semua dependency sudah
terpasang, tinggal jalankan kodenya:

```bash
conda activate DataAndAI
# atau: pip install -r requirements.txt
```

## Menjalankan

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8123
```

Dokumentasi interaktif (Swagger UI): http://127.0.0.1:8123/docs

## Testing

### 1. Otomatis (pytest)

```bash
pytest tests/ -v
```

Ada 8 test yang mencakup:

- `tests/test_health.py` — `GET /health` mengembalikan status ok dan model
  sudah termuat.
- `tests/test_tiling.py` — unit test murni untuk logika pembagian tile
  (`_tile_offsets`): raster lebih kecil dari satu tile, tile terakhir pas di
  tepi gambar, tidak ada celah antar tile.
- `tests/test_detect_validation.py` — file bukan `.tif` ditolak (400), file
  `.tif` yang isinya rusak/bukan raster valid juga ditolak (400).
- `tests/test_detect_pipeline.py` — end-to-end memakai `sawit.tif` asli:
  memastikan jumlah deteksi > 0, koordinat masuk akal (dalam bounding box
  Indonesia), dan mode `output=shapefile` menghasilkan zip berisi
  `.shp/.shx/.dbf/.prj` yang valid.

Test end-to-end ini memakai `sawit.tif` sungguhan sehingga menjalankan model
ONNX beneran (bukan mock), jadi totalnya makan waktu ±50 detik — sebagian
besar dihabiskan di dua test pipeline tersebut.

### 2. Manual lewat Swagger UI

Jalankan server (lihat bagian "Menjalankan"), buka
http://127.0.0.1:8123/docs, expand `POST /api/v1/detect`, klik **Try it
out**, pilih file `sawit.tif`, lalu **Execute**. Body respons langsung bisa
dilihat di halaman yang sama.

### 3. Manual lewat curl

```bash
# cek server hidup
curl http://127.0.0.1:8123/health

# deteksi, hasil JSON
curl -X POST "http://127.0.0.1:8123/api/v1/detect" \
  -F "file=@sawit.tif;type=image/tiff"

# deteksi, hasil shapefile (zip)
curl -X POST "http://127.0.0.1:8123/api/v1/detect?output=shapefile" \
  -F "file=@sawit.tif;type=image/tiff" -o result.zip
```

## Endpoint

### `GET /health`

```json
{"status": "ok", "model_loaded": true}
```

### `POST /api/v1/detect`

Upload file GeoTIFF via multipart form. Menjalankan pipeline lengkap
(tiling → deteksi → penghapusan duplikat → georeferencing) dan mengembalikan
hasil deteksi sebagai JSON (default) atau file shapefile terzip yang bisa
diunduh.

**Form field**
- `file` — orthophoto `.tif`/`.tiff` (wajib)

**Query parameter** (semua opsional, default diambil dari `app/config.py`)

| Parameter | Default | Keterangan |
|---|---|---|
| `tile_width` | 640 | lebar tile dalam piksel |
| `tile_height` | 640 | tinggi tile dalam piksel |
| `conf_threshold` | 0.3 | threshold skor gabungan objectness×class-score |
| `nms_threshold` | 0.3 | threshold IoU untuk non-max suppression |
| `min_distance` | 3.0 | jarak (dalam satuan CRS raster) untuk menggabungkan deteksi duplikat |
| `output` | `json` | `json` atau `shapefile` |

**Contoh**

```bash
curl -X POST "http://127.0.0.1:8123/api/v1/detect" \
  -F "file=@sawit.tif;type=image/tiff"
```

```json
{
  "filename": "sawit.tif",
  "crs": "EPSG:32749",
  "tile_width": 640,
  "tile_height": 640,
  "conf_threshold": 0.3,
  "nms_threshold": 0.3,
  "min_distance": 3.0,
  "total_detections": 499,
  "detections": [
    {
      "class_name": "sawit",
      "confidence": 0.7865703891187223,
      "x": -548560.3924535138,
      "y": 10062276.79500654,
      "lon": 101.61888958843046,
      "lat": 0.5558514282797347
    }
  ]
}
```

`x`/`y` dalam CRS asli raster (di sini UTM zone 49S); `lon`/`lat` sudah
direproyeksi ke WGS84 untuk kemudahan pembacaan.

```bash
curl -X POST "http://127.0.0.1:8123/api/v1/detect?output=shapefile" \
  -F "file=@sawit.tif;type=image/tiff" -o result.zip
```

Mengunduh `result.zip` berisi `.shp`/`.shx`/`.dbf`/`.prj`/`.cpg`.

### Penanganan error

- `400` — file bukan `.tif`/`.tiff`, atau bukan raster valid yang bisa
  dibuka `rasterio`.
- `503` — model belum selesai dimuat.

## Verifikasi yang sudah dilakukan

- `sawit.tif` (3657×3291 px, RGB, EPSG:32749, GSD ~6cm) dijalankan
  end-to-end lewat `POST /api/v1/detect`: menghasilkan 499 deteksi,
  koordinat berada dalam batas raster dan hasil reproyeksinya masuk akal
  (area Sumatra).
- Percobaan yang sama dengan `output=shapefile`: file zip dibuka ulang
  dengan `geopandas.read_file("zip://...")`, terkonfirmasi 499 fitur dan
  CRS yang benar.
- Terverifikasi respons `400` untuk nama file bukan TIFF dan file `.tif`
  yang rusak.
- Terverifikasi `GET /health` dan `GET /openapi.json` (dokumentasi Swagger).

## Keterbatasan / asumsi

- Nama class di `data.yaml` (`sawit`) adalah tebakan — tidak ada file label
  yang disertakan dalam materi ujian.
- Mengasumsikan raster north-up dan tidak terotasi (berlaku untuk
  `sawit.tif`); raster yang terotasi perlu perubahan pada konversi
  piksel→koordinat di `pipeline.py`.
- Inferensi model bersifat sinkron dan berjalan di CPU (`cv2.dnn`, sama
  seperti kode asli); raster yang sangat besar akan membuat satu request
  memakan waktu lebih lama (sawit.tif — 36 tile — memakan waktu ±23 detik
  end-to-end di mesin ini).
- **Raster berukuran sangat besar (dimensi piksel maupun ukuran file):**
  - *Pembacaan raster* sudah aman untuk ukuran berapa pun — `tiling.py`
    memakai windowed read dari `rasterio`, jadi raster tidak pernah dimuat
    penuh ke memori, hanya dibaca per-tile sesuai kebutuhan.
  - *Tapi* pipeline berjalan **sinkron dalam satu request HTTP** dan
    tile diproses **satu per satu secara berurutan** (dikunci
    `threading.Lock` di `PalmDetector`, karena `cv2.dnn.Net` tidak aman
    dipanggil paralel) — waktu proses naik linear terhadap jumlah tile.
    Raster besar (mis. orthomosaic ribuan tile) bisa membuat request
    berjalan sampai bermenit-menit, berisiko kena timeout di sisi
    client/reverse-proxy (banyak yang defaultnya 30–60 detik).
  - Tidak ada validasi/batas ukuran file upload — file yang sangat besar
    akan tetap diterima dan diproses dulu sebelum ketahuan gagal (atau
    memenuhi folder temp) daripada ditolak di awal.
  - Jika jumlah deteksi jadi sangat banyak (puluhan ribu titik), response
    `output=json` bisa menjadi besar dan lambat di-serialize/transfer,
    karena semua deteksi dikembalikan sekaligus dalam satu body JSON
    (tidak ada paginasi).

## Saran pengembangan

Kalau ke depannya API ini perlu menangani raster yang jauh lebih besar
dari `sawit.tif` (mis. orthomosaic drone skala puluhan-ratusan hektar),
berikut prioritas perubahan yang disarankan:

1. **Pola job asinkron** — pisahkan `POST /api/v1/detect` menjadi dua
   endpoint: satu untuk submit (langsung mengembalikan `job_id` tanpa
   menunggu pipeline selesai), satu lagi untuk polling status/hasil
   (`GET /api/v1/jobs/{job_id}`). Ini menghilangkan risiko timeout sama
   sekali, karena klien tidak lagi menunggu satu koneksi HTTP terbuka
   selama pipeline berjalan.
2. **Validasi ukuran upload & estimasi jumlah tile di awal** — tolak
   dengan `413`/`400` sebelum mulai memproses, bukan setelah raster
   ditulis penuh ke disk.
3. **Paralelisasi inferensi antar-tile** — saat ini tile diproses
   berurutan lewat satu lock. Bisa diganti dengan thread/process pool
   (atau batching beberapa tile sekaligus ke model) untuk memanfaatkan
   banyak core CPU, selama tetap menjaga akses ke `cv2.dnn.Net` aman.
4. **Default hasil besar ke file, bukan JSON inline** — kalau jumlah
   deteksi melewati ambang tertentu, arahkan klien untuk mengunduh
   `output=shapefile` (atau GeoJSON) alih-alih mengembalikan ribuan objek
   dalam satu response JSON.
5. **Dukungan raster terotasi** — generalisasi konversi piksel→koordinat
   di `pipeline.py` supaya tidak hanya berlaku untuk raster north-up.
