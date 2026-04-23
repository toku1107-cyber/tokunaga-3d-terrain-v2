"""(有)徳永測量 社屋 3D地形ビューア VER2（広域1km四方） 生成スクリプト
=== 機能 ===
- .laz 点群 → 標高グリッド（地理座標系で直接ビニング）
- 10K TIF オルソ → 同一座標系で自動位置合わせ（rot=0）
- 6カラースキーム / 5表示モード / 方角・高度プリセット
- 背景白黒切替 / localStorage保存 / クリック標高表示
=== 方針 ===
Metashape全出力は同一座標系（JGD2011 平面直角座標第Ⅱ系 EPSG:6670）
→ rot=0, scale=100%, offset=0 で自動位置合わせ
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import base64
import io
import time
import numpy as np
from pathlib import Path

# ---- 設定 ----
BASE_DIR = Path(__file__).parent
LAZ_PATH = Path(r'U:\Metashape 成果データ\2025 12 09 徳永測量\2025 12 09 徳永測量-写真点群.laz')
ORTHO_TIF = Path(r'U:\Metashape 成果データ\2025 12 09 徳永測量\数値写真　サイズ10K\2025 12 09 徳永測量-数値写真　サイズ10K.tif')
ORTHO_JPG_FALLBACK = Path(r'U:\Metashape 成果データ\2025 12 09 徳永測量\数値写真　JPEG 小サイズ出力\2025 12 09 徳永測量-数値写真　JPEG 小サイズ出力.jpg')
OUTPUT_HTML = BASE_DIR / '徳永測量社屋VER2_3dviewer.html'

ORTHO_MAX_PX = 5000  # オルソ画像の最大幅（ピクセル）
ORTHO_JPEG_QUALITY = 85  # JPEG品質

t0 = time.time()

# ==============================================================
# Phase 1: .laz 点群から標高データ抽出（地理座標系・チャンク読込み）
# ==============================================================
print('Phase 1/4: .laz 点群読取り中...')

import laspy

# ヘッダから座標範囲を先に取得（メモリ節約）
with laspy.open(str(LAZ_PATH)) as reader:
    header = reader.header
    n_points = header.point_count
    x_min, x_max = float(header.mins[0]), float(header.maxs[0])
    y_min, y_max = float(header.mins[1]), float(header.maxs[1])
    z_min_raw, z_max_raw = float(header.mins[2]), float(header.maxs[2])

print(f'  総ポイント数: {n_points:,}')
print(f'  LAZ撮影範囲: X[{x_min:.3f},{x_max:.3f}] ({x_max-x_min:.1f}m) / Y[{y_min:.3f},{y_max:.3f}] ({y_max-y_min:.1f}m)')
print(f'  Z: [{z_min_raw:.3f}, {z_max_raw:.3f}]')

# 【VER2】シーン範囲を中心±500m（1km四方）に強制上書き（広域モード）
SCENE_HALF = 500.0  # 半径(m)
_laz_cx = (x_min + x_max) / 2
_laz_cy = (y_min + y_max) / 2
print(f'  LAZ撮影範囲中心: X={_laz_cx:.3f}, Y={_laz_cy:.3f}')
x_min = _laz_cx - SCENE_HALF
x_max = _laz_cx + SCENE_HALF
y_min = _laz_cy - SCENE_HALF
y_max = _laz_cy + SCENE_HALF
MW = x_max - x_min  # 地理的幅（メートル・VER2: 1000m固定）
MH = y_max - y_min  # 地理的高さ（メートル・VER2: 1000m固定）
print(f'  VER2シーン範囲: X[{x_min:.1f},{x_max:.1f}] / Y[{y_min:.1f},{y_max:.1f}] ({MW:.0f}m×{MH:.0f}m)')

# グリッドサイズ: アスペクト比に合わせて自動計算
GRID_X = 1050
GRID_Y = int(round(GRID_X * MH / MW))
print(f'  グリッド: {GRID_X} x {GRID_Y} (自動計算)')

# チャンクごとにビニング（メモリ効率的）
GX, GY = GRID_X, GRID_Y
grid = np.full(GX * GY, -np.inf, dtype=np.float64)

CHUNK_SIZE = 5_000_000  # 500万点ずつ
print(f'  チャンク読込み ({CHUNK_SIZE:,}点ずつ)...')
total_read = 0
with laspy.open(str(LAZ_PATH)) as reader:
    for chunk in reader.chunk_iterator(CHUNK_SIZE):
        cx = np.array(chunk.x, dtype=np.float64)
        cy = np.array(chunk.y, dtype=np.float64)
        cz = np.array(chunk.z, dtype=np.float64)
        _chunk_n = len(cx)
        # 【VER2】シーン範囲(1km四方)外の点は除外
        _mask_in = (cx >= x_min) & (cx <= x_max) & (cy >= y_min) & (cy <= y_max)
        cx, cy, cz = cx[_mask_in], cy[_mask_in], cz[_mask_in]
        if len(cx) > 0:
            ix = np.clip(((cx - x_min) / (MW + 1e-10) * (GX - 1)).astype(np.int32), 0, GX - 1)
            iy = np.clip(((cy - y_min) / (MH + 1e-10) * (GY - 1)).astype(np.int32), 0, GY - 1)
            np.maximum.at(grid, iy * GX + ix, cz)
        total_read += _chunk_n
        print(f'\r    {total_read:,} / {n_points:,} ({total_read/n_points*100:.0f}%)', end='', flush=True)
print()

grid = grid.reshape(GY, GX)

# ==============================================================
# Phase 2: 補間 + 平滑化
# ==============================================================
print(f'Phase 2/4: 補間・平滑化中...')

valid = grid > -np.inf
original_valid = valid.copy()  # ★ 補間前のオリジナルマスクを保持（メッシュ境界用）
filled = valid.sum()
print(f'  データあり: {filled:,} / {GX*GY:,} ({filled/(GX*GY)*100:.1f}%)')

# 空セル補間（反復近傍平均）— 高さ値は補間するが、マスクはオリジナルを使用
print('  空セル補間中...')
for iteration in range(50):
    empty = ~valid
    if not empty.any():
        break
    padded = np.pad(grid, 1, mode='constant', constant_values=-np.inf)
    pv = np.pad(valid.astype(np.float64), 1, mode='constant', constant_values=0)
    # 4近傍＋4対角 = 8近傍の加重平均
    s = np.zeros_like(grid)
    c = np.zeros_like(grid)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            patch = padded[1+dy:GY+1+dy, 1+dx:GX+1+dx]
            pmask = pv[1+dy:GY+1+dy, 1+dx:GX+1+dx]
            s += np.where(pmask > 0, patch, 0)
            c += pmask
    fill_mask = empty & (c >= 2)
    if not fill_mask.any():
        break
    grid[fill_mask] = s[fill_mask] / c[fill_mask]
    valid |= fill_mask

print(f'  補間後: {valid.sum():,} セル ({valid.sum()/(GX*GY)*100:.1f}%)')

# 軽い平滑化（3x3、3回反復）
print('  平滑化中...')
smooth = grid.copy()
for _ in range(3):
    padded = np.pad(smooth, 1, mode='edge')
    pv = np.pad(valid.astype(np.float64), 1, mode='constant', constant_values=0)
    s = np.zeros_like(smooth)
    c = np.zeros_like(smooth)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            p = padded[1+dy:GY+1+dy, 1+dx:GX+1+dx]
            pm = pv[1+dy:GY+1+dy, 1+dx:GX+1+dx]
            s += np.where(pm > 0, p, 0)
            c += pm
    ok = valid & (c > 0)
    smooth[ok] = s[ok] / c[ok]
grid = smooth

# 統計量
h_valid = grid[valid]
h_min = float(h_valid.min())
h_max = float(h_valid.max())
h_mean = float(h_valid.mean())
h_p5 = float(np.percentile(h_valid, 5))
h_p50 = float(np.percentile(h_valid, 50))
h_p95 = float(np.percentile(h_valid, 95))
print(f'  標高: [{h_min:.4f}, {h_max:.4f}] 平均{h_mean:.4f}')

# 未補間セルを平均値で埋める（NaN防止: Three.jsのcomputeBoundingSphere対策）
grid[~valid] = h_mean

# base64エンコード（flipud: 行0=北(ymax)）
# ★ マスクはオリジナル（補間前）を使用 → 不規則な地形境界を維持
grid_flipped = np.flipud(grid).astype(np.float32)
mask_flipped = np.flipud(original_valid).astype(np.uint8)
h_b64 = base64.b64encode(grid_flipped.tobytes()).decode()
m_b64 = base64.b64encode(mask_flipped.tobytes()).decode()
print(f'  標高base64: {len(h_b64):,} chars / マスク: {len(m_b64):,} chars')

# ==============================================================
# Phase 3: オルソ画像処理（同一座標系シンプルクロップ方式）
# ==============================================================
# .lazとGeoTIFは同一座標系（EPSG:6670）→ rot=0, scale=100%
# 点群範囲に対応するオルソ領域を切り出してJPEG化するだけ
print('Phase 3/4: オルソ画像処理中...')

from PIL import Image

ortho_b64 = ''
ortho_source = 'none'

if ORTHO_TIF.exists():
    print(f'  10K TIF読込: {ORTHO_TIF.name}')
    img = Image.open(str(ORTHO_TIF))
    orig_w, orig_h = img.size
    print(f'  元サイズ: {orig_w}x{orig_h} ({img.mode})')

    # GeoTIFF座標読取り
    tags = img.tag_v2
    pixel_size = tags.get(33550, None)  # (x_res, y_res, 0)
    origin = tags.get(33922, None)       # (0, 0, 0, x_origin, y_origin, 0)

    if pixel_size and origin:
        px_sz = pixel_size[0]
        ortho_ox = origin[3]   # 画像左端のX座標
        ortho_oy = origin[4]   # 画像上端のY座標（GeoTIFはY軸上→下=北→南）
        print(f'  GeoTIFF: origin=({ortho_ox:.3f}, {ortho_oy:.3f}), px={px_sz:.6f}m')

        # 点群範囲 → オルソ上のピクセル座標に変換
        # X: 左端=ortho_ox → 右に+   Y: 上端=ortho_oy → 下に-
        crop_left   = (x_min - ortho_ox) / px_sz
        crop_right  = (x_max - ortho_ox) / px_sz
        crop_top    = (ortho_oy - y_max) / px_sz  # Y軸反転: 北(ymax)が上
        crop_bottom = (ortho_oy - y_min) / px_sz

        # 整数化（少し広めにクロップして端を確保）
        cl = max(0, int(np.floor(crop_left)))
        cr = min(orig_w, int(np.ceil(crop_right)))
        ct = max(0, int(np.floor(crop_top)))
        cb = min(orig_h, int(np.ceil(crop_bottom)))
        print(f'  クロップ: ({cl}, {ct}) → ({cr}, {cb}) = {cr-cl}x{cb-ct}px')

        # 点群範囲の座標カバー率
        cover_x = (cr - cl) * px_sz / MW * 100
        cover_y = (cb - ct) * px_sz / MH * 100
        print(f'  カバー率: X={cover_x:.1f}%, Y={cover_y:.1f}%')
    else:
        print('  GeoTIFFタグなし → オルソ全域を使用')
        cl, ct, cr, cb = 0, 0, orig_w, orig_h

    # RGBA→RGB変換（黒背景合成: alpha=0領域を黒にする）
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    # 点群範囲でクロップ
    cropped = img.crop((cl, ct, cr, cb))
    print(f'  クロップ後: {cropped.size[0]}x{cropped.size[1]}')

    # リサイズ（JPEG埋込み用）
    cw, ch = cropped.size
    if max(cw, ch) > ORTHO_MAX_PX:
        ratio = ORTHO_MAX_PX / max(cw, ch)
        new_w = int(cw * ratio)
        new_h = int(ch * ratio)
        cropped = cropped.resize((new_w, new_h), Image.LANCZOS)
        print(f'  リサイズ: {new_w}x{new_h}')

    # JPEG化 → base64
    buf = io.BytesIO()
    cropped.save(buf, format='JPEG', quality=ORTHO_JPEG_QUALITY, optimize=True)
    ortho_bytes = buf.getvalue()
    ortho_b64 = base64.b64encode(ortho_bytes).decode()
    ortho_source = f'GeoTIFクロップ ({cropped.size[0]}x{cropped.size[1]}, rot=0, Q{ORTHO_JPEG_QUALITY})'
    print(f'  JPEG: {len(ortho_bytes)/1024:.0f} KB → base64: {len(ortho_b64):,} chars')

    # 診断画像保存
    cropped.save(str(BASE_DIR / '_diag_cropped_ortho.png'))
    print(f'  診断画像: _diag_cropped_ortho.png')

    del img, cropped

elif ORTHO_JPG_FALLBACK.exists():
    print(f'  フォールバック: {ORTHO_JPG_FALLBACK.name}')
    with open(str(ORTHO_JPG_FALLBACK), 'rb') as f:
        ortho_bytes = f.read()
    ortho_b64 = base64.b64encode(ortho_bytes).decode()
    ortho_source = f'JPEG小 ({len(ortho_bytes)/1024:.0f} KB)'
    print(f'  JPEG: {len(ortho_bytes)/1024:.0f} KB → base64: {len(ortho_b64):,} chars')
else:
    print('  ⚠ オルソ画像なし — オルソモード無効')

# ==============================================================
# Phase 3.5: 国土地理院タイル取得（周辺コンテキスト用）
# ==============================================================
print('Phase 3.5: 国土地理院タイル取得中...')

gsi_b64 = ''
gsi_std_b64 = ''
gsi_enabled = False
gsi_scene_corners = {}
gsi_dem_ok = False

try:
    from pyproj import Transformer as _Transformer
    import requests as _req
    import math as _math

    _tr = _Transformer.from_crs("EPSG:6670", "EPSG:4326", always_xy=True)
    _tr_inv = _Transformer.from_crs("EPSG:4326", "EPSG:6670", always_xy=True)

    # 【VER2】拡張0 — シーン範囲=タイル取得範囲（1km四方そのまま）
    _ext = 0.0
    _gx0, _gy0 = x_min - _ext, y_min - _ext
    _gx1, _gy1 = x_max + _ext, y_max + _ext

    # 4隅 → WGS84
    _corners_wgs = [_tr.transform(ex, ny) for ex, ny in [(_gx0,_gy0),(_gx0,_gy1),(_gx1,_gy0),(_gx1,_gy1)]]
    _lat_min_w = min(c[1] for c in _corners_wgs)
    _lat_max_w = max(c[1] for c in _corners_wgs)
    _lon_min_w = min(c[0] for c in _corners_wgs)
    _lon_max_w = max(c[0] for c in _corners_wgs)
    _lon_c, _lat_c = _tr.transform((x_min+x_max)/2, (y_min+y_max)/2)
    print(f'  中心: lat={_lat_c:.6f}, lon={_lon_c:.6f}')
    print(f'  WGS84範囲: lat=[{_lat_min_w:.6f},{_lat_max_w:.6f}], lon=[{_lon_min_w:.6f},{_lon_max_w:.6f}]')

    # タイル番号計算
    _zoom = 16
    def _ll2t(lat, lon, z):
        n = 2 ** z
        tx = int((lon + 180) / 360 * n)
        ty = int((1 - _math.log(_math.tan(_math.radians(lat)) + 1/_math.cos(_math.radians(lat))) / _math.pi) / 2 * n)
        return tx, ty
    def _t2ll(tx, ty, z):
        n = 2 ** z
        lon_v = tx / n * 360 - 180
        lat_v = _math.degrees(_math.atan(_math.sinh(_math.pi * (1 - 2 * ty / n))))
        return lat_v, lon_v

    _tx0, _ty1 = _ll2t(_lat_min_w, _lon_min_w, _zoom)
    _tx1, _ty0 = _ll2t(_lat_max_w, _lon_max_w, _zoom)
    _tx0 = max(0, _tx0 - 1)
    _ty0 = max(0, _ty0 - 1)
    _tx1 += 1
    _ty1 += 1

    _ntx = _tx1 - _tx0 + 1
    _nty = _ty1 - _ty0 + 1
    _nt = _ntx * _nty
    print(f'  タイル: z={_zoom}, x=[{_tx0},{_tx1}], y=[{_ty0},{_ty1}] ({_ntx}x{_nty}={_nt}枚)')

    if _nt <= 64:
        _ts = 256
        _stitched = Image.new('RGB', (_ntx * _ts, _nty * _ts), (30, 40, 50))
        _fetched = 0
        for _ry in range(_ty0, _ty1 + 1):
            for _rx in range(_tx0, _tx1 + 1):
                try:
                    _r = _req.get(f'https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{_zoom}/{_rx}/{_ry}.jpg', timeout=10)
                    if _r.status_code == 200:
                        _tile = Image.open(io.BytesIO(_r.content))
                        _stitched.paste(_tile, ((_rx - _tx0) * _ts, (_ry - _ty0) * _ts))
                        _fetched += 1
                except Exception:
                    pass
                print(f'\r    取得中: {_fetched}/{_nt}', end='', flush=True)
        print()

        if _fetched > 0:
            # タイル領域のWGS84境界
            _tlat_max, _tlon_min = _t2ll(_tx0, _ty0, _zoom)
            _tlat_min, _tlon_max = _t2ll(_tx1 + 1, _ty1 + 1, _zoom)

            # タイル4隅 → EPSG:6670 → シーン座標
            _wxc = (x_min + x_max) / 2
            _wyc = (y_min + y_max) / 2
            _nw = _tr_inv.transform(_tlon_min, _tlat_max)
            _ne = _tr_inv.transform(_tlon_max, _tlat_max)
            _sw = _tr_inv.transform(_tlon_min, _tlat_min)
            _se = _tr_inv.transform(_tlon_max, _tlat_min)

            gsi_scene_corners = {
                'nw': (_nw[0] - _wxc, _nw[1] - _wyc),
                'ne': (_ne[0] - _wxc, _ne[1] - _wyc),
                'sw': (_sw[0] - _wxc, _sw[1] - _wyc),
                'se': (_se[0] - _wxc, _se[1] - _wyc),
            }
            print(f'  シーン NW: ({gsi_scene_corners["nw"][0]:.1f}, {gsi_scene_corners["nw"][1]:.1f})')
            print(f'  シーン SE: ({gsi_scene_corners["se"][0]:.1f}, {gsi_scene_corners["se"][1]:.1f})')

            _buf = io.BytesIO()
            _stitched.save(_buf, format='JPEG', quality=80, optimize=True)
            _gb = _buf.getvalue()
            gsi_b64 = base64.b64encode(_gb).decode()
            gsi_enabled = True
            print(f'  結合画像: {_stitched.size[0]}x{_stitched.size[1]} -> {len(_gb)/1024:.0f}KB (base64: {len(gsi_b64)/1024:.0f}KB)')
            _stitched.save(str(BASE_DIR / '_diag_gsi_tiles.jpg'), quality=80)

            # ---- 標準地図タイル取得 ----
            print('  標準地図取得中 (std z=16)...')
            _stitched_std = Image.new('RGB', (_ntx * _ts, _nty * _ts), (240, 240, 240))
            _std_fetched = 0
            for _ry in range(_ty0, _ty1 + 1):
                for _rx in range(_tx0, _tx1 + 1):
                    try:
                        _r = _req.get(f'https://cyberjapandata.gsi.go.jp/xyz/std/{_zoom}/{_rx}/{_ry}.png', timeout=10)
                        if _r.status_code == 200:
                            _tile = Image.open(io.BytesIO(_r.content)).convert('RGB')
                            _stitched_std.paste(_tile, ((_rx - _tx0) * _ts, (_ry - _ty0) * _ts))
                            _std_fetched += 1
                    except Exception:
                        pass
                    print(f'\r    標準地図: {_std_fetched}/{_nt}', end='', flush=True)
            print()
            if _std_fetched > 0:
                _buf_std = io.BytesIO()
                _stitched_std.save(_buf_std, format='JPEG', quality=85, optimize=True)
                _gb_std = _buf_std.getvalue()
                gsi_std_b64 = base64.b64encode(_gb_std).decode()
                print(f'  標準地図: {_stitched_std.size[0]}x{_stitched_std.size[1]} -> {len(_gb_std)/1024:.0f}KB (base64: {len(gsi_std_b64)/1024:.0f}KB)')
            else:
                print('  ⚠ 標準地図取得失敗')

            # ---- DEM取得（3D化用） ----
            print('  DEM取得中 (dem5a_png z=15)...')
            _dz = 15
            _dtx0, _dty1 = _ll2t(_tlat_min, _tlon_min, _dz)
            _dtx1, _dty0 = _ll2t(_tlat_max, _tlon_max, _dz)
            _dtx0 -= 1; _dty0 -= 1; _dtx1 += 1; _dty1 += 1
            _dntx = _dtx1 - _dtx0 + 1
            _dnty = _dty1 - _dty0 + 1
            _dnt = _dntx * _dnty
            print(f'    DEM タイル: z={_dz}, {_dntx}x{_dnty}={_dnt}枚')

            _dem_full = np.full((_dnty * 256, _dntx * 256), np.nan, dtype=np.float64)
            _dem_fetched = 0
            for _dry in range(_dty0, _dty1 + 1):
                for _drx in range(_dtx0, _dtx1 + 1):
                    try:
                        _dr = _req.get(f'https://cyberjapandata.gsi.go.jp/xyz/dem5a_png/{_dz}/{_drx}/{_dry}.png', timeout=10)
                        if _dr.status_code == 200:
                            _di = Image.open(io.BytesIO(_dr.content))
                            _da = np.array(_di)[:,:,:3]
                            _raw = _da[:,:,0].astype(np.int32)*65536 + _da[:,:,1].astype(np.int32)*256 + _da[:,:,2].astype(np.int32)
                            _nd = (_da[:,:,0]==128)&(_da[:,:,1]==0)&(_da[:,:,2]==0)
                            _ev = np.where(_raw < 2**23, _raw*0.01, (_raw.astype(np.float64)-2**24)*0.01)
                            _ev[_nd] = np.nan
                            _px = (_drx - _dtx0) * 256
                            _py = (_dry - _dty0) * 256
                            _dem_full[_py:_py+256, _px:_px+256] = _ev
                            _dem_fetched += 1
                    except Exception:
                        pass
                    print(f'\r    DEM取得: {_dem_fetched}/{_dnt}', end='', flush=True)
            print()

            if _dem_fetched > 0:
                # DEMタイル領域のWGS84境界
                _dlat_max, _dlon_min = _t2ll(_dtx0, _dty0, _dz)
                _dlat_min, _dlon_max = _t2ll(_dtx1+1, _dty1+1, _dz)

                # 航空写真タイル範囲にクロップ
                _dw_deg = _dlon_max - _dlon_min
                _dh_deg = _dlat_max - _dlat_min
                _cl = int(((_tlon_min - _dlon_min) / _dw_deg) * (_dntx * 256))
                _cr = int(((_tlon_max - _dlon_min) / _dw_deg) * (_dntx * 256))
                _ct = int(((_dlat_max - _tlat_max) / _dh_deg) * (_dnty * 256))
                _cb = int(((_dlat_max - _tlat_min) / _dh_deg) * (_dnty * 256))
                _dem_crop = _dem_full[max(0,_ct):min(_dem_full.shape[0],_cb), max(0,_cl):min(_dem_full.shape[1],_cr)]

                # ダウンサンプル
                _target_gy = 96
                _target_gx = int(round(_target_gy * _dem_crop.shape[1] / _dem_crop.shape[0]))
                _iy_idx = (np.arange(_target_gy) * _dem_crop.shape[0] / _target_gy).astype(int)
                _ix_idx = (np.arange(_target_gx) * _dem_crop.shape[1] / _target_gx).astype(int)
                _iy_idx = np.clip(_iy_idx, 0, _dem_crop.shape[0]-1)
                _ix_idx = np.clip(_ix_idx, 0, _dem_crop.shape[1]-1)
                gsi_dem = _dem_crop[np.ix_(_iy_idx, _ix_idx)].astype(np.float32)

                # NaN埋め
                _nan_mask = np.isnan(gsi_dem)
                if _nan_mask.any():
                    gsi_dem[_nan_mask] = h_min
                gsi_dem_b64 = base64.b64encode(gsi_dem.tobytes()).decode()

                # シーン寸法（PlaneGeometry用）
                gsi_w = abs(gsi_scene_corners['ne'][0] - gsi_scene_corners['sw'][0])
                gsi_h = abs(gsi_scene_corners['ne'][1] - gsi_scene_corners['sw'][1])
                gsi_cx = (gsi_scene_corners['sw'][0] + gsi_scene_corners['ne'][0]) / 2
                gsi_cy = (gsi_scene_corners['sw'][1] + gsi_scene_corners['ne'][1]) / 2
                gsi_dem_gx = _target_gx
                gsi_dem_gy = _target_gy
                gsi_dem_ok = True

                print(f'    DEM: {_dem_crop.shape[1]}x{_dem_crop.shape[0]} -> {_target_gx}x{_target_gy}, 標高[{np.nanmin(gsi_dem):.1f},{np.nanmax(gsi_dem):.1f}]')
                print(f'    base64: {len(gsi_dem_b64)/1024:.0f}KB')
            else:
                gsi_dem_ok = False
                print('    ⚠ DEM取得失敗 — フラット表示にフォールバック')
        else:
            print('  ⚠ タイル取得失敗')
    else:
        print(f'  ⚠ タイル数が多すぎます ({_nt}枚) — スキップ')

except ImportError as e:
    print(f'  ⚠ ライブラリ未インストール ({e}) — スキップ')
except Exception as e:
    print(f'  ⚠ GSIタイル取得エラー: {e}')

# ==============================================================
# Phase 3.6: 【VER2】DEM5aでLAZ空白セル（撮影範囲外）を補完
# ==============================================================
print('Phase 3.6: LAZ範囲外セルをDEM5aで補完中...')
if gsi_dem_ok:
    _dh, _dw = gsi_dem.shape
    # gsi_dem 行0=北(lat_max) / grid 行0=y_min=南 → flipud が必要
    # gsi_dem を grid 解像度(GY, GX)に最近傍リサンプル（flip込み）
    _iy_map = ((GY - 1 - np.arange(GY)) * (_dh - 1) / max(GY - 1, 1)).astype(int)
    _ix_map = (np.arange(GX) * (_dw - 1) / max(GX - 1, 1)).astype(int)
    _iy_map = np.clip(_iy_map, 0, _dh - 1)
    _ix_map = np.clip(_ix_map, 0, _dw - 1)
    _dem_rs = gsi_dem[np.ix_(_iy_map, _ix_map)].astype(np.float64)
    # LAZオリジナル有効セル以外を DEM値で上書き（Phase 2 で h_mean 埋めされた領域も置換）
    _dem_fill = ~original_valid  # LAZデータなかった全セル
    _n_filled = int(_dem_fill.sum())
    grid[_dem_fill] = _dem_rs[_dem_fill]
    # DEMで埋めたセルも「有効」扱いに格上げ（ただしマスクは original_valid のままで境界保持）
    print(f'  DEM補完セル: {_n_filled:,} / {GX*GY:,} ({_n_filled/(GX*GY)*100:.1f}%)')
    # 標高統計を再計算（DEM注入で範囲変化）
    h_min = float(grid.min())
    h_max = float(grid.max())
    h_mean = float(grid.mean())
    h_p5 = float(np.percentile(grid, 5))
    h_p50 = float(np.percentile(grid, 50))
    h_p95 = float(np.percentile(grid, 95))
    print(f'  補完後標高: [{h_min:.4f}, {h_max:.4f}] 平均{h_mean:.4f}')

    # grid_flipped / h_b64 を再生成（Phase 2 版を上書き）
    grid_flipped = np.flipud(grid).astype(np.float32)
    h_b64 = base64.b64encode(grid_flipped.tobytes()).decode()
    print(f'  標高base64更新: {len(h_b64):,} chars')
else:
    print('  ⚠ DEM未取得 — LAZ範囲外は平均標高のフラット表示')

# ==============================================================
# Phase 3.7: GCP（基準点）読み込み
# ==============================================================
print('Phase 3.7: GCP（基準点・水準点）読み込み中...')
SIMA_FILE = Path(r'__NOT_AVAILABLE__')
GCP_FILE = Path(r'__NOT_AVAILABLE__')
gcp_list = []
gcp_js = ''
gcp_toggle_html = ''
gcp_help_html = ''
gcp_source = ''

try:
    # SIMAファイル優先（正式点名: BM.2等）
    if SIMA_FILE.exists():
        with open(SIMA_FILE, 'r', encoding='cp932') as f:
            for line in f:
                line = line.strip()
                if line.startswith('A01,'):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 6:
                        name = parts[2].strip()
                        northing = float(parts[3])
                        easting = float(parts[4])
                        elev = float(parts[5])
                        gcp_list.append((name, easting, northing, elev))
        gcp_source = str(SIMA_FILE)
        print(f'  SIMA: {len(gcp_list)}点読み込み（正式点名）')
    # フォールバック: GCP.txt
    elif GCP_FILE.exists():
        with open(GCP_FILE, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    name = parts[0].strip()
                    northing = float(parts[1])
                    easting = float(parts[2])
                    elev = float(parts[3])
                    gcp_list.append((name, easting, northing, elev))
        gcp_source = str(GCP_FILE)
        print(f'  GCP: {len(gcp_list)}点読み込み')
    else:
        print('  GCPファイルなし — スキップ')

    # 共通処理: SIMA/GCPどちらのソースでも座標変換+JS生成
    if len(gcp_list) > 0:
        # シーン座標に変換（X=Easting, Y=Northing, 中心からの相対値）
        wxc = (x_min + x_max) / 2
        wyc = (y_min + y_max) / 2
        gcp_scene = []
        for name, ex, ny, el in gcp_list:
            sx = ex - wxc
            sy = ny - wyc
            gcp_scene.append((name, sx, sy, el))
            print(f'    {name}: scene({sx:.1f}, {sy:.1f}), h={el:.2f}m')

        # GCP JS生成
        gcp_data = ','.join([f'{{n:"{g[0]}",x:{g[1]:.2f},y:{g[2]:.2f},h:{g[3]:.3f}}}' for g in gcp_scene])
        gcp_js = f'''
// ---- GCP（基準点）マーカー ----
const gcpData=[{gcp_data}];
const gcpGroup=new THREE.Group();gcpGroup.visible=true;scene.add(gcpGroup);
function buildGcpMarkers(zS){{
  while(gcpGroup.children.length)gcpGroup.remove(gcpGroup.children[0]);
  gcpData.forEach(function(g){{
    const gz=(g.h-ZC)*zS;
    // ピン（垂直線）
    const lg=new THREE.BufferGeometry();
    lg.setAttribute('position',new THREE.Float32BufferAttribute([g.x,g.y,gz,g.x,g.y,gz+8*zS],3));
    gcpGroup.add(new THREE.LineSegments(lg,new THREE.LineBasicMaterial({{color:0xff4444,linewidth:2}})));
    // ドット
    const sg=new THREE.SphereGeometry(1.5,8,6);
    const sm=new THREE.MeshBasicMaterial({{color:0xff4444}});
    const sp=new THREE.Mesh(sg,sm);sp.position.set(g.x,g.y,gz+8*zS);gcpGroup.add(sp);
    // ラベル
    const cv=document.createElement('canvas');cv.width=256;cv.height=64;
    const ctx=cv.getContext('2d');ctx.fillStyle='rgba(0,0,0,0.75)';ctx.fillRect(0,0,256,64);
    ctx.fillStyle='#ff6666';ctx.font='bold 24px sans-serif';ctx.fillText(g.n,8,28);
    ctx.fillStyle='#ffffff';ctx.font='18px monospace';ctx.fillText(g.h.toFixed(3)+'m',8,52);
    const tx=new THREE.CanvasTexture(cv);tx.minFilter=THREE.LinearFilter;
    const lb=new THREE.Sprite(new THREE.SpriteMaterial({{map:tx,depthTest:false,transparent:true}}));
    lb.scale.set(16,4,1);lb.position.set(g.x,g.y,gz+8*zS+4);gcpGroup.add(lb);
  }});
}}
buildGcpMarkers(1);
window.toggleGcp=function(v){{gcpGroup.visible=v}};
window.updateGcpZ=function(zS){{buildGcpMarkers(zS)}};
'''
        gcp_toggle_html = '  <label style="display:block;font-size:11px;cursor:pointer;margin:2px 0"><input type="checkbox" id="showGcp" checked onchange="toggleGcp(this.checked)"> 基準点・水準点</label>'
        gcp_help_html = f'''
<h3>基準点・水準点</h3>
<p>UAV写真測量の精度を確保するための基準点・水準点（{len(gcp_list)}点）を表示しています。</p>
<table>
<tr><th>名称</th><th>Easting</th><th>Northing</th><th>標高(m)</th></tr>
''' + '\n'.join([f'<tr><td><span class="tag">{g[0]}</span></td><td>{g[1]:.3f}</td><td>{g[2]:.3f}</td><td>{g[3]:.3f}</td></tr>' for g in gcp_list]) + f'''
</table>
<p style="font-size:10px;color:#6e7681;margin-top:4px">参照: <code onclick="copyPath(this)" title="クリックでコピー">{gcp_source}</code></p>
<p style="font-size:10px;color:#d29922;margin-top:4px">※ 今後は、EXCELに整理し、そこから読み込む形式が望ましい。</p>
'''
except Exception as e:
    print(f'  ⚠ GCP読み込みエラー: {e}')

# ==============================================================
# Phase 3.8: 筆界（登記所備付地図）読み込み
# ==============================================================
print('Phase 3.8: 筆界（登記所備付地図）読み込み中...')
import zipfile as _zf
import xml.etree.ElementTree as _ET

CHIKUBND_ZIP = Path(r'N:\B 技術室\A 共有データ\小林市法務局データ\45205-3523-2025\45205-3523-20.zip')
chiku_js = ''
chiku_toggle_html = ''
chiku_help_html = ''

try:
    if CHIKUBND_ZIP.exists():
        _zmn = 'http://www.moj.go.jp/MINJI/tizuzumen'
        _moj = 'http://www.moj.go.jp/MINJI/tizuxml'

        # 地形モデル座標範囲（EPSG:6670） + 100mマージン
        _margin = 100
        _te_min, _te_max = x_min - _margin, x_max + _margin  # Easting
        _tn_min, _tn_max = y_min - _margin, y_max + _margin  # Northing

        # 対象ZIP検索: search-list.csvから大字坂元を含むZIPを特定
        _target_zips = set()
        with _zf.ZipFile(str(CHIKUBND_ZIP), 'r') as _zout:
            _csv_name = [n for n in _zout.namelist() if n.endswith('.csv')]
            if _csv_name:
                _csv_text = _zout.read(_csv_name[0]).decode('cp932')
                for _line in _csv_text.strip().split('\n')[1:]:
                    _parts = _line.split(',')
                    if len(_parts) >= 3:
                        _target_zips.add(_parts[0].strip())

            # 【VER2】ZIPネスト構造（苧畑方式）/ 直接XML構造（小林市方式）両対応
            _inner_zips = [n for n in _zout.namelist() if n.endswith('.zip')]
            _xml_entries = []  # [(表示名, XMLバイト列), ...]
            if _inner_zips:
                for _izn in sorted(_inner_zips):
                    try:
                        _inner_data = _zout.read(_izn)
                        with _zf.ZipFile(io.BytesIO(_inner_data)) as _z2:
                            _xml_entries.append((_izn, _z2.read(_z2.namelist()[0])))
                    except Exception:
                        pass
            else:
                for _xn in sorted([n for n in _zout.namelist() if n.endswith('.xml')]):
                    try:
                        _xml_entries.append((_xn, _zout.read(_xn)))
                    except Exception:
                        pass
            print(f'  XMLエントリ: {len(_xml_entries)}件（内部ZIP={len(_inner_zips)}）')

            _all_parcels = []  # [(oaza, chiban, [(e,n),...]), ...]
            _wxc = (x_min + x_max) / 2
            _wyc = (y_min + y_max) / 2

            for _izn, _xml_data in _xml_entries:
                try:
                    _root = _ET.fromstring(_xml_data)

                    # 座標範囲で粗フィルタ（全点走査）
                    _all_xs = [float(el.text) for el in _root.iter(f'{{{_zmn}}}X')]
                    _all_ys = [float(el.text) for el in _root.iter(f'{{{_zmn}}}Y')]
                    if not _all_xs:
                        continue
                    # XML: X=Northing, Y=Easting
                    _n_min, _n_max = min(_all_xs), max(_all_xs)
                    _e_min, _e_max = min(_all_ys), max(_all_ys)
                    if not (_e_max >= _te_min and _e_min <= _te_max and
                            _n_max >= _tn_min and _n_min <= _tn_max):
                        continue

                    # Point辞書: id → (easting, northing)
                    _points = {}
                    for _pt in _root.iter(f'{{{_zmn}}}GM_Point'):
                        _pid = _pt.get('id')
                        _xel = _pt.find(f'.//{{{_zmn}}}X')
                        _yel = _pt.find(f'.//{{{_zmn}}}Y')
                        if _xel is not None and _yel is not None:
                            _points[_pid] = (float(_yel.text), float(_xel.text))

                    # Curve辞書: id → [(e,n), ...]
                    _curves = {}
                    for _curve in _root.iter(f'{{{_zmn}}}GM_Curve'):
                        _cid = _curve.get('id')
                        _pts = []
                        for _col in _curve.iter(f'{{{_zmn}}}GM_PointArray.column'):
                            _direct = _col.find(f'.//{{{_zmn}}}GM_Position.direct')
                            if _direct is not None:
                                _xel = _direct.find(f'{{{_zmn}}}X')
                                _yel = _direct.find(f'{{{_zmn}}}Y')
                                if _xel is not None and _yel is not None:
                                    _pts.append((float(_yel.text), float(_xel.text)))
                                continue
                            _indirect = _col.find(f'.//{{{_zmn}}}GM_PointRef.point')
                            if _indirect is not None:
                                _ref = _indirect.get('idref')
                                if _ref in _points:
                                    _pts.append(_points[_ref])
                        if _pts:
                            _curves[_cid] = _pts

                    # Surface辞書: id → [curve_id, ...]
                    _surfaces = {}
                    for _surf in _root.iter(f'{{{_zmn}}}GM_Surface'):
                        _sid = _surf.get('id')
                        _cids = [g.get('idref') for g in _surf.iter(f'{{{_zmn}}}GM_CompositeCurve.generator')]
                        _surfaces[_sid] = _cids

                    # 筆 → ポリゴン復元 → 範囲フィルタ
                    _file_count = 0
                    for _fude in _root.iter(f'{{{_moj}}}筆'):
                        _oaza_el = _fude.find(f'{{{_moj}}}大字名')
                        _chiban_el = _fude.find(f'{{{_moj}}}地番')
                        _shape_el = _fude.find(f'{{{_moj}}}形状')
                        _oaza = _oaza_el.text if _oaza_el is not None else ''
                        _chiban = _chiban_el.text if _chiban_el is not None else ''
                        _sid = _shape_el.get('idref') if _shape_el is not None else ''

                        if _sid in _surfaces:
                            _polygon_pts = []
                            for _cid in _surfaces[_sid]:
                                if _cid in _curves:
                                    _polygon_pts.extend(_curves[_cid])
                            if _polygon_pts:
                                _es = [p[0] for p in _polygon_pts]
                                _ns = [p[1] for p in _polygon_pts]
                                if (max(_es) >= _te_min and min(_es) <= _te_max and
                                    max(_ns) >= _tn_min and min(_ns) <= _tn_max):
                                    _all_parcels.append((_oaza, _chiban, _polygon_pts))
                                    _file_count += 1

                    if _file_count > 0:
                        print(f'  {_izn}: {_file_count}筆')
                except Exception as _e:
                    pass  # 個別ZIPエラーはスキップ

        if _all_parcels:
            print(f'  合計: {len(_all_parcels)}筆の筆界ポリゴンを抽出')

            # シーン座標でのポリゴン線分データを生成
            # 各筆の外周をclosed polylineとしてLineSegments用ペアに変換
            _line_pairs = []  # [(x1,y1,x2,y2), ...] シーン座標
            _label_data = []  # [(cx,cy,oaza,chiban), ...] ラベル用（重心座標）
            for _oaza, _chiban, _pts in _all_parcels:
                if len(_pts) < 2:
                    continue
                # シーン座標に変換
                _scene_pts = [(e - _wxc, n - _wyc) for e, n in _pts]
                # ポリゴンを閉じる（最初と最後が異なる場合）
                if _scene_pts[0] != _scene_pts[-1]:
                    _scene_pts.append(_scene_pts[0])
                # 連続線分ペア
                for _i in range(len(_scene_pts) - 1):
                    _line_pairs.append((_scene_pts[_i][0], _scene_pts[_i][1],
                                        _scene_pts[_i+1][0], _scene_pts[_i+1][1]))
                # 重心（ラベル配置用）
                _cx = sum(p[0] for p in _scene_pts[:-1]) / max(1, len(_scene_pts) - 1)
                _cy = sum(p[1] for p in _scene_pts[:-1]) / max(1, len(_scene_pts) - 1)
                _label_data.append((_cx, _cy, _oaza, _chiban))

            # JS用データ: Float32Array用のフラットな配列（x1,y1,x2,y2,...）
            _line_flat = []
            for _x1, _y1, _x2, _y2 in _line_pairs:
                _line_flat.extend([_x1, _y1, _x2, _y2])
            _chiku_line_b64 = base64.b64encode(np.array(_line_flat, dtype=np.float32).tobytes()).decode()

            # ラベル用JSON（地番情報）
            _label_json = ','.join([f'{{x:{ld[0]:.1f},y:{ld[1]:.1f},n:"{ld[3]}"}}' for ld in _label_data])

            print(f'  線分: {len(_line_pairs)}本, ラベル: {len(_label_data)}件')
            print(f'  base64: {len(_chiku_line_b64)/1024:.0f}KB')

            chiku_js = f'''
// ---- 筆界（登記所備付地図） ----
const chikuGroup=new THREE.Group();chikuGroup.visible=true;scene.add(chikuGroup);
const chikuLblGroup=new THREE.Group();chikuLblGroup.visible=false;scene.add(chikuLblGroup);
const chikuLabels=[{_label_json}];
const _chikuB64=atob('__CHIKU_B64__');
const _chikuBuf=new ArrayBuffer(_chikuB64.length);
const _chikuU8=new Uint8Array(_chikuBuf);
for(let i=0;i<_chikuB64.length;i++)_chikuU8[i]=_chikuB64.charCodeAt(i);
const chikuLines=new Float32Array(_chikuBuf);
const CHIKU_N={len(_line_pairs)};
// 標高補間（2段階フォールバック: 地形グリッド → GSI DEM → フラット）
function smpHSafe(wx,wy,zS){{
  // 1. 地形グリッド内かつ有効マスクの場合
  const fx=(wx-WXmin)/MW*(GX-1),fy=(WYmax-wy)/MH*(GY-1);
  if(fx>=0&&fx<GX-1&&fy>=0&&fy<GY-1){{
    const ix=Math.floor(fx),iy=Math.floor(fy),mi=iy*GX+ix;
    if(!(M[mi]===0&&M[mi+1]===0&&M[mi+GX]===0&&M[mi+GX+1]===0)){{
      const tx=fx-ix,ty=fy-iy;
      return(H[mi]*(1-tx)*(1-ty)+H[mi+1]*tx*(1-ty)+H[mi+GX]*(1-tx)*ty+H[mi+GX+1]*tx*ty-ZC)*zS+0.5;
    }}
  }}
  // 2. GSI DEMがある場合（地形外の筆界を地理院DEMに沿わせる）
  if(typeof gsiDemH!=='undefined'){{
    const sx=wx-WXc,sy=wy-WYc;
    const gcol=(sx-gsiCX+gsiW/2)/gsiW*(gsiDGX-1);
    const grow=(gsiH/2-(sy-gsiCY))/gsiH*(gsiDGY-1);
    if(gcol>=0&&gcol<gsiDGX-1&&grow>=0&&grow<gsiDGY-1){{
      const ic=Math.floor(gcol),ir=Math.floor(grow),gi=ir*gsiDGX+ic;
      const tc=gcol-ic,tr=grow-ir;
      const gh=gsiDemH[gi]*(1-tc)*(1-tr)+gsiDemH[gi+1]*tc*(1-tr)+gsiDemH[gi+gsiDGX]*(1-tc)*tr+gsiDemH[gi+gsiDGX+1]*tc*tr;
      return(gh-ZC)*zS+0.5;
    }}
  }}
  // 3. いずれもない場合
  return(HMIN-ZC)*zS-1;
}}
function buildChiku(zS){{
  while(chikuGroup.children.length)chikuGroup.remove(chikuGroup.children[0]);
  const pts=[];
  for(let i=0;i<CHIKU_N;i++){{
    const x1=chikuLines[i*4],y1=chikuLines[i*4+1],x2=chikuLines[i*4+2],y2=chikuLines[i*4+3];
    const wx1=x1+WXc,wy1=y1+WYc,wx2=x2+WXc,wy2=y2+WYc;
    const z1=smpHSafe(wx1,wy1,zS);
    const z2=smpHSafe(wx2,wy2,zS);
    pts.push(x1,y1,z1,x2,y2,z2);
  }}
  const bg=new THREE.BufferGeometry();
  bg.setAttribute('position',new THREE.Float32BufferAttribute(pts,3));
  const bm=new THREE.LineBasicMaterial({{color:0xffcc00,transparent:true,opacity:0.7,depthTest:true}});
  chikuGroup.add(new THREE.LineSegments(bg,bm));
}}
buildChiku(1);
// 地番ラベル生成
function buildChikuLabels(zS){{
  while(chikuLblGroup.children.length)chikuLblGroup.remove(chikuLblGroup.children[0]);
  chikuLabels.forEach(function(lb){{
    const wx=lb.x+WXc,wy=lb.y+WYc;
    const z=smpHSafe(wx,wy,zS)+1.5;
    const cv=document.createElement('canvas');cv.width=128;cv.height=32;
    const ctx=cv.getContext('2d');
    ctx.fillStyle='rgba(0,0,0,0.6)';ctx.fillRect(0,0,128,32);
    ctx.fillStyle='#ffee88';ctx.font='bold 18px sans-serif';ctx.textAlign='center';
    ctx.fillText(lb.n,64,22);
    const tx=new THREE.CanvasTexture(cv);tx.minFilter=THREE.LinearFilter;
    const sp=new THREE.Sprite(new THREE.SpriteMaterial({{map:tx,depthTest:false,transparent:true}}));
    sp.scale.set(8,2,1);sp.position.set(lb.x,lb.y,z);
    chikuLblGroup.add(sp);
  }});
}}
buildChikuLabels(1);
window.toggleChiku=function(v){{chikuGroup.visible=v}};
window.toggleChikuLbl=function(v){{chikuLblGroup.visible=v}};
window.updateChikuZ=function(zS){{buildChiku(zS);buildChikuLabels(zS)}};
'''.replace('__CHIKU_B64__', _chiku_line_b64)

            chiku_toggle_html = '  <label style="display:block;font-size:11px;cursor:pointer;margin:2px 0"><input type="checkbox" id="showChiku" checked onchange="toggleChiku(this.checked)"> 筆界（用地区画）</label>\n  <label style="display:block;font-size:11px;cursor:pointer;margin:2px 0;padding-left:16px"><input type="checkbox" id="showChikuLbl" onchange="toggleChikuLbl(this.checked)"> 地番</label>'

            # 大字別集計
            from collections import Counter as _Counter
            _oaza_count = _Counter(p[0] for p in _all_parcels)
            _oaza_summary = '、'.join([f'{o} {c}筆' for o, c in _oaza_count.most_common()])

            chiku_help_html = f'''
<h3>筆界（用地区画）</h3>
<p>登記所備付地図データ（法務省）から、対象区域内の筆界ポリゴン（{len(_all_parcels)}筆）を抽出・表示しています。</p>
<p>{_oaza_summary}</p>
<table>
<tr><th>項目</th><th>内容</th></tr>
<tr><td>データソース</td><td>G空間情報センター 登記所備付地図データ（えびの市）</td></tr>
<tr><td>座標系</td><td>公共座標2系（EPSG:6670）</td></tr>
<tr><td>データ形式</td><td>地図XMLフォーマット</td></tr>
<tr><td>抽出範囲</td><td>地形モデル範囲 ±100m</td></tr>
</table>
<p style="font-size:10px;color:#6e7681;margin-top:4px">参照: <code onclick="copyPath(this)" title="クリックでコピー">{str(CHIKUBND_ZIP.parent)}</code></p>
'''
        else:
            print('  範囲内の筆界データなし — スキップ')
    else:
        print('  筆界ZIPファイルなし — スキップ')
except Exception as e:
    print(f'  ⚠ 筆界読み込みエラー: {e}')

# ==============================================================
# Phase 3.9: 路線測量データ + コリドー
# ==============================================================
print('Phase 3.9: 路線測量データ読み込み中...')
ROUTE_SIMA = Path(r'__NOT_AVAILABLE__')
COORD_SIMA = Path(r'__NOT_AVAILABLE__')
FENCE_HEIGHT = 2.0   # 柵の高さ(m)
FENCE_WIDTH  = 0.05  # 柵の厚さ(m) — 断面プロファイル幅
route_js = ''
route_toggle_html = ''
route_help_html = ''

try:
    if ROUTE_SIMA.exists():
        # ---- 路線SIMA読み込み（結線情報あり、Z=0） ----
        with open(ROUTE_SIMA, 'r', encoding='cp932') as f:
            sima_lines = f.readlines()

        route_points = {}  # 番号→(名前, easting, northing)
        for line in sima_lines:
            line = line.strip()
            if line.startswith('A01,'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    pnum = int(parts[1])
                    pname = parts[2].strip()
                    northing = float(parts[3])
                    easting = float(parts[4])
                    route_points[pnum] = (pname, easting, northing)

        # ---- 座標SIMA読み込み（Z値あり、結線なし） ----
        # XY座標+Z値をリストで保持（最近傍マッチング用）
        coord_xyz = []  # [(easting, northing, z, pnum, name), ...]
        if COORD_SIMA.exists():
            with open(COORD_SIMA, 'r', encoding='cp932') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('A01,'):
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 6:
                            c_pnum = int(parts[1])
                            c_name = parts[2].strip()
                            c_northing = float(parts[3])
                            c_easting = float(parts[4])
                            c_z = float(parts[5])
                            coord_xyz.append((c_easting, c_northing, c_z, c_pnum, c_name))
            print(f'  座標SIMA: {len(coord_xyz)}点（XY最近傍でZ値マッチング）')
        else:
            print(f'  座標SIMAなし — 地形標高を使用')

        def find_nearest_z(easting, northing):
            """路線点のXY座標から、座標SIMAの最近傍点のZ値を返す"""
            if not coord_xyz:
                return 0.0, -1, '', float('inf')
            best_dist = float('inf')
            best_z = 0.0
            best_pnum = -1
            best_name = ''
            for ce, cn, cz, cpn, cname in coord_xyz:
                d = (easting - ce)**2 + (northing - cn)**2
                if d < best_dist:
                    best_dist = d
                    best_z = cz
                    best_pnum = cpn
                    best_name = cname
            return best_z, best_pnum, best_name, best_dist**0.5

        # ---- 中心線（F00,3）からチェーン順のポイント列を取得 ----
        in_centerline = False
        centerline_pts = []
        current_pnum = None
        for line in sima_lines:
            line = line.strip()
            if line.startswith('F00,3,'):
                in_centerline = True
                continue
            if in_centerline:
                if line.startswith('F99'):
                    break
                if line.startswith('B01,'):
                    parts = [p.strip() for p in line.split(',')]
                    current_pnum = int(parts[1])
                elif line.startswith('B03,') and current_pnum is not None:
                    parts = [p.strip() for p in line.split(',')]
                    chainage = float(parts[1])
                    centerline_pts.append((current_pnum, chainage))
                    current_pnum = None

        print(f'  中心線: {len(centerline_pts)}点（チェーン順）')

        # ---- シーン座標に変換（Z値: XY最近傍マッチング） ----
        wxc = (x_min + x_max) / 2
        wyc = (y_min + y_max) / 2
        route_scene = []  # [(name, sx, sy, z, chainage), ...]
        max_match_dist = 0.0
        for pnum, chainage in centerline_pts:
            if pnum in route_points:
                name, ex, ny = route_points[pnum]
                sx = ex - wxc
                sy = ny - wyc
                z, matched_pnum, matched_name, dist = find_nearest_z(ex, ny)
                if dist > max_match_dist:
                    max_match_dist = dist
                route_scene.append((name, sx, sy, z, chainage))

        print(f'  変換済み: {len(route_scene)}点')
        if coord_xyz:
            print(f'  Z値マッチング: XY最近傍（最大距離 {max_match_dist:.4f}m）')
        z_vals = [r[3] for r in route_scene if r[3] != 0.0]
        if z_vals:
            print(f'  標高: [{min(z_vals):.3f}, {max(z_vals):.3f}]m')
        print(f'  総延長: {route_scene[-1][4]:.1f}m（{route_scene[0][0]} → {route_scene[-1][0]}）')

        # ---- ラベルデータ: BP, EP, 主要NO測点（NO1〜NO26）のみ ----
        route_labels = []
        import re as _re
        for name, sx, sy, z, chainage in route_scene:
            if name in ('BP', 'EP'):
                route_labels.append((name, sx, sy, z, chainage))
            elif _re.match(r'^NO\d+$', name):
                route_labels.append((name, sx, sy, z, chainage))

        print(f'  線分: {len(route_scene)-1}本, ラベル: {len(route_labels)}点')

        # ---- JSON生成 ----
        # 路線点（x, y, z）の配列
        route_pts_json = ','.join([f'{{x:{r[1]:.2f},y:{r[2]:.2f},z:{r[3]:.3f}}}' for r in route_scene])
        route_lbl_json = ','.join([f'{{n:"{r[0]}",x:{r[1]:.2f},y:{r[2]:.2f},z:{r[3]:.3f},c:{r[4]:.1f}}}' for r in route_labels])

        # ---- JS生成（路線 + コリドー + ラベル） ----
        route_js = f'''
// ---- 路線 + コリドー ----
const routeGroup=new THREE.Group();routeGroup.visible=true;scene.add(routeGroup);
const corridorGroup=new THREE.Group();corridorGroup.visible=true;scene.add(corridorGroup);
const routeLblGroup=new THREE.Group();routeLblGroup.visible=true;scene.add(routeLblGroup);
const routePts=[{route_pts_json}];
const routeLabels=[{route_lbl_json}];
const FENCE_H={FENCE_HEIGHT};

function buildRoute(zS){{
  while(routeGroup.children.length)routeGroup.remove(routeGroup.children[0]);
  // 路線（パス）: SIMA Z値で描画
  const pts=[];
  for(let i=0;i<routePts.length-1;i++){{
    const p1=routePts[i],p2=routePts[i+1];
    const z1=(p1.z-ZC)*zS+0.3, z2=(p2.z-ZC)*zS+0.3;
    pts.push(p1.x,p1.y,z1, p2.x,p2.y,z2);
  }}
  const bg=new THREE.BufferGeometry();
  bg.setAttribute('position',new THREE.Float32BufferAttribute(pts,3));
  routeGroup.add(new THREE.LineSegments(bg,new THREE.LineBasicMaterial({{color:0xff3333,opacity:0.9,transparent:true,depthTest:true}})));
  // 柵上端ライン（白）
  const tpts=[];
  for(let i=0;i<routePts.length-1;i++){{
    const p1=routePts[i],p2=routePts[i+1];
    const z1=(p1.z+FENCE_H-ZC)*zS, z2=(p2.z+FENCE_H-ZC)*zS;
    tpts.push(p1.x,p1.y,z1, p2.x,p2.y,z2);
  }}
  const tbg=new THREE.BufferGeometry();
  tbg.setAttribute('position',new THREE.Float32BufferAttribute(tpts,3));
  corridorGroup.add(new THREE.LineSegments(tbg,new THREE.LineBasicMaterial({{color:0xffaaaa,opacity:0.6,transparent:true,depthTest:true}})));
}}
buildRoute(1);

function buildCorridor(zS){{
  while(corridorGroup.children.length)corridorGroup.remove(corridorGroup.children[0]);
  // コリドー: 断面プロファイル(幅0.05m×高2.0m)をパスに沿ってスイープ（平行方向）
  const N=routePts.length;
  if(N<2)return;
  // 各点でパス方向に直交する水平オフセット（±FENCE_WIDTH/2）を計算
  const hw={FENCE_WIDTH}/2;
  const pos=[];const idx=[];
  for(let i=0;i<N;i++){{
    const p=routePts[i];
    // 前後の点からパスの水平方向ベクトルを算出
    let dx=0,dy=0;
    if(i<N-1){{dx+=routePts[i+1].x-p.x;dy+=routePts[i+1].y-p.y;}}
    if(i>0){{dx+=p.x-routePts[i-1].x;dy+=p.y-routePts[i-1].y;}}
    const len=Math.sqrt(dx*dx+dy*dy)||1;
    // 法線（水平面内で90度回転）
    const nx=-dy/len*hw, ny=dx/len*hw;
    const zBot=(p.z-ZC)*zS;
    const zTop=(p.z+FENCE_H-ZC)*zS;
    // 4頂点: 内側下,内側上,外側下,外側上
    const vi=pos.length/3;
    pos.push(p.x-nx,p.y-ny,zBot, p.x-nx,p.y-ny,zTop, p.x+nx,p.y+ny,zBot, p.x+nx,p.y+ny,zTop);
    if(i>0){{
      const pv=vi-4;
      // 内側面(2三角形)
      idx.push(pv+0,pv+1,vi+0, vi+0,pv+1,vi+1);
      // 外側面(2三角形)
      idx.push(pv+2,vi+2,pv+3, vi+2,vi+3,pv+3);
      // 上面(2三角形)
      idx.push(pv+1,pv+3,vi+1, vi+1,pv+3,vi+3);
    }}
  }}
  const geo=new THREE.BufferGeometry();
  geo.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
  geo.setIndex(idx);geo.computeVertexNormals();
  const mat=new THREE.MeshLambertMaterial({{color:0xff4444,transparent:true,opacity:0.45,side:THREE.DoubleSide,depthWrite:false}});
  const mesh=new THREE.Mesh(geo,mat);
  mesh.renderOrder=2;
  corridorGroup.add(mesh);
}}
buildCorridor(1);

function buildRouteLabels(zS){{
  while(routeLblGroup.children.length)routeLblGroup.remove(routeLblGroup.children[0]);
  routeLabels.forEach(function(lb){{
    const z=(lb.z+FENCE_H-ZC)*zS+1.5;
    const isEnd=(lb.n==='BP'||lb.n==='EP');
    const cw=isEnd?128:96;const ch=isEnd?36:28;
    const cv=document.createElement('canvas');cv.width=cw;cv.height=ch;
    const ctx=cv.getContext('2d');
    ctx.fillStyle=isEnd?'rgba(180,0,0,0.8)':'rgba(0,0,0,0.7)';
    ctx.fillRect(0,0,cw,ch);
    ctx.fillStyle=isEnd?'#ffffff':'#ff9999';
    ctx.font=isEnd?'bold 22px sans-serif':'bold 16px sans-serif';
    ctx.textAlign='center';
    ctx.fillText(lb.n,cw/2,isEnd?26:20);
    const tx=new THREE.CanvasTexture(cv);tx.minFilter=THREE.LinearFilter;
    const sp=new THREE.Sprite(new THREE.SpriteMaterial({{map:tx,depthTest:false,transparent:true}}));
    sp.scale.set(isEnd?10:7,isEnd?3:2,1);sp.position.set(lb.x,lb.y,z);
    routeLblGroup.add(sp);
  }});
}}
buildRouteLabels(1);
window.toggleRoute=function(v){{routeGroup.visible=v}};
window.toggleCorridor=function(v){{corridorGroup.visible=v}};
window.toggleRouteLbl=function(v){{routeLblGroup.visible=v}};
window.updateRouteZ=function(zS){{buildRoute(zS);buildCorridor(zS);buildRouteLabels(zS)}};
'''

        route_toggle_html = '  <label style="display:block;font-size:11px;cursor:pointer;margin:2px 0"><input type="checkbox" id="showRoute" checked onchange="toggleRoute(this.checked)"> 路線</label>\n  <label style="display:block;font-size:11px;cursor:pointer;margin:2px 0"><input type="checkbox" id="showCorridor" checked onchange="toggleCorridor(this.checked)"> 柵（コリドー）</label>\n  <label style="display:block;font-size:11px;cursor:pointer;margin:2px 0;padding-left:16px"><input type="checkbox" id="showRouteLbl" checked onchange="toggleRouteLbl(this.checked)"> 測点名</label>'

        # ---- 全点テーブルHTML生成 ----
        route_detail_rows = ''
        for i, (name, sx, sy, z, chainage) in enumerate(route_scene):
            # 実座標に戻す
            ex_real = sx + wxc
            ny_real = sy + wyc
            row_style = ' style="background:#1a2332"' if name.startswith('NO') or name in ('BP','EP') else ''
            route_detail_rows += f'<tr{row_style}><td>{i+1}</td><td><b>{name}</b></td><td style="text-align:right">{chainage:.1f}</td><td style="text-align:right">{z:.3f}</td><td style="text-align:right;font-size:9px">{ex_real:.3f}</td><td style="text-align:right;font-size:9px">{ny_real:.3f}</td></tr>\n'

        route_help_html = f'''
<h3>路線</h3>
<p>計画路線（{len(route_scene)}点、総延長{route_scene[-1][4]:.1f}m）を3Dコリドーとして表示しています。</p>
<table>
<tr><th>項目</th><th>内容</th></tr>
<tr><td>起点</td><td>BP（追距 0.000m、標高 {route_scene[0][3]:.3f}m）</td></tr>
<tr><td>終点</td><td>EP（追距 {route_scene[-1][4]:.1f}m、標高 {route_scene[-1][3]:.3f}m）</td></tr>
<tr><td>柵高さ</td><td>{FENCE_HEIGHT}m（断面プロファイルから）</td></tr>
<tr><td>柵厚さ</td><td>{FENCE_WIDTH}m（断面プロファイルから）</td></tr>
<tr><td>標高データ</td><td>座標SIMAのXY最近傍点のZ値を採用</td></tr>
<tr><td>コリドー方式</td><td>断面スイープ（平行方向）</td></tr>
<tr><td>IP点（折れ点）</td><td>{len([r for r in route_scene if r[0].startswith('IP')])}点</td></tr>
<tr><td>NO測点</td><td>{len([r for r in route_scene if r[0].startswith('NO') and r[0][2:].isdigit()])}測点（20m間隔）</td></tr>
</table>
<details style="margin-top:8px">
<summary style="cursor:pointer;color:#58a6ff;font-size:11px;padding:4px 8px;background:#161b22;border:1px solid #30363d;border-radius:4px;display:inline-block">
全{len(route_scene)}点の詳細を表示</summary>
<div style="max-height:300px;overflow-y:auto;margin-top:4px;border:1px solid #30363d;border-radius:4px">
<table style="margin:0">
<tr><th>#</th><th>測点名</th><th>追距(m)</th><th>標高(m)</th><th>Easting</th><th>Northing</th></tr>
{route_detail_rows}</table>
</div>
</details>
<p style="font-size:10px;color:#6e7681;margin-top:4px">路線: <code onclick="copyPath(this)" title="クリックでコピー">{str(ROUTE_SIMA)}</code><br>
座標: <code onclick="copyPath(this)" title="クリックでコピー">{str(COORD_SIMA)}</code></p>
<p style="font-size:10px;color:#d29922;margin-top:4px">※ 今後は、EXCELに整理し、そこから読み込む形式が望ましい。</p>
'''
        print(f'  路線測量データ読み込み完了')
    else:
        print(f'  路線SIMAファイルなし — スキップ')
except Exception as e:
    print(f'  ⚠ 路線読み込みエラー: {e}')

# ==============================================================
# Phase 4: HTML生成
# ==============================================================
print('Phase 4/4: HTML生成中...')

ZC = h_mean          # Z中心
cam_dist = max(MW, MH) * 1.5

# オルソ有無でUIを調整
ortho_options = ''
if ortho_b64:
    ortho_options = '''
      <option value="ortho_hs" selected>オルソ＋陰影</option>'''

ortho_controls = ''  # ベイク方式: 回転は焼き込み済みのため操作パネル不要

ortho_js_data = ''
if ortho_b64:
    ortho_js_data = f"const orthoB64='{ortho_b64}';"

# テクスチャマッピング方式: Three.jsのテクスチャパイプラインでsRGBを正しく処理
ortho_js_loader = ''
if ortho_b64:
    ortho_js_loader = """
// ---- オルソ画像テクスチャ（Three.jsテクスチャマッピング方式）----
let orthoTex=null;
(function(){const img=new Image();img.onload=function(){
  const cv=document.createElement('canvas');cv.width=img.width;cv.height=img.height;
  cv.getContext('2d').drawImage(img,0,0);
  orthoTex=new THREE.CanvasTexture(cv);
  orthoTex.colorSpace=THREE.SRGBColorSpace;
  orthoTex.minFilter=THREE.LinearFilter;
  orthoTex.magFilter=THREE.LinearFilter;
  omat.map=orthoTex;omat.needsUpdate=true;
  console.log('オルソテクスチャ読込完了:',img.width,'x',img.height);
  applyAll();
};img.src='data:image/jpeg;base64,'+orthoB64})();"""

# テクスチャ方式: オルソ色はテクスチャが処理するため、頂点カラーは陰影のみ
ortho_js_color = """
      {
        const bl=parseFloat(document.getElementById('blendS').value)/100;
        const sh0=.3+.7*hs;const sh=1-bl*(1-sh0);
        r=sh;g=sh;b=sh;
      }""" if ortho_b64 else "      r=.3;g=.3;b=.3;"

# 回転/スケール/フリップの操作パネル不要
ortho_save_keys = ''
ortho_save_flip = "\n  if(d.showTerrain!=null){const te=document.getElementById('showTerrain');if(te){te.checked=d.showTerrain;toggleTerrain(d.showTerrain)}}"
ortho_save_flip += "\n  if(d.showOvl!=null){const oe=document.getElementById('showOvl');if(oe){oe.checked=d.showOvl;toggleOverlay(d.showOvl)}}\n  if(d.gridMode){const gm=document.getElementById('gridMode');if(gm)gm.value=d.gridMode}\n  if(d.gridElev!=null){const ge=document.getElementById('gridElev');if(ge)ge.value=d.gridElev}"
ortho_save_flip_w = "\n  {const te=document.getElementById('showTerrain');if(te)d.showTerrain=te.checked;}"
ortho_save_flip_w += "\n  d.showOvl=document.getElementById('showOvl').checked;\n  const gme=document.getElementById('gridMode');if(gme)d.gridMode=gme.value;\n  const gee=document.getElementById('gridElev');if(gee)d.gridElev=gee.value;"

# GSI設定の保存/復元
if gsi_enabled:
    ortho_save_flip += "\n  if(d.showGsi!=null){const ge=document.getElementById('showGsi');if(ge){ge.checked=d.showGsi;toggleGsi(d.showGsi)}}"
    ortho_save_flip += "\n  if(d.gsiType){const gt=document.getElementById('gsiType');if(gt){gt.value=d.gsiType;if(window.switchGsiType)switchGsiType(d.gsiType)}}"
    ortho_save_flip += "\n  if(d.gsiCutout!=null){const gc=document.getElementById('gsiCutout');if(gc){gc.checked=d.gsiCutout;toggleGsiCutout(d.gsiCutout)}}"
    ortho_save_flip_w += "\n  {const ge=document.getElementById('showGsi');if(ge)d.showGsi=ge.checked;}"
    ortho_save_flip_w += "\n  {const gt=document.getElementById('gsiType');if(gt)d.gsiType=gt.value;}"
    ortho_save_flip_w += "\n  {const gc=document.getElementById('gsiCutout');if(gc)d.gsiCutout=gc.checked;}"

# 筆界設定の保存/復元
if chiku_toggle_html:
    ortho_save_flip += "\n  if(d.showChiku!=null){const ce=document.getElementById('showChiku');if(ce){ce.checked=d.showChiku;toggleChiku(d.showChiku)}}"
    ortho_save_flip += "\n  if(d.showChikuLbl!=null){const cl=document.getElementById('showChikuLbl');if(cl){cl.checked=d.showChikuLbl;toggleChikuLbl(d.showChikuLbl)}}"
    ortho_save_flip_w += "\n  {const ce=document.getElementById('showChiku');if(ce)d.showChiku=ce.checked;}"
    ortho_save_flip_w += "\n  {const cl=document.getElementById('showChikuLbl');if(cl)d.showChikuLbl=cl.checked;}"

# 路線設定の保存/復元
if route_toggle_html:
    ortho_save_flip += "\n  if(d.showRoute!=null){const re=document.getElementById('showRoute');if(re){re.checked=d.showRoute;toggleRoute(d.showRoute)}}"
    ortho_save_flip += "\n  if(d.showCorridor!=null){const ce=document.getElementById('showCorridor');if(ce){ce.checked=d.showCorridor;toggleCorridor(d.showCorridor)}}"
    ortho_save_flip += "\n  if(d.showRouteLbl!=null){const rl=document.getElementById('showRouteLbl');if(rl){rl.checked=d.showRouteLbl;toggleRouteLbl(d.showRouteLbl)}}"
    ortho_save_flip_w += "\n  {const re=document.getElementById('showRoute');if(re)d.showRoute=re.checked;}"
    ortho_save_flip_w += "\n  {const ce=document.getElementById('showCorridor');if(ce)d.showCorridor=ce.checked;}"
    ortho_save_flip_w += "\n  {const rl=document.getElementById('showRouteLbl');if(rl)d.showRouteLbl=rl.checked;}"

ortho_debounce = ''
ortho_debounce_display = ''

# ブレンドスライダー（オルソ有効時のみ）
blend_slider_html = ''
if ortho_b64:
    blend_slider_html = '  <div class="rw" id="blendRow"><label>陰影:</label><input type="range" id="blendS" min="0" max="100" value="30"><span id="blendV" style="width:36px;text-align:right">30%</span></div>'
    ortho_save_keys = ",'blendS'"
    ortho_debounce = ",'blendS'"

# ---- オーバーレイJS: グリッド / 目盛 / トンボ ----
overlay_js = """
const ovl=new THREE.Group();scene.add(ovl);
window.toggleOverlay=function(v){ovl.visible=v};
function niceInt(r){r/=6;const p=Math.pow(10,Math.floor(Math.log10(r)));return r/p<1.5?p:r/p<3.5?2*p:5*p}
const WXc=(WXmin+WXmax)/2,WYc=(WYmin+WYmax)/2;
const iXY=niceInt(Math.max(MW,MH)),iZ=niceInt(HMAX-HMIN);
function smpH(wx,wy){
  const fx=(wx-WXmin)/MW*(GX-1),fy=(WYmax-wy)/MH*(GY-1);
  const ix=Math.max(0,Math.min(GX-2,Math.floor(fx))),iy=Math.max(0,Math.min(GY-2,Math.floor(fy)));
  const tx=fx-ix,ty=fy-iy,i=iy*GX+ix;
  return H[i]*(1-tx)*(1-ty)+H[i+1]*tx*(1-ty)+H[i+GX]*(1-tx)*ty+H[i+GX+1]*tx*ty;
}
function mkLbl(t,c){
  const cv=document.createElement('canvas'),x=cv.getContext('2d');
  x.font='bold 36px monospace';const w=x.measureText(t).width;cv.width=w+8;cv.height=44;
  x.font='bold 36px monospace';x.fillStyle=c||'#8b949e';x.fillText(t,4,34);
  const tx=new THREE.CanvasTexture(cv);tx.minFilter=THREE.LinearFilter;
  const s=new THREE.Sprite(new THREE.SpriteMaterial({map:tx,depthTest:false,transparent:true}));
  s.scale.set(cv.width/cv.height*3,3,1);return s;
}
function buildOvl(zS){
  while(ovl.children.length)ovl.remove(ovl.children[0]);
  zS=zS||1;const N=80;
  const gMode=document.getElementById('gridMode')?document.getElementById('gridMode').value:'flat';
  const gridElevInput=document.getElementById('gridElev');
  const userElev=gridElevInput?parseFloat(gridElevInput.value):HMAX;
  const flatZ=((!isNaN(userElev)?userElev:HMAX)-ZC)*zS+1.5;
  const gM=new THREE.LineBasicMaterial({color:gMode==='flat'?0x8ab4d8:0x6a8598,transparent:true,opacity:gMode==='flat'?0.55:0.50,depthTest:true});
  const aM=new THREE.LineBasicMaterial({color:0x718096,transparent:true,opacity:0.6});
  const tMat=new THREE.LineBasicMaterial({color:0xff6b6b,transparent:true,opacity:0.7});
  function addLS(pts,m){const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(pts,3));ovl.add(new THREE.LineSegments(g,m))}
  const gP=[];
  if(gMode==='flat'){
    for(let wy=Math.ceil(WYmin/iXY)*iXY;wy<=WYmax;wy+=iXY){gP.push(-MW/2,wy-WYc,flatZ,MW/2,wy-WYc,flatZ)}
    for(let wx=Math.ceil(WXmin/iXY)*iXY;wx<=WXmax;wx+=iXY){gP.push(wx-WXc,-MH/2,flatZ,wx-WXc,MH/2,flatZ)}
  }else{
    for(let wy=Math.ceil(WYmin/iXY)*iXY;wy<=WYmax;wy+=iXY){const my=wy-WYc;for(let i=0;i<N;i++){const w0=WXmin+MW*i/N,w1=WXmin+MW*(i+1)/N;gP.push(w0-WXc,my,(smpH(w0,wy)-ZC)*zS+.2,w1-WXc,my,(smpH(w1,wy)-ZC)*zS+.2)}}
    for(let wx=Math.ceil(WXmin/iXY)*iXY;wx<=WXmax;wx+=iXY){const mx=wx-WXc;for(let j=0;j<N;j++){const w0=WYmin+MH*j/N,w1=WYmin+MH*(j+1)/N;gP.push(mx,w0-WYc,(smpH(wx,w0)-ZC)*zS+.2,mx,w1-WYc,(smpH(wx,w1)-ZC)*zS+.2)}}
  }
  if(gP.length)addLS(gP,gM);
  const isFlat=gMode==='flat';
  const aP=[];
  if(isFlat){
    aP.push(-MW/2,-MH/2,flatZ,MW/2,-MH/2,flatZ);
    aP.push(-MW/2,-MH/2,flatZ,-MW/2,MH/2,flatZ);
  }else{
    for(let i=0;i<N;i++){const w0=WXmin+MW*i/N,w1=WXmin+MW*(i+1)/N;aP.push(w0-WXc,-MH/2,(smpH(w0,WYmin)-ZC)*zS+.3,w1-WXc,-MH/2,(smpH(w1,WYmin)-ZC)*zS+.3)}
    for(let j=0;j<N;j++){const w0=WYmin+MH*j/N,w1=WYmin+MH*(j+1)/N;aP.push(-MW/2,w0-WYc,(smpH(WXmin,w0)-ZC)*zS+.3,-MW/2,w1-WYc,(smpH(WXmin,w1)-ZC)*zS+.3)}
  }
  if(aP.length)addLS(aP,aM);
  for(let wx=Math.ceil(WXmin/iXY)*iXY;wx<=WXmax;wx+=iXY){const mx=wx-WXc,mz=isFlat?flatZ:(smpH(wx,WYmin)-ZC)*zS+.3;addLS([mx,-MH/2-2,mz,mx,-MH/2+2,mz],aM);const l=mkLbl(wx.toFixed(0));l.position.set(mx,-MH/2-6,mz);ovl.add(l)}
  for(let wy=Math.ceil(WYmin/iXY)*iXY;wy<=WYmax;wy+=iXY){const my=wy-WYc,mz=isFlat?flatZ:(smpH(WXmin,wy)-ZC)*zS+.3;addLS([-MW/2-2,my,mz,-MW/2+2,my,mz],aM);const l=mkLbl(wy.toFixed(0));l.position.set(-MW/2-8,my,mz);ovl.add(l)}
  const zB=(HMIN-ZC)*zS,zT=(HMAX-ZC)*zS;
  addLS([-MW/2-3,-MH/2-3,zB,-MW/2-3,-MH/2-3,zT],aM);
  for(let wz=Math.ceil(HMIN/iZ)*iZ;wz<=HMAX;wz+=iZ){const mz=(wz-ZC)*zS;addLS([-MW/2-5,-MH/2-3,mz,-MW/2-1,-MH/2-3,mz],aM);const l=mkLbl(wz.toFixed(0)+'m');l.position.set(-MW/2-12,-MH/2-3,mz);ovl.add(l)}
  const tL=Math.min(MW,MH)*.05;
  [[-MW/2,-MH/2,1,1],[MW/2,-MH/2,-1,1],[-MW/2,MH/2,1,-1],[MW/2,MH/2,-1,-1]].forEach(function(c){const cx=c[0],cy=c[1],dx=c[2],dy=c[3],cz=isFlat?flatZ:(smpH(cx+WXc,cy+WYc)-ZC)*zS+.5;addLS([cx,cy,cz,cx+tL*dx,cy,cz,cx,cy,cz,cx,cy+tL*dy,cz],tMat)});
}
buildOvl(1);
window.buildOvl=buildOvl;
window.onGridModeChange=function(){
  const mode=document.getElementById('gridMode').value;
  const row=document.getElementById('gridElevRow');
  if(row)row.style.display=mode==='flat'?'flex':'none';
  buildOvl(parseFloat(document.getElementById('zS').value));
};
(function(){const ge=document.getElementById('gridElev');if(ge)ge.addEventListener('change',function(){buildOvl(parseFloat(document.getElementById('zS').value))})})();
"""

# ---- ヘルプモーダル zoom/pan JS ----
modal_zoom_js = """
(function(){
  const md=document.getElementById('helpModal');if(!md)return;
  let sc=1,px=0,py=0,dr=false,sx,sy;
  function up(){md.style.transform='scale('+sc+') translate('+px+'px,'+py+'px)'}
  md.addEventListener('wheel',function(e){e.preventDefault();e.stopPropagation();
    sc=Math.max(0.5,Math.min(5,sc*(e.deltaY>0?0.9:1.1)));up()},{passive:false});
  md.addEventListener('mousedown',function(e){
    if(e.target.tagName==='BUTTON'||e.target.tagName==='A'||e.target.tagName==='CODE')return;
    dr=true;sx=e.clientX/sc-px;sy=e.clientY/sc-py;md.style.cursor='grabbing';e.preventDefault()});
  window.addEventListener('mousemove',function(e){if(!dr)return;px=e.clientX/sc-sx;py=e.clientY/sc-sy;up()});
  window.addEventListener('mouseup',function(){dr=false;if(md)md.style.cursor='grab'});
  window.closeHelp=function(){document.getElementById('helpOverlay').classList.remove('show');sc=1;px=0;py=0;up()};
})();
"""

# ---- 右パネル zoom/pan JS ----
panel_zoom_js = """
(function(){
  const rp=document.getElementById('R');if(!rp)return;
  let sc=1,px=0,py=0,dr=false,sx,sy;
  function up(){rp.style.transform='scale('+sc+') translate('+px+'px,'+py+'px)'}
  rp.addEventListener('wheel',function(e){e.preventDefault();e.stopPropagation();
    sc=Math.max(0.5,Math.min(3,sc*(e.deltaY>0?0.9:1.1)));up()},{passive:false});
  rp.addEventListener('mousedown',function(e){
    if(e.target.tagName==='BUTTON'||e.target.tagName==='A'||e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;
    dr=true;sx=e.clientX/sc-px;sy=e.clientY/sc-py;rp.style.cursor='grabbing';e.preventDefault()});
  window.addEventListener('mousemove',function(e){if(!dr)return;px=e.clientX/sc-sx;py=e.clientY/sc-sy;up()});
  window.addEventListener('mouseup',function(){dr=false;if(rp)rp.style.cursor='grab'});
})();
"""

# ---- 方角パネル zoom/pan JS ----
vp_zoom_js = """
(function(){
  const vp=document.getElementById('VP');if(!vp)return;
  let sc=1,px=0,py=0,dr=false,sx,sy;
  function up(){vp.style.transform='scale('+sc+') translate('+px+'px,'+py+'px)'}
  vp.addEventListener('wheel',function(e){e.preventDefault();e.stopPropagation();
    sc=Math.max(0.5,Math.min(3,sc*(e.deltaY>0?0.9:1.1)));up()},{passive:false});
  vp.addEventListener('mousedown',function(e){
    if(e.target.tagName==='BUTTON'||e.target.tagName==='A'||e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;
    dr=true;sx=e.clientX/sc-px;sy=e.clientY/sc-py;vp.style.cursor='grabbing';e.preventDefault()});
  window.addEventListener('mousemove',function(e){if(!dr)return;px=e.clientX/sc-sx;py=e.clientY/sc-sy;up()});
  window.addEventListener('mouseup',function(){dr=false;if(vp)vp.style.cursor='grab'});
})();
"""

# ---- 国土地理院背景JS/HTML ----
gsi_js = ''
gsi_toggle_html = ''
gsi_help_html = ''
gsi_updz = ''

if gsi_enabled:
    if gsi_dem_ok:
        # 3D版（DEM地形付き + 中央くり抜き + デュアルテクスチャ）
        gsi_js_template = """
// ---- 国土地理院 背景 (3D, 高解像度くり抜き) ----
const gsiB64='__GSI_B64__';
const gsiStdB64='__GSI_STD_B64__';
const gsiDemH=b64f32('__GSI_DEM_B64__');
const gsiDGX=__GSI_DGX__,gsiDGY=__GSI_DGY__;
const gsiW=__GSI_W__,gsiH=__GSI_H__,gsiCX=__GSI_CX__,gsiCY=__GSI_CY__;
const GSI_SUB=4;
const gsiSX=gsiDGX*GSI_SUB,gsiSY=gsiDGY*GSI_SUB;
let gsiMesh=null,gsiTexPhoto=null,gsiTexStd=null,gsiHi=null,gsiFullIdx=null,gsiCutIdx=null;
function gsiLoadTex(b64,cb){
  if(!b64)return;
  const img=new Image();img.onload=function(){
    const cv=document.createElement('canvas');cv.width=img.width;cv.height=img.height;
    cv.getContext('2d').drawImage(img,0,0);
    const tex=new THREE.CanvasTexture(cv);
    tex.colorSpace=THREE.SRGBColorSpace;tex.minFilter=THREE.LinearFilter;
    cb(tex);
  };img.src='data:image/jpeg;base64,'+b64;
}
function gsiDemInterp(fx,fy){
  const ix=Math.min(Math.floor(fx),gsiDGX-2),iy=Math.min(Math.floor(fy),gsiDGY-2);
  const tx=fx-ix,ty=fy-iy;
  return gsiDemH[iy*gsiDGX+ix]*(1-tx)*(1-ty)+gsiDemH[iy*gsiDGX+ix+1]*tx*(1-ty)+gsiDemH[(iy+1)*gsiDGX+ix]*(1-tx)*ty+gsiDemH[(iy+1)*gsiDGX+ix+1]*tx*ty;
}
(function(){
  gsiLoadTex(gsiB64,function(tex){
    gsiTexPhoto=tex;
    const geo=new THREE.PlaneGeometry(gsiW,gsiH,gsiSX-1,gsiSY-1);
    const pos=geo.attributes.position.array;
    gsiHi=new Float32Array(gsiSX*gsiSY);
    for(let iy=0;iy<gsiSY;iy++){for(let ix=0;ix<gsiSX;ix++){
      const vi=iy*gsiSX+ix;
      const fx=ix/(gsiSX-1)*(gsiDGX-1),fy=iy/(gsiSY-1)*(gsiDGY-1);
      const h=gsiDemInterp(fx,fy);
      gsiHi[vi]=h;pos[vi*3+2]=(h-ZC)*1-0.3;
    }}
    geo.attributes.position.needsUpdate=true;geo.computeVertexNormals();
    // 地形マスク連動くり抜き（高解像度）
    const oi=geo.index.array;
    gsiFullIdx=Array.from(oi);
    const ni=[];
    for(let i=0;i<oi.length;i+=3){
      let allUnder=true;
      for(let j=0;j<3;j++){const vi=oi[i+j];
        const sx=pos[vi*3]+gsiCX,sy=pos[vi*3+1]+gsiCY;
        const col=Math.round((sx+MW/2)/MW*(GX-1));
        const row=Math.round((MH/2-sy)/MH*(GY-1));
        if(col<0||col>=GX||row<0||row>=GY||M[row*GX+col]===0){allUnder=false;break}}
      if(!allUnder)ni.push(oi[i],oi[i+1],oi[i+2]);
    }
    gsiCutIdx=ni;
    geo.setIndex(ni);geo.computeVertexNormals();
    const mat=new THREE.MeshLambertMaterial({map:tex,side:THREE.DoubleSide,polygonOffset:true,polygonOffsetFactor:2,polygonOffsetUnits:2});
    gsiMesh=new THREE.Mesh(geo,mat);
    gsiMesh.renderOrder=-1;
    gsiMesh.position.set(gsiCX,gsiCY,0);
    gsiMesh.visible=document.getElementById('showGsi')?.checked??true;
    scene.add(gsiMesh);
    console.log('GSI 3D背景:',gsiSX,'x',gsiSY,'全faces:',gsiFullIdx.length/3,'くり抜き後:',ni.length/3);
  });
  gsiLoadTex(gsiStdB64,function(tex){gsiTexStd=tex});
})();
window.toggleGsi=function(v){if(gsiMesh)gsiMesh.visible=v};
window.toggleGsiCutout=function(v){
  if(!gsiMesh||!gsiFullIdx||!gsiCutIdx)return;
  gsiMesh.geometry.setIndex(v?gsiCutIdx:gsiFullIdx);
  gsiMesh.geometry.computeVertexNormals();
};
window.switchGsiType=function(type){
  if(!gsiMesh)return;
  if(type==='std'&&gsiTexStd){gsiMesh.material.map=gsiTexStd;gsiMesh.material.needsUpdate=true}
  else if(gsiTexPhoto){gsiMesh.material.map=gsiTexPhoto;gsiMesh.material.needsUpdate=true}
};
window.updateGsiZ=function(zS){
  if(!gsiMesh||!gsiHi)return;
  const pos=gsiMesh.geometry.attributes.position.array;
  for(let i=0;i<gsiSX*gsiSY;i++){pos[i*3+2]=(gsiHi[i]-ZC)*zS-0.3*zS}
  gsiMesh.geometry.attributes.position.needsUpdate=true;
  gsiMesh.geometry.computeVertexNormals();
};
"""
        gsi_js = gsi_js_template.replace('__GSI_B64__', gsi_b64)
        gsi_js = gsi_js.replace('__GSI_STD_B64__', gsi_std_b64)
        gsi_js = gsi_js.replace('__GSI_DEM_B64__', gsi_dem_b64)
        gsi_js = gsi_js.replace('__GSI_DGX__', str(gsi_dem_gx))
        gsi_js = gsi_js.replace('__GSI_DGY__', str(gsi_dem_gy))
        gsi_js = gsi_js.replace('__GSI_W__', f'{gsi_w:.2f}')
        gsi_js = gsi_js.replace('__GSI_H__', f'{gsi_h:.2f}')
        gsi_js = gsi_js.replace('__GSI_CX__', f'{gsi_cx:.2f}')
        gsi_js = gsi_js.replace('__GSI_CY__', f'{gsi_cy:.2f}')
    else:
        # フラットフォールバック（DEMなし）
        _sc = gsi_scene_corners
        gsi_js_template = """
// ---- 国土地理院 航空写真タイル背景 (flat) ----
const gsiB64='__GSI_B64__';
let gsiMesh=null;
(function(){const img=new Image();img.onload=function(){
  const cv=document.createElement('canvas');cv.width=img.width;cv.height=img.height;
  cv.getContext('2d').drawImage(img,0,0);
  const tex=new THREE.CanvasTexture(cv);
  tex.colorSpace=THREE.SRGBColorSpace;tex.minFilter=THREE.LinearFilter;
  const geo=new THREE.BufferGeometry();
  const zG=(HMIN-ZC)*1-2;
  const v=new Float32Array([__SW_X__,__SW_Y__,zG,__SE_X__,__SE_Y__,zG,__NW_X__,__NW_Y__,zG,__NE_X__,__NE_Y__,zG]);
  geo.setAttribute('position',new THREE.BufferAttribute(v,3));
  geo.setAttribute('uv',new THREE.BufferAttribute(new Float32Array([0,0,1,0,0,1,1,1]),2));
  geo.setIndex([0,1,2,1,3,2]);
  const mat=new THREE.MeshBasicMaterial({map:tex,transparent:true,opacity:0.9,side:THREE.DoubleSide});
  gsiMesh=new THREE.Mesh(geo,mat);
  gsiMesh.visible=document.getElementById('showGsi')?.checked??true;
  scene.add(gsiMesh);
};img.src='data:image/jpeg;base64,'+gsiB64})();
window.toggleGsi=function(v){if(gsiMesh)gsiMesh.visible=v};
window.switchGsiType=function(){};
window.updateGsiZ=function(zS){if(!gsiMesh)return;const p=gsiMesh.geometry.attributes.position.array;const z=(HMIN-ZC)*zS-2;p[2]=p[5]=p[8]=p[11]=z;gsiMesh.geometry.attributes.position.needsUpdate=true};
"""
        gsi_js = gsi_js_template.replace('__GSI_B64__', gsi_b64)
        gsi_js = gsi_js.replace('__SW_X__', f'{_sc["sw"][0]:.2f}').replace('__SW_Y__', f'{_sc["sw"][1]:.2f}')
        gsi_js = gsi_js.replace('__SE_X__', f'{_sc["se"][0]:.2f}').replace('__SE_Y__', f'{_sc["se"][1]:.2f}')
        gsi_js = gsi_js.replace('__NW_X__', f'{_sc["nw"][0]:.2f}').replace('__NW_Y__', f'{_sc["nw"][1]:.2f}')
        gsi_js = gsi_js.replace('__NE_X__', f'{_sc["ne"][0]:.2f}').replace('__NE_Y__', f'{_sc["ne"][1]:.2f}')

    gsi_toggle_html = '  <label style="display:block;font-size:11px;cursor:pointer;margin:2px 0"><input type="checkbox" id="showGsi" checked onchange="toggleGsi(this.checked)"> 国土地理院 背景</label>\n  <label style="display:block;font-size:11px;cursor:pointer;margin:2px 0;padding-left:16px"><input type="checkbox" id="gsiCutout" checked onchange="toggleGsiCutout(this.checked)"> くり抜き（地形モデル部分）</label>\n  <div class="rw" style="margin-top:2px"><label>地図:</label><select id="gsiType" onchange="switchGsiType(this.value)" style="flex:1"><option value="photo">航空写真</option><option value="std">標準地図</option></select></div>'

    gsi_help_html = """
<h3>国土地理院 背景タイル</h3>
<p>3Dモデルの<b>周囲</b>に、国土地理院の地図タイルを3D背景として表示しています（中央は地形モデル本体）。</p>
<table>
<tr><th>項目</th><th>内容</th></tr>
<tr><td>航空写真</td><td><span class="tag">seamlessphoto</span> シームレス空中写真（z=16, 約2.4m/px）</td></tr>
<tr><td>標準地図</td><td><span class="tag">std</span> 電子国土基本図・標準地図（z=16）</td></tr>
<tr><td>標高データ</td><td><span class="tag">dem5a_png</span> 5mメッシュDEM → 3D地形化（z=15）</td></tr>
<tr><td>表示範囲</td><td>3Dモデル周囲の約3倍エリアを立体表示（中央くり抜き）</td></tr>
<tr><td>座標変換</td><td><span class="tag">pyproj</span> EPSG:6670 → WGS84</td></tr>
<tr><td>用途</td><td>地区周辺の地形・道路・建物等の立体的な位置関係を把握</td></tr>
</table>
<p style="font-size:10px;color:#6e7681;margin-top:4px">出典: 国土地理院（https://maps.gsi.go.jp/development/ichiran.html）<br>
この地図の作成に当たっては、国土地理院長の承認を得て、同院発行の基盤地図情報を使用した。</p>
"""

    gsi_updz = ';if(window.updateGsiZ)updateGsiZ(s)'

html = f'''<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>(有)徳永測量 社屋 VER2 広域1km - 3D地形ビューア</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0d1117;overflow:hidden;font-family:'Segoe UI','Yu Gothic UI',sans-serif;color:#c9d1d9;font-size:12px}}
canvas{{display:block}}
.panel{{position:absolute;background:rgba(13,17,23,0.93);padding:10px 14px;border-radius:8px;
  border:1px solid rgba(48,54,61,0.8);backdrop-filter:blur(8px);line-height:1.5}}
#info{{margin-bottom:6px;padding-bottom:6px;border-bottom:1px solid #21262d}}
#info h2{{color:#58a6ff;font-size:13px;margin-bottom:3px}}
#info .d{{color:#8b949e;font-size:10px}}
.key{{display:inline-block;background:#21262d;border:1px solid #30363d;border-radius:3px;
  padding:0 4px;font-size:9px;color:#79c0ff;margin:0 1px;line-height:1.7}}
#R{{position:absolute;top:10px;right:10px;width:255px;max-height:calc(100vh - 20px);overflow:hidden;cursor:grab;transform-origin:top right}}
#R h3{{color:#58a6ff;font-size:11px;margin:6px 0 3px 0}}
#R h3:first-child{{margin-top:0}}
.rw{{display:flex;align-items:center;gap:5px;margin:3px 0}}
.rw label{{white-space:nowrap;font-size:10px;color:#8b949e;min-width:50px}}
.rw input[type=number]{{width:72px;background:#0d1117;border:1px solid #30363d;
  color:#79c0ff;padding:2px 5px;border-radius:3px;font-size:11px;font-family:monospace}}
.rw select{{background:#0d1117;border:1px solid #30363d;color:#c9d1d9;padding:2px 5px;
  border-radius:3px;font-size:11px;flex:1}}
.rw input[type=range]{{flex:1;accent-color:#58a6ff}}
.br{{display:flex;gap:3px;margin:4px 0;flex-wrap:wrap}}
.br button{{flex:1;padding:3px 4px;font-size:10px;border-radius:4px;cursor:pointer;
  border:1px solid #30363d;min-width:0}}
.bg{{background:#1f6f2f !important;color:#fff !important}}
.bg:hover{{background:#2a8a3a !important}}
.bd{{background:#21262d !important;color:#c9d1d9 !important}}
.bd:hover{{background:#30363d !important}}
.sv{{color:#3fb950;font-size:9px;margin-top:2px;opacity:0;transition:opacity .3s}}
.st{{margin-top:4px;font-size:9px;color:#6e7681;border-top:1px solid #21262d;padding-top:3px}}
.st td{{padding:0 4px}}.st .v{{text-align:right;font-family:monospace}}
#BL{{bottom:10px;left:10px;display:none}}
#VP{{position:absolute;bottom:10px;right:10px;width:200px;cursor:grab;transform-origin:bottom right}}
#VP h3{{color:#58a6ff;font-size:11px;margin-bottom:4px}}
.dg{{display:grid;grid-template-columns:repeat(3,1fr);gap:3px;margin-bottom:6px}}
.dg button{{padding:4px 2px;font-size:10px;background:#21262d;color:#c9d1d9;
  border:1px solid #30363d;border-radius:4px;cursor:pointer}}
.dg button:hover,.ag button:hover{{background:#30363d}}
.dg button.act{{background:#1f6f2f;color:#fff;border-color:#3fb950}}
.ag{{display:flex;gap:3px}}
.ag button{{flex:1;padding:4px 2px;font-size:9px;background:#21262d;color:#c9d1d9;
  border:1px solid #30363d;border-radius:4px;cursor:pointer}}
.ag button.act{{background:#1f6f2f;color:#fff;border-color:#3fb950}}
#leg{{position:absolute;right:14px;top:50%;transform:translateY(-50%);text-align:center}}
#leg .lb{{font-size:9px;color:#8b949e;margin:2px 0}}
#lbar{{width:14px;height:140px;border-radius:3px;
  background:linear-gradient(to bottom,#f0e6c8,#d4a843,#7ab648,#3a8a2a,#1a4a14)}}
#et{{display:none;position:absolute;pointer-events:none;background:rgba(0,0,0,0.90);color:#79c0ff;
  padding:6px 10px;border-radius:5px;font-size:12px;font-family:monospace;
  border:1px solid #30363d;white-space:nowrap;z-index:100}}
#em{{display:none;position:absolute;pointer-events:none;width:12px;height:12px;
  border:2px solid #ff6b6b;border-radius:50%;transform:translate(-50%,-50%);z-index:99;
  box-shadow:0 0 5px rgba(255,107,107,0.5)}}
#helpBtn{{display:none}}
#helpOverlay{{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.7);z-index:200;
  justify-content:center;align-items:center;backdrop-filter:blur(3px)}}
#helpOverlay.show{{display:flex}}
#helpModal{{background:#161b22;border:1px solid #30363d;border-radius:12px;
  max-width:680px;width:90%;max-height:none;overflow:hidden;padding:24px 28px;
  color:#c9d1d9;box-shadow:0 8px 32px rgba(0,0,0,0.5)}}
#helpModal h2{{color:#58a6ff;font-size:16px;margin:0 0 12px 0;padding-bottom:8px;border-bottom:1px solid #21262d}}
#helpModal h3{{color:#79c0ff;font-size:13px;margin:16px 0 6px 0}}
#helpModal p,#helpModal li{{font-size:12px;line-height:1.7;color:#c9d1d9}}
#helpModal ul{{padding-left:18px;margin:4px 0}}
#helpModal table{{width:100%;border-collapse:collapse;margin:6px 0;font-size:11px}}
#helpModal th{{text-align:left;color:#58a6ff;padding:4px 8px;border-bottom:1px solid #30363d;font-weight:600}}
#helpModal td{{padding:4px 8px;border-bottom:1px solid #21262d}}
#helpModal .tag{{display:inline-block;background:#1f6f2f;color:#fff;padding:1px 6px;border-radius:3px;font-size:10px;margin:1px}}
#helpModal .close{{position:sticky;top:0;float:right;background:#21262d;border:1px solid #30363d;
  color:#c9d1d9;border-radius:6px;padding:4px 12px;cursor:pointer;font-size:12px}}
#helpModal .close:hover{{background:#30363d}}
#helpModal{{cursor:grab;transform-origin:center center}}
#helpModal code{{cursor:pointer;transition:background .2s;border-radius:3px;padding:1px 4px;position:relative}}
#helpModal code:hover{{background:#30363d;text-decoration:underline}}
#helpModal code::after{{content:'\\1F4CB';font-size:9px;margin-left:3px;opacity:0.5}}
#helpModal code:hover::after{{opacity:1}}
#copyToast{{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%) scale(0.8);
  background:rgba(31,111,47,0.95);color:#fff;padding:10px 18px;border-radius:8px;
  font-size:13px;z-index:999;pointer-events:none;opacity:0;transition:opacity .3s,transform .3s;
  box-shadow:0 4px 16px rgba(0,0,0,0.4);white-space:nowrap}}
#copyToast.show{{opacity:1;transform:translate(-50%,-50%) scale(1)}}
.cpre-wrap{{position:relative}}
.cpre-btn{{position:absolute;top:4px;right:4px;background:#21262d;border:1px solid #30363d;
  color:#8b949e;padding:2px 8px;border-radius:4px;font-size:10px;cursor:pointer}}
.cpre-btn:hover{{background:#30363d;color:#c9d1d9}}
</style>
</head>
<body>
<div id="R" class="panel">
  <div id="info">
    <h2>(有)徳永測量 社屋 VER2 広域1km - 3D地形モデル</h2>
    <div class="d">{MW:.1f} x {MH:.1f} m / {GX}x{GY} ({filled:,}セル) / {n_points:,}点</div>
  </div>
  <h3>表示モード</h3>
  <div class="rw">
    <select id="dMode" onchange="applyAll()">
      <option value="elev_hs">標高カラー + 陰影</option>
      <option value="slope">傾斜量図</option>
      <option value="cs">微地形強調（CS風）</option>{ortho_options}
    </select>
  </div>
{blend_slider_html}

  <h3>カラースキーム</h3>
  <div class="rw">
    <select id="cScheme" onchange="applyAll()">
      <option value="terrain">地形（緑-茶）</option>
      <option value="gray">グレースケール</option>
      <option value="warm">暖色（黄-赤-茶）</option>
      <option value="cool">寒色（青-緑-白）</option>
      <option value="rainbow">虹色（レインボー）</option>
      <option value="red_relief">赤色立体図風</option>
    </select>
  </div>
  <div class="rw" style="margin-top:4px">
    <label>背景:</label>
    <button id="bgBlk" class="bd" onclick="setBg('dark')" style="flex:1;padding:3px 4px;font-size:10px;border-radius:4px;cursor:pointer">黒</button>
    <button id="bgWht" class="bd" onclick="setBg('light')" style="flex:1;padding:3px 4px;font-size:10px;border-radius:4px;cursor:pointer">白</button>
  </div>

{ortho_controls}
  <div class="rw"><label>太陽高度:</label><input type="range" id="sunA" min="3" max="75" value="20"><span id="sunV" style="width:28px;text-align:right">20°</span></div>
  <div class="rw"><label>微地形:</label><input type="range" id="mZ" min="1" max="30" value="8"><span id="mZV" style="width:28px;text-align:right">8x</span></div>

  <h3>表示切替</h3>
  <label style="display:block;font-size:11px;cursor:pointer;margin:2px 0"><input type="checkbox" id="showTerrain" checked onchange="toggleTerrain(this.checked)"> 3D地形モデル</label>
  <label style="display:block;font-size:11px;cursor:pointer;margin:2px 0"><input type="checkbox" id="showOvl" checked onchange="toggleOverlay(this.checked)"> グリッド / 目盛 / トンボ</label>
  <div class="rw" style="margin-top:2px"><label>グリッド:</label><select id="gridMode" onchange="onGridModeChange()" style="flex:1"><option value="flat" selected>標高指定</option><option value="terrain">地形追従</option></select></div>
  <div class="rw" id="gridElevRow" style="margin-top:1px"><label>標高(m):</label><input type="number" id="gridElev" step="0.1" value="{h_max:.1f}" style="width:72px;background:#0d1117;border:1px solid #30363d;color:#79c0ff;padding:2px 5px;border-radius:3px;font-size:11px;font-family:monospace"><button class="bd" onclick="document.getElementById('gridElev').value={h_max:.1f};buildOvl(parseFloat(document.getElementById('zS').value))" style="padding:2px 6px;font-size:9px;border-radius:3px;cursor:pointer">既定</button></div>
{gsi_toggle_html}
{gcp_toggle_html}
{chiku_toggle_html}
{route_toggle_html}

  <div class="st">
    <table>
      <tr><td>最小</td><td class="v">{h_min:.2f}</td><td>5%</td><td class="v">{h_p5:.2f}</td></tr>
      <tr><td>最大</td><td class="v">{h_max:.2f}</td><td>50%</td><td class="v">{h_p50:.2f}</td></tr>
      <tr><td>範囲</td><td class="v">{h_max-h_min:.2f}</td><td>95%</td><td class="v">{h_p95:.2f}</td></tr>
    </table>
  </div>

  <h3>着目標高 (m)</h3>
  <div class="rw"><label>下限:</label><input type="number" id="cMin" step="0.1" value="{h_min:.1f}"></div>
  <div class="rw"><label>上限:</label><input type="number" id="cMax" step="0.1" value="{h_max:.1f}"></div>
  <div class="br">
    <button class="bg" onclick="applyAll()">適用</button>
    <button class="bd" onclick="rstRange()">全範囲</button>
  </div>

  <h3>Z誇張 / 表示</h3>
  <div class="rw"><label>Z誇張:</label><input type="range" id="zS" min="0.5" max="15" value="1" step="0.5"><span id="zV" style="width:36px;text-align:right">1.0x</span></div>
  <label style="display:block;font-size:11px;cursor:pointer;margin:2px 0"><input type="checkbox" id="wf"> 三角網</label>
  <label style="display:block;font-size:11px;cursor:pointer;margin:2px 0"><input type="checkbox" id="fs"> フラット</label>

  <h3>操作方法</h3>
  <div style="font-size:10px;color:#8b949e;line-height:1.8">
    <span class="key">左ドラッグ</span> 回転 &nbsp;
    <span class="key">中ドラッグ</span> パン<br>
    <span class="key">ホイール</span> ズーム &nbsp;
    <span class="key">左クリック</span> 標高 &nbsp;
    <span class="key">右クリック</span> 消去<br>
    <span class="key">中ボタン2連打</span> 初期視点
  </div>

  <div style="margin-top:8px;border-top:1px solid #21262d;padding-top:6px;display:flex;gap:4px">
    <button class="bg" onclick="saveDef()" style="flex:1;padding:5px;font-size:11px">全設定を保存</button>
    <button class="bd" onclick="document.getElementById('helpOverlay').classList.add('show')" style="padding:5px 10px;font-size:11px;border-radius:4px;cursor:pointer">? 説明</button>
  </div>
  <div class="sv" id="svM">保存しました</div>
</div>

<div id="BL" class="panel">
  <label>Z誇張: <input type="range" id="zS" min="0.5" max="15" value="1" step="0.5"><span id="zV">1.0</span>x</label>
  <label><input type="checkbox" id="wf"> 三角網</label>
  <label><input type="checkbox" id="fs"> フラット</label>
</div>

<div id="VP" class="panel">
  <h3 title="カメラの水平方向（どの方角から見るか）">視点の方角 <span style="font-size:9px;color:#8b949e;font-weight:normal">（どの方角から見るか）</span></h3>
  <div class="dg">
    <button onclick="setDir(315)" id="d315" title="北西側から見る">北西から</button>
    <button onclick="setDir(0)" id="d0" title="北側から見る">北から</button>
    <button onclick="setDir(45)" id="d45" title="北東側から見る">北東から</button>
    <button onclick="setDir(270)" id="d270" title="西側から見る">西から</button>
    <button onclick="setDir(-1)" id="dtop" title="真上（天頂）から見下ろす">真上</button>
    <button onclick="setDir(90)" id="d90" title="東側から見る">東から</button>
    <button onclick="setDir(225)" id="d225" title="南西側から見る">南西から</button>
    <button onclick="setDir(180)" id="d180" title="南側から見る">南から</button>
    <button onclick="setDir(135)" id="d135" title="南東側から見る">南東から</button>
  </div>
  <h3 title="カメラの仰角（どの高さから見るか）">視点の高さ <span style="font-size:9px;color:#8b949e;font-weight:normal">（どの角度で見下ろすか）</span></h3>
  <div class="ag">
    <button onclick="setAlt(88)" id="a88" title="真上から見下ろす（88°）">真上</button>
    <button onclick="setAlt(65)" id="a65" title="高い角度から見る（65°）">高角</button>
    <button onclick="setAlt(45)" id="a45" title="中くらいの角度（45°）">中角</button>
    <button onclick="setAlt(25)" id="a25" title="低い角度から見る（25°）">低角</button>
    <button onclick="setAlt(8)" id="a8" title="地表すれすれ（8°）">地表</button>
  </div>
  <div style="text-align:center;margin-top:8px;border-top:1px solid #21262d;padding-top:6px">
    <svg viewBox="0 0 60 60" width="50" height="50">
      <circle cx="30" cy="30" r="28" fill="rgba(13,17,23,0.85)" stroke="rgba(48,54,61,0.8)" stroke-width="1"/>
      <g id="compassNeedle">
        <polygon points="30,6 27,28 33,28" fill="#e53e3e"/>
        <polygon points="30,54 27,32 33,32" fill="#4a5568"/>
        <circle cx="30" cy="30" r="3" fill="#a0aec0"/>
        <text x="30" y="16" text-anchor="middle" fill="#fff" font-size="9" font-weight="bold" font-family="sans-serif">N</text>
      </g>
    </svg>
  </div>
</div>

<div id="leg" class="panel">
  <div class="lb" id="lgMx">{h_max:.1f}</div>
  <div id="lbar"></div>
  <div class="lb" id="lgMn">{h_min:.1f}</div>
</div>
<div id="et"></div><div id="em"></div>
<div id="copyToast"></div>
<button id="helpBtn" onclick="document.getElementById('helpOverlay').classList.add('show')" title="説明">?</button>
<button id="saveBtn" onclick="saveOffline()" title="オフライン用に保存" style="position:fixed;bottom:14px;right:52px;width:36px;height:36px;border-radius:50%;background:rgba(13,17,23,0.9);color:#58a6ff;border:1px solid #30363d;font-size:18px;cursor:pointer;z-index:99">&#x2B73;</button>
<div id="helpOverlay" onclick="if(event.target===this)this.classList.remove('show')">
<div id="helpModal">
<button class="close" onclick="closeHelp()">閉じる</button>
<button class="close" onclick="printHelp()" style="right:90px;background:#1f6feb;border-color:#388bfd" title="説明をPDF印刷（A4縦）">&#x1F5B6; 印刷</button>
<h2>(有)徳永測量 社屋 3D地形ビューア VER2（広域1km四方） — 説明</h2>

<h3>使用データ</h3>
<table>
<tr><th>ファイル</th><th>参照フォルダ</th><th>内容</th><th>技術的位置づけ</th></tr>
<tr><td>写真点群.laz</td><td><code style="font-size:9px" onclick="copyPath(this)" title="クリックでコピー">Metashape成果フォルダ</code></td><td>UAV写真測量による3次元点群<br><span class="tag">{n_points:,}点</span> <span class="tag">66.6MB</span></td>
<td>標高データの源。点群をグリッドビニングして{GX}x{GY}セルの標高メッシュを生成</td></tr>
<tr><td>数値写真 サイズ10K.tif</td><td><code style="font-size:9px" onclick="copyPath(this)" title="クリックでコピー">Metashape成果フォルダ</code></td><td>高解像度オルソモザイク画像<br><span class="tag">GeoTIFF</span> <span class="tag">約245MB</span></td>
<td>3D地形にテクスチャマッピングする航空写真。GeoTIFF座標で自動位置合わせ</td></tr>
<tr><td>scheme.prj / .kml</td><td><code style="font-size:9px" onclick="copyPath(this)" title="クリックでコピー">Metashape成果フォルダ</code></td><td>座標系定義ファイル</td>
<td>JGD2011 平面直角座標第II系 (EPSG:6670) を定義。全データの空間参照</td></tr>
<tr><td>レポート.pdf</td><td><code style="font-size:9px" onclick="copyPath(this)" title="クリックでコピー">Metashape成果フォルダ</code></td><td>Metashape処理レポート</td>
<td>座標系・処理パラメータ・標定結果等の記録</td></tr>
</table>

<h3>3D地形モデルの構成</h3>
<p>本ビューアの3D地形モデルは、<b>正規格子（グリッド）ベースの三角形メッシュ</b>で構成されています。<br>
不整三角網（TIN）ではなく、点群を等間隔の格子に変換し、各格子を対角線で2つの三角形に分割する方式です。</p>

<table>
<tr><th>#</th><th>構成要素</th><th>内容</th><th>使用データ</th></tr>
<tr><td>1</td><td><b>標高グリッド</b></td><td>LAZ点群を {GX}×{GY} セルの等間隔格子に分割し、各セル内の点群平均標高を算出（グリッドビニング）</td><td>写真点群.laz<br><span class="tag">{n_points:,}点</span></td></tr>
<tr><td>2</td><td><b>三角形メッシュ</b></td><td>格子の各マスを対角線で2分割し三角形を生成（PlaneGeometry）。点群が存在しない範囲はマスク処理で除去し、地形の輪郭を形成</td><td>（標高グリッドから自動生成）</td></tr>
<tr><td>3</td><td><b>テクスチャ</b></td><td>GeoTIFFオルソ画像を3D地形面に貼り付け。座標系が同一（EPSG:6670）のため回転・縮尺の調整不要で自動位置合わせ</td><td>数値写真 サイズ10K.tif<br><span class="tag">GeoTIFF</span></td></tr>
<tr><td>4</td><td><b>頂点カラー</b></td><td>標高値を色に変換。表示モード（標高カラー・傾斜量図・微地形強調等）に応じて各頂点の色を切替</td><td>（標高グリッドから計算）</td></tr>
<tr><td>5</td><td><b>陰影起伏</b></td><td>地表面の法線ベクトルと仮想太陽光の内積で陰影を計算。地形の凹凸を視覚的に強調</td><td>（標高グリッドから計算）</td></tr>
<tr><td>6</td><td><b>三角網表示</b></td><td>三角形メッシュの辺を黒線で描画（「三角網」チェックボックスで表示切替）</td><td>（メッシュ構造の可視化）</td></tr>
<tr><td>7</td><td><b>ライティング</b></td><td>環境光 + 指向性光源×2 による立体感の付与（Phong陰影モデル）</td><td>（レンダリング設定）</td></tr>
</table>

<p style="margin-top:8px;font-size:11px"><b>処理の流れ</b>:</p>
<p style="font-size:11px;line-height:2;margin:4px 0">
<span class="tag" style="background:#1f6f2f">LAZ点群</span> <span style="color:#6e7681">{n_points:,}点</span>
→ <span class="tag" style="background:#1f6f2f">グリッドビニング</span> <span style="color:#6e7681">{GX}×{GY}セル</span>
→ <span class="tag" style="background:#1f6f2f">三角形メッシュ</span>
→ <span class="tag" style="background:#1f6f2f">マスク適用</span> <span style="color:#6e7681">有効{filled:,}セル</span>
→ <span class="tag" style="background:#1f6f2f">テクスチャ/色付け</span>
→ <span class="tag" style="background:#1f6f2f">Three.jsで3D表示</span>
</p>

<p style="margin-top:6px;font-size:10px;color:#6e7681">
※ グリッドのセルサイズは点群の範囲（{MW:.1f}m × {MH:.1f}m）を格子数で割った値。<br>
※ 「三角網」チェックONで、三角形の辺が黒線で確認できます。
</p>

<h3>座標系・位置合わせ</h3>
<p>Metashape成果の全出力（点群・オルソ・DSM）は<b>同一座標系 EPSG:6670</b>を共有。<br>
そのため回転=0°、スケール=100%、オフセット=0で<b>自動位置合わせ</b>が成立。手動調整は不要。</p>

<h3>使用技術</h3>
<table>
<tr><th>技術</th><th>用途</th></tr>
<tr><td><span class="tag">Three.js r0.160</span></td><td>WebGL 3Dレンダリング（PlaneGeometry + OrbitControls）</td></tr>
<tr><td><span class="tag">CanvasTexture + sRGB</span></td><td>オルソ画像の正確な色再現（頂点カラー方式では色褪せが発生するため）</td></tr>
<tr><td><span class="tag">laspy + チャンク読込</span></td><td>2GB超の.lazファイルをメモリ効率的に処理（500万点/チャンク）</td></tr>
<tr><td><span class="tag">GeoTIFFクロップ</span></td><td>オルソ画像を点群範囲でピクセル精度でクロップ</td></tr>
<tr><td><span class="tag">陰影起伏解析</span></td><td>地表面の法線ベクトルと仮想太陽光の内積で陰影を計算</td></tr>
</table>

<h3>表示モード</h3>
<table>
<tr><th>モード</th><th>内容</th></tr>
<tr><td>オルソ画像</td><td>航空写真をそのまま3D地形にマッピング</td></tr>
<tr><td>オルソ＋陰影</td><td>航空写真 × 陰影起伏（地形の凹凸が強調される）</td></tr>
<tr><td>標高カラー＋陰影</td><td>標高値を色分け + 陰影起伏</td></tr>
<tr><td>傾斜量図</td><td>地表の傾斜角度を可視化</td></tr>
<tr><td>微地形強調（CS風）</td><td>曲率+傾斜で微細な地形変化を検出</td></tr>
</table>

<h3>操作方法</h3>
<table>
<tr><th>操作</th><th>機能</th></tr>
<tr><td><b>左ドラッグ</b></td><td>回転</td></tr>
<tr><td><b>中ドラッグ</b> / <b>右ドラッグ</b></td><td>パン（平行移動）</td></tr>
<tr><td><b>両ドラッグ</b>（左右同時押し）</td><td>ズーム（上→拡大、下→縮小）※Jw_cad風</td></tr>
<tr><td><b>マウスホイール</b></td><td>ズーム</td></tr>
<tr><td><b>左クリック</b></td><td>その地点の標高値を表示</td></tr>
<tr><td><b>右クリック</b></td><td>標高表示を消去</td></tr>
<tr><td><b>中ボタンダブルクリック</b></td><td>初期視点にリセット</td></tr>
</table>

<h3>データ仕様</h3>
<ul>
<li>座標系: JGD2011 平面直角座標第II系 (EPSG:6670)</li>
<li>実寸: {MW:.1f}m × {MH:.1f}m</li>
<li>標高: {h_min:.1f}m 〜 {h_max:.1f}m</li>
<li>グリッド: {GX} × {GY} セル（有効 {filled:,} セル）</li>
<li>オルソ解像度: {ortho_source}</li>
</ul>

{gsi_help_html}

{gcp_help_html}

{chiku_help_html}

{route_help_html}

<h3>開発方法</h3>
<p>本ビューアは、<b>Claude Code</b>（Anthropic社のAIプログラミングアシスタント）が、Pythonスクリプト
<code onclick="copyPath(this)" title="クリックでコピー">generate_3d_viewer.py</code> を自動生成・更新し、Metashape成果データ（.laz点群・GeoTIFFオルソ・GCP座標等）を
読み込んで、<b>単一HTMLファイル</b>（Three.js 3Dビューア）に変換する方式で開発されています。</p>
<p>開発者が要件や修正を日本語で指示すると、Claude Codeがコードの調査・設計・実装・検証を一貫して行い、
Playwright自動テストによる品質保証まで含めて1つのワークフローとして実行します。</p>
<p>今後、別の地区・工区でも同じ方式で3Dビューアを作成できます。必要なデータ（.laz点群・GeoTIFFオルソ・SIMA座標等）を用意し、
同様の手順でPythonスクリプトを実行すれば、新しい3Dビューアを自動生成できます。</p>

<h3>GitHub Pages による公開</h3>
<p>本ビューアは <b>GitHub Pages</b> を利用してインターネット上に公開されています。</p>
<table>
<tr><th>項目</th><th>内容</th></tr>
<tr><td>GitHubとは</td><td>プログラムのソースコードをバージョン管理・共有するためのクラウドサービス（Microsoft社）</td></tr>
<tr><td>GitHub Pagesとは</td><td>GitHubのリポジトリ（保管庫）にHTMLファイルを置くだけで、Webサイトとして無料で公開できる仕組み</td></tr>
<tr><td>仕組み</td><td>① Pythonスクリプトが単一HTMLファイルを生成 → ② GitHubリポジトリにアップロード（push） → ③ GitHub Pagesが自動的にWebサイトとして配信</td></tr>
<tr><td>URL</td><td><a href="https://toku1107-cyber.github.io/tokunaga-3d-terrain-v2/" target="_blank" style="color:#58a6ff">https://toku1107-cyber.github.io/tokunaga-3d-terrain-v2/</a></td></tr>
<tr><td>アクセス</td><td>GitHub PagesのPUBLICリポジトリなのでURLを知っていれば誰でもアクセス可能</td></tr>
<tr><td>更新方法</td><td>Claude Codeが新しいHTMLを生成し、GitHubにpushすると数分で自動反映</td></tr>
</table>

<h3>スキル（経験の蓄積）と本ビューアへの具体的な貢献</h3>
<p>Claude Codeは、作業中に遭遇した問題・解決策を<b>スキルファイル</b>（<code onclick="copyPath(this)" title="クリックでコピー">.agent/skills/</code>）として自動記録します。
スキルは「過去の経験の結晶」であり、次回同種の作業を行う際に同じ失敗を繰り返さないための仕組みです。
本ビューアの開発では、以下のスキルが<b>具体的に</b>活用されました。</p>
<table>
<tr><th>スキル名</th><th>格納先</th><th>本ビューアへの具体的な貢献</th></tr>
<tr><td><b>3d-terrain-ortho-viewer</b></td><td><code onclick="copyPath(this)" title="クリックでコピー">.agent/skills/3d-terrain-ortho-viewer/</code></td>
<td><b>本ビューアの中核スキル</b>。LAZ点群→グリッドビニング、GeoTIFFアライメント、Three.js PlaneGeometry、CanvasTexture sRGB色補正、陰影起伏解析など、全ての3D表示技術の知見を蓄積。<br>新しい地区で作成する際、このスキルを読み込むことで初回と同じ試行錯誤を回避できる</td></tr>
<tr><td><b>10-agent-architecture</b></td><td><code onclick="copyPath(this)" title="クリックでコピー">.agent/skills/10-agent-architecture/</code></td>
<td>品質保証ワークフロー。PM（要件整理）→ アーキテクト（調査）→ ディスパッチャー（タスク分割）→ 実装 → セルフレビュー → QAの一貫プロセスを適用。<br>「いきなりコードを書かない、まず調査」の原則により、既存コードとの整合性を確保</td></tr>
<tr><td><b>windows-terminal-emoji-encode-fix</b></td><td><code onclick="copyPath(this)" title="クリックでコピー">.agent/skills/windows-terminal-emoji-encode-fix/</code></td>
<td>Windows環境でのPython実行時、cp932エンコードエラーを回避。<br><code>sys.stdout.reconfigure(encoding='utf-8')</code>をスクリプト冒頭に必ず適用するルール</td></tr>
<tr><td><b>no-overconfidence-protocol</b></td><td><code onclick="copyPath(this)" title="クリックでコピー">.agent/skills/no-overconfidence-protocol/</code></td>
<td>「原因が特定できました」の禁止ルール。AIが100%の断定をせず、仮説ベースで報告する義務。<br>「修正します→できていません」の往復ループを防止</td></tr>
<tr><td><b>excel-screenshot-verification</b></td><td><code onclick="copyPath(this)" title="クリックでコピー">.agent/skills/excel-screenshot-verification/</code></td>
<td>スクリーンショットによる目視検証手法を、HTMLビューアの品質確認に応用。<br>Playwrightでスクリーンショットを撮影→Read toolで画像確認→レイアウト崩れ等を検出</td></tr>
</table>
<p style="margin-top:8px"><b>スキルの仕組み</b>: 難航した経験（差し戻し2回以上、試行錯誤30分以上等）が自動的にスキルとして記録される。
次のセッションでClaude Codeは作業前にスキルを読み込み、「前回のハマりポイント」を事前に把握して着手するため、同じ問題での停滞が激減する。</p>

<h3>次回、同種の3Dビューアを作成するには</h3>
<p>別の地区・工区でも同じ方式で3Dビューアを作成できます。以下に、人間側の準備とClaude Codeへの指示方法をまとめます。</p>

<h4>① 人間が事前に用意するもの</h4>
<table>
<tr><th>#</th><th>ファイル</th><th>入手元</th><th>必須/任意</th><th>備考</th></tr>
<tr><td>1</td><td><b>写真点群.laz</b></td><td>Metashape成果</td><td>★必須</td><td>2GB超でも処理可能（チャンク読込）</td></tr>
<tr><td>2</td><td><b>数値写真（オルソ）.tif</b></td><td>Metashape成果</td><td>★必須</td><td>サイズ10K推奨。GeoTIFF形式（座標埋込）</td></tr>
<tr><td>3</td><td><b>scheme.prj / .kml</b></td><td>Metashape成果</td><td>★必須</td><td>座標系定義。オルソTIFと同じフォルダ</td></tr>
<tr><td>4</td><td><b>GCP/SIMA座標</b></td><td>現場測量</td><td>任意</td><td>基準点・水準点を地図上に表示</td></tr>
<tr><td>5</td><td><b>筆界データ</b></td><td>G空間情報センター</td><td>任意</td><td>登記所備付地図XML（ZIP形式）</td></tr>
<tr><td>6</td><td><b>路線データ</b></td><td>路線測量SIMA</td><td>任意</td><td>結線情報ありのSIMAファイル</td></tr>
</table>

<h4>② Claude Codeへのコピペ指示文（テンプレート）</h4>
<p>新しいセッションで以下をコピペして貼り付けるだけで、Claude Codeが作業を開始します。<br>
【】内を実際のパスに置き換えてください。</p>
<div class="cpre-wrap"><button class="cpre-btn" onclick="copyPre(this)">コピー</button><pre style="background:#161b22;padding:12px;border-radius:6px;font-size:10px;overflow-x:auto;white-space:pre-wrap">
3D地形ビューアを作成してください。
CLAUDE.mdを厳守してください。

■ スキル読込（最初に必ず実行）
.agent/skills/3d-terrain-ortho-viewer/SKILL.md

■ 作業フォルダ
【G:\マイドライブ\00_MyAI_Workspace\...\作業フォルダ名】

■ 入力データ
- 点群: 【フォルダ内の】写真点群.laz
- オルソ: 【フォルダ内の】数値写真 サイズ10K.tif
- 座標系: 【フォルダ内の】scheme.prj, scheme.kml
- GCP: 【パス】.sim （任意）
- 筆界: 【パス】.zip （任意・G空間情報センター）
- 路線: 【パス】.sim （任意・結線ありSIMA）

■ 出力
- 単一HTMLファイル（Three.js 3Dビューア）
- GitHub Pages配信（希望する場合）

■ 地区名・タイトル
【例: (有)徳永測量 社屋】 / 【例: R7○○地区】
</pre></div>

<h4>③ Claude Codeが質問する内容（Ask User Question）</h4>
<p>作業中に以下のような質問をする場合があります。事前に把握しておくとスムーズです。</p>
<table>
<tr><th>質問</th><th>意味</th><th>回答例</th></tr>
<tr><td>SIMA座標系は平面直角II系ですか？</td><td>座標の解釈に必要</td><td>「はい、II系です」</td></tr>
<tr><td>筆界ZIPに複数地域が含まれますが対象は？</td><td>必要な範囲を絞り込む</td><td>「対象地区周辺のみ」</td></tr>
<tr><td>路線のZ値は計画高ですか、現況高ですか？</td><td>表示方法に影響</td><td>「全てZ=0なので地形に沿わせて」</td></tr>
<tr><td>GitHub Pagesで公開しますか？</td><td>インターネット公開の要否</td><td>「はい」/「今回は不要」</td></tr>
<tr><td>国土地理院背景は必要ですか？</td><td>周辺の地図表示</td><td>「はい、航空写真+標準地図」</td></tr>
</table>

<p style="margin-top:16px;font-size:10px;color:#6e7681;border-top:1px solid #21262d;padding-top:8px">
生成: generate_3d_viewer.py (v23) | Metashape成果データ → HTML変換<br>
(有)徳永測量 社屋 — 自社案件
</p>
</div>
</div>

<script type="importmap">
{{"imports":{{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}}}
</script>
<script type="module">
import * as THREE from 'three';
import {{OrbitControls}} from 'three/addons/controls/OrbitControls.js';

function b64f32(b){{const s=atob(b),a=new Uint8Array(s.length);for(let i=0;i<s.length;i++)a[i]=s.charCodeAt(i);return new Float32Array(a.buffer)}}
function b64u8(b){{const s=atob(b),a=new Uint8Array(s.length);for(let i=0;i<s.length;i++)a[i]=s.charCodeAt(i);return a}}
const H=b64f32('{h_b64}');
const M=b64u8('{m_b64}');
const GX={GX},GY={GY},MW={MW:.4f},MH={MH:.4f};
const HMIN={h_min:.4f},HMAX={h_max:.4f},ZC={ZC:.4f},XC=0.0000,YC=0.0000;
const WXmin={x_min:.4f},WXmax={x_max:.4f},WYmin={y_min:.4f},WYmax={y_max:.4f};

const scene=new THREE.Scene();scene.background=new THREE.Color(0x0d1117);
let curBg='dark';
window.setBg=function(mode){{
  curBg=mode;
  const isDark=mode==='dark';
  const bgCol=isDark?0x0d1117:0xffffff;
  const panelBg=isDark?'rgba(13,17,23,0.93)':'rgba(240,240,240,0.93)';
  const panelBdr=isDark?'rgba(48,54,61,0.8)':'rgba(180,180,180,0.8)';
  const txtCol=isDark?'#c9d1d9':'#1a1a1a';
  const subTxt=isDark?'#8b949e':'#555';
  const keyBg=isDark?'#21262d':'#e8e8e8';
  const keyBdr=isDark?'#30363d':'#ccc';
  const keyTxt=isDark?'#79c0ff':'#0066cc';
  const btnBg=isDark?'#21262d':'#e0e0e0';
  const btnBdr=isDark?'#30363d':'#bbb';
  const btnTxt=isDark?'#c9d1d9':'#333';
  const inpBg=isDark?'#0d1117':'#fff';
  const inpBdr=isDark?'#30363d':'#bbb';
  scene.background=new THREE.Color(bgCol);
  document.body.style.background=isDark?'#0d1117':'#fff';
  document.body.style.color=txtCol;
  document.querySelectorAll('.panel').forEach(p=>{{p.style.background=panelBg;p.style.borderColor=panelBdr;p.style.color=txtCol}});
  document.querySelectorAll('.key').forEach(k=>{{k.style.background=keyBg;k.style.borderColor=keyBdr;k.style.color=keyTxt}});
  document.querySelectorAll('#info .d').forEach(d=>d.style.color=subTxt);
  document.querySelectorAll('.rw label, .lb, .st, .st td').forEach(e=>e.style.color=subTxt);
  document.querySelectorAll('.rw select, .rw input[type=number]').forEach(e=>{{e.style.background=inpBg;e.style.borderColor=inpBdr;e.style.color=isDark?'#79c0ff':'#0066cc'}});
  document.querySelectorAll('.dg button, .ag button').forEach(b=>{{if(!b.classList.contains('act')){{b.style.background=btnBg;b.style.borderColor=btnBdr;b.style.color=btnTxt}}}});
  document.querySelectorAll('.bd').forEach(b=>{{b.style.background=btnBg+'!important';b.style.borderColor=btnBdr;b.style.color=btnTxt}});
  document.getElementById('bgBlk').style.background=isDark?'#1f6f2f':btnBg;
  document.getElementById('bgBlk').style.color=isDark?'#fff':btnTxt;
  document.getElementById('bgBlk').style.borderColor=isDark?'#3fb950':btnBdr;
  document.getElementById('bgWht').style.background=isDark?btnBg:'#1f6f2f';
  document.getElementById('bgWht').style.color=isDark?btnTxt:'#fff';
  document.getElementById('bgWht').style.borderColor=isDark?btnBdr:'#3fb950';
}};
const cam=new THREE.PerspectiveCamera(45,innerWidth/innerHeight,0.1,2000);
cam.up.set(0,0,1);
const D={cam_dist:.1f};
const ren=new THREE.WebGLRenderer({{antialias:true}});
ren.setSize(innerWidth,innerHeight);ren.setPixelRatio(devicePixelRatio);
document.body.appendChild(ren.domElement);
const ctl=new OrbitControls(cam,ren.domElement);
ctl.enableDamping=true;ctl.dampingFactor=0.08;ctl.target.set(0,0,0);
ctl.mouseButtons={{LEFT:THREE.MOUSE.ROTATE,MIDDLE:THREE.MOUSE.PAN,RIGHT:THREE.MOUSE.PAN}};
ctl.minPolarAngle=0.03;ctl.maxPolarAngle=Math.PI*0.48;

// ---- 視点制御 ----
let curAz=225, curAlt=55;
function setCam(px,py,pz){{
  const s={{x:cam.position.x,y:cam.position.y,z:cam.position.z}};
  const dur=350,t0=performance.now();
  (function step(){{
    const t=Math.min(1,(performance.now()-t0)/dur);
    const e=t<.5?2*t*t:1-(-2*t+2)**2/2;
    cam.position.set(s.x+(px-s.x)*e,s.y+(py-s.y)*e,s.z+(pz-s.z)*e);
    ctl.update();if(t<1)requestAnimationFrame(step);
  }})();
}}
function applyBtnTheme(el,isAct){{
  if(!el)return;
  if(isAct){{el.style.background='#1f6f2f';el.style.color='#fff';el.style.borderColor='#3fb950'}}
  else if(curBg==='dark'){{el.style.background='#21262d';el.style.color='#c9d1d9';el.style.borderColor='#30363d'}}
  else{{el.style.background='#e0e0e0';el.style.color='#333';el.style.borderColor='#bbb'}}
}}
function viewUpdate(){{
  document.querySelectorAll('.dg button,.ag button').forEach(b=>{{b.classList.remove('act');applyBtnTheme(b,false)}});
  const setAct=function(el){{if(el){{el.classList.add('act');applyBtnTheme(el,true)}}}};
  if(curAz===-1)setAct(document.getElementById('dtop'));
  else setAct(document.getElementById('d'+curAz));
  setAct(document.getElementById('a'+curAlt));
  if(curAz===-1){{ setCam(0,0,D); return }}
  const r=D*0.85;
  const aR=curAlt*Math.PI/180, azR=curAz*Math.PI/180;
  setCam(r*Math.cos(aR)*Math.sin(azR), r*Math.cos(aR)*Math.cos(azR), r*Math.sin(aR));
}}
window.setDir=function(az){{curAz=az;if(az===-1)curAlt=88;else if(curAlt>=85)curAlt=55;viewUpdate()}};
window.setAlt=function(al){{curAlt=al;if(curAz===-1)curAz=225;viewUpdate()}};
viewUpdate();

let midT=0;
ren.domElement.addEventListener('mousedown',e=>{{if(e.button===1){{const n=Date.now();if(n-midT<350){{curAz=225;curAlt=55;viewUpdate()}}midT=n}}}});

// ---- 両ドラッグ（左右同時押し: Jw_cad風ズーム） ----
let _bd=false,_bsy=0,_bsd=0;
ren.domElement.addEventListener('mousedown',function(e){{
  if(e.buttons===3){{_bd=true;_bsy=e.clientY;_bsd=cam.position.distanceTo(ctl.target);ctl.enabled=false;e.preventDefault()}}
}});
window.addEventListener('mousemove',function(e){{
  if(!_bd)return;
  const dy=e.clientY-_bsy;
  const nd=_bsd*Math.exp(dy*0.005);
  const dr=cam.position.clone().sub(ctl.target).normalize();
  cam.position.copy(ctl.target).addScaledVector(dr,nd);ctl.update();
}});
window.addEventListener('mouseup',function(){{if(_bd){{_bd=false;ctl.enabled=true}}}});

// ---- Geometry ----
const geo=new THREE.PlaneGeometry(MW,MH,GX-1,GY-1);
const pos=geo.attributes.position.array;
const origZ=new Float32Array(GX*GY);
for(let i=0;i<GX*GY;i++){{origZ[i]=H[i]-ZC;pos[i*3+2]=origZ[i]}}
geo.setAttribute('color',new THREE.BufferAttribute(new Float32Array(GX*GY*3),3));
const oi=geo.index.array,ni=[];
for(let i=0;i<oi.length;i+=3){{if(M[oi[i]]&&M[oi[i+1]]&&M[oi[i+2]])ni.push(oi[i],oi[i+1],oi[i+2])}}
geo.setIndex(ni);geo.computeVertexNormals();
const mat=new THREE.MeshPhongMaterial({{vertexColors:true,side:THREE.DoubleSide,shininess:12}});
const omat=new THREE.MeshBasicMaterial({{vertexColors:true,side:THREE.DoubleSide}});
const ter=new THREE.Mesh(geo,mat);scene.add(ter);
const wM=new THREE.MeshBasicMaterial({{color:0x000000,wireframe:true,transparent:true,opacity:0.03}});
const wir=new THREE.Mesh(geo,wM);wir.visible=false;scene.add(wir);
window.toggleTerrain=function(v){{ter.visible=v;wir.visible=v&&document.getElementById('dMode')?.value==='wireframe'}};

scene.add(new THREE.AmbientLight(0x606878,0.6));
const sn=new THREE.DirectionalLight(0xfff5e0,1);sn.position.set(40,-20,60);scene.add(sn);
const sn2=new THREE.DirectionalLight(0x8090b0,0.4);sn2.position.set(-30,40,30);scene.add(sn2);

{ortho_js_data}
{ortho_js_loader}

{overlay_js}

{gsi_js}

{gcp_js}

{chiku_js}

{route_js}

// ---- 微地形解析 ----
const cW=MW/(GX-1),cH=MH/(GY-1);
const gx_=new Float32Array(GX*GY),gy_=new Float32Array(GX*GY),cv=new Float32Array(GX*GY);
for(let iy=1;iy<GY-1;iy++)for(let ix=1;ix<GX-1;ix++){{
  const i=iy*GX+ix;
  gx_[i]=(H[i+1]-H[i-1])/(2*cW);gy_[i]=(H[i+GX]-H[i-GX])/(2*cH);
  cv[i]=(H[i+1]+H[i-1]-2*H[i])/(cW*cW)+(H[i+GX]+H[i-GX]-2*H[i])/(cH*cH);
}}

// ---- カラースキーム ----
const SCHEMES={{
  terrain:[[0,.10,.29,.08],[.15,.18,.45,.12],[.3,.30,.60,.18],[.45,.48,.71,.28],[.6,.65,.76,.30],[.75,.80,.68,.28],[.9,.90,.82,.55],[1,.96,.93,.82]],
  gray:[[0,.12,.12,.14],[.5,.55,.55,.57],[1,.95,.95,.93]],
  warm:[[0,.18,.06,.04],[.2,.45,.12,.05],[.4,.72,.30,.08],[.6,.88,.55,.12],[.8,.95,.78,.35],[1,.98,.93,.75]],
  cool:[[0,.04,.08,.28],[.2,.08,.22,.48],[.4,.15,.42,.55],[.6,.30,.62,.52],[.8,.60,.82,.68],[1,.92,.97,.98]],
  rainbow:[[0,.15,.0,.42],[.2,.0,.18,.78],[.4,.0,.65,.35],[.6,.50,.75,.0],[.8,.88,.60,.0],[1,.85,.12,.12]],
  red_relief:[[0,.12,.0,.0],[.2,.35,.05,.03],[.4,.58,.12,.06],[.6,.78,.28,.12],[.8,.92,.55,.30],[1,.98,.88,.75]]
}};
function schemeColor(t,sch){{
  const S=SCHEMES[sch]||SCHEMES.terrain;
  for(let i=0;i<S.length-1;i++){{if(t<=S[i+1][0]){{const f=(t-S[i][0])/(S[i+1][0]-S[i][0]);
    return[S[i][1]+f*(S[i+1][1]-S[i][1]),S[i][2]+f*(S[i+1][2]-S[i][2]),S[i][3]+f*(S[i+1][3]-S[i][3])]}}}}
  const L=S[S.length-1];return[L[1],L[2],L[3]];
}}

// ---- 統合カラー更新 ----
let applyTimer=null;
window.applyAll=function(){{
  clearTimeout(applyTimer);
  applyTimer=setTimeout(_applyAll,50);
}};
function _applyAll(){{
  const isOrthoMode=document.getElementById('dMode').value.startsWith('ortho');
  const oc=document.getElementById('orthoCtl');if(oc)oc.style.display=isOrthoMode?'':'none';
  const br=document.getElementById('blendRow');if(br)br.style.display=isOrthoMode?'':'none';
  const bv=document.getElementById('blendV');if(bv)bv.textContent=document.getElementById('blendS').value+'%';
  const cMin=parseFloat(document.getElementById('cMin').value);
  const cMax=parseFloat(document.getElementById('cMax').value);
  const mode=document.getElementById('dMode').value;
  const sch=document.getElementById('cScheme').value;
  const sunAlt=parseFloat(document.getElementById('sunA').value);
  const mZv=parseFloat(document.getElementById('mZ').value);
  document.getElementById('sunV').textContent=sunAlt+'\\u00b0';
  document.getElementById('mZV').textContent=mZv+'x';
  document.getElementById('lgMn').textContent=cMin.toFixed(1);
  document.getElementById('lgMx').textContent=cMax.toFixed(1);

  const aR=sunAlt*Math.PI/180,azR=315*Math.PI/180;
  const sx=Math.cos(aR)*Math.sin(azR),sy=Math.cos(aR)*Math.cos(azR),sz=Math.sin(aR);
  const col=geo.attributes.color.array;
  const hR=cMax-cMin||.001;
  for(let i=0;i<GX*GY;i++){{
    const h=H[i];
    const dx=gx_[i]*mZv,dy=gy_[i]*mZv;
    const len=Math.sqrt(dx*dx+dy*dy+1);
    const hs=Math.max(0,(-dx*sx-dy*sy+sz)/len);
    const sl=Math.sqrt(dx*dx+dy*dy);
    let r,g,b;
    if(mode==='ortho_hs'){{
{ortho_js_color}
    }}else if(mode==='elev_hs'){{
      const t=Math.max(0,Math.min(1,(h-cMin)/hR));
      [r,g,b]=schemeColor(t,sch);
      const sh=.25+.75*hs;r*=sh;g*=sh;b*=sh;
    }}else if(mode==='slope'){{
      const s=Math.min(1,sl*.8);
      [r,g,b]=schemeColor(1-s,sch);
      const sh=.4+.6*hs;r*=sh;g*=sh;b*=sh;
    }}else{{
      const cn=Math.max(-1,Math.min(1,cv[i]*mZv*500));
      const sn2=Math.min(1,sl*1.5);
      if(cn<0){{r=.5+.5*(-cn);g=.45*(1+cn);b=.4*(1+cn)}}
      else{{r=.45*(1-cn);g=.45*(1-cn);b=.5+.5*cn}}
      const gy2=.55;r=gy2+(r-gy2)*(.3+.7*sn2);g=gy2+(g-gy2)*(.3+.7*sn2);b=gy2+(b-gy2)*(.3+.7*sn2);
      const sh=.3+.7*hs;r*=sh;g*=sh;b*=sh;
    }}
    col[i*3]=Math.max(0,Math.min(1,r));col[i*3+1]=Math.max(0,Math.min(1,g));col[i*3+2]=Math.max(0,Math.min(1,b));
  }}
  geo.attributes.color.needsUpdate=true;
  const isOrtho=mode==='ortho_hs';
  ter.material=isOrtho?omat:mat;
  document.getElementById('leg').style.display=isOrtho?'none':'';
  const stops=SCHEMES[sch]||SCHEMES.terrain;
  let grad='linear-gradient(to bottom,';
  for(let i=stops.length-1;i>=0;i--){{
    const s=stops[i];
    grad+='rgb('+Math.round(s[1]*255)+','+Math.round(s[2]*255)+','+Math.round(s[3]*255)+')';
    if(i>0)grad+=',';
  }}
  document.getElementById('lbar').style.background=grad+')';
}}

// ---- 左クリック→標高 ----
const rc=new THREE.Raycaster(),v2=new THREE.Vector2();
let mp={{x:0,y:0}};
ren.domElement.addEventListener('mousedown',e=>{{if(e.button===0)mp={{x:e.clientX,y:e.clientY}}}});
ren.domElement.addEventListener('mouseup',e=>{{
  if(e.button!==0||Math.sqrt((e.clientX-mp.x)**2+(e.clientY-mp.y)**2)>5)return;
  v2.x=(e.clientX/innerWidth)*2-1;v2.y=-(e.clientY/innerHeight)*2+1;
  rc.setFromCamera(v2,cam);
  const hits=rc.intersectObject(ter);
  const tip=document.getElementById('et'),mk=document.getElementById('em');
  if(hits.length){{
    const p=hits[0].point,zs=parseFloat(document.getElementById('zS').value)||1;
    const rZ=p.z/zs+ZC,rX=p.x+XC,rY=p.y+YC;
    tip.style.display='block';tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY-32)+'px';
    tip.innerHTML='標高: <b style="color:#3fb950;font-size:14px">'+rZ.toFixed(3)+'</b> m<br><span style="color:#8b949e;font-size:10px">X:'+rX.toFixed(3)+' Y:'+rY.toFixed(3)+'</span>';
    mk.style.display='block';mk.style.left=e.clientX+'px';mk.style.top=e.clientY+'px';
  }}else{{tip.style.display='none';mk.style.display='none'}}
}});
// 右クリック→標高表示消去
ren.domElement.addEventListener('contextmenu',e=>{{
  const tip=document.getElementById('et'),mk=document.getElementById('em');
  if(tip.style.display==='block'){{tip.style.display='none';mk.style.display='none';e.preventDefault()}}
}});

// ---- localStorage ----
const SK='tokunaga_shaya_uav_v2';
function loadDef(){{try{{const d=JSON.parse(localStorage.getItem(SK));if(!d)return;
  ['cMin','cMax','dMode','cScheme','sunA','mZ'{ortho_save_keys}].forEach(k=>{{if(d[k]!=null)document.getElementById(k).value=d[k]}});{ortho_save_flip}
  if(d.bg)setBg(d.bg);
  ['showTerrain','showOvl','wf','fs','showGcp','showChiku','showChikuLbl','showRoute','showCorridor','showRouteLbl','showGsi','gsiCutout'].forEach(k=>{{
    const el=document.getElementById(k);if(el&&d[k]!=null){{el.checked=d[k];if(el.onchange)el.onchange({{target:el}});else{{const ev=new Event('change');el.dispatchEvent(ev)}}}}
  }});
  if(d.gsiType!=null){{const gt=document.getElementById('gsiType');if(gt){{gt.value=d.gsiType;if(window.switchGsiType)switchGsiType(d.gsiType)}}}}
  if(d.zS!=null){{const zEl=document.getElementById('zS');if(zEl){{zEl.value=d.zS;const zV=document.getElementById('zV');if(zV)zV.textContent=parseFloat(d.zS).toFixed(1);updZ(parseFloat(d.zS))}}}}
  if(d.curAz!=null&&d.curAlt!=null){{curAz=d.curAz;curAlt=d.curAlt;viewUpdate()}}
}}catch(e){{}}}}
window.saveDef=function(){{try{{const d={{}};
  ['cMin','cMax','dMode','cScheme','sunA','mZ'{ortho_save_keys}].forEach(k=>d[k]=document.getElementById(k).value);{ortho_save_flip_w}
  d.bg=curBg;
  ['showTerrain','showOvl','wf','fs','showGcp','showChiku','showChikuLbl','showRoute','showCorridor','showRouteLbl','showGsi','gsiCutout'].forEach(k=>{{
    const el=document.getElementById(k);if(el)d[k]=el.checked;
  }});
  const gt=document.getElementById('gsiType');if(gt)d.gsiType=gt.value;
  const zEl=document.getElementById('zS');if(zEl)d.zS=zEl.value;
  d.curAz=curAz;d.curAlt=curAlt;
  localStorage.setItem(SK,JSON.stringify(d))}}catch(e){{}}
  const m=document.getElementById('svM');m.style.opacity='1';setTimeout(()=>m.style.opacity='0',2000);
}};
window.rstRange=function(){{document.getElementById('cMin').value={h_min:.1f};document.getElementById('cMax').value={h_max:.1f};applyAll()}};
loadDef();applyAll();

// スライダー debounce
['sunA','mZ'{ortho_debounce}].forEach(id=>{{
  const el=document.getElementById(id);if(el)el.addEventListener('input',function(){{
    {ortho_debounce_display}
    applyAll();
  }});
}});

// ---- Z/wireframe/flat ----
function updZ(s){{const p=geo.attributes.position.array;for(let i=0;i<origZ.length;i++)p[i*3+2]=origZ[i]*s;
  geo.attributes.position.needsUpdate=true;geo.computeVertexNormals();buildOvl(s){gsi_updz};if(window.updateGcpZ)updateGcpZ(s);if(window.updateChikuZ)updateChikuZ(s);if(window.updateRouteZ)updateRouteZ(s)}}
document.getElementById('zS').addEventListener('input',e=>{{
  const v=parseFloat(e.target.value);document.getElementById('zV').textContent=v.toFixed(1);updZ(v)}});
document.getElementById('wf').addEventListener('change',e=>wir.visible=e.target.checked);
document.getElementById('fs').addEventListener('change',e=>{{
  mat.flatShading=e.target.checked;mat.needsUpdate=true;geo.computeVertexNormals()}});

window.addEventListener('resize',()=>{{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();ren.setSize(innerWidth,innerHeight)}});
(function anim(){{requestAnimationFrame(anim);ctl.update();
  const _dx=ctl.target.x-cam.position.x,_dy=ctl.target.y-cam.position.y;
  const _az=Math.atan2(_dx,_dy)*180/Math.PI;
  const cn=document.getElementById('compassNeedle');
  if(cn)cn.setAttribute('transform','rotate('+(-_az)+' 30 30)');
  ren.render(scene,cam)}})();
// ---- オフライン保存 ----
window.saveOffline=function(){{
  const blob=new Blob([document.documentElement.outerHTML],{{type:'text/html;charset=utf-8'}});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);
  a.download='徳永測量社屋VER2_3dviewer.html';a.click();URL.revokeObjectURL(a.href);
}};

// ---- 説明欄印刷（A4縦 PDF） ----
window.printHelp=function(){{
  const modal=document.getElementById('helpModal');if(!modal)return;
  const w=window.open('','_blank');
  w.document.write('<html><head><meta charset="UTF-8"><title>(有)徳永測量 社屋 3D地形ビューア VER2 - 説明</title>');
  w.document.write('<style>');
  w.document.write('@page{{size:A4 portrait;margin:15mm 12mm}}');
  w.document.write('body{{font-family:"Yu Gothic","Meiryo",sans-serif;font-size:11px;line-height:1.6;color:#222;max-width:180mm}}');
  w.document.write('h2{{font-size:16px;border-bottom:2px solid #333;padding-bottom:4px;margin:12px 0 8px}}');
  w.document.write('h3{{font-size:13px;border-bottom:1px solid #999;padding-bottom:2px;margin:10px 0 6px;page-break-after:avoid}}');
  w.document.write('h4{{font-size:12px;margin:8px 0 4px;page-break-after:avoid}}');
  w.document.write('table{{border-collapse:collapse;width:100%;margin:6px 0;page-break-inside:avoid}}');
  w.document.write('th,td{{border:1px solid #aaa;padding:3px 6px;font-size:10px;text-align:left}}');
  w.document.write('th{{background:#e8e8e8;font-weight:bold}}');
  w.document.write('pre{{background:#f4f4f4!important;color:#222!important;padding:8px;border:1px solid #ccc;font-size:9px;white-space:pre-wrap;page-break-inside:avoid}}');
  w.document.write('code{{background:#f0f0f0;padding:1px 3px;font-size:9.5px}}');
  w.document.write('.tag{{display:inline-block;background:#e0e0e0;border-radius:3px;padding:1px 5px;font-size:9px;margin:1px}}');
  w.document.write('p{{margin:4px 0}}');
  w.document.write('ul{{margin:4px 0;padding-left:18px}}');
  w.document.write('li{{margin:2px 0}}');
  w.document.write('a{{color:#0366d6;text-decoration:none}}');
  w.document.write('button,.close,.cpre-btn{{display:none!important}}');
  w.document.write('code::after{{content:none!important}}');
  w.document.write('</style></head><body>');
  w.document.write(modal.innerHTML);
  w.document.write('</body></html>');
  w.document.close();
  setTimeout(function(){{w.print();w.close()}},500);
}};
// ---- パスコピー機能 ----
window.copyPath=function(el){{
  const txt=el.textContent.trim();
  const hasBackslash=txt.includes('\\\\');
  const isFile=hasBackslash&&/\\.\\w{{1,5}}$/.test(txt);
  const folderPath=isFile?txt.replace(/\\\\[^\\\\]+$/,''):txt;
  navigator.clipboard.writeText(folderPath).then(function(){{
    const t=document.getElementById('copyToast');
    t.innerHTML=isFile
      ?'<b>'+folderPath+'</b><br><span style="font-size:11px">フォルダパスをコピーしました。Win+E → アドレスバーに貼り付けて開けます</span>'
      :'<b>'+folderPath+'</b><br><span style="font-size:11px">Win+E → アドレスバーに貼り付けて開けます</span>';
    t.classList.add('show');
    setTimeout(function(){{t.classList.remove('show')}},3000);
  }});
}};
window.copyPre=function(btn){{
  const pre=btn.parentElement.querySelector('pre');
  if(!pre)return;
  navigator.clipboard.writeText(pre.textContent.trim()).then(function(){{
    const t=document.getElementById('copyToast');
    t.textContent='テンプレートをコピーしました';
    t.classList.add('show');
    btn.textContent='コピー済';
    setTimeout(function(){{t.classList.remove('show');btn.textContent='コピー'}},2500);
  }});
}};
{modal_zoom_js}
{panel_zoom_js}
{vp_zoom_js}
</script>
</body>
</html>'''

with open(str(OUTPUT_HTML), 'w', encoding='utf-8') as f:
    f.write(html)

elapsed = time.time() - t0
size_mb = OUTPUT_HTML.stat().st_size / 1024 / 1024
print(f'\n=== 完了 ({elapsed:.1f}秒) ===')
print(f'出力: {OUTPUT_HTML}')
print(f'ファイルサイズ: {size_mb:.1f} MB')
print(f'グリッド: {GX}x{GY} ({filled:,}セル)')
print(f'標高: [{h_min:.4f}, {h_max:.4f}]')
print(f'オルソ: {ortho_source}')
