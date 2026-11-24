"""
features.py  -  영상처리 실습 프로그램의 '영상처리 기능' 모음 (tkinter 와 무관한 순수 처리 코드)

구조
  S  : 슬라이더 정의      (label, 최소, 최대, 기본값, 간격)
  P  : 패널(결과 이미지 1장) 정의  (제목, 처리함수, 슬라이더 목록)
  F  : 기능(화면 1개) 정의  (이름, 설명, 패널 목록, 추가 버튼)
  Ctx: 현재 불러온 이미지를 담는 객체

처리함수 규칙:  func(ctx, *슬라이더값) -> 이미지  또는  (이미지, '캡션 글자')
"""
import math
from dataclasses import dataclass, field
from typing import Callable, List

import cv2
import numpy as np

MAX_SIDE = 480            # 실습용 작업 이미지의 긴 변 최대 크기(속도를 위해 축소)
GRAY_BG = (128, 128, 128)  # 기하 변환에서 비는 영역을 채울 색


# ----------------------------------------------------------------------
# 데이터 구조
# ----------------------------------------------------------------------
@dataclass
class S:
    label: str
    lo: float
    hi: float
    default: float
    res: float = 1


@dataclass
class P:
    title: str
    func: Callable
    sliders: List[S] = field(default_factory=list)
    heavy: bool = False          # True 이면 슬라이더를 놓을 때만 다시 계산


@dataclass
class F:
    name: str
    desc: str
    panels: List[P]
    buttons: list = field(default_factory=list)   # [(버튼 글자, 동작 이름)]
    extra: bool = False                            # True 이면 '추가기능' 표시


class Ctx:
    """불러온 이미지 정보 (원본 / 작업용 축소본 / 그레이 / 합성용 두 번째 이미지)"""

    def __init__(self, buf):
        self.buf = buf
        self.orig = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if self.orig is None:
            raise ValueError("이미지를 해석할 수 없습니다.")
        self.unchanged = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        h, w = self.orig.shape[:2]
        s = MAX_SIDE / max(h, w)
        if s < 1:
            self.img = cv2.resize(self.orig, (max(1, round(w * s)), max(1, round(h * s))),
                                  interpolation=cv2.INTER_AREA)
        else:
            self.img = self.orig.copy()
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.img2_custom = None

    def img2(self):
        """영상 합성에 쓰는 두 번째 이미지 (기본: 무지개 그라데이션)"""
        h, w = self.img.shape[:2]
        if self.img2_custom is not None:
            return cv2.resize(self.img2_custom, (w, h), interpolation=cv2.INTER_AREA)
        hue = np.tile(np.linspace(0, 179, w, dtype=np.float32), (h, 1)).astype(np.uint8)
        sat = np.full((h, w), 200, np.uint8)
        val = np.tile(np.linspace(255, 120, h, dtype=np.float32)[:, None], (1, w)).astype(np.uint8)
        return cv2.cvtColor(cv2.merge([hue, sat, val]), cv2.COLOR_HSV2BGR)


# ----------------------------------------------------------------------
# 공통 도우미
# ----------------------------------------------------------------------
def odd(k, lo=1):
    k = max(int(k), lo)
    return k if k % 2 == 1 else k + 1


def norm_u8(a):
    a = np.asarray(a, np.float32)
    mn, mx = float(a.min()), float(a.max())
    if mx - mn < 1e-9:
        return np.zeros(a.shape, np.uint8)
    return ((a - mn) * (255.0 / (mx - mn))).astype(np.uint8)


def clip_u8(a):
    return np.clip(a, 0, 255).astype(np.uint8)


def on_canvas(img, hw, bg=235):
    """img 를 (H,W) 크기 캔버스 가운데에 실제 크기 그대로 올린다(넘치면 잘림)."""
    H, W = hw
    h, w = img.shape[:2]
    out = np.full((H, W) + img.shape[2:], bg, img.dtype)
    ys, xs = (H - h) // 2, (W - w) // 2
    sy0, sx0 = max(0, -ys), max(0, -xs)
    dy0, dx0 = max(0, ys), max(0, xs)
    hh, ww = min(h - sy0, H - dy0), min(w - sx0, W - dx0)
    if hh > 0 and ww > 0:
        out[dy0:dy0 + hh, dx0:dx0 + ww] = img[sy0:sy0 + hh, sx0:sx0 + ww]
    return out


def warp_fit(img, M, border=GRAY_BG):
    """변환 후 이미지가 잘리지 않도록 출력 크기를 자동 계산해서 warpAffine"""
    M = np.asarray(M, np.float64)
    h, w = img.shape[:2]
    pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])[None]
    t = cv2.transform(pts, M)[0]
    x0, y0 = t.min(0)
    x1, y1 = t.max(0)
    M2 = M.copy()
    M2[0, 2] -= x0
    M2[1, 2] -= y0
    nw, nh = int(math.ceil(x1 - x0)), int(math.ceil(y1 - y0))
    return cv2.warpAffine(img, M2, (max(nw, 1), max(nh, 1)), borderValue=border), M2


def fmt_mat(M):
    return "\n".join("[" + "  ".join(f"{v:8.3f}" for v in row) + "]" for row in np.asarray(M))


def encode_image(img, ext, params=()):
    ok, buf = cv2.imencode(ext, img, list(params))
    if not ok:
        raise ValueError(f"{ext} 형식으로 저장할 수 없습니다.")
    return buf


def gamma_lut(g):
    return np.array([((i / 255.0) ** (1.0 / g)) * 255 for i in range(256)], np.uint8)


def hist_plot(g, log=0):
    """그레이 이미지의 히스토그램을 256x256 그림으로 그린다."""
    h = cv2.calcHist([g], [0], None, [256], [0, 256]).ravel()
    if log:
        h = np.log1p(h)
    canvas = np.full((256, 256, 3), 255, np.uint8)
    if h.max() > 0:
        h = h / h.max() * 240
    for x in range(256):
        cv2.line(canvas, (x, 255), (x, 255 - int(h[x])), (90, 90, 90), 1)
    for v in (0, 64, 128, 192, 255):
        cv2.line(canvas, (v, 248), (v, 255), (0, 0, 255), 1)
    return canvas


def draw_pts(img, pts, color, close=True):
    out = img.copy()
    p = np.int32(np.round(pts))
    cv2.polylines(out, [p.reshape(-1, 1, 2)], close, color, 2)
    for i, (x, y) in enumerate(p):
        cv2.circle(out, (int(x), int(y)), 6, color, -1)
        cv2.putText(out, str(i + 1), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return out


# ======================================================================
# 1. 화소단위처리
# ======================================================================
# ---- 1-1 회색 음영 이미지 생성 -------------------------------------
def _quant(a, levels):
    levels = int(np.clip(levels, 2, 256))
    q = np.floor(a / (256.0 / levels))
    return np.clip(q * (255.0 / (levels - 1)), 0, 255).astype(np.uint8)


def px_gray_h(c, lv):
    a = np.tile(np.linspace(0, 255, 256, dtype=np.float32), (256, 1))
    return _quant(a, lv), f"가로 방향 0→255 / 밝기 {int(lv)}단계"


def px_gray_v(c, lv):
    a = np.tile(np.linspace(0, 255, 256, dtype=np.float32)[:, None], (1, 256))
    return _quant(a, lv), f"세로 방향 0→255 / 밝기 {int(lv)}단계"


def px_gray_radial(c, pct):
    yy, xx = np.mgrid[0:256, 0:256].astype(np.float32)
    r = np.hypot(xx - 127.5, yy - 127.5)
    R = 181.0 * pct / 100.0
    return clip_u8(255 * (1 - np.clip(r / R, 0, 1))), "중심 255 → 바깥쪽 0"


def px_gray_checker(c, size):
    yy, xx = np.mgrid[0:256, 0:256]
    s = max(1, int(size))
    return (((xx // s + yy // s) % 2) * 255).astype(np.uint8), f"{s}x{s} 화소 체스판"


def px_gray_noise(c, mean):
    n = np.random.RandomState(0).normal(mean, 40, (256, 256))
    return clip_u8(n), f"평균 {int(mean)}, 표준편차 40 인 가우시안 난수"


# ---- 1-2 RGB 컬러 이미지 생성 --------------------------------------
def rgb_solid(c, r, g, b):
    a = np.zeros((256, 256, 3), np.uint8)
    a[:] = (b, g, r)           # OpenCV 는 B,G,R 순서!
    return a, f"R={int(r)}, G={int(g)}, B={int(b)}  (저장 순서 B,G,R)"


def rgb_gradient(c, b):
    x = np.tile(np.linspace(0, 255, 256, dtype=np.float32), (256, 1))
    y = x.T
    img = np.dstack([np.full((256, 256), b, np.float32), y, x])
    return clip_u8(img), "R: 가로로 증가, G: 세로로 증가, B: 고정"


def rgb_split(c, ch):
    idx = {1: 2, 2: 1, 3: 0}[int(ch)]
    out = np.zeros_like(c.img)
    out[..., idx] = c.img[..., idx]
    return out, ["", "R 채널만 남김", "G 채널만 남김", "B 채널만 남김"][int(ch)]


def rgb_swap(c):
    return c.img[..., ::-1].copy(), "B↔R 순서를 바꾸면 색이 뒤바뀝니다 (BGR→RGB 착각 예시)"


PX_GRAY = F(
    "회색 음영 이미지 생성",
    """■ 회색 음영(그레이스케일) 이미지 생성
→ 수식과 NumPy 배열만으로 이미지를 직접 만들어 보며 '이미지 = 숫자 배열'임을 익힙니다.
→ 그레이 이미지는 (높이, 너비) 크기의 2차원 uint8 배열이고, 0=검정 ~ 255=흰색입니다.

◇ 가로 / 세로 그라데이션
 - **np.linspace(0, 255, 256)** 로 만든 값을 행(또는 열)마다 복사해서 만듭니다.
 - 슬라이더 '밝기 단계'를 줄이면 값을 양자화(quantization)해서 계단 모양이 생깁니다. (비트 수를 줄이는 효과)

◇ 방사형 그라데이션
 - 중심에서의 거리 r = sqrt(x²+y²) 를 구해 **255 × (1 - r/R)** 로 밝기를 정합니다.

◇ 체스판 패턴
 - ((x // 칸) + (y // 칸)) % 2 가 0 / 1 인 칸을 0 / 255 로 칠합니다.

◇ 랜덤 노이즈
 - np.random.normal 로 평균 밝기 주변에 퍼진 난수를 만들고 0~255 로 자릅니다(clip).

※ 이 화면의 이미지는 불러온 사진과 무관하게 생성됩니다. 이미지를 더블클릭하면 저장할 수 있습니다.""",
    [
        P("가로 그라데이션", px_gray_h, [S("밝기 단계 수", 2, 256, 256)]),
        P("세로 그라데이션", px_gray_v, [S("밝기 단계 수", 2, 256, 16)]),
        P("방사형 그라데이션", px_gray_radial, [S("반지름(%)", 10, 150, 100)]),
        P("체스판 패턴", px_gray_checker, [S("칸 크기(px)", 4, 128, 32)]),
        P("랜덤 노이즈", px_gray_noise, [S("평균 밝기", 0, 255, 128)]),
    ],
)

PX_RGB = F(
    "RGB 컬러 이미지 생성",
    """■ RGB 컬러 이미지 생성
→ 컬러 이미지는 (높이, 너비, 3) 배열이며, 3개의 채널이 각각 한 가지 색의 세기를 나타냅니다.
→ ⚠ OpenCV 는 **B, G, R 순서**로 저장합니다. (matplotlib / PIL 은 R, G, B 순서)

◇ 단색 컬러
 - 슬라이더 R, G, B 값으로 한 가지 색으로 채운 이미지를 만듭니다. (예: 255,0,0 → 빨강)

◇ RGB 그라데이션
 - R 채널은 가로 방향, G 채널은 세로 방향으로 증가시키고 B 는 슬라이더 값으로 고정합니다.

◇ 채널 분리
 - cv2.split 과 같은 원리로 한 채널만 남기고 나머지는 0 으로 만듭니다.
 - 밝은 부분일수록 해당 색 성분이 많다는 뜻입니다.

◇ 채널 순서 바꾸기
 - img[..., ::-1] 로 B 와 R 을 맞바꿉니다. 색이 이상하게 나오는 가장 흔한 실수를 보여줍니다.""",
    [
        P("단색 컬러 (R,G,B 지정)", rgb_solid, [S("R", 0, 255, 255), S("G", 0, 255, 128), S("B", 0, 255, 0)]),
        P("RGB 그라데이션", rgb_gradient, [S("B 값", 0, 255, 128)]),
        P("채널 분리", rgb_split, [S("채널 (1=R, 2=G, 3=B)", 1, 3, 1)]),
        P("채널 순서 바꾸기 (BGR↔RGB)", rgb_swap),
    ],
)


# ---- 1-3 그림파일 읽기 ----------------------------------------------
def rd_color(c):
    return c.img, f"cv2.IMREAD_COLOR  →  shape={c.img.shape}, dtype=uint8"


def rd_gray(c):
    return c.gray, f"cv2.IMREAD_GRAYSCALE  →  shape={c.gray.shape}"


def rd_unchanged(c):
    u = c.unchanged
    cap = f"cv2.IMREAD_UNCHANGED  →  shape={u.shape}, dtype={u.dtype}"
    if u.dtype != np.uint8:
        u = norm_u8(u)
    if u.ndim == 3 and u.shape[2] == 4:       # 알파(투명도) 채널이 있으면 흰 배경 위에 합성
        a = u[..., 3:4].astype(np.float32) / 255
        u = clip_u8(u[..., :3] * a + 255 * (1 - a))
    h, w = c.img.shape[:2]
    return cv2.resize(u, (w, h), interpolation=cv2.INTER_AREA), cap


def rd_reduced(c, n):
    flag = {1: cv2.IMREAD_REDUCED_COLOR_2, 2: cv2.IMREAD_REDUCED_COLOR_4, 3: cv2.IMREAD_REDUCED_COLOR_8}[int(n)]
    r = cv2.imdecode(c.buf, flag)
    rel = r.shape[1] / c.orig.shape[1]
    h, w = c.img.shape[:2]
    small = cv2.resize(c.img, (max(1, round(w * rel)), max(1, round(h * rel))), interpolation=cv2.INTER_AREA)
    return on_canvas(small, (h, w)), f"1/{2 ** int(n)} 크기로 읽음: {r.shape[1]}x{r.shape[0]} (원본 {c.orig.shape[1]}x{c.orig.shape[0]})"


PX_READ = F(
    "그림파일 읽기",
    """■ 그림파일 읽기 (cv2.imread / cv2.imdecode)
→ 같은 파일도 읽는 옵션(flag)에 따라 서로 다른 배열이 만들어집니다.

◇ cv2.IMREAD_COLOR (기본값)
 - 3채널 B,G,R 컬러로 읽습니다. 투명도(알파)는 버립니다.

◇ cv2.IMREAD_GRAYSCALE
 - 바로 1채널 그레이스케일로 읽습니다. 컬러로 읽은 뒤 변환하는 것보다 메모리를 절약합니다.

◇ cv2.IMREAD_UNCHANGED
 - 파일에 들어 있는 그대로(알파 채널, 16비트 등 포함) 읽습니다. 아래 캡션의 shape 이 (H,W,4) 이면 알파 채널이 있는 PNG 입니다.

◇ cv2.IMREAD_REDUCED_COLOR_2 / 4 / 8
 - 이미지를 1/2, 1/4, 1/8 크기로 줄여서 읽습니다. 큰 이미지를 빨리 미리보기할 때 유용합니다. (슬라이더 1=1/2, 2=1/4, 3=1/8)

※ 한글 경로의 파일은 cv2.imread 가 실패할 수 있어 이 프로그램은
   np.fromfile + cv2.imdecode 방식으로 읽습니다.
※ 화면에는 작업용으로 줄인 이미지가 표시되며, 캡션의 shape 은 해당 flag 로 읽은 실제 크기입니다.""",
    [
        P("IMREAD_COLOR", rd_color),
        P("IMREAD_GRAYSCALE", rd_gray),
        P("IMREAD_UNCHANGED", rd_unchanged),
        P("IMREAD_REDUCED_COLOR_n", rd_reduced, [S("축소 단계 (1=1/2, 2=1/4, 3=1/8)", 1, 3, 1)]),
    ],
)


# ---- 1-4 그림파일 쓰기 ----------------------------------------------
def _wr(c, ext, params, note=""):
    buf = encode_image(c.img, ext, params)
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    raw = c.img.nbytes
    return dec, f"{ext}  {len(buf) / 1024:.1f} KB  (비압축 {raw / 1024:.0f} KB 의 {len(buf) / raw:.1%}) {note}"


def wr_jpg(c, q):
    return _wr(c, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, int(q)], "손실 압축")


def wr_png(c, lv):
    return _wr(c, ".png", [cv2.IMWRITE_PNG_COMPRESSION, int(lv)], "무손실")


def wr_bmp(c):
    return _wr(c, ".bmp", [], "무압축")


def wr_webp(c, q):
    return _wr(c, ".webp", [cv2.IMWRITE_WEBP_QUALITY, int(q)], "손실 압축")


def wr_jpg_diff(c, q):
    dec = cv2.imdecode(encode_image(c.img, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, int(q)]), cv2.IMREAD_COLOR)
    d = cv2.absdiff(c.img, dec)
    return clip_u8(d.astype(np.float32) * 10), "원본과의 차이를 10배 증폭 (밝을수록 손실이 큼)"


PX_WRITE = F(
    "그림파일 쓰기",
    """■ 그림파일 쓰기 (cv2.imwrite / cv2.imencode)
→ 저장 형식과 옵션에 따라 파일 크기와 화질이 달라집니다. 각 패널은 '저장 → 다시 읽은' 결과입니다.

◇ JPEG (.jpg)
 - **손실 압축**. IMWRITE_JPEG_QUALITY(0~100)가 낮을수록 용량이 작아지고 블록 노이즈가 생깁니다.

◇ PNG (.png)
 - **무손실 압축**. IMWRITE_PNG_COMPRESSION(0~9)은 압축 강도일 뿐 화질은 변하지 않고 용량·속도만 달라집니다.

◇ BMP (.bmp)
 - 압축하지 않아 용량이 가장 큽니다.

◇ WebP (.webp)
 - JPEG 보다 효율이 좋은 손실 압축 형식입니다. IMWRITE_WEBP_QUALITY 로 품질을 조절합니다.

◇ JPEG 압축 오차
 - 원본과 JPEG 복원본의 차이(absdiff)를 10배 확대해 보여줍니다. 품질을 낮출수록 에지 주변 오차가 커집니다.

※ 아래 [현재 이미지를 파일로 저장] 버튼으로 불러온 원본을 원하는 형식으로 저장할 수 있습니다.
※ 각 결과 이미지를 더블클릭해도 그 결과를 저장할 수 있습니다.""",
    [
        P("JPEG", wr_jpg, [S("품질 (0~100)", 1, 100, 30)]),
        P("PNG", wr_png, [S("압축 레벨 (0~9)", 0, 9, 3)]),
        P("BMP", wr_bmp),
        P("WebP", wr_webp, [S("품질 (0~100)", 1, 100, 30)]),
        P("JPEG 압축 오차 (x10)", wr_jpg_diff, [S("품질 (0~100)", 1, 100, 30)]),
    ],
    [("현재 이미지를 파일로 저장...", "save_orig")],
)


# ---- 1-5 그레이스케일 + 밝기 변환 ------------------------------------
def gb_gray(c):
    return c.gray, "cv2.cvtColor(img, COLOR_BGR2GRAY)   Y = 0.299R + 0.587G + 0.114B"


def gb_bright(c, beta):
    return cv2.convertScaleAbs(c.gray, alpha=1.0, beta=int(beta)), f"g(x,y) = f(x,y) + ({int(beta)})  (0~255 로 자름)"


def gb_gamma(c, g10):
    g = g10 / 10.0
    return cv2.LUT(c.gray, gamma_lut(g)), f"g = 255 × (f/255)^(1/{g:.1f})   γ>1 이면 밝아짐"


def gb_negative(c):
    return 255 - c.gray, "g(x,y) = 255 - f(x,y)"


def gb_threshold(c, t):
    _, b = cv2.threshold(c.gray, int(t), 255, cv2.THRESH_BINARY)
    return b, f"f > {int(t)} 이면 255, 아니면 0"


PX_GRAYBRIGHT = F(
    "그레이스케일\n+ 밝기 변환",
    """■ 그레이스케일 + 밝기 변환 (화소 단위 점 연산)
→ 각 화소를 다른 화소와 상관없이 같은 함수 g = T(f) 로 변환하는 처리입니다.

◇ 그레이스케일
 - 컬러 3채널을 밝기 1채널로 바꿉니다. **Y = 0.299R + 0.587G + 0.114B** (사람 눈이 초록에 가장 민감해서 가중치가 다름)

◇ 밝기 조절
 - 모든 화소에 같은 값(beta)을 더합니다. 0~255 를 넘으면 잘립니다(saturate). cv2.convertScaleAbs 사용.

◇ 감마 보정
 - **g = 255 × (f/255)^(1/γ)** 형태의 비선형 변환입니다. 룩업 테이블(cv2.LUT)로 빠르게 적용합니다.
 - γ > 1 : 어두운 부분이 밝아짐 / γ < 1 : 어두워짐

◇ 반전 (네거티브)
 - 255 에서 화소값을 뺍니다. 검정↔흰색이 뒤바뀝니다.

◇ 이진화
 - 임계값보다 크면 255, 아니면 0 으로 만들어 흑백 두 값만 남깁니다. (cv2.threshold)""",
    [
        P("그레이스케일 변환", gb_gray),
        P("밝기 조절 (덧셈)", gb_bright, [S("밝기 값 (beta)", -150, 150, 60)]),
        P("감마 보정", gb_gamma, [S("감마 γ ×10", 1, 40, 20)]),
        P("반전 (네거티브)", gb_negative),
        P("이진화 (threshold)", gb_threshold, [S("임계값", 0, 255, 127)]),
    ],
)


# ---- 1-6 영상 합성 ------------------------------------------------
def _circle_mask(c, pct):
    h, w = c.img.shape[:2]
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (w // 2, h // 2), int(min(h, w) / 2 * pct / 100), 255, -1)
    return m


def cp_second(c):
    return c.img2(), "합성에 사용하는 두 번째 이미지 (아래 버튼으로 바꿀 수 있음)"


def cp_weighted(c, a):
    a = a / 100.0
    return cv2.addWeighted(c.img, a, c.img2(), 1 - a, 0), f"dst = {a:.2f}·img1 + {1 - a:.2f}·img2"


def cp_add(c):
    return cv2.add(c.img, c.img2()), "cv2.add : 더한 값이 255 를 넘으면 255 로 고정(포화)"


def cp_sub(c):
    return cv2.subtract(c.img, c.img2()), "cv2.subtract : 음수가 되면 0 으로 고정"


def cp_diff(c):
    return cv2.absdiff(c.img, c.img2()), "cv2.absdiff : |img1 - img2|  (두 영상의 차이 찾기)"


def cp_mask(c, pct):
    m = _circle_mask(c, pct)
    return np.where(m[..., None] > 0, c.img, c.img2()).astype(np.uint8), "원형 마스크 안은 img1, 밖은 img2"


def cp_and(c, pct):
    m = _circle_mask(c, pct)
    return cv2.bitwise_and(c.img, c.img, mask=m), "cv2.bitwise_and + mask : 마스크 영역만 남김"


PX_COMPOSE = F(
    "영상 합성",
    """■ 영상 합성 (두 이미지를 화소 단위로 결합)
→ 두 이미지의 같은 위치 화소끼리 연산합니다. 크기가 다르면 자동으로 같은 크기로 맞춥니다.
→ 기본 두 번째 이미지는 무지개 그라데이션이며, [합성할 이미지 선택] 버튼으로 다른 사진을 쓸 수 있습니다.

◇ 가중합 cv2.addWeighted
 - **dst = α·img1 + (1-α)·img2** 로 부드럽게 섞습니다. 크로스 페이드(디졸브) 효과입니다.

◇ 덧셈 cv2.add / 뺄셈 cv2.subtract
 - 포화(saturation) 연산입니다. 255 를 넘으면 255, 0 미만이면 0 으로 고정됩니다.
 - NumPy 의 a + b 는 256 이 되면 0 으로 돌아가므로(오버플로) 결과가 다릅니다.

◇ 차이 cv2.absdiff
 - |img1 - img2| : 두 이미지에서 달라진 부분을 찾을 때 씁니다.

◇ 마스크 합성
 - 원형 마스크(슬라이더로 크기 조절) 안쪽은 img1, 바깥쪽은 img2 로 채웁니다.

◇ 비트 AND + 마스크
 - cv2.bitwise_and(img, img, mask=mask) 로 마스크가 흰색인 영역만 남기고 나머지는 검정으로 만듭니다.""",
    [
        P("합성용 두 번째 이미지", cp_second),
        P("가중합 (addWeighted)", cp_weighted, [S("img1 비율 α (%)", 0, 100, 50)]),
        P("덧셈 (add)", cp_add),
        P("뺄셈 (subtract)", cp_sub),
        P("차이 (absdiff)", cp_diff),
        P("마스크 합성", cp_mask, [S("마스크 반지름 (%)", 5, 100, 70)]),
        P("비트 AND + 마스크", cp_and, [S("마스크 반지름 (%)", 5, 100, 70)]),
    ],
    [("합성할 이미지 선택...", "load_img2"), ("기본 이미지로 되돌리기", "reset_img2")],
)


# ---- 1-7 명암 대비 조절 ---------------------------------------------
def ct_linear(c, a10, beta):
    a = a10 / 10.0
    return cv2.convertScaleAbs(c.img, alpha=a, beta=int(beta)), f"g = {a:.1f}·f + ({int(beta)})"


def ct_stretch(c, p):
    out = np.empty_like(c.img)
    for i in range(3):
        ch = c.img[..., i].astype(np.float32)
        lo, hi = np.percentile(ch, [p, 100 - p])
        if hi <= lo:
            hi = lo + 1
        out[..., i] = clip_u8((ch - lo) * 255.0 / (hi - lo))
    return out, f"하위/상위 {int(p)}% 를 버리고 나머지를 0~255 로 확장"


def ct_gamma(c, g10):
    g = g10 / 10.0
    return cv2.LUT(c.img, gamma_lut(g)), f"γ = {g:.1f}"


def ct_sigmoid(c, gain, mid):
    x = c.img.astype(np.float32) / 255.0
    m = mid / 100.0

    def s(v):
        return 1.0 / (1.0 + np.exp(-gain * (v - m)))

    y = (s(x) - s(0.0)) / (s(1.0) - s(0.0))
    return clip_u8(y * 255), "S자 곡선: 중간 톤의 대비를 키움"


PX_CONTRAST = F(
    "명암 대비 조절",
    """■ 명암 대비(Contrast) 조절
→ 밝은 곳과 어두운 곳의 차이를 키우거나 줄이는 처리입니다.

◇ 선형 대비 / 밝기  (g = α·f + β)
 - **α(곱)** 는 대비, **β(덧셈)** 는 밝기를 조절합니다. α>1 이면 대비 증가, α<1 이면 감소.
 - cv2.convertScaleAbs 가 결과를 0~255 로 잘라 줍니다.

◇ 대비 스트레칭
 - 화소값 분포의 양끝(하위·상위 p%)을 버리고 나머지 구간을 0~255 로 늘립니다.
 - 흐릿하고 칙칙한 이미지(분포가 좁은 이미지)에 효과적입니다.

◇ 감마 대비
 - 비선형 변환으로 어두운 영역 또는 밝은 영역의 대비를 선택적으로 조절합니다.

◇ 시그모이드(S자) 곡선
 - 중간 밝기 부근을 가파르게, 양 끝을 완만하게 만들어 자연스럽게 대비를 높입니다.
 - '게인'은 곡선의 가파른 정도, '중심'은 가장 가파른 밝기 위치입니다.""",
    [
        P("선형 대비 (α, β)", ct_linear, [S("대비 α ×10", 1, 40, 15), S("밝기 β", -100, 100, 0)]),
        P("대비 스트레칭", ct_stretch, [S("버릴 비율 p (%)", 0, 20, 2)]),
        P("감마 변환", ct_gamma, [S("감마 γ ×10", 1, 40, 15)]),
        P("시그모이드 곡선", ct_sigmoid, [S("게인", 1, 20, 8), S("중심 (%)", 10, 90, 50)]),
    ],
)


# ---- 1-8 히스토그램 평활화 -------------------------------------------
def he_orig(c):
    return c.gray, "원본(그레이스케일)"


def he_eq(c):
    return cv2.equalizeHist(c.gray), "cv2.equalizeHist"


def he_clahe(c, clip10, tile):
    cl = cv2.createCLAHE(clipLimit=clip10 / 10.0, tileGridSize=(int(tile), int(tile)))
    return cl.apply(c.gray), f"CLAHE clipLimit={clip10 / 10:.1f}, tile={int(tile)}x{int(tile)}"


def he_color(c):
    ycc = cv2.cvtColor(c.img, cv2.COLOR_BGR2YCrCb)
    ycc[..., 0] = cv2.equalizeHist(ycc[..., 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR), "YCrCb 의 밝기(Y)만 평활화 → 색 왜곡 최소화"


def he_hist_orig(c, log):
    return hist_plot(c.gray, log), "원본 히스토그램 (가로: 밝기 0~255, 세로: 화소 수)"


def he_hist_eq(c, log):
    return hist_plot(cv2.equalizeHist(c.gray), log), "평활화 후 히스토그램: 더 넓고 고르게 퍼짐"


PX_HISTEQ = F(
    "히스토그램 평활화",
    """■ 히스토그램 평활화 (Histogram Equalization)
→ 히스토그램: 밝기값(0~255)별로 화소가 몇 개 있는지 센 분포 그래프입니다.
→ 평활화: 밝기 분포가 한쪽에 몰린 이미지를 0~255 전체에 고르게 퍼뜨려 대비를 자동으로 높입니다.

◇ cv2.equalizeHist
 - 누적분포함수(CDF)를 변환 함수로 사용합니다. **g = 255 × CDF(f)**
 - 8비트 단일 채널(그레이)에만 적용됩니다.

◇ CLAHE (cv2.createCLAHE)
 - 이미지를 작은 타일로 나눠 각각 평활화하는 '적응형' 방식입니다.
 - clipLimit 으로 대비 증폭 한계를 정해 노이즈 과증폭을 막습니다.

◇ 컬러 이미지 평활화
 - RGB 각 채널을 따로 하면 색이 틀어지므로 YCrCb 로 바꿔 **밝기(Y)만** 평활화하고 되돌립니다.

◇ 히스토그램 그래프
 - 원본과 평활화 결과의 분포를 비교합니다. '로그 스케일'을 1 로 하면 작은 값도 잘 보입니다.
 - 흑백 이미지처럼 0 과 255 에만 몰린 경우엔 평활화 효과가 거의 없습니다.""",
    [
        P("원본 (그레이)", he_orig),
        P("히스토그램 평활화", he_eq),
        P("CLAHE", he_clahe, [S("clipLimit ×10", 5, 100, 30), S("타일 수", 2, 16, 8)]),
        P("컬러 평활화 (YCrCb)", he_color),
        P("원본 히스토그램", he_hist_orig, [S("세로축 (0=선형, 1=로그)", 0, 1, 0)]),
        P("평활화 후 히스토그램", he_hist_eq, [S("세로축 (0=선형, 1=로그)", 0, 1, 0)]),
    ],
)

PIXEL_FEATURES = [PX_GRAY, PX_RGB, PX_READ, PX_WRITE, PX_GRAYBRIGHT, PX_COMPOSE, PX_CONTRAST, PX_HISTEQ]


# ======================================================================
# 2. 영역처리
# ======================================================================
# ---- 2-1 블러링 ----------------------------------------------------
def bl_blur(c, k):
    k = max(1, int(k))
    return cv2.blur(c.img, (k, k)), f"{k}x{k} 평균"


def bl_gauss(c, k):
    k = odd(k)
    return cv2.GaussianBlur(c.img, (k, k), 0), f"{k}x{k} 가우시안 (홀수만 가능)"


def bl_box(c, k):
    k = max(1, int(k))
    return cv2.boxFilter(c.img, -1, (k, k), normalize=True), f"{k}x{k} box (normalize=True)"


def bl_median(c, k):
    k = odd(k)
    return cv2.medianBlur(c.img, k), f"{k}x{k} 중간값 (홀수만 가능)"


AR_BLUR = F(
    "블러링",
    """■ 다양한 블러(흐림) 필터 비교
→ 이미지를 흐리게 만들어 노이즈를 제거하거나 부드럽게 표현할 때 사용합니다.
→ 슬라이더로 커널(창) 크기를 바꿉니다. 커널이 클수록 더 많이 흐려집니다.

◇ cv2.blur
 - **평균 블러**: 주변 픽셀의 산술 평균으로 부드럽게 만듭니다.

◇ cv2.GaussianBlur
 - **가우시안 블러**: 중심에 더 큰 가중치를 주는 방식입니다.
 - 가장 자연스럽고 부드러운 흐림 효과를 줍니다. 커널 크기는 홀수여야 합니다.

◇ cv2.boxFilter
 - **박스 필터**: blur 와 비슷하지만 normalize 옵션 등 고급 설정이 가능합니다.
 - normalize=False 면 합계를 그대로 써서 매우 밝아집니다. 정밀하게 제어할 때 유리합니다.

◇ cv2.medianBlur
 - **중간값 블러**: 주변 픽셀을 정렬해 중간값을 사용합니다.
 - 소금-후추(점 모양) 노이즈를 제거하는 데 탁월하고 에지는 비교적 잘 보존합니다.""",
    [
        P("cv2.blur", bl_blur, [S("커널 크기 (ksize)", 1, 51, 5)]),
        P("cv2.GaussianBlur", bl_gauss, [S("커널 크기 (ksize)", 1, 51, 5)]),
        P("cv2.boxFilter", bl_box, [S("커널 크기 (ksize)", 1, 51, 5)]),
        P("cv2.medianBlur", bl_median, [S("커널 크기 (ksize)", 1, 51, 5)]),
    ],
)


# ---- 2-2 샤프닝 ----------------------------------------------------
def sh_orig(c):
    return c.img, "원본"


def sh_unsharp(c, sigma, a10):
    a = a10 / 10.0
    blur = cv2.GaussianBlur(c.img, (0, 0), sigma)
    return cv2.addWeighted(c.img, 1 + a, blur, -a, 0), f"원본 + {a:.1f} × (원본 - 블러σ={int(sigma)})"


def sh_kernel(c, s10):
    s = s10 / 10.0
    k = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32) * s
    k[1, 1] += 1
    return cv2.filter2D(c.img, -1, k), f"3x3 커널 (중앙 {k[1, 1]:.1f}, 상하좌우 {-s:.1f})"


def sh_laplacian(c, c10):
    img = c.img.astype(np.float32)
    lap = cv2.Laplacian(img, cv2.CV_32F, ksize=1)
    return clip_u8(img - (c10 / 10.0) * lap), f"g = f - {c10 / 10:.1f} × Laplacian(f)"


def sh_highboost(c, A10, k):
    A = A10 / 10.0
    k = odd(k)
    blur = cv2.GaussianBlur(c.img, (k, k), 0)
    return cv2.addWeighted(c.img, A, blur, -1, 0), f"g = {A:.1f}·f - 블러 (A=1 이면 고주파만, A>1 이면 원본 유지)"


AR_SHARPEN = F(
    "샤프닝",
    """■ 샤프닝(선명화)
→ 에지와 세부 묘사를 강조해 이미지를 또렷하게 만드는 처리입니다. 원리: **원본 + (원본의 고주파 성분)**

◇ 언샤프 마스크 (Unsharp Masking)
 - 블러 처리본을 원본에서 빼면 에지(고주파)만 남습니다. 이것을 원본에 더합니다.
 - **g = f + a·(f - blur(f))**  (σ: 블러 정도, a: 강도)

◇ 샤프닝 커널
 - 중앙 값이 크고 주변이 음수인 3x3 커널을 cv2.filter2D 로 컨볼루션합니다.
 - 강도가 0 이면 원본과 같고, 커질수록 선명해지지만 노이즈도 커집니다.

◇ 라플라시안 샤프닝
 - 2차 미분(라플라시안)으로 에지를 검출하고 원본에서 **빼서** 에지 부분을 강조합니다.

◇ 하이부스트 필터
 - **g = A·f - blur(f)**. A=1 이면 순수 고주파(에지)만 남고, A 가 클수록 원본이 더 유지됩니다.

※ 샤프닝은 노이즈도 함께 강조하므로 필요하면 먼저 블러로 노이즈를 줄입니다.""",
    [
        P("원본", sh_orig),
        P("언샤프 마스크", sh_unsharp, [S("블러 σ", 1, 15, 3), S("강도 a ×10", 0, 50, 15)]),
        P("샤프닝 커널 (filter2D)", sh_kernel, [S("강도 ×10", 0, 30, 10)]),
        P("라플라시안 샤프닝", sh_laplacian, [S("강도 c ×10", 0, 30, 10)]),
        P("하이부스트 필터", sh_highboost, [S("증폭 A ×10", 10, 40, 15), S("블러 커널", 3, 21, 5)]),
    ],
)


# ---- 2-3 에지 검출_1 (1차 미분) -------------------------------------
def _mag_out(gx, gy, th):
    m = norm_u8(np.sqrt(gx * gx + gy * gy))
    if th > 0:
        m = np.where(m > th, 255, 0).astype(np.uint8)
    return m


def _sobel_k(k):
    return min(odd(k), 7)


def e1_sobel_x(c, k):
    k = _sobel_k(k)
    return norm_u8(np.abs(cv2.Sobel(c.gray, cv2.CV_32F, 1, 0, ksize=k))), f"x 방향 미분 → 세로 에지 (ksize={k})"


def e1_sobel_y(c, k):
    k = _sobel_k(k)
    return norm_u8(np.abs(cv2.Sobel(c.gray, cv2.CV_32F, 0, 1, ksize=k))), f"y 방향 미분 → 가로 에지 (ksize={k})"


def e1_sobel_mag(c, k, th):
    k = _sobel_k(k)
    gx = cv2.Sobel(c.gray, cv2.CV_32F, 1, 0, ksize=k)
    gy = cv2.Sobel(c.gray, cv2.CV_32F, 0, 1, ksize=k)
    return _mag_out(gx, gy, th), f"크기 √(gx²+gy²)  ksize={k}, 임계값={int(th)} (0=이진화 안 함)"


def e1_prewitt(c, th):
    g = c.gray.astype(np.float32)
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
    ky = kx.T.copy()
    return _mag_out(cv2.filter2D(g, -1, kx), cv2.filter2D(g, -1, ky), th), "Prewitt 3x3 (가중치 없는 평균 미분)"


def e1_roberts(c, th):
    g = c.gray.astype(np.float32)
    kx = np.array([[1, 0], [0, -1]], np.float32)
    ky = np.array([[0, 1], [-1, 0]], np.float32)
    return _mag_out(cv2.filter2D(g, -1, kx), cv2.filter2D(g, -1, ky), th), "Roberts 2x2 (대각선 방향 차이)"


def e1_scharr(c, th):
    gx = cv2.Scharr(c.gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(c.gray, cv2.CV_32F, 0, 1)
    return _mag_out(gx, gy, th), "Scharr 3x3 (Sobel 보다 회전 정확도가 높음)"


AR_EDGE1 = F(
    "에지 검출_1",
    """■ 에지 검출 1 : 1차 미분(기울기, Gradient) 기반
→ 에지 = 밝기가 급격히 변하는 곳. 밝기의 변화율(1차 미분)이 큰 곳을 찾습니다.
→ x 방향 미분 gx 와 y 방향 미분 gy 를 구하고 **크기 = √(gx² + gy²)** 로 에지 세기를 계산합니다.

◇ Sobel (cv2.Sobel)
 - 가운데 행/열에 2배 가중치를 줘서 약간의 블러 효과가 있어 노이즈에 강합니다.
 - **X 방향**은 세로 에지, **Y 방향**은 가로 에지를 잘 찾습니다. ksize 는 1,3,5,7 (홀수).
 - '크기' 패널은 두 방향을 합친 결과이며, 임계값을 주면 흑백 에지 맵이 됩니다.

◇ Prewitt
 - Sobel 과 비슷하지만 가중치가 모두 1 인 커널(cv2.filter2D 로 직접 적용)입니다.

◇ Roberts
 - 2x2 대각선 커널로 계산이 가장 간단하지만 노이즈에 약합니다.

◇ Scharr (cv2.Scharr)
 - 3x3 Sobel 을 개선한 커널로 방향에 따른 정확도가 더 높습니다.

※ 임계값 0 은 '그레이 세기 그대로', 1 이상은 그 값보다 큰 곳만 흰색으로 표시합니다.""",
    [
        P("Sobel X", e1_sobel_x, [S("ksize (1,3,5,7)", 1, 7, 3)]),
        P("Sobel Y", e1_sobel_y, [S("ksize (1,3,5,7)", 1, 7, 3)]),
        P("Sobel 크기 (X+Y)", e1_sobel_mag, [S("ksize (1,3,5,7)", 1, 7, 3), S("임계값 (0=사용 안 함)", 0, 255, 0)]),
        P("Prewitt", e1_prewitt, [S("임계값 (0=사용 안 함)", 0, 255, 0)]),
        P("Roberts", e1_roberts, [S("임계값 (0=사용 안 함)", 0, 255, 0)]),
        P("Scharr", e1_scharr, [S("임계값 (0=사용 안 함)", 0, 255, 0)]),
    ],
)


# ---- 2-4 에지 검출_2 (2차 미분 / Canny) ------------------------------
def e2_canny(c, lo, hi):
    return cv2.Canny(c.gray, int(lo), int(hi)), f"threshold1={int(lo)}, threshold2={int(hi)}"


def e2_autocanny(c, s):
    s = s / 100.0
    med = float(np.median(c.gray))
    lo, hi = int(max(0, (1 - s) * med)), int(min(255, (1 + s) * med))
    return cv2.Canny(c.gray, lo, hi), f"중앙값={med:.0f} → 자동 임계값 {lo} ~ {hi}"


def e2_laplacian(c, k):
    k = _sobel_k(k)
    return norm_u8(np.abs(cv2.Laplacian(c.gray, cv2.CV_32F, ksize=k))), f"Laplacian ksize={k}"


def e2_log(c, s):
    g = cv2.GaussianBlur(c.gray, (0, 0), s)
    return norm_u8(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))), f"LoG: 가우시안 σ={int(s)} → 라플라시안"


def e2_dog(c, s1, s2):
    a, b = sorted((float(s1), float(s2)))
    if a == b:
        b = a + 1
    d = cv2.GaussianBlur(c.gray, (0, 0), a).astype(np.float32) - cv2.GaussianBlur(c.gray, (0, 0), b).astype(np.float32)
    return norm_u8(np.abs(d)), f"DoG: 가우시안(σ={a:.0f}) - 가우시안(σ={b:.0f})"


def e2_overlay(c, lo, hi):
    e = cv2.Canny(c.gray, int(lo), int(hi))
    out = c.img.copy()
    out[e > 0] = (0, 0, 255)
    return out, "Canny 에지를 원본 위에 빨간색으로 표시"


AR_EDGE2 = F(
    "에지 검출_2",
    """■ 에지 검출 2 : 2차 미분 / 다단계(Canny) 방식
→ 1차 미분의 극대점 = 2차 미분이 0 을 지나는 곳(zero-crossing). 이를 이용하거나, 여러 단계로 정제합니다.

◇ Canny (cv2.Canny)
 - 가우시안 블러 → 기울기 계산 → 비최대 억제(가는 선) → 이중 임계값 + 히스테리시스 순서로 처리합니다.
 - **threshold2(높음)**: 이 값 이상이면 확실한 에지 / **threshold1(낮음)**: 확실한 에지와 연결될 때만 에지로 인정.
 - 가장 널리 쓰이는 에지 검출기입니다.

◇ Auto Canny
 - 이미지 밝기의 중앙값을 기준으로 (1±σ)×중앙값 을 임계값으로 자동 설정합니다.

◇ Laplacian (cv2.Laplacian)
 - 2차 미분 연산자 ∇²f = ∂²f/∂x² + ∂²f/∂y². 방향에 상관없이 에지를 검출하지만 노이즈에 매우 민감합니다.

◇ LoG (Laplacian of Gaussian)
 - 먼저 가우시안으로 노이즈를 줄인 뒤 라플라시안을 적용합니다. σ 가 클수록 굵은 에지만 남습니다.

◇ DoG (Difference of Gaussians)
 - σ 가 다른 두 가우시안 블러의 차이로 LoG 를 빠르게 근사합니다.

◇ 에지 오버레이
 - Canny 결과를 원본에 겹쳐 에지가 어디서 검출되는지 확인합니다.""",
    [
        P("Canny", e2_canny, [S("threshold1 (낮음)", 0, 255, 50), S("threshold2 (높음)", 0, 255, 150)]),
        P("Auto Canny", e2_autocanny, [S("σ ×100", 5, 100, 33)]),
        P("Laplacian", e2_laplacian, [S("ksize (1,3,5,7)", 1, 7, 3)]),
        P("LoG", e2_log, [S("가우시안 σ", 1, 10, 2)]),
        P("DoG", e2_dog, [S("σ1", 1, 10, 1), S("σ2", 1, 20, 3)]),
        P("에지 오버레이 (Canny)", e2_overlay, [S("threshold1", 0, 255, 50), S("threshold2", 0, 255, 150)]),
    ],
)


# ---- 2-5 필터링 ----------------------------------------------------
def _sp_noise(img, amount):
    rs = np.random.RandomState(1)
    out = img.copy()
    m = rs.rand(*img.shape[:2])
    p = amount / 100.0
    out[m < p / 2] = 0
    out[m > 1 - p / 2] = 255
    return out


def _gauss_noise(img, sigma):
    n = np.random.RandomState(2).normal(0, sigma, img.shape)
    return clip_u8(img.astype(np.float32) + n)


def fl_salt(c, amt):
    return _sp_noise(c.img, amt), f"소금-후추 노이즈 {int(amt)}% (임의의 화소가 0 또는 255)"


def fl_gnoise(c, s):
    return _gauss_noise(c.img, s), f"가우시안 노이즈 σ={int(s)}"


def fl_median(c, amt, k):
    k = odd(k)
    return cv2.medianBlur(_sp_noise(c.img, amt), k), f"소금-후추 {int(amt)}% 노이즈 → 중간값 {k}x{k} 로 제거"


def fl_bilateral(c, d, sc):
    out = cv2.bilateralFilter(_gauss_noise(c.img, 20), int(d), float(sc), float(sc))
    return out, f"가우시안 노이즈(σ=20) → 양방향 필터 d={int(d)}, σColor=σSpace={int(sc)}"


def fl_nlm(c, h):
    noisy = _gauss_noise(c.img, 20)
    return cv2.fastNlMeansDenoisingColored(noisy, None, float(h), float(h), 7, 21), f"가우시안 노이즈(σ=20) → NLM 제거 h={int(h)}"


_KERNELS = [
    ("항등(원본 그대로)", np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.float32), 0),
    ("엠보싱", np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], np.float32), 128),
    ("외곽선 추출", np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32), 0),
    ("샤프닝", np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32), 0),
    ("평균 3x3", np.ones((3, 3), np.float32) / 9, 0),
    ("가우시안 3x3", np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16, 0),
]


def fl_custom(c, idx):
    name, k, delta = _KERNELS[int(idx)]
    rows = " / ".join(" ".join(f"{v:g}" if abs(v) >= 1 or v == 0 else f"{v:.2f}" for v in r) for r in k)
    return cv2.filter2D(c.img, -1, k, delta=delta), f"{name}  커널: {rows}"


AR_FILTER = F(
    "필터링",
    """■ 필터링 (노이즈 제거 + 사용자 정의 필터)
→ 영역 처리의 핵심은 커널(마스크)을 이미지 위에서 이동하며 계산하는 **컨볼루션**입니다.

◇ 소금-후추 노이즈 / 가우시안 노이즈
 - 노이즈가 있는 이미지를 만들어 필터 효과를 비교할 수 있게 합니다.
 - 소금-후추: 임의의 화소가 검정/흰색으로 튐 / 가우시안: 모든 화소에 정규분포 오차가 더해짐.

◇ 중간값 필터 (medianBlur)
 - 소금-후추 노이즈 제거에 최적입니다. 평균 필터는 튄 값이 주변에 번지지만 중간값은 무시합니다.

◇ 양방향 필터 (cv2.bilateralFilter)
 - 거리와 **밝기 차이** 둘 다 고려해서 평탄한 영역만 흐리고 에지는 보존합니다. (피부 보정 등)
 - d: 이웃 지름, σColor: 클수록 밝기가 많이 다른 화소도 섞임.

◇ Non-Local Means (cv2.fastNlMeansDenoisingColored)
 - 이미지 전체에서 비슷한 패턴을 찾아 평균을 냅니다. 품질이 좋지만 느립니다.
 - 슬라이더를 놓을 때 계산합니다. h 가 클수록 노이즈는 더 지워지지만 디테일도 사라집니다.

◇ 사용자 정의 커널 (cv2.filter2D)
 - 슬라이더 0~5 : 항등 / 엠보싱 / 외곽선 / 샤프닝 / 평균 / 가우시안.
 - 커널 숫자를 바꾸기만 해도 전혀 다른 효과가 나는 것을 확인할 수 있습니다.""",
    [
        P("소금-후추 노이즈 추가", fl_salt, [S("노이즈 비율 (%)", 1, 30, 5)]),
        P("가우시안 노이즈 추가", fl_gnoise, [S("σ", 1, 80, 20)]),
        P("중간값 필터로 제거", fl_median, [S("노이즈 비율 (%)", 1, 30, 5), S("커널 크기", 1, 15, 3)]),
        P("양방향 필터", fl_bilateral, [S("d (지름)", 1, 15, 9), S("σColor = σSpace", 5, 200, 75)]),
        P("Non-Local Means", fl_nlm, [S("h (제거 강도)", 1, 40, 10)], heavy=True),
        P("사용자 정의 커널 (filter2D)", fl_custom, [S("커널 종류 (0~5)", 0, 5, 1)]),
    ],
)


# ---- 2-6 모폴로지 --------------------------------------------------
_SHAPES = [cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS]
_SHAPE_NAME = ["사각형", "타원", "십자"]


def _bin(c, thr=0):
    if thr and thr > 0:
        _, b = cv2.threshold(c.gray, int(thr), 255, cv2.THRESH_BINARY)
    else:
        _, b = cv2.threshold(c.gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b


def _se(shape, k):
    k = max(1, int(k))
    return cv2.getStructuringElement(_SHAPES[int(shape)], (k, k))


def mo_bin(c, thr):
    return _bin(c, thr), "임계값 0 이면 Otsu 자동 이진화 (흰색=객체)"


def mo_erode(c, k, shape, it):
    return cv2.erode(_bin(c), _se(shape, k), iterations=int(it)), f"침식 {_SHAPE_NAME[int(shape)]} {int(k)}x{int(k)} × {int(it)}회: 흰 영역 축소"


def mo_dilate(c, k, shape, it):
    return cv2.dilate(_bin(c), _se(shape, k), iterations=int(it)), f"팽창 {_SHAPE_NAME[int(shape)]} {int(k)}x{int(k)} × {int(it)}회: 흰 영역 확대"


def _mo(op, text):
    def f(c, k, shape):
        return cv2.morphologyEx(_bin(c), op, _se(shape, k)), f"{text} ({_SHAPE_NAME[int(shape)]} {int(k)}x{int(k)})"
    return f


mo_open = _mo(cv2.MORPH_OPEN, "열림 = 침식→팽창: 작은 흰 점 제거")
mo_close = _mo(cv2.MORPH_CLOSE, "닫힘 = 팽창→침식: 작은 검은 구멍 메움")
mo_grad = _mo(cv2.MORPH_GRADIENT, "그래디언트 = 팽창 - 침식: 윤곽선")
mo_tophat = _mo(cv2.MORPH_TOPHAT, "탑햇 = 원본 - 열림: 밝고 작은 부분")
mo_blackhat = _mo(cv2.MORPH_BLACKHAT, "블랙햇 = 닫힘 - 원본: 어둡고 작은 부분")

_SH = S("커널 모양 (0=사각 1=타원 2=십자)", 0, 2, 1)

AR_MORPH = F(
    "모폴로지",
    """■ 모폴로지(Morphology) 연산
→ 구조 요소(커널)의 모양을 이용해 이진 이미지의 형태를 변형하는 처리입니다. 객체의 잡음 제거, 구멍 메우기, 윤곽 추출에 씁니다.
→ 먼저 Otsu 방법(또는 직접 고른 임계값)으로 이진화하고, 흰색을 객체로 봅니다.

◇ 침식 (Erosion, cv2.erode)
 - 커널 안이 **전부 흰색**일 때만 흰색으로 남깁니다. 흰 영역이 줄어들고 작은 흰 점이 사라집니다.

◇ 팽창 (Dilation, cv2.dilate)
 - 커널 안에 **흰색이 하나라도** 있으면 흰색으로 만듭니다. 흰 영역이 커지고 끊긴 부분이 이어집니다.

◇ 열림 (Opening) = 침식 → 팽창
 - 객체 크기는 유지하면서 작은 흰색 잡음(점)을 제거합니다.

◇ 닫힘 (Closing) = 팽창 → 침식
 - 객체 내부의 작은 구멍과 틈을 메웁니다.

◇ 그래디언트 (Gradient) = 팽창 - 침식
 - 객체의 윤곽선만 남깁니다.

◇ 탑햇 (Top-hat) = 원본 - 열림
 - 커널보다 작고 **밝은** 세부 요소만 추출합니다.

◇ 블랙햇 (Black-hat) = 닫힘 - 원본
 - 커널보다 작고 **어두운** 세부 요소만 추출합니다.

※ 커널 모양(사각/타원/십자)과 크기에 따라 결과가 달라집니다. 반복 횟수는 침식·팽창 패널에서 조절합니다.""",
    [
        P("이진화 (Otsu)", mo_bin, [S("임계값 (0=Otsu 자동)", 0, 255, 0)]),
        P("침식 (erode)", mo_erode, [S("커널 크기", 1, 31, 5), _SH, S("반복 횟수", 1, 10, 1)]),
        P("팽창 (dilate)", mo_dilate, [S("커널 크기", 1, 31, 5), _SH, S("반복 횟수", 1, 10, 1)]),
        P("열림 (open)", mo_open, [S("커널 크기", 1, 31, 5), _SH]),
        P("닫힘 (close)", mo_close, [S("커널 크기", 1, 31, 5), _SH]),
        P("그래디언트", mo_grad, [S("커널 크기", 1, 31, 5), _SH]),
        P("탑햇 (top-hat)", mo_tophat, [S("커널 크기", 1, 31, 9), _SH]),
        P("블랙햇 (black-hat)", mo_blackhat, [S("커널 크기", 1, 31, 9), _SH]),
    ],
)

AREA_FEATURES = [AR_BLUR, AR_SHARPEN, AR_EDGE1, AR_EDGE2, AR_FILTER, AR_MORPH]


# ======================================================================
# 3. 기하학적처리
# ======================================================================
# ---- 3-1 확대/축소 -------------------------------------------------
def _canvas_note(c, out, extra=""):
    h, w = c.img.shape[:2]
    return on_canvas(out, (h, w)), f"결과 크기 {out.shape[1]}x{out.shape[0]} (작업 이미지 {w}x{h}) {extra}"


def rs_scale(c, pct):
    s = pct / 100.0
    h, w = c.img.shape[:2]
    out = cv2.resize(c.img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_LINEAR)
    return _canvas_note(c, out, f"| 배율 {s:.2f}배")


def rs_fxfy(c, fx, fy):
    h, w = c.img.shape[:2]
    out = cv2.resize(c.img, (max(1, round(w * fx / 100)), max(1, round(h * fy / 100))), interpolation=cv2.INTER_LINEAR)
    return _canvas_note(c, out, f"| fx={fx / 100:.2f}, fy={fy / 100:.2f}")


def rs_size(c, w, h):
    out = cv2.resize(c.img, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
    return _canvas_note(c, out, "| dsize 로 크기 직접 지정")


def rs_pyrdown(c, n):
    out = c.img
    for _ in range(int(n)):
        out = cv2.pyrDown(out)
    return _canvas_note(c, out, f"| pyrDown {int(n)}회")


def rs_pyr_updown(c, n):
    h, w = c.img.shape[:2]
    out = c.img
    for _ in range(int(n)):
        out = cv2.pyrDown(out)
    for _ in range(int(n)):
        out = cv2.pyrUp(out)
    return cv2.resize(out, (w, h)), f"pyrDown {int(n)}회 후 pyrUp {int(n)}회: 줄였다 키우면 세부 정보가 사라짐"


GM_RESIZE = F(
    "확대/축소",
    """■ 확대 / 축소 (Scaling, cv2.resize)
→ 이미지 크기를 바꾸는 가장 기본적인 기하 변환입니다. 결과는 작업 이미지와 같은 크기의 캔버스 가운데에 **실제 크기 비율 그대로** 표시됩니다. (커지면 가장자리가 잘려 보입니다.) 정확한 크기는 아래 글자를 보세요.

◇ 비율 지정 (fx = fy)
 - 가로·세로를 같은 비율로 확대/축소합니다. 1.0 보다 작으면 축소, 크면 확대입니다.

◇ 가로 / 세로 따로 지정 (fx, fy)
 - 비율이 다르면 이미지가 찌그러집니다(종횡비 변경).

◇ 크기 직접 지정 (dsize)
 - 원하는 픽셀 크기 (너비, 높이) 를 직접 지정합니다. 비율이 맞지 않으면 역시 찌그러집니다.

◇ 이미지 피라미드 (cv2.pyrDown / pyrUp)
 - pyrDown : 가우시안 블러 후 가로·세로를 1/2 로 줄입니다. (여러 번 반복 가능)
 - pyrUp : 2배로 키웁니다. 줄였다가 키우면 정보가 복원되지 않는 것을 확인할 수 있습니다.
 - 크기별로 처리해야 하는 객체 검출, 블렌딩 등에 쓰입니다.

※ 축소 시에는 cv2.INTER_AREA, 확대 시에는 INTER_CUBIC / INTER_LINEAR 가 일반적으로 좋습니다. 자세한 비교는 '보간법' 메뉴를 참고하세요.""",
    [
        P("비율 지정 (fx=fy)", rs_scale, [S("배율 (%)", 10, 300, 60)]),
        P("가로/세로 비율 따로", rs_fxfy, [S("fx 가로 (%)", 10, 300, 100), S("fy 세로 (%)", 10, 300, 50)]),
        P("크기 직접 지정 (dsize)", rs_size, [S("너비 (px)", 20, 800, 300), S("높이 (px)", 20, 800, 200)]),
        P("pyrDown (1/2씩 축소)", rs_pyrdown, [S("반복 횟수", 0, 4, 1)]),
        P("pyrDown → pyrUp", rs_pyr_updown, [S("반복 횟수", 0, 4, 2)]),
    ],
)


# ---- 3-2 보간법 ----------------------------------------------------
def _interp(flag, name):
    def f(c, small):
        h, w = c.img.shape[:2]
        sw = max(4, int(small))
        sh = max(4, int(round(sw * h / w)))
        sm = cv2.resize(c.img, (sw, sh), interpolation=cv2.INTER_AREA)
        return cv2.resize(sm, (w, h), interpolation=flag), f"{name}: {sw}x{sh} 로 줄인 이미지를 {w}x{h} 로 확대"
    return f


def ip_orig(c):
    return c.img, "원본 (비교용)"


GM_INTERP = F(
    "보간법",
    """■ 보간법 (Interpolation) 비교
→ 이미지를 키우거나 회전하면 원본에 없던 위치의 화소값을 '추정'해서 채워야 합니다. 그 추정 방법이 보간법입니다.
→ 이 화면은 원본을 아주 작게 줄였다가(슬라이더 값) 각 보간법으로 다시 원래 크기로 키워서 품질을 비교합니다.

◇ INTER_NEAREST (최근접 이웃)
 - 가장 가까운 화소값을 그대로 복사합니다. 가장 빠르지만 계단 현상(모자이크)이 생깁니다.

◇ INTER_LINEAR (양선형, 기본값)
 - 주변 4개 화소를 거리에 비례해 섞습니다. 속도와 품질의 균형이 좋습니다.

◇ INTER_CUBIC (쌍3차)
 - 주변 4x4=16개 화소를 사용해 더 부드럽고 선명합니다. LINEAR 보다 느립니다.

◇ INTER_AREA (영역 평균)
 - 화소 영역 관계를 이용합니다. **축소**에 가장 좋고, 확대할 때는 NEAREST 와 비슷합니다.

◇ INTER_LANCZOS4
 - 주변 8x8 화소를 사용하는 고품질 방식입니다. 가장 느리고 링잉(가장자리 출렁임)이 생길 수 있습니다.

※ 슬라이더 값이 작을수록(더 작게 줄일수록) 방법별 차이가 뚜렷하게 보입니다.""",
    [
        P("원본", ip_orig),
        P("INTER_NEAREST", _interp(cv2.INTER_NEAREST, "최근접"), [S("줄일 너비 (px)", 8, 200, 40)]),
        P("INTER_LINEAR", _interp(cv2.INTER_LINEAR, "양선형"), [S("줄일 너비 (px)", 8, 200, 40)]),
        P("INTER_CUBIC", _interp(cv2.INTER_CUBIC, "쌍3차"), [S("줄일 너비 (px)", 8, 200, 40)]),
        P("INTER_AREA", _interp(cv2.INTER_AREA, "영역"), [S("줄일 너비 (px)", 8, 200, 40)]),
        P("INTER_LANCZOS4", _interp(cv2.INTER_LANCZOS4, "란초스"), [S("줄일 너비 (px)", 8, 200, 40)]),
    ],
)


# ---- 3-3 평행이동 --------------------------------------------------
def _translate(mode, name):
    def f(c, tx, ty):
        h, w = c.img.shape[:2]
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        out = cv2.warpAffine(c.img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=mode, borderValue=GRAY_BG)
        return out, f"{name}\nM = [[1, 0, {int(tx)}], [0, 1, {int(ty)}]]"
    return f


_TS = [S("x 이동 tx (px)", -240, 240, 80), S("y 이동 ty (px)", -240, 240, 50)]

GM_TRANSLATE = F(
    "평행이동",
    """■ 평행이동 (Translation, cv2.warpAffine)
→ 이미지를 가로 tx, 세로 ty 픽셀만큼 옮깁니다. 2x3 변환 행렬을 사용합니다.
→ **M = [[1, 0, tx], [0, 1, ty]]**  (tx>0: 오른쪽, ty>0: 아래쪽)

◇ 경계 처리 (borderMode)
 - 이동하면 한쪽에 빈 영역이 생기는데, 그 영역을 어떻게 채울지 정합니다.
 - **BORDER_CONSTANT** : 일정한 색(여기서는 회색)으로 채움
 - **BORDER_REPLICATE** : 가장자리 화소를 계속 복제
 - **BORDER_REFLECT** : 가장자리를 거울처럼 반사
 - **BORDER_WRAP** : 반대편 이미지가 이어 붙음 (타일처럼)

※ warpAffine 의 출력 크기(dsize)를 원본과 같게 하면 밀려나간 부분은 잘려 나갑니다.""",
    [
        P("BORDER_CONSTANT", _translate(cv2.BORDER_CONSTANT, "상수(회색)"), _TS),
        P("BORDER_REPLICATE", _translate(cv2.BORDER_REPLICATE, "가장자리 복제"), _TS),
        P("BORDER_REFLECT", _translate(cv2.BORDER_REFLECT, "반사"), _TS),
        P("BORDER_WRAP", _translate(cv2.BORDER_WRAP, "반복(wrap)"), _TS),
    ],
)


# ---- 3-4 회전 ------------------------------------------------------
def ro_center(c, ang, sc):
    h, w = c.img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc / 100.0)
    return cv2.warpAffine(c.img, M, (w, h), borderValue=GRAY_BG), "M =\n" + fmt_mat(M)


def ro_expand(c, ang):
    h, w = c.img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    out, _ = warp_fit(c.img, M)
    return out, f"결과 크기 {out.shape[1]}x{out.shape[0]} (잘림 없이 캔버스 확장)"


def ro_90(c, k):
    k = int(k) % 4
    code = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE][k]
    out = c.img if code is None else cv2.rotate(c.img, code)
    return out, f"cv2.rotate : 시계 방향 {k * 90}°"


def ro_pivot(c, ang, cx, cy):
    h, w = c.img.shape[:2]
    center = (w * cx / 100.0, h * cy / 100.0)
    M = cv2.getRotationMatrix2D(center, ang, 1.0)
    out = cv2.warpAffine(c.img, M, (w, h), borderValue=GRAY_BG)
    cv2.circle(out, (int(center[0]), int(center[1])), 7, (0, 0, 255), -1)
    return out, f"회전 중심 ({center[0]:.0f}, {center[1]:.0f}) 을 빨간 점으로 표시"


GM_ROTATE = F(
    "회전",
    """■ 회전 (Rotation)
→ 이미지를 한 점(중심)을 기준으로 각도만큼 돌립니다. cv2.getRotationMatrix2D 로 2x3 행렬을 만들고 warpAffine 으로 적용합니다.
→ 각도가 양수이면 **반시계 방향**입니다. 행렬: [[α, β, (1-α)cx - β·cy], [-β, α, β·cx + (1-α)cy]] (α=s·cosθ, β=s·sinθ)

◇ 중심 회전 + 배율
 - 이미지 중심을 기준으로 회전하면서 동시에 확대/축소(scale)도 할 수 있습니다. 출력 크기가 같으므로 모서리가 잘립니다.

◇ 캔버스 확장 회전
 - 회전한 이미지의 네 모서리를 계산해 출력 크기를 키워 **잘림 없이** 전체를 보여줍니다.

◇ 90° 단위 회전 (cv2.rotate)
 - 90°, 180°, 270° 는 보간 없이 화소를 재배열하므로 화질 손실이 없고 빠릅니다.

◇ 임의의 중심 회전
 - 중심 좌표를 이미지 크기의 비율(%)로 정해 원하는 점을 축으로 돌립니다. 빨간 점이 회전축입니다.

※ 회전하면 원본에 없던 위치의 값이 필요하므로 보간이 적용되고, 빈 영역은 회색으로 채워집니다.""",
    [
        P("중심 회전 + 배율", ro_center, [S("각도 (°)", -180, 180, 30), S("배율 (%)", 10, 200, 100)]),
        P("캔버스 확장 회전", ro_expand, [S("각도 (°)", -180, 180, 30)]),
        P("90° 단위 회전", ro_90, [S("횟수 (0~3 × 90°)", 0, 3, 1)]),
        P("임의의 중심 회전", ro_pivot, [S("각도 (°)", -180, 180, 45), S("중심 x (%)", 0, 100, 0), S("중심 y (%)", 0, 100, 0)]),
    ],
)


# ---- 3-5 전단 ------------------------------------------------------
def sr_x(c, s):
    s = s / 100.0
    out, M = warp_fit(c.img, [[1, s, 0], [0, 1, 0]])
    return out, f"x 방향 전단 shx={s:.2f}\nM =\n" + fmt_mat(M)


def sr_y(c, s):
    s = s / 100.0
    out, M = warp_fit(c.img, [[1, 0, 0], [s, 1, 0]])
    return out, f"y 방향 전단 shy={s:.2f}\nM =\n" + fmt_mat(M)


def sr_xy(c, sx, sy):
    sx, sy = sx / 100.0, sy / 100.0
    out, M = warp_fit(c.img, [[1, sx, 0], [sy, 1, 0]])
    return out, f"shx={sx:.2f}, shy={sy:.2f}\nM =\n" + fmt_mat(M)


GM_SHEAR = F(
    "전단",
    """■ 전단 변환 (Shear / Skew)
→ 직사각형을 평행사변형처럼 한쪽으로 기울이는 변환입니다. 글자를 이탤릭체로 만드는 것과 같은 원리입니다.

◇ x 방향 전단
 - 높이(y)에 비례해 x 좌표를 밀어냅니다. **M = [[1, shx, 0], [0, 1, 0]]**
 - 위쪽과 아래쪽이 서로 반대로 밀리며 수직선이 기울어집니다.

◇ y 방향 전단
 - 너비(x)에 비례해 y 좌표를 밀어냅니다. **M = [[1, 0, 0], [shy, 1, 0]]**

◇ x, y 동시 전단
 - 두 방향의 전단을 한 행렬로 동시에 적용합니다. **M = [[1, shx, 0], [shy, 1, 0]]**

※ 전단은 직선을 직선으로, 평행선을 평행선으로 유지하는 **어파인 변환**의 한 종류입니다.
※ 이 프로그램은 잘리지 않도록 출력 캔버스 크기를 자동 계산합니다. 빈 영역은 회색입니다.""",
    [
        P("x 방향 전단", sr_x, [S("shx ×100", -100, 100, 40)]),
        P("y 방향 전단", sr_y, [S("shy ×100", -100, 100, 40)]),
        P("x, y 동시 전단", sr_xy, [S("shx ×100", -100, 100, 30), S("shy ×100", -100, 100, 30)]),
    ],
)


# ---- 3-6 어파인 변환 -----------------------------------------------
def _aff_src(c):
    h, w = c.img.shape[:2]
    return np.float32([[w * 0.25, h * 0.25], [w * 0.75, h * 0.25], [w * 0.25, h * 0.75]])


def af_src(c):
    return draw_pts(c.img, _aff_src(c), (0, 200, 0)), "기준이 되는 3개의 점 (초록색)"


def af_points(c, dx2, dy2, dx3, dy3):
    h, w = c.img.shape[:2]
    src = _aff_src(c)
    dst = src.copy()
    dst[1] += (dx2, dy2)
    dst[2] += (dx3, dy3)
    M = cv2.getAffineTransform(src, dst)
    out = cv2.warpAffine(c.img, M, (w, h), borderValue=GRAY_BG)
    return draw_pts(out, dst, (0, 0, 255)), "점 2, 3 을 옮긴 위치(빨간색)에 맞춘 결과\nM =\n" + fmt_mat(M)


def af_matrix(c, ang, sc, tx, ty):
    h, w = c.img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc / 100.0)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(c.img, M, (w, h), borderValue=GRAY_BG), "회전 + 확대 + 이동을 하나의 행렬로 합성\nM =\n" + fmt_mat(M)


GM_AFFINE = F(
    "어파인 변환",
    """■ 어파인(Affine) 변환
→ **이동 + 회전 + 확대/축소 + 전단**을 한 번에 표현하는 2x3 행렬 변환입니다.
→ 성질: 직선은 직선으로, 평행선은 평행선으로 유지됩니다. (길이와 각도는 바뀔 수 있음)
→ **[x', y'] = [[a, b, c], [d, e, f]] · [x, y, 1]ᵀ**

◇ 3점 지정 (cv2.getAffineTransform)
 - 변환 전 3개 점과 변환 후 3개 점을 알려주면 그 관계를 만족하는 행렬을 자동으로 계산합니다.
 - 첫 패널: 기준 3점(초록). 둘째 패널: 점 2, 3 을 슬라이더로 옮기면 이미지가 그에 맞춰 변형됩니다. (빨간 점이 이동 후 위치)

◇ 행렬 직접 구성
 - 회전각, 배율, 이동량으로 행렬을 만들어 한 번에 적용합니다. (cv2.getRotationMatrix2D + 이동 성분 더하기)

※ 점이 3개이면 변환이 유일하게 결정됩니다. 점이 4개이면 어파인으로는 부족하고 '원근 변환'이 필요합니다.""",
    [
        P("기준 3점", af_src),
        P("3점 지정 변환", af_points, [S("점2 x 이동", -150, 150, 40), S("점2 y 이동", -150, 150, 30),
                                     S("점3 x 이동", -150, 150, -20), S("점3 y 이동", -150, 150, 30)]),
        P("행렬 직접 구성", af_matrix, [S("회전 (°)", -180, 180, 15), S("배율 (%)", 20, 200, 80),
                                    S("x 이동", -200, 200, 30), S("y 이동", -200, 200, 20)]),
    ],
)


# ---- 3-7 원근 투시 변환 ---------------------------------------------
def _corners(c):
    h, w = c.img.shape[:2]
    return np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])


def _persp(c, dst):
    h, w = c.img.shape[:2]
    M = cv2.getPerspectiveTransform(_corners(c), np.float32(dst))
    out = cv2.warpPerspective(c.img, M, (w, h), borderValue=GRAY_BG)
    return out, M


def pp_src(c):
    return draw_pts(c.img, _corners(c), (0, 200, 0)), "변환 전 네 모서리 (초록색, 시계 방향 1→4)"


def pp_squeeze(c, t, b, l, r):
    h, w = c.img.shape[:2]
    t, b, l, r = t / 100.0, b / 100.0, l / 100.0, r / 100.0
    dst = [[t * w / 2, l * h / 2], [w - 1 - t * w / 2, r * h / 2],
           [w - 1 - b * w / 2, h - 1 - r * h / 2], [b * w / 2, h - 1 - l * h / 2]]
    out, M = _persp(c, dst)
    return draw_pts(out, dst, (0, 0, 255)), "네 모서리를 안쪽으로 모은 결과\nM =\n" + fmt_mat(M)


def pp_rotate3d(c, ang, dist):
    h, w = c.img.shape[:2]
    cx, cy = w / 2, h / 2
    th, D = math.radians(ang), w * dist / 100.0
    dst = []
    for x, y in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        X, Y = x * w / 2, y * h / 2
        s = D / (D + X * math.sin(th))
        dst.append([cx + X * math.cos(th) * s, cy + Y * s])
    out, M = _persp(c, dst)
    return out, f"가로 축을 중심으로 {int(ang)}° 돌린 것처럼 보이는 투시\nM =\n" + fmt_mat(M)


GM_PERSPECTIVE = F(
    "원근 투시 변환",
    """■ 원근(Perspective) 투시 변환
→ 사진을 비스듬히 찍었을 때처럼 **멀리 있는 부분은 작게, 가까운 부분은 크게** 보이게 하는 변환입니다.
→ 3x3 행렬(호모그래피)을 사용하며, 직선은 직선으로 유지되지만 **평행선이 평행하지 않게** 됩니다. (어파인과의 차이)
→ 네 점의 대응 관계로 행렬을 구합니다: cv2.getPerspectiveTransform(변환 전 4점, 변환 후 4점) → cv2.warpPerspective

◇ 기준 네 모서리
 - 변환 전 이미지의 네 모서리 좌표입니다.

◇ 모서리 모으기
 - 윗변·아랫변·왼쪽·오른쪽 변을 각각 안쪽으로 줄입니다. 윗변만 줄이면 사다리꼴(멀리서 본 모양)이 됩니다.
 - 빨간 점이 변환 후 네 모서리 위치입니다.

◇ 3D 회전 시뮬레이션
 - 이미지를 세로축으로 돌렸을 때 카메라에서 보이는 모양을 계산해 만든 네 점으로 변환합니다.
 - '카메라 거리'가 가까울수록 원근 효과가 강해집니다.

※ 활용 예: 비스듬히 찍은 문서·명함을 반듯하게 펴기(문서 스캔), 번호판 정면화, 가상 시점 변환.""",
    [
        P("기준 네 모서리", pp_src),
        P("모서리 모으기", pp_squeeze, [S("윗변 축소 (%)", 0, 80, 35), S("아랫변 축소 (%)", 0, 80, 0),
                                   S("왼쪽 변 축소 (%)", 0, 80, 0), S("오른쪽 변 축소 (%)", 0, 80, 0)]),
        P("3D 회전 시뮬레이션", pp_rotate3d, [S("회전 각도 (°)", -70, 70, 35), S("카메라 거리 (%)", 60, 400, 150)]),
    ],
)


# ---- 3-8 사상 ------------------------------------------------------
def _grid(c):
    h, w = c.img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return xx, yy, w, h


def _remap(c, mx, my):
    return cv2.remap(c.img, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=GRAY_BG)


def mp_flip(c, mode):
    xx, yy, w, h = _grid(c)
    mode = int(mode)
    if mode == 0:
        mx, my, t = w - 1 - xx, yy, "좌우 반전: x' = W-1-x"
    elif mode == 1:
        mx, my, t = xx, h - 1 - yy, "상하 반전: y' = H-1-y"
    elif mode == 2:
        mx, my, t = w - 1 - xx, h - 1 - yy, "상하좌우 반전 (180° 회전)"
    else:
        mx, my, t = yy * (w / h), xx * (h / w), "대각선 대칭 (x↔y 맞바꿈)"
    return _remap(c, mx, my), "cv2.remap  " + t


def mp_wave(c, amp, wl):
    xx, yy, w, h = _grid(c)
    return _remap(c, xx + amp * np.sin(2 * np.pi * yy / wl), yy), f"x' = x + {int(amp)}·sin(2πy/{int(wl)})"


def mp_lens(c, k100):
    xx, yy, w, h = _grid(c)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    nx, ny = (xx - cx) / cx, (yy - cy) / cy
    f = 1 + (k100 / 100.0) * (nx * nx + ny * ny)
    return _remap(c, cx + nx * f * cx, cy + ny * f * cy), f"r' = r·(1 + k·r²), k={k100 / 100:.2f}  (+: 가운데 수축, -: 가운데 팽창)"


def mp_swirl(c, ang):
    xx, yy, w, h = _grid(c)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    R = min(h, w) / 2.0
    dx, dy = xx - cx, yy - cy
    r = np.sqrt(dx * dx + dy * dy)
    th = np.radians(ang) * np.clip(1 - r / R, 0, 1) ** 2
    mx = cx + np.cos(th) * dx - np.sin(th) * dy
    my = cy + np.sin(th) * dx + np.cos(th) * dy
    return _remap(c, mx, my), f"중심에서 멀어질수록 비틀림이 줄어드는 소용돌이 (최대 {int(ang)}°)"


GM_REMAP = F(
    "사상",
    """■ 사상 (Mapping, cv2.remap)
→ '결과 이미지의 (x, y) 위치에는 원본의 어느 위치 (map_x, map_y) 화소를 가져올까?' 를 **화소마다 직접** 지정하는 가장 일반적인 변환입니다.
→ 앞의 이동·회전·어파인·원근 변환은 모두 사상의 특수한 경우입니다. 수식으로 map_x, map_y 를 만들면 어떤 왜곡도 가능합니다.
→ 결과 → 원본 방향으로 좌표를 계산하는 것을 '역방향 사상'이라 하며, 빈틈이 생기지 않아 실제로 이 방식을 씁니다.

◇ 반전
 - 좌우: x' = W-1-x / 상하: y' = H-1-y / 상하좌우 / 대각선 대칭 (슬라이더 0~3)

◇ 물결 효과
 - **x' = x + A·sin(2πy/λ)** : 행마다 좌우로 흔들어 물결 모양을 만듭니다. (A: 진폭, λ: 파장)

◇ 렌즈 왜곡
 - 중심에서의 거리 r 에 따라 샘플링 위치를 **r' = r(1 + k·r²)** 로 바꿔 어안/핀쿠션 왜곡을 흉내냅니다.

◇ 소용돌이
 - 중심에 가까울수록 더 많이 회전시켜 소용돌이처럼 비틀어 놓습니다.

※ 범위를 벗어나는 위치는 회색으로 채우고, 값은 양선형 보간(INTER_LINEAR)으로 구합니다.""",
    [
        P("반전 (remap)", mp_flip, [S("모드 (0=좌우 1=상하 2=둘다 3=대각)", 0, 3, 0)]),
        P("물결 효과", mp_wave, [S("진폭 (px)", 0, 40, 10), S("파장 (px)", 10, 200, 60)]),
        P("렌즈 왜곡", mp_lens, [S("왜곡 계수 k ×100", -80, 120, 50)]),
        P("소용돌이", mp_swirl, [S("최대 각도 (°)", 0, 720, 180)]),
    ],
)

GEOM_FEATURES = [GM_RESIZE, GM_INTERP, GM_TRANSLATE, GM_ROTATE, GM_SHEAR, GM_AFFINE, GM_PERSPECTIVE, GM_REMAP]


# ======================================================================
# 4. 주파수영역처리
# ======================================================================
def _fft(g):
    return np.fft.fftshift(np.fft.fft2(g.astype(np.float32)))


def _ifft(F_):
    return np.fft.ifft2(np.fft.ifftshift(F_))


def _dist(shape):
    H, W = shape
    y, x = np.ogrid[:H, :W]
    return np.sqrt((y - H // 2) ** 2 + (x - W // 2) ** 2).astype(np.float32)


# ---- 4-1 푸리에 변환 실습 --------------------------------------------
def ft_orig(c):
    return c.gray, "공간 영역 (그레이스케일)"


def ft_mag(c, gain):
    mag = np.log1p(np.abs(_fft(c.gray)))
    return clip_u8(norm_u8(mag).astype(np.float32) * gain / 100.0), "log(1+|F|) : 중앙=저주파(DC), 바깥=고주파"


def ft_phase(c):
    ph = np.angle(_fft(c.gray))
    return clip_u8((ph + np.pi) / (2 * np.pi) * 255), "위상 스펙트럼 (-π ~ π 를 0~255 로 표시)"


def ft_recon(c, keep):
    F_ = _fft(c.gray)
    D = _dist(c.gray.shape)
    R = math.hypot(*c.gray.shape) / 2.0 * keep / 100.0
    out = clip_u8(_ifft(F_ * (D <= R)).real)
    return out, f"역변환(ifft): 저주파 {int(keep)}% 만 사용 (반지름 {R:.0f}px 안)"


def ft_mag_only(c):
    r = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(np.abs(_fft(c.gray)))).real)
    return norm_u8(np.log1p(np.abs(r))), "진폭만 사용(위상=0) : 영상의 구조가 사라짐"


def ft_phase_only(c):
    r = np.abs(_ifft(np.exp(1j * np.angle(_fft(c.gray)))))
    r = np.clip(r, 0, np.percentile(r, 99))
    return norm_u8(r), "위상만 사용(진폭=1) : 윤곽(에지) 정보가 남음"


FQ_FOURIER = F(
    "푸리에 변환 실습",
    """■ 푸리에 변환 (Fourier Transform) 실습
→ 이미지를 '밝기가 변하는 속도(주파수)'가 다른 수많은 파동(사인 곡선)의 합으로 분해하는 변환입니다.
→ 2차원 이산 푸리에 변환(DFT) : np.fft.fft2,  np.fft.fftshift 로 저주파를 가운데로 옮겨서 봅니다.

◇ 진폭 스펙트럼
 - 각 주파수 성분의 세기 |F(u,v)| 를 log(1+|F|) 로 압축해 보여줍니다. (값 차이가 너무 커서 로그를 씁니다.)
 - **가운데(저주파)**: 완만한 밝기 변화(전체적인 형태) / **바깥쪽(고주파)**: 급격한 변화(에지, 노이즈, 세부)
 - 이미지에 수직선이 많으면 스펙트럼에서 수평 방향 선이 나타나는 등 방향 정보도 보입니다.

◇ 위상 스펙트럼
 - 각 성분의 위치/모양 정보(각도)입니다. 눈으로는 무늬 없는 잡음처럼 보이지만 실제로는 구조 정보가 담겨 있습니다.

◇ 역변환(저주파 제한 복원)
 - np.fft.ifft2 로 되돌립니다. 100% 이면 원본과 같고, 비율을 줄이면 고주파가 사라져 **흐려집니다**(저역통과 필터 효과).

◇ 진폭만 / 위상만 복원
 - 위상을 버리면 영상의 구조가 완전히 사라지고, 진폭을 버려도(1로 고정) 윤곽은 남습니다.
 - **영상의 구조는 진폭보다 위상에 더 많이 들어 있음**을 보여주는 대표적인 실험입니다.""",
    [
        P("원본 (그레이)", ft_orig),
        P("진폭 스펙트럼", ft_mag, [S("밝기 배율 (%)", 50, 300, 100)]),
        P("위상 스펙트럼", ft_phase),
        P("역변환 (저주파만 복원)", ft_recon, [S("사용할 저주파 비율 (%)", 1, 100, 100)]),
        P("진폭만으로 복원", ft_mag_only),
        P("위상만으로 복원", ft_phase_only),
    ],
)


# ---- 4-2 주파수 필터링 -----------------------------------------------
def _ff(c, H):
    return clip_u8(_ifft(_fft(c.gray) * H).real)


def fl_ilpf(c, d0):
    return _ff(c, (_dist(c.gray.shape) <= d0).astype(np.float32)), f"이상적 LPF 차단 반지름 {int(d0)} (링잉 발생)"


def fl_glpf(c, d0):
    D = _dist(c.gray.shape)
    return _ff(c, np.exp(-(D ** 2) / (2.0 * d0 ** 2))), f"가우시안 LPF σ={int(d0)} (링잉 없음)"


def fl_blpf(c, d0, n):
    D = _dist(c.gray.shape)
    return _ff(c, 1.0 / (1.0 + (D / d0) ** (2 * int(n)))), f"버터워스 LPF D0={int(d0)}, 차수 n={int(n)}"


def fl_ihpf(c, d0):
    return _ff(c, (_dist(c.gray.shape) > d0).astype(np.float32)), f"이상적 HPF 차단 반지름 {int(d0)}"


def fl_ghpf(c, d0):
    D = _dist(c.gray.shape)
    return _ff(c, 1.0 - np.exp(-(D ** 2) / (2.0 * d0 ** 2))), f"가우시안 HPF σ={int(d0)}"


def _ring(c, lo, hi):
    lo, hi = sorted((float(lo), float(hi)))
    D = _dist(c.gray.shape)
    return ((D >= lo) & (D <= hi)).astype(np.float32), lo, hi


def fl_bpf(c, lo, hi):
    H, lo, hi = _ring(c, lo, hi)
    return _ff(c, H), f"대역통과: 반지름 {lo:.0f} ~ {hi:.0f} 사이 주파수만 통과"


def fl_brf(c, lo, hi):
    H, lo, hi = _ring(c, lo, hi)
    return _ff(c, 1.0 - H), f"대역차단: 반지름 {lo:.0f} ~ {hi:.0f} 사이 주파수를 제거"


_BAND = [S("안쪽 반지름", 1, 150, 20), S("바깥쪽 반지름", 2, 250, 60)]

FQ_FILTER = F(
    "주파수 필터링",
    """■ 주파수 영역 필터링
→ 이미지를 푸리에 변환한 뒤 **스펙트럼에 필터 H(u,v) 를 곱하고** 역변환하는 방법입니다.
→ **G(u,v) = F(u,v) × H(u,v)**  →  ifft 로 공간 영역으로 복원.
→ 스펙트럼 가운데(저주파)로부터의 거리 D 를 기준으로 필터를 만듭니다. 슬라이더 값 = 차단(컷오프) 반지름(px)입니다.

◇ 저역통과 필터 (LPF) : 저주파만 통과 → 이미지가 **흐려짐**, 노이즈 제거
 - **이상적(Ideal)** : D ≤ D0 이면 1, 아니면 0. 경계가 급격해서 물결무늬 **링잉(ringing)** 이 생깁니다.
 - **가우시안(Gaussian)** : H = exp(-D²/2σ²). 부드럽게 줄어들어 링잉이 없습니다.
 - **버터워스(Butterworth)** : H = 1/(1+(D/D0)^2n). 차수 n 이 커질수록 이상적 필터에 가까워집니다.

◇ 고역통과 필터 (HPF) : 고주파만 통과 → **에지·세부만 남음** (어두운 배경에 윤곽선)
 - 이상적 HPF 와 가우시안 HPF (= 1 - 가우시안 LPF).

◇ 대역통과 필터 (BPF) : 안쪽~바깥쪽 반지름 사이의 주파수만 통과
 - 너무 낮은 주파수(전체 밝기)와 너무 높은 주파수(노이즈)를 함께 제거해서 중간 크기의 무늬만 남깁니다.

◇ 대역차단 필터 (BRF) : 특정 대역의 주파수만 제거
 - 주기적인 줄무늬 노이즈를 지울 때 사용합니다.

※ 공간 영역의 컨볼루션(블러/샤프닝)과 수학적으로 같은 처리입니다. (컨볼루션 정리)""",
    [
        P("원본 (그레이)", ft_orig),
        P("이상적 저역통과 (ILPF)", fl_ilpf, [S("차단 반지름 D0", 1, 250, 40)]),
        P("가우시안 저역통과", fl_glpf, [S("σ (D0)", 1, 250, 40)]),
        P("버터워스 저역통과", fl_blpf, [S("차단 반지름 D0", 1, 250, 40), S("차수 n", 1, 10, 2)]),
        P("이상적 고역통과 (IHPF)", fl_ihpf, [S("차단 반지름 D0", 1, 250, 20)]),
        P("가우시안 고역통과", fl_ghpf, [S("σ (D0)", 1, 250, 20)]),
        P("대역통과 (BPF)", fl_bpf, _BAND),
        P("대역차단 (BRF)", fl_brf, _BAND),
    ],
)

FREQ_FEATURES = [FQ_FOURIER, FQ_FILTER]

# ======================================================================
# 메인 메뉴 구성 (화면의 버튼 배치 순서)
# ======================================================================
MENU = [
    ("화소단위처리", PIXEL_FEATURES),
    ("영역처리", AREA_FEATURES),
    ("기하학적처리", GEOM_FEATURES),
    ("주파수영역처리", FREQ_FEATURES),
]


# ======================================================================
# 5. 추가기능  (원본 프로그램에 없던 기능)
# ======================================================================
# ---- 5-1 색 공간 변환 -----------------------------------------------
_SPACES = [
    ("HSV", cv2.COLOR_BGR2HSV, ["색조 H", "채도 S", "명도 V"]),
    ("Lab", cv2.COLOR_BGR2Lab, ["밝기 L", "a (초록↔빨강)", "b (파랑↔노랑)"]),
    ("YCrCb", cv2.COLOR_BGR2YCrCb, ["밝기 Y", "색차 Cr", "색차 Cb"]),
    ("HLS", cv2.COLOR_BGR2HLS, ["색조 H", "밝기 L", "채도 S"]),
]


def _hsv(c):
    return cv2.cvtColor(c.img, cv2.COLOR_BGR2HSV)


def ex_cs_view(c, mode):
    name, code, _ = _SPACES[int(mode)]
    return cv2.cvtColor(c.img, code), f"{name} 값을 B,G,R 자리에 그대로 표시 (가짜 색)"


def ex_cs_channel(c, space, ch):
    name, code, chs = _SPACES[int(space)]
    return cv2.cvtColor(c.img, code)[..., int(ch) - 1], f"{name} 의 {chs[int(ch) - 1]} 채널"


def ex_hue_shift(c, shift):
    hsv = _hsv(c)
    hsv[..., 0] = ((hsv[..., 0].astype(np.int32) + int(shift)) % 180).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), f"색조(H)를 {int(shift)} 만큼 회전 (OpenCV 의 H 는 0~179)"


def ex_saturation(c, pct):
    hsv = _hsv(c)
    hsv[..., 1] = clip_u8(hsv[..., 1].astype(np.float32) * pct / 100.0)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), f"채도(S) × {pct / 100:.2f}  (0 이면 흑백)"


def ex_value(c, pct):
    hsv = _hsv(c)
    hsv[..., 2] = clip_u8(hsv[..., 2].astype(np.float32) * pct / 100.0)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), f"명도(V) × {pct / 100:.2f}  (색은 유지하고 밝기만 변경)"


EX_COLORSPACE = F(
    "색 공간 변환",
    """■ [추가기능] 색 공간 변환 (Color Space)
→ 같은 색을 표현하는 방법은 여러 가지입니다. 목적에 맞는 색 공간으로 바꾸면 처리가 훨씬 쉬워집니다. (cv2.cvtColor)

◇ 색 공간 보기
 - HSV / Lab / YCrCb / HLS 로 변환한 값을 B,G,R 처럼 그대로 표시합니다. 색이 이상해 보이는 것이 정상입니다.

◇ 채널 보기
 - **HSV**: 색조(H, 어떤 색인가) / 채도(S, 얼마나 선명한가) / 명도(V, 얼마나 밝은가)
 - **Lab**: L=밝기, a·b=색 방향. 사람이 느끼는 색 차이와 거리가 비슷한 공간입니다.
 - **YCrCb**: Y=밝기, Cr·Cb=색차. JPEG, 영상 압축에서 사용합니다.

◇ 색조 회전 / 채도 조절 / 명도 조절
 - RGB 에서는 어려운 '색만 바꾸기', '선명도만 바꾸기', '밝기만 바꾸기'가 HSV 에서는 한 채널만 건드리면 됩니다.

※ OpenCV 의 HSV 범위: H 0~179, S 0~255, V 0~255 (H 는 8비트에 맞추려 절반으로 줄인 값).""",
    [
        P("색 공간 보기", ex_cs_view, [S("0=HSV 1=Lab 2=YCrCb 3=HLS", 0, 3, 0)]),
        P("채널 보기", ex_cs_channel, [S("색 공간 (0=HSV 1=Lab 2=YCrCb)", 0, 2, 0), S("채널 (1~3)", 1, 3, 1)]),
        P("색조 회전", ex_hue_shift, [S("H 이동량", -90, 90, 40)]),
        P("채도 조절", ex_saturation, [S("채도 (%)", 0, 300, 150)]),
        P("명도 조절", ex_value, [S("명도 (%)", 20, 250, 130)]),
    ],
    extra=True,
)


# ---- 5-2 색상 영역 추출 -----------------------------------------------
def _range_mask(c, h1, h2, smin, vmin):
    lo = np.array([min(h1, h2), smin, vmin], np.uint8)
    hi = np.array([max(h1, h2), 255, 255], np.uint8)
    return cv2.inRange(_hsv(c), lo, hi)


def ex_mask(c, h1, h2, smin, vmin):
    m = _range_mask(c, h1, h2, smin, vmin)
    return m, f"cv2.inRange: 색조 {int(min(h1, h2))}~{int(max(h1, h2))} 에 해당하는 화소 = 흰색 ({int((m > 0).mean() * 100)}%)"


def ex_extract(c, h1, h2, smin, vmin):
    m = _range_mask(c, h1, h2, smin, vmin)
    return cv2.bitwise_and(c.img, c.img, mask=m), "마스크 영역의 원본 색만 남기고 나머지는 검정"


def ex_splash(c, h1, h2, smin, vmin):
    m = _range_mask(c, h1, h2, smin, vmin)
    g3 = cv2.cvtColor(c.gray, cv2.COLOR_GRAY2BGR)
    return np.where(m[..., None] > 0, c.img, g3).astype(np.uint8), "선택한 색만 컬러, 나머지는 흑백 (컬러 스플래시)"


_HSV_S = [S("색조 최소 (H)", 0, 179, 35), S("색조 최대 (H)", 0, 179, 85),
          S("채도 최소 (S)", 0, 255, 60), S("명도 최소 (V)", 0, 255, 50)]

EX_COLORMASK = F(
    "색상 영역 추출",
    """■ [추가기능] 색상 영역 추출 (Color Segmentation)
→ HSV 색 공간에서 색조(H) 범위로 '특정 색 물체'만 골라내는 대표적인 방법입니다. 신호등, 공, 마커 추적 등에 쓰입니다.

◇ 마스크 (cv2.inRange)
 - H, S, V 가 지정한 범위 안에 있으면 255(흰색), 아니면 0(검정)으로 만든 흑백 마스크입니다.
 - 대략적인 H: 빨강 0~10 및 170~179 / 주황 10~25 / 노랑 25~35 / 초록 35~85 / 파랑 100~130 / 보라 130~160

◇ 추출 결과 (cv2.bitwise_and)
 - 마스크가 흰색인 곳만 원본 색을 남깁니다.

◇ 컬러 스플래시
 - 선택한 색은 컬러로 두고 나머지는 흑백으로 만드는 사진 효과입니다.

※ 채도 최소·명도 최소를 올리면 회색·검정·흰색처럼 색이 없는 부분이 제외됩니다. 사진에 맞게 H 범위를 조절해 보세요.""",
    [
        P("마스크 (inRange)", ex_mask, _HSV_S),
        P("추출 결과", ex_extract, _HSV_S),
        P("컬러 스플래시", ex_splash, _HSV_S),
    ],
    extra=True,
)


# ---- 5-3 이진화 비교 ---------------------------------------------------
def ex_bin_fixed(c, t):
    _, b = cv2.threshold(c.gray, int(t), 255, cv2.THRESH_BINARY)
    return b, f"고정 임계값 {int(t)}"


def ex_bin_otsu(c):
    t, b = cv2.threshold(c.gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b, f"Otsu 가 찾은 임계값 = {t:.0f}"


def ex_bin_triangle(c):
    t, b = cv2.threshold(c.gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    return b, f"Triangle 이 찾은 임계값 = {t:.0f}"


def _adaptive(method, name):
    def f(c, block, C):
        block = odd(block, 3)
        return cv2.adaptiveThreshold(c.gray, 255, method, cv2.THRESH_BINARY, block, int(C)), f"{name}  블록 {block}x{block}, C={int(C)}"
    return f


EX_BINARY = F(
    "이진화 비교",
    """■ [추가기능] 이진화 방법 비교 (Thresholding)
→ 그레이 이미지를 검정(0)과 흰색(255) 두 값으로만 나누는 처리입니다. 물체 분리, 문서 스캔, 문자 인식의 첫 단계입니다.

◇ 고정 임계값 (cv2.threshold)
 - 화소값이 임계값보다 크면 255, 아니면 0. 사진 전체에 같은 기준을 적용하므로 조명이 고르지 않으면 실패합니다.

◇ Otsu 방법
 - 히스토그램을 보고 두 그룹(배경/물체)의 분산이 가장 잘 나뉘는 임계값을 **자동으로** 찾습니다.

◇ Triangle 방법
 - 히스토그램의 최고점과 끝점을 이은 직선에서 가장 멀리 떨어진 지점을 임계값으로 씁니다. 한쪽에 크게 치우친 분포에 유리합니다.

◇ 적응형(Adaptive) 평균 / 가우시안
 - 화소마다 **주변 블록의 평균(또는 가우시안 가중 평균) - C** 를 임계값으로 씁니다.
 - 그림자나 조명이 불균일한 문서 사진에 효과적입니다. 블록이 클수록 넓은 영역을 기준으로 삼습니다.""",
    [
        P("고정 임계값", ex_bin_fixed, [S("임계값", 0, 255, 127)]),
        P("Otsu 자동", ex_bin_otsu),
        P("Triangle 자동", ex_bin_triangle),
        P("적응형 평균", _adaptive(cv2.ADAPTIVE_THRESH_MEAN_C, "평균"), [S("블록 크기", 3, 99, 31), S("상수 C", -20, 30, 8)]),
        P("적응형 가우시안", _adaptive(cv2.ADAPTIVE_THRESH_GAUSSIAN_C, "가우시안"), [S("블록 크기", 3, 99, 31), S("상수 C", -20, 30, 8)]),
    ],
    extra=True,
)


# ---- 5-4 윤곽선 검출 ---------------------------------------------------
def _find_cnts(c, inv, min_area, mode):
    flag = (cv2.THRESH_BINARY_INV if int(inv) else cv2.THRESH_BINARY) + cv2.THRESH_OTSU
    _, b = cv2.threshold(c.gray, 0, 255, flag)
    cnts = cv2.findContours(b, mode, cv2.CHAIN_APPROX_SIMPLE)[-2]
    return [k for k in cnts if cv2.contourArea(k) >= min_area]


def ex_ct_external(c, inv, ma):
    cnts = _find_cnts(c, inv, ma, cv2.RETR_EXTERNAL)
    out = c.img.copy()
    cv2.drawContours(out, cnts, -1, (0, 200, 0), 2)
    for k in cnts:
        m = cv2.moments(k)
        if m["m00"] > 0:
            cv2.circle(out, (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])), 5, (0, 0, 255), -1)
    return out, f"가장 바깥 윤곽선 {len(cnts)}개 (빨간 점 = 무게중심)"


def ex_ct_all(c, inv, ma):
    cnts = _find_cnts(c, inv, ma, cv2.RETR_LIST)
    out = c.img.copy()
    for i, k in enumerate(cnts):
        col = cv2.cvtColor(np.uint8([[[(i * 37) % 180, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        cv2.drawContours(out, [k], -1, tuple(int(v) for v in col), 2)
    return out, f"안쪽 구멍까지 모든 윤곽선 {len(cnts)}개 (RETR_LIST)"


def ex_ct_box(c, inv, ma):
    cnts = _find_cnts(c, inv, ma, cv2.RETR_EXTERNAL)
    out = c.img.copy()
    for k in cnts:
        x, y, w, h = cv2.boundingRect(k)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0), 2)
        box = np.int32(cv2.boxPoints(cv2.minAreaRect(k)))
        cv2.polylines(out, [box], True, (0, 0, 255), 2)
    return out, "초록 = 경계 사각형(boundingRect), 빨강 = 최소 회전 사각형(minAreaRect)"


def ex_ct_hull(c, inv, ma):
    cnts = _find_cnts(c, inv, ma, cv2.RETR_EXTERNAL)
    out = c.img.copy()
    for k in cnts:
        cv2.polylines(out, [cv2.convexHull(k)], True, (255, 0, 0), 2)
        (x, y), r = cv2.minEnclosingCircle(k)
        cv2.circle(out, (int(x), int(y)), int(r), (0, 160, 255), 1)
    return out, "파랑 = 볼록 껍질(convexHull), 주황 = 최소 외접원"


_CT_S = [S("물체 색 (0=흰색, 1=검은색)", 0, 1, 1), S("최소 면적 (px²)", 0, 5000, 200)]

EX_CONTOUR = F(
    "윤곽선 검출",
    """■ [추가기능] 윤곽선 검출 (Contours)
→ 이진 이미지에서 같은 값이 이어진 영역의 **경계를 좌표 목록으로** 구합니다. 에지 검출이 '픽셀 맵'이라면 윤곽선은 '도형 데이터'라서 면적·둘레·위치를 계산할 수 있습니다.
→ 처리 순서: 그레이 → Otsu 이진화 → cv2.findContours → 그리기/분석. '물체 색'은 물체가 흰색이면 0, 검은색이면 1 로 선택하세요.

◇ 바깥 윤곽선 (RETR_EXTERNAL) + 무게중심
 - 가장 바깥 경계만 찾습니다. cv2.moments 로 무게중심을 구해 빨간 점으로 표시합니다.

◇ 모든 윤곽선 (RETR_LIST)
 - 물체 안쪽의 구멍 경계까지 모두 찾아 서로 다른 색으로 그립니다.

◇ 경계 사각형
 - boundingRect : 축에 평행한 가장 작은 사각형 / minAreaRect : 회전된 가장 작은 사각형.

◇ 볼록 껍질 / 최소 외접원
 - convexHull : 윤곽선을 감싸는 가장 작은 볼록 다각형(고무줄로 감싼 모양) / minEnclosingCircle : 감싸는 가장 작은 원.

※ 최소 면적을 올리면 작은 잡음 윤곽선을 걸러냅니다.""",
    [
        P("바깥 윤곽선 + 무게중심", ex_ct_external, _CT_S),
        P("모든 윤곽선", ex_ct_all, _CT_S),
        P("경계 사각형", ex_ct_box, _CT_S),
        P("볼록 껍질 / 외접원", ex_ct_hull, _CT_S),
    ],
    extra=True,
)


# ---- 5-5 허프 변환 -----------------------------------------------------
def _hough_edges(c):
    return cv2.Canny(cv2.GaussianBlur(c.gray, (5, 5), 0), 50, 150)


def ex_hough_p(c, th, minlen, gap):
    lines = cv2.HoughLinesP(_hough_edges(c), 1, np.pi / 180, int(th), minLineLength=int(minlen), maxLineGap=int(gap))
    out = c.img.copy()
    arr = np.zeros((0, 4)) if lines is None else np.asarray(lines).reshape(-1, 4)   # OpenCV 버전별 모양 차이 대응
    n = len(arr)
    for x1, y1, x2, y2 in arr[:300]:
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    return out, f"확률적 허프 직선(HoughLinesP): {n}개 검출"


def ex_hough_std(c, th):
    lines = cv2.HoughLines(_hough_edges(c), 1, np.pi / 180, int(th))
    out = c.img.copy()
    arr = np.zeros((0, 2)) if lines is None else np.asarray(lines).reshape(-1, 2)[:, :2]
    n = len(arr)
    for rho, theta in arr[:100]:
        a, b = math.cos(theta), math.sin(theta)
        x0, y0 = a * rho, b * rho
        p1 = (int(x0 + 2000 * (-b)), int(y0 + 2000 * a))
        p2 = (int(x0 - 2000 * (-b)), int(y0 - 2000 * a))
        cv2.line(out, p1, p2, (255, 0, 0), 1)
    return out, f"표준 허프 직선(HoughLines): {n}개 (무한 직선, 최대 100개 표시)"


def ex_hough_circle(c, p2, rmin):
    g = cv2.medianBlur(c.gray, 5)
    circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, 1.2, 30, param1=100, param2=int(p2),
                               minRadius=int(rmin), maxRadius=0)
    out = c.img.copy()
    n = 0
    if circles is not None:
        arr = np.asarray(circles).reshape(-1, 3)
        n = len(arr)
        for x, y, r in arr[:200]:
            cv2.circle(out, (int(x), int(y)), int(r), (0, 200, 0), 2)
            cv2.circle(out, (int(x), int(y)), 3, (0, 0, 255), -1)
    return out, f"허프 원 검출(HoughCircles): {n}개 (초록=원, 빨강=중심)"


EX_HOUGH = F(
    "허프 변환",
    """■ [추가기능] 허프 변환 (Hough Transform)
→ 에지 점들이 **직선이나 원 같은 도형을 이루는지** 투표 방식으로 찾아내는 알고리즘입니다. 도로 차선, 건물 선, 동전/눈동자 검출에 쓰입니다.
→ 원리: 에지 점 하나를 지나는 모든 직선(ρ, θ)에 한 표씩 던지고, 표가 많이 모인 (ρ, θ) 를 직선으로 인정합니다.
→ 먼저 Canny 에지(50/150)를 구한 뒤 적용합니다.

◇ 표준 허프 직선 (cv2.HoughLines)
 - (ρ, θ) 로 표현되는 **무한히 긴 직선**을 찾습니다. 임계값(투표 수)이 높을수록 확실한 직선만 남습니다.

◇ 확률적 허프 직선 (cv2.HoughLinesP)
 - **시작점과 끝점이 있는 선분**을 찾습니다. 최소 선분 길이, 선분 사이 최대 틈(maxLineGap)을 지정할 수 있어 실제로 더 많이 씁니다.

◇ 허프 원 검출 (cv2.HoughCircles)
 - 원의 중심과 반지름 3개 변수를 투표로 찾습니다. param2 가 작으면 가짜 원이 많이 나오고, 크면 확실한 원만 남습니다.
 - 입력 노이즈를 줄이려고 먼저 중간값 블러를 적용합니다.""",
    [
        P("직선 (HoughLinesP)", ex_hough_p, [S("투표 임계값", 10, 200, 60), S("최소 선분 길이", 5, 200, 40), S("최대 틈", 1, 50, 10)]),
        P("직선 (HoughLines)", ex_hough_std, [S("투표 임계값", 30, 300, 120)]),
        P("원 (HoughCircles)", ex_hough_circle, [S("param2 (작을수록 많이 검출)", 10, 100, 40), S("최소 반지름", 5, 150, 15)]),
    ],
    extra=True,
)


# ---- 5-6 코너 검출 -----------------------------------------------------
def ex_harris(c, k100, th):
    d = cv2.dilate(cv2.cornerHarris(np.float32(c.gray), 2, 3, k100 / 100.0), None)
    mask = d > (th / 100.0) * d.max() if d.max() > 0 else np.zeros(d.shape, bool)
    out = c.img.copy()
    out[mask] = (0, 0, 255)
    n = cv2.connectedComponents(mask.astype(np.uint8))[0] - 1
    return out, f"Harris 코너 영역 약 {n}곳 (k={k100 / 100:.2f}, 최댓값의 {int(th)}% 이상)"


def ex_shitomasi(c, maxc, q):
    pts = cv2.goodFeaturesToTrack(c.gray, int(maxc), q / 100.0, 10)
    out = c.img.copy()
    arr = np.zeros((0, 2)) if pts is None else np.asarray(pts).reshape(-1, 2)
    n = len(arr)
    for x, y in arr:
        cv2.circle(out, (int(x), int(y)), 5, (0, 200, 0), 2)
    return out, f"Shi-Tomasi 코너 {n}개 (최대 {int(maxc)}개, 품질 {q / 100:.2f})"


def ex_fast(c, t):
    kps = cv2.FastFeatureDetector_create(threshold=int(t), nonmaxSuppression=True).detect(c.gray, None)
    out = c.img.copy()
    for kp in kps:
        cv2.circle(out, (int(kp.pt[0]), int(kp.pt[1])), 4, (255, 0, 0), 1)
    return out, f"FAST 특징점 {len(kps)}개 (임계값 {int(t)})"


EX_CORNER = F(
    "코너 검출",
    """■ [추가기능] 코너(모서리) 검출
→ 코너 = 어느 방향으로 움직여도 주변 밝기가 크게 변하는 점입니다. 에지는 한 방향으로만 변하지만 코너는 두 방향 모두 변합니다.
→ 이미지 정합(파노라마), 물체 추적, 3D 복원처럼 '위치를 정확히 맞춰야 하는' 작업의 기준점으로 쓰입니다.

◇ Harris 코너 (cv2.cornerHarris)
 - 작은 창을 움직였을 때 변화량을 행렬로 계산하고 코너 점수 R = det(M) - k·trace(M)² 를 구합니다.
 - k 는 보통 0.04~0.06. 임계값이 낮을수록 코너가 많이 잡힙니다. (빨간 영역)

◇ Shi-Tomasi (cv2.goodFeaturesToTrack)
 - Harris 를 개선해 고윳값의 최솟값으로 점수를 매기며, 가장 좋은 점부터 N개만 골라 줍니다. (품질 수준, 최소 거리 10px)

◇ FAST (cv2.FastFeatureDetector)
 - 중심 화소 주변 16개 원형 화소와 밝기를 비교하는 매우 빠른 방법입니다. 실시간 영상에 적합합니다.""",
    [
        P("Harris", ex_harris, [S("k ×100", 1, 10, 4), S("임계값 (최댓값의 %)", 1, 50, 5)]),
        P("Shi-Tomasi", ex_shitomasi, [S("최대 개수", 1, 300, 50), S("품질 ×100", 1, 50, 10)]),
        P("FAST", ex_fast, [S("임계값", 5, 100, 30)]),
    ],
    extra=True,
)


# ---- 5-7 이미지 효과 ---------------------------------------------------
def ex_sketch(c, k):
    k = odd(k, 3)
    blur = cv2.GaussianBlur(255 - c.gray, (k, k), 0)
    return cv2.divide(c.gray, 255 - blur, scale=256), f"연필 스케치: 반전 블러 {k}x{k} 와 나누기(color dodge)"


def ex_sepia(c, pct):
    m = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]], np.float32)
    sep = cv2.transform(c.img.astype(np.float32), m)
    a = pct / 100.0
    return clip_u8(c.img * (1 - a) + sep * a), f"세피아 {int(pct)}%: 색 변환 행렬로 갈색 톤을 만듦"


def ex_cartoon(c, levels, block):
    smooth = cv2.bilateralFilter(c.img, 9, 75, 75)
    step = max(1, 256 // int(levels))
    color = clip_u8((smooth // step) * step + step // 2)
    edges = cv2.adaptiveThreshold(cv2.medianBlur(c.gray, 7), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, odd(block, 3), 7)
    return cv2.bitwise_and(color, color, mask=edges), f"양방향 필터 + 색 {int(levels)}단계 + 적응형 에지 선"


def ex_mosaic(c, b):
    h, w = c.img.shape[:2]
    b = max(1, int(b))
    small = cv2.resize(c.img, (max(1, w // b), max(1, h // b)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST), f"{b}x{b} 블록 모자이크 (축소 후 최근접 확대)"


def ex_vignette(c, strength):
    h, w = c.img.shape[:2]
    sig = 1.25 - strength / 100.0
    mask = cv2.getGaussianKernel(h, h * sig) @ cv2.getGaussianKernel(w, w * sig).T
    mask = mask / mask.max()
    return clip_u8(c.img * mask[..., None]), "가장자리를 어둡게 (가우시안 마스크 곱)"


EX_EFFECT = F(
    "이미지 효과",
    """■ [추가기능] 이미지 효과 (Photo Effects)
→ 앞에서 배운 기본 연산(블러, 에지, 색 변환, 마스크)을 조합하면 사진 앱의 필터 효과를 직접 만들 수 있습니다.

◇ 연필 스케치
 - 그레이를 반전해 블러 처리한 뒤 원본을 그 값으로 나눕니다(color dodge). 밝기가 급변하는 에지만 어둡게 남습니다.

◇ 세피아
 - 3x3 색 변환 행렬로 B,G,R 을 섞어 갈색 톤 옛날 사진 느낌을 냅니다. (cv2.transform) 강도 100% 가 완전한 세피아입니다.

◇ 카툰
 - 양방향 필터로 색을 단순화 → 색 단계 줄이기(포스터화) → 적응형 이진화로 만든 에지 선을 곱합니다.

◇ 모자이크
 - 이미지를 작게 줄였다가 최근접 보간으로 키워 큰 블록으로 만듭니다. 개인정보 가리기에 쓰입니다.

◇ 비네팅
 - 중심이 밝고 가장자리가 어두운 가우시안 마스크를 곱해 시선을 가운데로 모읍니다.""",
    [
        P("연필 스케치", ex_sketch, [S("블러 크기", 3, 51, 21)]),
        P("세피아", ex_sepia, [S("강도 (%)", 0, 100, 100)]),
        P("카툰", ex_cartoon, [S("색 단계 수", 2, 16, 8), S("선 블록 크기", 3, 25, 9)]),
        P("모자이크", ex_mosaic, [S("블록 크기 (px)", 2, 60, 16)]),
        P("비네팅", ex_vignette, [S("강도", 10, 100, 60)]),
    ],
    extra=True,
)


# ---- 5-8 K-means 색 분할 -----------------------------------------------
_km_cache = {}


def _kmeans(c, K):
    key = (c.img.shape, int(c.img.sum()), int(K))
    if key in _km_cache:
        return _km_cache[key]
    h, w = c.img.shape[:2]
    data = c.img.reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, int(K), None, crit, 2, cv2.KMEANS_PP_CENTERS)
    order = np.argsort(centers.sum(axis=1))           # 어두운 군집부터 번호 부여
    rank = np.empty(len(order), np.int32)
    rank[order] = np.arange(len(order))
    res = (rank[labels.ravel()].reshape(h, w), clip_u8(centers[order]))
    if len(_km_cache) > 12:
        _km_cache.clear()
    _km_cache[key] = res
    return res


def ex_km_simplify(c, K):
    lab, cen = _kmeans(c, K)
    return cen[lab], f"모든 화소를 가장 가까운 대표색 {int(K)}개 중 하나로 교체"


def ex_km_cluster(c, K, idx):
    lab, _ = _kmeans(c, K)
    idx = min(int(idx), int(K)) - 1
    m = lab == idx
    out = np.where(m[..., None], c.img, (c.img * 0.15).astype(np.uint8)).astype(np.uint8)
    return out, f"군집 {idx + 1}/{int(K)} 에 속한 화소 {m.mean() * 100:.1f}% (어두운 군집이 앞 번호)"


def ex_poster(c, bits):
    s = 8 - int(bits)
    return ((c.img >> s) << s).astype(np.uint8), f"채널당 {int(bits)}비트 → 채널당 {2 ** int(bits)}단계, 총 {2 ** (3 * int(bits))}색"


EX_KMEANS = F(
    "K-means 색 분할",
    """■ [추가기능] K-means 군집화로 색 분할 / 색 줄이기
→ 군집화(Clustering)란 비슷한 것끼리 K개의 그룹으로 묶는 기계학습 방법입니다. 각 화소의 (B,G,R) 값을 3차원 점으로 보고 비슷한 색끼리 묶습니다.
→ 알고리즘: ① 중심 K개 선택 → ② 각 화소를 가장 가까운 중심에 배정 → ③ 중심을 그룹 평균으로 이동 → 변화가 없을 때까지 ②③ 반복 (cv2.kmeans)
→ 계산량이 커서 슬라이더를 놓을 때 결과가 갱신됩니다.

◇ 색 단순화
 - 모든 화소를 K개의 대표색으로 교체합니다. K 가 작을수록 만화/포스터 같은 느낌이 나고 파일 용량도 줄어듭니다. (색 양자화)

◇ 군집 하나만 보기
 - 선택한 군집에 속한 부분만 원래 색으로 보이고 나머지는 어둡게 합니다. 하늘, 숲, 피부처럼 색으로 영역을 나누는 영상 분할입니다.

◇ 포스터화 (비트 줄이기)
 - K-means 없이 각 채널의 하위 비트를 버려 색 단계를 줄입니다. 빠르지만 색 분포를 고려하지 않아 K-means 보다 품질이 낮습니다.""",
    [
        P("K-means 색 단순화", ex_km_simplify, [S("군집 수 K", 2, 12, 4)], heavy=True),
        P("군집 하나만 보기", ex_km_cluster, [S("군집 수 K", 2, 12, 4), S("보려는 군집 번호", 1, 12, 1)], heavy=True),
        P("포스터화 (비트 줄이기)", ex_poster, [S("채널당 비트 수", 1, 7, 2)]),
    ],
    extra=True,
)

EXTRA_FEATURES = [EX_COLORSPACE, EX_COLORMASK, EX_BINARY, EX_CONTOUR, EX_HOUGH, EX_CORNER, EX_EFFECT, EX_KMEANS]
MENU.append(("추가기능", EXTRA_FEATURES))
