"""
영상처리 실습 프로그램  (Python + tkinter + OpenCV)

실행:   python app.py
필요:   pip install opencv-python pillow numpy
구성:   app.py      - 화면(GUI)
        features.py - 영상처리 기능 + 설명 문구
"""
import math
import os
import traceback
import urllib.request
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import cv2
import numpy as np
from PIL import Image, ImageTk

import features as ft

# ----------------------------------------------------------------------
# 디자인 설정 (색상 / 글꼴)  -  화면 모양을 바꾸고 싶으면 여기만 고치면 됩니다
# ----------------------------------------------------------------------
FONT = "맑은 고딕"
BG = "#eef1f5"          # 전체 배경
CARD = "#ffffff"        # 카드(흰 상자) 배경
NAVY = "#1f2d3d"        # 상단 바
TEXT = "#1f2937"        # 기본 글자색
SUB = "#6b7280"         # 보조 글자색
BORDER = "#d6dbe3"      # 테두리
ACCENT = "#3b82f6"      # 주요 버튼 색
ACCENT_D = "#2563eb"
GRAY_BTN = "#64748b"
GRAY_BTN_D = "#475569"
# 메뉴 분류별 색상 (화소 / 영역 / 기하 / 주파수 / 추가기능 순서)
CAT_COLORS = ["#3b82f6", "#10b981", "#8b5cf6", "#ef4444", "#f59e0b"]

IMG_TYPES = [("Image files", ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")),
             ("All files", "*.*")]


# ----------------------------------------------------------------------
# 이미지 입출력 / 표시 도우미 (한글 경로 지원)
# ----------------------------------------------------------------------
def read_bytes(path):
    return np.fromfile(path, np.uint8)


def save_image(path, img):
    ext = os.path.splitext(path)[1].lower() or ".png"
    ft.encode_image(img, ext).tofile(path)


def to_rgb_u8(img):
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = ft.norm_u8(a)
    if a.ndim == 2:
        return cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
    if a.shape[2] == 4:
        return cv2.cvtColor(a, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def make_photo(img, size):
    """이미지를 size x size 정사각형 칸에 비율을 유지하며 맞춰 PhotoImage 로 만든다."""
    rgb = to_rgb_u8(img)
    h, w = rgb.shape[:2]
    s = min(size / w, size / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 238, np.uint8)
    y, x = (size - nh) // 2, (size - nw) // 2
    canvas[y:y + nh, x:x + nw] = rgb
    return ImageTk.PhotoImage(Image.fromarray(canvas))


# ----------------------------------------------------------------------
# 공통 위젯 도우미
# ----------------------------------------------------------------------
def make_button(parent, text, command, bg=ACCENT, hover=ACCENT_D, fg="white",
                font=None, padx=18, pady=9, width=None):
    """평평한 디자인 + 마우스를 올리면 색이 바뀌는 버튼"""
    b = tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                  activebackground=hover, activeforeground=fg, relief="flat", bd=0,
                  cursor="hand2", font=font or (FONT, 10, "bold"), padx=padx, pady=pady,
                  highlightthickness=0)
    if width:
        b.configure(width=width)
    b.bind("<Enter>", lambda e: b.configure(bg=hover))
    b.bind("<Leave>", lambda e: b.configure(bg=bg))
    return b


def make_menu_button(parent, text, command, color):
    """메뉴용 버튼: 평소엔 흰색, 마우스를 올리면 분류 색으로 변함"""
    b = tk.Button(parent, text=text, command=command, bg=CARD, fg=TEXT,
                  activebackground=color, activeforeground="white", relief="flat", bd=0,
                  cursor="hand2", font=(FONT, 10), width=21, height=2,
                  highlightthickness=1, highlightbackground=BORDER, highlightcolor=color)

    def on_enter(_e):
        b.configure(bg=color, fg="white", highlightbackground=color)

    def on_leave(_e):
        b.configure(bg=CARD, fg=TEXT, highlightbackground=BORDER)

    b.bind("<Enter>", on_enter)
    b.bind("<Leave>", on_leave)
    return b


def hline(parent, color=BORDER):
    return tk.Frame(parent, bg=color, height=1)


# ----------------------------------------------------------------------
# 세로 스크롤 가능한 프레임
# ----------------------------------------------------------------------
class ScrollFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        self.vsb = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self.canvas, bg=BG)
        self.win = self.canvas.create_window((0, 0), window=self.inner, anchor="n")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.coords(self.win, e.width // 2, 0))
        self.canvas.bind_all("<MouseWheel>", self._wheel)
        self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-2, "units"))
        self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(2, "units"))

    def _wheel(self, e):
        try:
            self.canvas.yview_scroll(int(-e.delta / 120) * 2, "units")
        except tk.TclError:
            pass

    def destroy(self):
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            try:
                self.canvas.unbind_all(seq)
            except tk.TclError:
                pass
        super().destroy()


# ----------------------------------------------------------------------
# 결과 이미지 1개 + 슬라이더들  (흰색 카드 모양)
# ----------------------------------------------------------------------
class PanelView(tk.Frame):
    def __init__(self, parent, page, panel, size, color=ACCENT):
        super().__init__(parent, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        self.page, self.panel, self.size = page, panel, size
        self.last = None
        self.photo = None
        self._job = None

        # 카드 맨 위의 색 띠 + 제목
        tk.Frame(self, bg=color, height=4).pack(fill="x")
        tk.Label(self, text=panel.title, font=(FONT, 11, "bold"), fg=TEXT, bg=CARD).pack(pady=(8, 6), padx=10)

        frame = tk.Frame(self, bg=BORDER, padx=1, pady=1)
        frame.pack(padx=12)
        self.img_label = tk.Label(frame, bg=CARD, cursor="hand2", bd=0)
        self.img_label.pack()
        self.img_label.bind("<Double-Button-1>", self.save_result)

        self.scales = []
        for s in panel.sliders:
            sc = tk.Scale(self, from_=s.lo, to=s.hi, resolution=s.res, orient="horizontal",
                          label=s.label, length=size, font=(FONT, 9), fg=TEXT, bg=CARD,
                          troughcolor="#dbe2ea", activebackground=color, bd=0, sliderrelief="flat",
                          highlightthickness=0)
            sc.set(s.default)
            sc.pack(pady=(4, 0), padx=12)
            if panel.heavy:
                sc.bind("<ButtonRelease-1>", lambda e: self.refresh())
            else:
                sc.configure(command=lambda v: self.schedule())
            self.scales.append((s, sc))

        self.cap = tk.Label(self, text="", font=(FONT, 9), fg=SUB, bg=CARD,
                            wraplength=size, justify="left", anchor="nw")
        self.cap.pack(pady=(6, 10), padx=12, anchor="w")

    @staticmethod
    def _value(s, sc):
        v = float(sc.get())
        return int(round(v)) if float(s.res).is_integer() else v

    def schedule(self):
        if self._job is not None:
            self.after_cancel(self._job)
        self._job = self.after(15, self.refresh)

    def refresh(self):
        self._job = None
        vals = [self._value(s, sc) for s, sc in self.scales]
        try:
            res = self.panel.func(self.page.ctx, *vals)
            cap = ""
            if isinstance(res, tuple):
                res, cap = res
            self.last = res
            self.photo = make_photo(res, self.size)
            self.img_label.configure(image=self.photo)
            self.cap.configure(text=cap, fg=SUB)
        except Exception as e:                      # 한 패널의 오류가 프로그램 전체를 멈추지 않게
            traceback.print_exc()
            self.cap.configure(text=f"오류: {e}", fg="#dc2626")

    def save_result(self, _event=None):
        if self.last is None:
            return
        path = filedialog.asksaveasfilename(title="결과 이미지 저장", defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")])
        if not path:
            return
        try:
            img = self.last
            if img.dtype != np.uint8:
                img = ft.norm_u8(img)
            save_image(path, img)
            messagebox.showinfo("저장 완료", f"저장했습니다.\n{path}")
        except Exception as e:
            messagebox.showerror("저장 실패", str(e))


# ----------------------------------------------------------------------
# 기능 화면
# ----------------------------------------------------------------------
class FeaturePage(tk.Frame):
    def __init__(self, app, feature):
        super().__init__(app.container, bg=BG)
        self.app, self.feature, self.ctx = app, feature, app.ctx
        self.desc_win = None
        title = feature.name.replace("\n", " ")
        color = CAT_COLORS[-1] if feature.extra else app.color_of(feature)

        # ---- 위쪽 제목 줄 ----
        top = tk.Frame(self, bg=CARD)
        top.pack(side="top", fill="x")
        tk.Frame(top, bg=color, width=6).pack(side="left", fill="y")
        if feature.extra:
            tk.Label(top, text="추가기능", font=(FONT, 9, "bold"), fg="white", bg=color,
                     padx=8, pady=2).pack(side="left", padx=(14, 0), pady=14)
            title_text = title
        else:
            title_text = title
        tk.Label(top, text=title_text, font=(FONT, 16, "bold"), fg=TEXT, bg=CARD).pack(side="left", padx=14, pady=12)
        tk.Label(top, text="슬라이더를 움직이면 결과가 바로 바뀝니다  ·  결과 이미지를 더블클릭하면 저장",
                 font=(FONT, 9), fg=SUB, bg=CARD).pack(side="right", padx=18)
        hline(self).pack(side="top", fill="x")

        # ---- 아래쪽 버튼 줄 ----
        hline(self).pack(side="bottom", fill="x")
        bar = tk.Frame(self, bg=CARD)
        bar.pack(side="bottom", fill="x")
        inner = tk.Frame(bar, bg=CARD)
        inner.pack(pady=10)
        for label, action in feature.buttons:
            make_button(inner, label, lambda a=action: self.do_action(a),
                        bg=GRAY_BTN, hover=GRAY_BTN_D).pack(side="left", padx=6)
        make_button(inner, "설명 보기", self.show_desc, bg=ACCENT, hover=ACCENT_D, width=14).pack(side="left", padx=6)
        make_button(inner, "뒤로", app.show_menu, bg=NAVY, hover="#0f1a26", width=14).pack(side="left", padx=6)

        # ---- 가운데 결과 카드들 ----
        sf = ScrollFrame(self)
        sf.pack(fill="both", expand=True)

        n = len(feature.panels)
        cols = n if n <= 4 else (3 if n <= 6 else 4)
        rows = math.ceil(n / cols)
        size = (300 if cols <= 3 else 275) if rows == 1 else 250
        self.views = []
        for i, panel in enumerate(feature.panels):
            v = PanelView(sf.inner, self, panel, size, color)
            v.grid(row=i // cols, column=i % cols, padx=10, pady=12, sticky="n")
            self.views.append(v)

        self.update_idletasks()
        for v in self.views:
            v.refresh()
            self.update_idletasks()

    def refresh_all(self):
        for v in self.views:
            v.refresh()

    # ---- 버튼 동작 ----------------------------------------------------
    def do_action(self, action):
        if action == "load_img2":
            path = filedialog.askopenfilename(title="합성할 이미지 선택", filetypes=IMG_TYPES)
            if not path:
                return
            img = cv2.imdecode(read_bytes(path), cv2.IMREAD_COLOR)
            if img is None:
                messagebox.showerror("오류", "이미지를 열 수 없습니다.")
                return
            self.ctx.img2_custom = img
            self.refresh_all()
        elif action == "reset_img2":
            self.ctx.img2_custom = None
            self.refresh_all()
        elif action == "save_orig":
            path = filedialog.asksaveasfilename(title="이미지 저장", defaultextension=".png",
                                                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"),
                                                           ("BMP", "*.bmp"), ("WebP", "*.webp")])
            if not path:
                return
            try:
                save_image(path, self.ctx.orig)
                messagebox.showinfo("저장 완료", f"저장했습니다.\n{path}")
            except Exception as e:
                messagebox.showerror("저장 실패", str(e))

    def show_desc(self):
        if self.desc_win is not None and self.desc_win.winfo_exists():
            self.desc_win.lift()
            return
        win = tk.Toplevel(self)
        win.title("설명 보기")
        win.geometry("720x660")
        win.configure(bg=CARD)
        self.desc_win = win

        make_button(win, "닫기", win.destroy, bg=GRAY_BTN, hover=GRAY_BTN_D, width=12).pack(side="bottom", pady=10)
        hline(win).pack(side="bottom", fill="x")
        frame = tk.Frame(win, bg=CARD)
        frame.pack(fill="both", expand=True)
        sb = tk.Scrollbar(frame)
        sb.pack(side="right", fill="y")
        txt = tk.Text(frame, wrap="word", font=(FONT, 11), bg=CARD, fg=TEXT, bd=0, highlightthickness=0,
                      padx=22, pady=18, yscrollcommand=sb.set, spacing1=3, spacing3=3, cursor="arrow")
        txt.pack(side="left", fill="both", expand=True)
        sb.config(command=txt.yview)
        txt.tag_configure("b", font=(FONT, 11, "bold"))
        txt.tag_configure("h", font=(FONT, 15, "bold"), foreground=NAVY, spacing3=10)
        txt.tag_configure("sec", font=(FONT, 11, "bold"), foreground=ACCENT_D, spacing1=12)
        txt.tag_configure("note", foreground=SUB)
        for line in self.feature.desc.split("\n"):
            stripped = line.lstrip()
            if line.startswith("■"):
                txt.insert("end", line + "\n", "h")
            elif stripped.startswith("◇"):
                txt.insert("end", line + "\n", "sec")
            elif stripped.startswith("※"):
                txt.insert("end", line + "\n", "note")
            else:
                for i, part in enumerate(line.split("**")):       # **굵게** 표시
                    txt.insert("end", part, "b" if i % 2 else "")
                txt.insert("end", "\n")
        txt.configure(state="disabled")


# ----------------------------------------------------------------------
# 메인 애플리케이션
# ----------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("영상처리 실습 프로그램")
        self.geometry("1360x880")
        self.minsize(900, 600)
        self.configure(bg=BG)

        # 항상 보이는 상단 바
        self.header = tk.Frame(self, bg=NAVY)
        self.header.pack(side="top", fill="x")
        tk.Label(self.header, text="영상처리 실습 프로그램", font=(FONT, 13, "bold"), fg="white",
                 bg=NAVY).pack(side="left", padx=18, pady=10)
        self.info_label = tk.Label(self.header, text="", font=(FONT, 9), fg="#cbd5e1", bg=NAVY)
        self.info_label.pack(side="right", padx=18)

        self.container = tk.Frame(self, bg=BG)
        self.container.pack(fill="both", expand=True)
        self.ctx = None
        self.source = "local"
        self.preview_photo = None
        self.show_welcome()

    def clear(self):
        for w in self.container.winfo_children():
            w.destroy()

    def color_of(self, feature):
        """기능이 속한 메뉴 분류의 색상"""
        for i, (_cat, feats) in enumerate(ft.MENU):
            if feature in feats:
                return CAT_COLORS[i % len(CAT_COLORS)]
        return ACCENT

    def _card(self, pady=(0, 0)):
        """화면 가운데에 놓이는 흰색 카드"""
        card = tk.Frame(self.container, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        card.place(relx=0.5, rely=0.5, anchor="center")
        return card

    # ---- 1) 시작 화면 ---------------------------------------------------
    def show_welcome(self):
        self.clear()
        self.info_label.configure(text="")
        card = self._card()
        tk.Frame(card, bg=ACCENT, height=6).pack(fill="x")
        tk.Label(card, text="환영합니다! 영상처리 실습 프로그램입니다.", font=(FONT, 20, "bold"), fg=NAVY,
                 bg=CARD).pack(padx=70, pady=(46, 10))
        tk.Label(card, text="버튼을 눌러 실습에 사용할 사진을 불러와주세요.", font=(FONT, 13), fg=SUB,
                 bg=CARD).pack(padx=70, pady=(0, 30))
        row = tk.Frame(card, bg=CARD)
        row.pack(padx=70, pady=(0, 50))
        make_button(row, "로컬 파일 불러오기", self.load_local, font=(FONT, 12, "bold"),
                    width=20, pady=14).pack(side="left", padx=10)
        make_button(row, "링크를 통해 불러오기", self.load_link, bg=GRAY_BTN, hover=GRAY_BTN_D,
                    font=(FONT, 12, "bold"), width=20, pady=14).pack(side="left", padx=10)

    # ---- 이미지 불러오기 ------------------------------------------------
    def _set_image(self, buf, source):
        try:
            self.ctx = ft.Ctx(buf)
        except Exception:
            messagebox.showerror("오류", "이미지를 열 수 없습니다.\n(jpg, png, bmp 등 이미지 파일인지 확인하세요)")
            return
        self.source = source
        h, w = self.ctx.orig.shape[:2]
        self.info_label.configure(text=f"불러온 이미지  {w} x {h}")
        self.show_preview()

    def load_local(self):
        path = filedialog.askopenfilename(title="열기", filetypes=IMG_TYPES)
        if not path:
            return
        try:
            self._set_image(read_bytes(path), "local")
        except Exception as e:
            messagebox.showerror("오류", f"파일을 읽지 못했습니다.\n{e}")

    def load_link(self):
        url = simpledialog.askstring("링크로 불러오기", "이미지 주소(URL)를 입력하세요:", parent=self)
        if not url:
            return
        try:
            req = urllib.request.Request(url.strip(), headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                data = r.read()
            self._set_image(np.frombuffer(data, np.uint8), "link")
        except Exception as e:
            messagebox.showerror("오류", f"링크에서 이미지를 불러오지 못했습니다.\n{e}")

    # ---- 2) 선택된 이미지 확인 화면 --------------------------------------
    def show_preview(self):
        self.clear()
        card = self._card()
        tk.Frame(card, bg=ACCENT, height=6).pack(fill="x")
        tk.Label(card, text="선택된 이미지", font=(FONT, 17, "bold"), fg=NAVY, bg=CARD).pack(padx=60, pady=(26, 14))
        self.preview_photo = make_photo(self.ctx.orig, 420)
        frame = tk.Frame(card, bg=BORDER, padx=1, pady=1)
        frame.pack(padx=60)
        tk.Label(frame, image=self.preview_photo, bg=CARD, bd=0).pack()
        h, w = self.ctx.orig.shape[:2]
        tk.Label(card, text=f"원본 크기 {w} x {h}   (실습에는 긴 변 {ft.MAX_SIDE}px 이하로 줄여 사용)",
                 font=(FONT, 9), fg=SUB, bg=CARD).pack(pady=(10, 16))
        row = tk.Frame(card, bg=CARD)
        row.pack(padx=60, pady=(0, 30))
        if self.source == "link":
            make_button(row, "링크에서 다시 선택", self.load_link, bg=GRAY_BTN, hover=GRAY_BTN_D,
                        width=22, pady=11).pack(side="left", padx=8)
        else:
            make_button(row, "로컬 파일에서 다시 선택", self.load_local, bg=GRAY_BTN, hover=GRAY_BTN_D,
                        width=22, pady=11).pack(side="left", padx=8)
        make_button(row, "다음", self.show_menu, width=22, pady=11).pack(side="left", padx=8)

    # ---- 3) 기법 선택 화면 -----------------------------------------------
    def show_menu(self):
        self.clear()
        box = tk.Frame(self.container, bg=BG)
        box.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(box, text="실습할 영상처리 기법을 선택하세요.", font=(FONT, 18, "bold"), fg=NAVY,
                 bg=BG).pack(pady=(0, 6))
        tk.Label(box, text="마우스를 올리면 색이 바뀌고, 클릭하면 실습 화면으로 이동합니다.",
                 font=(FONT, 10), fg=SUB, bg=BG).pack(pady=(0, 22))
        cols = tk.Frame(box, bg=BG)
        cols.pack()
        for ci, (cat, feats) in enumerate(ft.MENU):
            color = CAT_COLORS[ci % len(CAT_COLORS)]
            col = tk.Frame(cols, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
            col.grid(row=0, column=ci, padx=8, sticky="n")
            tk.Label(col, text=cat, font=(FONT, 12, "bold"), fg="white", bg=color,
                     pady=9).pack(fill="x")
            body = tk.Frame(col, bg=CARD)
            body.pack(padx=10, pady=10)
            for f in feats:
                make_menu_button(body, f.name, lambda f=f: self.show_feature(f), color).pack(pady=3)
        make_button(box, "다른 이미지 선택", self.show_welcome, bg=GRAY_BTN, hover=GRAY_BTN_D,
                    width=20, pady=8).pack(pady=(26, 0))

    # ---- 4) 기능 화면 ----------------------------------------------------
    def show_feature(self, feature):
        self.clear()
        FeaturePage(self, feature).pack(fill="both", expand=True)


if __name__ == "__main__":
    App().mainloop()
