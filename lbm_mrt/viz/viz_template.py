#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_template.py  –  统一科研制图模板（支持“英文期刊图”一键切“博士论文中文图”）
======================================================================================

你这版模板的“关键增强”（解决你刚才 tick 字号改不动的问题，并减少脚本里的重复代码）：
1) ✅ 修复：_format_ticks() 在设置 ticklabel 字体时，把 size/weight 一起写死，避免被 Matplotlib 回退为 medium
2) ✅ 新增 3 个“省事窗口”（你后续脚本可以少写很多辅助函数）：
   (A) init_style(...)    : 一行完成 build_fontpack + set_global_style，并缓存为全局默认
   (B) format_axes(...)   : 一行完成 _format_spines + _format_ticks（最常用）
   (C) set_* 支持 fontpack=None : set_xlabel/set_ylabel/set_title/set_axis_labels 可省略传 fontpack

使用建议（你后续绘图脚本最简结构）：


CN_FONT_FILES = [
    "assets/fonts/simsun.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
]

V.init_style(mode="thesis_cn", base_fontsize=22, bold=True, axis_linewidth=2.0, cn_font_files=CN_FONT_FILES)

fig, ax = V.create_figure_ax(figsize=(8, 6))
ax.plot(x, y)

V.set_xlabel(ax, "x (m)", fontsize=22, lang="en")
V.set_ylabel(ax, "温度 (K)", fontsize=22, lang="cn")

V.format_axes(ax, tick_font="cn", tick_labelsize=20, tick_weight="bold")

V.save_figure(fig, "demo.pdf")
------------------------------------------------

注意：
- “lang=cn/en/auto” 功能仍保留（用于中英文混排）
- tick_font 用于控制“刻度数字”使用中文/英文的字体（论文常用 cn，期刊常用 en）

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.ticker import MultipleLocator

# =============================================================================
# 0) 字体管理：英文/中文分离 + 自动回退 + 可选“按文件强制加载”
# =============================================================================

# 下面两个全局变量，是为了让你的绘图脚本“少写很多 fontpack 传参”
_DEFAULT_FONTPACK: Optional["FontPack"] = None
_DEFAULT_MODE: str = "journal"
_VIZ_TICK_FAMILY: str = ""


@dataclass(frozen=True)
class FontPack:
    """保存一组英文字体/中文字体的 FontProperties，便于全局复用。"""

    en: fm.FontProperties
    cn: fm.FontProperties


def _contains_cjk(text: str) -> bool:
    """粗略判断字符串是否包含中文/日文/韩文（CJK）字符。"""
    for ch in text:
        code = ord(ch)
        # CJK Unified Ideographs + Extensions（常用范围足够）
        if (0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF):
            return True
    return False


def _addfont_if_exists(font_file: str | Path) -> Optional[str]:
    """
    若 font_file 存在，则 addfont 并返回该字体在 matplotlib 中的 name；不存在返回 None。
    说明：addfont 后字体会进入 matplotlib fontManager，可用于 family=name 的方式调用。
    """
    p = Path(font_file).expanduser().resolve()
    if not p.exists():
        return None
    fm.fontManager.addfont(str(p))
    name = fm.FontProperties(fname=str(p)).get_name()
    return name


def _pick_first_available_font_name(candidates: Sequence[str]) -> Optional[str]:
    """
    在 matplotlib 已知字体列表里，按候选名字顺序找第一个可用字体 name。
    注意：依赖 fontManager.ttflist；在无桌面/字体索引不全环境下可能扫不到系统字体。
    这种情况应走“按文件强制加载”（cn_font_files / en_font_files）。
    """
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


def build_fontpack(
    *,
    # [强烈推荐] 指定字体文件路径（最稳）
    cn_font_files: Optional[Sequence[str | Path]] = None,
    en_font_files: Optional[Sequence[str | Path]] = None,
    # [可选] 如果不指定文件，则按字体“名字”尝试匹配（依赖 fontManager 扫描）
    cn_font_names: Optional[Sequence[str]] = None,
    en_font_names: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> FontPack:
    """
    构建 FontPack（英文 + 中文）。

    优先级：
      1) 如果提供 font_files：按文件 addfont -> 必定可用（只要文件存在）
      2) 否则按 font_names：尝试从 ttflist 匹配
      3) 再否则兜底使用 matplotlib 默认字体（不推荐，但不会崩）
    """

    # ---------- 默认候选（可按你机器习惯调整） ----------
    default_en_names = (
        list(en_font_names)
        if en_font_names
        else [
            "Times New Roman",
            "Times",
            "Liberation Serif",
            "DejaVu Serif",
        ]
    )

    default_cn_names = (
        list(cn_font_names)
        if cn_font_names
        else [
            "SimSun",  # Windows
            "宋体",
            "STSong",  # macOS/LaTeX
            "Songti SC",
            "Noto Sans CJK SC",
            "Noto Serif CJK SC",
            "AR PL UKai CN",
            "AR PL UMing TW MBE",
            "WenQuanYi Micro Hei",
            "Microsoft YaHei",
        ]
    )

    # ---------- 1) 先尝试“按文件强制加载” ----------
    en_name = None
    cn_name = None

    if en_font_files:
        for fp in en_font_files:
            en_name = _addfont_if_exists(fp)
            if en_name:
                break

    if cn_font_files:
        for fp in cn_font_files:
            cn_name = _addfont_if_exists(fp)
            if cn_name:
                break

    # ---------- 2) 如果没提供文件/或文件不存在，则按名字匹配 ----------
    if en_name is None:
        en_name = _pick_first_available_font_name(default_en_names)

    if cn_name is None:
        cn_name = _pick_first_available_font_name(default_cn_names)

    # ---------- 3) 兜底 ----------
    if en_name is None:
        en_name = "DejaVu Serif"
    if cn_name is None:
        # 若中文字体仍找不到，中文会变方块；你应提供 cn_font_files 解决
        cn_name = "DejaVu Sans"

    en_prop = fm.FontProperties(family=en_name)
    cn_prop = fm.FontProperties(family=cn_name)

    if verbose:
        print(f"[viz_template] EN font = {en_name}")
        print(f"[viz_template] CN font = {cn_name}")

    return FontPack(en=en_prop, cn=cn_prop)


# =============================================================================
# 1) 全局风格设置（rcParams）：英文期刊 vs 中文论文
# =============================================================================


def set_global_style(
    *,
    mode: str = "journal",
    base_fontsize: int = 24,
    bold: bool = True,
    axis_linewidth: float = 2.0,
    fontpack: Optional[FontPack] = None,
) -> FontPack:
    """
    一次性设置 Matplotlib 全局 rcParams，并返回 FontPack（英文/中文）。

    mode:
      - "journal"   : 英文期刊风格（Times 为主；刻度数字默认英文）
      - "thesis_cn" : 中文论文风格（刻度数字默认中文；中文标签更像宋体）
      - "custom"    : 你自己传 fontpack，或 build_fontpack 自己配

    注：即使全局 mode=thesis_cn，你仍可在 set_xlabel/set_ylabel/set_title 里用 lang="en"
        强制英文文本使用英文衬线字体（Times/Liberation/DejaVu Serif）。
    """
    global _DEFAULT_FONTPACK, _DEFAULT_MODE

    if fontpack is None:
        # ✅ 1) 如果已经 init_style / set_global_style 过，就复用缓存，绝不重建
        if _DEFAULT_FONTPACK is not None:
            fontpack = _DEFAULT_FONTPACK
        else:
            # ✅ 2) 第一次调用才构建兜底 fontpack（并把 simsun.ttc 放到最前）
            cn_files = [
                "assets/fonts/simsun.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
                "/usr/share/fonts/truetype/arphic/ukai.ttc",
                "/usr/share/fonts/truetype/arphic/uming.ttc",
            ]
            fontpack = build_fontpack(cn_font_files=cn_files, verbose=True)

    weight = "bold" if bold else "normal"

    # mode 决定 rcParams 默认 family（注意只是兜底）
    if mode == "journal":
        default_family = fontpack.en.get_name()
        tick_family = fontpack.en.get_name()
    elif mode == "thesis_cn":
        default_family = fontpack.cn.get_name()
        tick_family = fontpack.cn.get_name()
    elif mode == "custom":
        default_family = fontpack.en.get_name()
        tick_family = fontpack.en.get_name()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # --- 写入 rcParams（这里主要控制线宽、默认字号、legend 等） ---
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [default_family, "DejaVu Sans"],
            "font.weight": weight,
            "axes.labelsize": base_fontsize,
            "axes.titlesize": base_fontsize,
            "axes.labelweight": weight,
            # PDF/PS 嵌入（避免 PDF 中文丢失）
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # 轴与刻度线
            "axes.linewidth": axis_linewidth,
            "xtick.major.width": axis_linewidth,
            "ytick.major.width": axis_linewidth,
            "xtick.minor.width": axis_linewidth * 0.75,
            "ytick.minor.width": axis_linewidth * 0.75,
            # —— 刻度长度 ——
            "xtick.major.size": 8,
            "ytick.major.size": 8,
            "xtick.minor.size": 6,
            "ytick.minor.size": 6,
            # —— 图例 ——
            "legend.frameon": False,
            "legend.fontsize": int(base_fontsize * 0.66),
            "axes.unicode_minus": False,
        }
    )

    # 默认刻度字号（注意：真正稳的是 _format_ticks 会对 ticklabel 再写一次 size）
    mpl.rcParams["xtick.labelsize"] = base_fontsize
    mpl.rcParams["ytick.labelsize"] = base_fontsize

    # 记录“默认 tick 字体家族”（内部使用）
    _DEFAULT_MODE = mode
    _DEFAULT_FONTPACK = fontpack

    # 兼容旧逻辑：如果你之前脚本用过这个 key，也能工作
    # mpl.rcParams["_viz_tick_family"] = tick_family  # type: ignore
    global _VIZ_TICK_FAMILY
    _VIZ_TICK_FAMILY = tick_family
    return fontpack


# =============================================================================
# 省事窗口 (A)：init_style —— 一行初始化字体+风格，并缓存为默认 fontpack
# =============================================================================


def init_style(
    *,
    mode: str = "journal",
    base_fontsize: int = 24,
    bold: bool = True,
    axis_linewidth: float = 2.0,
    cn_font_files: Optional[Sequence[str | Path]] = None,
    en_font_files: Optional[Sequence[str | Path]] = None,
    cn_font_names: Optional[Sequence[str]] = None,
    en_font_names: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> FontPack:
    """
    你后续绘图脚本最推荐的入口：
      - 一行完成：build_fontpack + set_global_style
      - 并缓存为全局默认：后续 set_xlabel/set_ylabel/_format_ticks 都可以不传 fontpack
    """
    fp = build_fontpack(
        cn_font_files=cn_font_files,
        en_font_files=en_font_files,
        cn_font_names=cn_font_names,
        en_font_names=en_font_names,
        verbose=verbose,
    )
    set_global_style(
        mode=mode,
        base_fontsize=base_fontsize,
        bold=bold,
        axis_linewidth=axis_linewidth,
        fontpack=fp,
    )
    return fp


# =============================================================================
# 2) 画布 / 坐标系管理
# =============================================================================


def create_figure_ax(
    figsize: Tuple[int, int] = (10, 8),
    **kwargs: object,
) -> Tuple[Figure, Axes]:
    """生成 (fig, ax)，支持传 constrained_layout、dpi 等参数。"""
    fig, ax = plt.subplots(figsize=figsize, **kwargs)  # type: ignore[call-overload]
    return fig, ax


def add_secondary_y(
    ax: Axes,
    *,
    ylabel: str,
    fontpack: Optional[FontPack] = None,
    ylim: Optional[Tuple[float, float]] = None,
    label_fontsize: Optional[int] = None,
    label_weight: str = "bold",
    lang: str = "auto",  # "auto" | "en" | "cn"
    tick_font: str = "auto",
    tick_labelsize: Optional[int] = None,
) -> Axes:
    """右侧添加次 y 轴。"""
    ax2 = ax.twinx()
    set_ylabel(
        ax2,
        ylabel,
        fontpack=fontpack,
        fontsize=label_fontsize,
        weight=label_weight,
        lang=lang,
    )
    if ylim is not None:
        ax2.set_ylim(*ylim)
    _format_spines(ax2)
    _format_ticks(ax2, fontpack=fontpack, tick_font=tick_font, labelsize=tick_labelsize)
    return ax2


# =============================================================================
# 3) 文本设置：标题/坐标轴标签（自动中英切换）—— 支持 fontpack=None
# =============================================================================


def _get_default_fontpack() -> FontPack:
    """获取全局默认 FontPack；若尚未 init_style/set_global_style，则构建一个兜底版本。"""
    global _DEFAULT_FONTPACK
    if _DEFAULT_FONTPACK is None:
        _DEFAULT_FONTPACK = build_fontpack(
            cn_font_files=[
                "assets/fonts/simsun.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/arphic/ukai.ttc",
                "/usr/share/fonts/truetype/arphic/uming.ttc",
            ],
            verbose=True,
        )
    return _DEFAULT_FONTPACK


def _choose_prop(
    text: str, fontpack: Optional[FontPack], lang: str
) -> fm.FontProperties:
    """
    lang:
      - "en"  : 强制英文字体
      - "cn"  : 强制中文字体
      - "auto": 按字符判断（含 CJK -> 中文字体，否则英文）
    """
    fp = fontpack if fontpack is not None else _get_default_fontpack()

    if lang == "en":
        return fp.en
    if lang == "cn":
        return fp.cn
    return fp.cn if _contains_cjk(text) else fp.en


def set_title(
    ax: Axes,
    title: str,
    *,
    fontpack: Optional[FontPack] = None,
    fontsize: Optional[int] = None,
    weight: str = "bold",
    lang: str = "auto",
) -> None:
    prop = _choose_prop(title, fontpack, lang)
    ax.set_title(title, fontproperties=prop, fontsize=fontsize, fontweight=weight)


def set_xlabel(
    ax: Axes,
    xlabel: str,
    *,
    fontpack: Optional[FontPack] = None,
    fontsize: Optional[int] = None,
    weight: str = "bold",
    lang: str = "auto",
) -> None:
    prop = _choose_prop(xlabel, fontpack, lang)
    ax.set_xlabel(xlabel, fontproperties=prop, fontsize=fontsize, fontweight=weight)


def set_ylabel(
    ax: Axes,
    ylabel: str,
    *,
    fontpack: Optional[FontPack] = None,
    fontsize: Optional[int] = None,
    weight: str = "bold",
    lang: str = "auto",
) -> None:
    prop = _choose_prop(ylabel, fontpack, lang)
    ax.set_ylabel(ylabel, fontproperties=prop, fontsize=fontsize, fontweight=weight)


def set_axis_labels(
    ax: Axes,
    xlabel: str,
    ylabel: str,
    *,
    fontpack: Optional[FontPack] = None,
    fontsize: int = 24,
    weight: str = "bold",
    xlang: str = "auto",
    ylang: str = "auto",
) -> None:
    set_xlabel(
        ax, xlabel, fontpack=fontpack, fontsize=fontsize, weight=weight, lang=xlang
    )
    set_ylabel(
        ax, ylabel, fontpack=fontpack, fontsize=fontsize, weight=weight, lang=ylang
    )


# =============================================================================
# 4) 坐标轴格式化：spines / ticks（重点修复：ticklabel size 不再被覆盖）
# =============================================================================


def _format_spines(ax: Axes, lw: float | None = None) -> None:
    """四条边框统一线宽。"""
    lw = lw or plt.rcParams["axes.linewidth"]
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_linewidth(lw)


def _format_ticks(
    ax: Axes,
    *,
    fontpack: Optional[FontPack] = None,
    x_minor: float | None = None,
    y_minor: float | None = None,
    tick_font: str = "auto",  # "auto" | "en" | "cn"
    labelsize: Optional[int] = None,
    tick_weight: str = "bold",
) -> None:
    """
    设置主/次刻度，并设置“刻度数字”的字体。

    ✅ 关键修复（你刚才的问题）：
    - 以前只 set_fontproperties(family=...) 不带 size，Matplotlib 有时会回退 ticklabel 字号为 medium
    - 现在：我们显式构造 FontProperties(family=..., size=labelsize, weight=...) 并写回每个 ticklabel
      这样 tick_labelsize 一定生效，不会被覆盖

    tick_font:
      - "cn": 刻度数字用中文字体（论文常用）
      - "en": 刻度数字用英文字体（期刊常用）
      - "auto": thesis_cn -> cn; journal -> en
    """
    fp = fontpack if fontpack is not None else _get_default_fontpack()

    if x_minor is not None:
        ax.xaxis.set_minor_locator(MultipleLocator(x_minor))
    if y_minor is not None:
        ax.yaxis.set_minor_locator(MultipleLocator(y_minor))

    if labelsize is None:
        labelsize = int(plt.rcParams.get("xtick.labelsize", 12))

    # 先用 tick_params 统一长度/线宽/字号（保持刻度线风格）
    ax.tick_params(axis="both", which="major", length=8, width=2, labelsize=labelsize)
    ax.tick_params(axis="both", which="minor", length=6, width=1.5)

    # 选择刻度字体 family
    if tick_font == "en":
        family = fp.en.get_name()
    elif tick_font == "cn":
        family = fp.cn.get_name()
    else:
        family = fp.cn.get_name() if _DEFAULT_MODE == "thesis_cn" else fp.en.get_name()

    # ✅ 把 size/weight 写死到 FontProperties 里，避免回退
    prop = fm.FontProperties(family=family, size=labelsize, weight=tick_weight)

    for lab in ax.get_xticklabels(which="both"):
        lab.set_fontproperties(prop)
    for lab in ax.get_yticklabels(which="both"):
        lab.set_fontproperties(prop)


# =============================================================================
# 省事窗口 (B)：format_axes —— 一行 spines + ticks（你后续基本只用这个）
# =============================================================================


def format_axes(
    ax: Axes,
    *,
    fontpack: Optional[FontPack] = None,
    tick_font: str = "auto",
    tick_labelsize: Optional[int] = None,
    tick_weight: str = "bold",
    x_minor: float | None = None,
    y_minor: float | None = None,
    spine_lw: float | None = None,
) -> None:
    """
    一行完成坐标轴常规格式化：
      - 边框线宽统一
      - 主/次刻度 locator
      - tick 字体/字号/粗细稳定生效
    """
    _format_spines(ax, lw=spine_lw)
    _format_ticks(
        ax,
        fontpack=fontpack,
        x_minor=x_minor,
        y_minor=y_minor,
        tick_font=tick_font,
        labelsize=tick_labelsize,
        tick_weight=tick_weight,
    )


# =============================================================================
# 其它常用小工具
# =============================================================================


def set_axis_limits(
    ax: Axes,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)


def set_legend(
    ax: Axes,
    *,
    fontpack: Optional[FontPack] = None,
    loc: str = "best",
    lang: str = "auto",
    fontsize: Optional[int] = None,
    weight: str = "bold",
) -> Legend:
    """
    图例字体也按中英分离，并支持 fontpack=None。
    """
    fp = fontpack if fontpack is not None else _get_default_fontpack()
    leg = ax.legend(loc=loc)

    # 如果用户没传 fontsize，则保持 matplotlib 默认 legend fontsize
    fs = fontsize if fontsize is not None else None

    for txt in leg.get_texts():
        prop0 = _choose_prop(txt.get_text(), fp, lang)

        # 与 tick 一样：为了稳，给 legend 也写死 size/weight（可选）
        if fs is None:
            # 继承当前 legend 的 fontsize
            prop = fm.FontProperties(
                family=prop0.get_name(),
                size=txt.get_fontsize(),
                weight=weight,
            )
        else:
            prop = fm.FontProperties(
                family=prop0.get_name(),
                size=fs,
                weight=weight,
            )
        txt.set_fontproperties(prop)

    return leg


# =============================================================================
# 5) 图片保存
# =============================================================================


def save_figure(
    fig: Figure,
    path: str | Path,
    *,
    dpi: int = 600,
    transparent: bool = False,
    tight: bool = True,
) -> None:
    """
    统一保存接口：
    - tight=True：tight_layout + bbox_inches="tight"（你大多数论文图都希望这样）
    - tight=False：完全按照当前 subplots_adjust 的布局输出（适合你 PT panels 那类）
    """
    path = Path(path)
    if tight:
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=transparent)
    else:
        fig.savefig(path, dpi=dpi, transparent=transparent)
    print(f"[viz_template] Saved: {path}")


# =============================================================================
# 6) 快速示例：英文期刊 vs 中文论文
# =============================================================================

if __name__ == "__main__":
    x = np.linspace(0, 1, 200)
    y1 = np.sin(2 * np.pi * x)

    # ---------------------------------------------------------
    # A) 英文期刊图（Times New Roman 风格）
    # ---------------------------------------------------------
    init_style(
        mode="journal", base_fontsize=18, bold=True, axis_linewidth=2.0, verbose=True
    )

    fig, ax = create_figure_ax(figsize=(8, 6))
    ax.plot(x, y1, lw=2, label="sin")

    set_title(ax, "Journal-style Title", fontsize=20, lang="en")
    set_axis_labels(ax, "x", "sin(x)", fontsize=20, xlang="en", ylang="en")

    format_axes(ax, tick_font="en", tick_labelsize=16, x_minor=0.1, y_minor=0.2)
    set_axis_limits(ax, (0, 1), (-1.2, 1.2))
    set_legend(ax, loc="upper right", lang="en", fontsize=16)

    save_figure(fig, "demo_journal.pdf", dpi=300)
    plt.close(fig)

    # ---------------------------------------------------------
    # B) 论文中文图（刻度数字/中文标题尽量“宋体风格”，英文仍可 Times）
    # ---------------------------------------------------------
    init_style(
        mode="thesis_cn", base_fontsize=20, bold=True, axis_linewidth=2.0, verbose=True
    )

    fig, ax = create_figure_ax(figsize=(8, 6))
    ax.plot(x, y1, lw=2, label="正弦 sin")

    set_title(ax, "中文标题：距离-时间", fontsize=22, lang="cn")
    set_axis_labels(ax, "距离 (m)", "时间 (min)", fontsize=22, xlang="cn", ylang="cn")

    # 论文：刻度数字用中文字体（并且 tick_labelsize 一定生效）
    format_axes(ax, tick_font="cn", tick_labelsize=18, x_minor=0.1, y_minor=0.2)
    set_axis_limits(ax, (0, 1), (-1.2, 1.2))
    set_legend(ax, loc="upper right", lang="auto", fontsize=18)

    save_figure(fig, "demo_thesis_cn.png", dpi=300)
    save_figure(fig, "demo_thesis_cn.pdf", dpi=300)
    plt.close(fig)

    print("[viz_template] Demo done.")
