#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os, re
from typing import Dict, List, Optional
import numpy as np, pandas as pd

pd.options.display.float_format = "{:,.6f}".format

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import PercentFormatter, FuncFormatter

HOUSE_GREY = "#4a4a4a"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 16,
    "axes.titleweight": "medium",
    "axes.labelsize": 11,
    "axes.edgecolor": HOUSE_GREY,
    "xtick.color": HOUSE_GREY,
    "ytick.color": HOUSE_GREY,
    "text.color": HOUSE_GREY,
    "axes.labelcolor": HOUSE_GREY,
    "legend.edgecolor": "none",
})


def _legend_if_any(ax, **kwargs):
    """Safely add a legend only if labels exist."""
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(**kwargs)



def read_fama_french_csv(path: str) -> pd.DataFrame:
    import re
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    start_idx = None
    for i, line in enumerate(lines):
        parts = [p.strip() for p in line.strip().split(",")]
        if parts and re.fullmatch(r"\d{6}", parts[0] or ""):
            start_idx = i
            break
    if start_idx is None or start_idx == 0:
        return pd.read_csv(path)
    header_idx = start_idx - 1
    while header_idx >= 0 and (not lines[header_idx].strip()):
        header_idx -= 1
    if header_idx < 0:
        header_idx = start_idx - 1
    nrows = 0
    for j in range(start_idx, len(lines)):
        L = lines[j].strip()
        if not L:
            break
        low = L.lower()
        if low.startswith("annual") or low.startswith("monthly") or low.startswith("copyright") or low.startswith("notes"):
            break
        nrows += 1
    df = pd.read_csv(path, header=0, skiprows=header_idx, nrows=(nrows if nrows>0 else None), engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df

def auto_load_table(path: str, sheet: Optional[str]) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path, sheet_name=(sheet if sheet else 0))
    try:
        df = pd.read_csv(path)
    except Exception:
        df = read_fama_french_csv(path)
    else:
        first_col = str(df.columns[0]).lower().replace(" ", "")
        if "thisfilewascreated" in first_col or "crsp" in first_col:
            df = read_fama_french_csv(path)
    return df

def detect_date_column(df: pd.DataFrame) -> str:
    candidates = ["date", "yyyymm", "year", "caldt"]
    norm = {c.lower().strip(): c for c in df.columns}
    for key in candidates:
        if key in norm:
            return norm[key]
    for c in df.columns:
        if str(c).lower().startswith("unnamed"):
            s = df[c].astype(str).str.strip()
            if s.str.fullmatch(r"\d{6}").any() or pd.to_datetime(df[c], errors="coerce").notna().any():
                return c
    for c in df.columns:
        s = df[c].astype(str).str.strip()
        if s.str.fullmatch(r"\d{6}").any():
            return c
    return df.columns[0]

def coerce_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    raw = df[date_col].astype(str).str.strip()
    if raw.str.fullmatch(r"\d{6}").all():
        df[date_col] = pd.to_datetime(raw.str[:4] + "-" + raw.str[4:] + "-01", errors="raise")
    else:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    return df

def filter_years(df: pd.DataFrame, begin: Optional[int], end: Optional[int]) -> pd.DataFrame:
    if begin is not None:
        df = df[df.index.year >= int(begin)]
    if end is not None:
        df = df[df.index.year <= int(end)]
    return df

def to_real_series(nominal: pd.Series, cpi: pd.Series) -> pd.Series:
    aligned = pd.concat([nominal, cpi], axis=1, join="inner").dropna()
    cpi_infl = aligned.iloc[:, 1].pct_change()
    real = ((1.0 + aligned.iloc[:, 0]) / (1.0 + cpi_infl)) - 1.0
    real.index = aligned.index
    return real.reindex(nominal.index)

def monthly_to_annual(df: pd.DataFrame) -> pd.DataFrame:
    compound = lambda x: (1.0 + x).prod() - 1.0
    annual = df.resample("YE").apply(compound)
    annual.index = annual.index.year
    return annual

def cumulative_growth(monthly: pd.DataFrame) -> pd.DataFrame:
    return (1.0 + monthly).cumprod()

def drawdown_from_cum(cum: pd.DataFrame) -> pd.DataFrame:
    running_max = cum.cummax()
    return (cum / running_max) - 1.0

def simulate_contributions(monthly: pd.Series, start_value: float, annual_contrib: float) -> pd.Series:
    mv = start_value; out = []; m_contrib = annual_contrib/12.0
    for r in monthly:
        mv += m_contrib; mv *= (1.0 + r); out.append(mv)
    return pd.Series(out, index=monthly.index)

def simulate_withdrawals(monthly: pd.Series, start_value: float, annual_withdraw_pct: float, mode: str="initial") -> pd.Series:
    mv = start_value; out = []
    for _, r in monthly.items():
        w = (start_value if mode=="initial" else mv) * annual_withdraw_pct / 12.0
        mv = max(0.0, mv - w); mv *= (1.0 + r); out.append(mv)
    return pd.Series(out, index=monthly.index)

def title_from_weights(weights: Dict[str, float]) -> str:
    return " | ".join([f"{k} {v:.0f}%" for k,v in weights.items()])

def set_percent_axis(ax):
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

def fmt_currency_func(currency: str):
    sym_map = {
        "USD": "$", "GBP": "£", "EUR": "€",
        "AUD": "A$", "CAD": "C$", "NZD": "NZ$", "JPY": "¥"
    }
    sym = sym_map.get(currency.upper(), "")
    if sym:
        return FuncFormatter(lambda x, pos: f"{sym}{x:,.0f}")
    else:
        # Fallback: append label
        label = currency if currency else ""
        return FuncFormatter(lambda x, pos: f"{x:,.0f} {label}".strip())

def set_value_axis_currency(ax, currency: str):
    ax.yaxis.set_major_formatter(fmt_currency_func(currency))

def chart_annual(annual: pd.DataFrame, out: str, dpi: int, svg: bool, title: str):
    fig, ax = plt.subplots(figsize=(11,5.8))
    years = annual.index.astype(int); col = annual.columns[0]
    ax.bar(years, annual[col].values, width=0.7, linewidth=0.6, edgecolor=HOUSE_GREY)
    mean_val = annual[col].mean()
    ax.axhline(mean_val, linestyle="--", linewidth=1, alpha=0.7, color=HOUSE_GREY,
               label=f"Average: {mean_val*100:.1f}%")
    set_percent_axis(ax); _legend_if_any(ax, loc="upper left")
    ax.set_xlabel("Year"); ax.set_ylabel("Annual return")
    ax.set_title(f"Annual Returns — {title}", loc="center")
    fig.tight_layout(); fig.savefig(out, dpi=dpi, bbox_inches="tight")
    if svg: fig.savefig(out.replace(".png",".svg"), bbox_inches="tight")
    plt.close(fig)

def chart_cum_currency(cum: pd.DataFrame, out: str, dpi: int, svg: bool, title: str, currency: str):
    # Layperson-friendly: no log; show in currency terms with commas/symbols.
    fig, ax = plt.subplots(figsize=(11,5.8))
    for col in cum.columns:
        ax.plot(cum.index, cum[col], linewidth=1.6)
    set_value_axis_currency(ax, currency)
    ax.set_xlabel("Date"); ax.set_ylabel(f"Value of 1 ({currency})")
    ax.set_title(f"Long-Term Returns (Cumulative) — {title}", loc="center")
    _legend_if_any(ax, ncol=min(4,len(cum.columns)))
    fig.tight_layout(); fig.savefig(out, dpi=dpi, bbox_inches="tight")
    if svg: fig.savefig(out.replace(".png",".svg"), bbox_inches="tight")
    plt.close(fig)

def chart_dd(dd: pd.DataFrame, out: str, dpi: int, svg: bool, title: str):
    fig, ax = plt.subplots(figsize=(11,5.3))
    for col in dd.columns:
        ax.plot(dd.index, dd[col], linewidth=1.2)
        ax.fill_between(dd.index, np.minimum(dd[col], 0), 0, alpha=0.18, color="#1f77b4")
    set_percent_axis(ax)
    ax.set_xlabel("Date"); ax.set_ylabel("Drawdown")
    ax.set_title(f"Drawdowns — {title}", loc="center")
    fig.tight_layout(); fig.savefig(out, dpi=dpi, bbox_inches="tight")
    if svg: fig.savefig(out.replace(".png",".svg"), bbox_inches="tight")
    plt.close(fig)

def chart_contrib(monthly: pd.DataFrame, start: float, contrib: float, out: str, dpi: int, svg: bool, title: str, currency: str):
    fig, ax = plt.subplots(figsize=(11,5.8))
    for col in monthly.columns:
        mv = simulate_contributions(monthly[col].dropna(), start, contrib)
        ax.plot(mv.index, mv.values, linewidth=1.6)
    set_value_axis_currency(ax, currency)
    _legend_if_any(ax, ncol=min(4,len(monthly.columns)))
    ax.set_xlabel("Date"); ax.set_ylabel(f"Portfolio value ({currency})")
    ax.set_title(f"Contributions Simulation — {title}", loc="center")
    fig.tight_layout(); fig.savefig(out, dpi=dpi, bbox_inches="tight")
    if svg: fig.savefig(out.replace(".png",".svg"), bbox_inches="tight")
    plt.close(fig)

def chart_withdraw(monthly: pd.DataFrame, start: float, pct: float, mode: str, out: str, dpi: int, svg: bool, title: str, currency: str):
    fig, ax = plt.subplots(figsize=(11,5.8))
    for col in monthly.columns:
        mv = simulate_withdrawals(monthly[col].dropna(), start, pct, mode)
        ax.plot(mv.index, mv.values, linewidth=1.6)
    set_value_axis_currency(ax, currency)
    _legend_if_any(ax, ncol=min(4,len(monthly.columns)))
    ax.set_xlabel("Date"); ax.set_ylabel(f"Portfolio value ({currency})")
    wtitle = f"{pct*100:.1f}% {mode}"
    ax.set_title(f"Withdrawals Simulation ({wtitle}) — {title}", loc="center")
    fig.tight_layout(); fig.savefig(out, dpi=dpi, bbox_inches="tight")
    if svg: fig.savefig(out.replace(".png",".svg"), bbox_inches="tight")
    plt.close(fig)

def chart_heatmap(annual: pd.DataFrame, out: str, dpi: int, svg: bool, title: str):
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.ticker import FuncFormatter

    s = annual.iloc[:, 0].dropna()
    years = s.index.astype(int).tolist()
    n = len(years)

    # build matrix of annualised returns (start year × horizon)
    mat = np.full((n, n), np.nan)
    growth = (1.0 + s).cumprod()
    for i in range(n):
        for h in range(1, n - i + 1):
            g_start = 1.0 if i == 0 else growth.iloc[i - 1]
            g_end = growth.iloc[i + h - 1]
            total = g_end / g_start - 1.0
            ann = (1.0 + total) ** (1.0 / h) - 1.0
            mat[i, h - 1] = ann

    # ---- robust, symmetric color limits
    finite = np.isfinite(mat)
    if not finite.any():
        vlim = 0.10
    else:
        p98 = float(np.nanpercentile(np.abs(mat[finite]), 98))
        vlim = max(0.05, min(0.60, p98))  # keep in a sensible window
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=vlim)

    fig, ax = plt.subplots(figsize=(11.8, 6.8))
    im = ax.imshow(mat, aspect="auto", origin="upper",
                   cmap="RdYlGn", norm=norm, interpolation="nearest")

    # tidy ticks
    step_y = max(1, n // 25)
    y_ticks = list(range(0, n, step_y))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([years[i] for i in y_ticks])

    step_x = max(1, n // 30)
    x_ticks = list(range(0, n, step_x))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(i + 1) for i in x_ticks])

    ax.set_xlabel("Holding period (years)")
    ax.set_ylabel("Start year")

    # dynamic tick step for the colorbar
    tick_step = 0.10 if vlim > 0.30 else 0.05
    ticks = np.arange(-vlim, vlim + 1e-9, tick_step)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_ticks(ticks)
    cbar.set_label("Annualised return")
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f}%"))

    # helpful end labels
    cbar.ax.text(0.5, 1.02, f"Best ≈ {vlim*100:.0f}%", transform=cbar.ax.transAxes,
                 ha="center", va="bottom", fontsize=9, color=HOUSE_GREY)
    cbar.ax.text(0.5, -0.04, f"Worst ≈ -{vlim*100:.0f}%", transform=cbar.ax.transAxes,
                 ha="center", va="top", fontsize=9, color=HOUSE_GREY)

    ax.set_title(f"Holding-Period Heat Map — {title}", loc="center")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    if svg:
        fig.savefig(out.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)


def export_excel(path: str, monthly: pd.DataFrame, annual: pd.DataFrame, cum: pd.DataFrame, dd: pd.DataFrame):
    try:
        with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
            monthly.to_excel(xw, sheet_name="Monthly"); annual.to_excel(xw, sheet_name="Annual")
            cum.to_excel(xw, sheet_name="Cumulative"); dd.to_excel(xw, sheet_name="Drawdowns")
    except ModuleNotFoundError:
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            monthly.to_excel(xw, sheet_name="Monthly"); annual.to_excel(xw, sheet_name="Annual")
            cum.to_excel(xw, sheet_name="Cumulative"); dd.to_excel(xw, sheet_name="Drawdowns")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FactorGraphs GUI")
        self.geometry("1100x820")
        self.file_path=tk.StringVar(); self.sheet=tk.StringVar()
        self.is_percent=tk.BooleanVar(value=True); self.rf_col=tk.StringVar()
        self.cpi_col=tk.StringVar(); self.want_real=tk.BooleanVar(value=False)
        self.begin_year=tk.StringVar(); self.end_year=tk.StringVar(); self.currency=tk.StringVar(value="USD")
        self.out_dir=tk.StringVar(); self.out_prefix=tk.StringVar(value="portfolio")
        self.dpi=tk.IntVar(value=170); self.save_svg=tk.BooleanVar(value=False); self.log_cum=tk.BooleanVar(value=False)
        self.start_val=tk.DoubleVar(value=10000.0); self.annual_contrib=tk.DoubleVar(value=6000.0)
        self.withdraw_pct=tk.DoubleVar(value=0.04); self.withdraw_mode=tk.StringVar(value="initial")
        self.columns: List[str] = []; self.include_vars: Dict[str, tk.BooleanVar] = {}; self.weight_vars: Dict[str, tk.DoubleVar] = {}
        self.currency_display = tk.StringVar(value="Currency: USD")
        self._build_ui()
        self.currency.trace_add("write", lambda *_: self.currency_display.set(f"Currency: {self.currency.get().strip() or 'USD'}"))
    def _build_ui(self):
        nb=ttk.Notebook(self); nb.pack(fill="both", expand=True, padx=8, pady=8)
        self.tab_data=ttk.Frame(nb); self.tab_port=ttk.Frame(nb); self.tab_outputs=ttk.Frame(nb)
        nb.add(self.tab_data,text="Data"); nb.add(self.tab_port,text="Portfolio"); nb.add(self.tab_outputs,text="Outputs")
        frm=ttk.Frame(self.tab_data,padding=8); frm.pack(fill="both", expand=True); row=0
        ttk.Label(frm,text="Input file (CSV/XLSX):").grid(row=row,column=0,sticky="w")
        ttk.Entry(frm,textvariable=self.file_path,width=65).grid(row=row,column=1,sticky="we",padx=4)
        ttk.Button(frm,text="Browse...",command=self.browse_file).grid(row=row,column=2,padx=4); row+=1
        ttk.Label(frm,text="Sheet (Excel, optional):").grid(row=row,column=0,sticky="w")
        ttk.Entry(frm,textvariable=self.sheet,width=20).grid(row=row,column=1,sticky="w",padx=4)
        ttk.Label(frm,text="(If CSV, leave blank)").grid(row=row,column=2,sticky="w"); row+=1
        ttk.Checkbutton(frm,text="Returns are in percent",variable=self.is_percent).grid(row=row,column=1,sticky="w"); row+=1
        ttk.Label(frm,text="Risk-free column (optional):").grid(row=row,column=0,sticky="w")
        ttk.Entry(frm,textvariable=self.rf_col,width=20).grid(row=row,column=1,sticky="w",padx=4); row+=1
        ttk.Label(frm,text="CPI index level column (optional):").grid(row=row,column=0,sticky="w")
        ttk.Entry(frm,textvariable=self.cpi_col,width=20).grid(row=row,column=1,sticky="w",padx=4)
        ttk.Checkbutton(frm,text="Compute REAL returns",variable=self.want_real).grid(row=row,column=2,sticky="w"); row+=1
        ttk.Label(frm,text="Begin year (optional):").grid(row=row,column=0,sticky="w")
        ttk.Entry(frm,textvariable=self.begin_year,width=10).grid(row=row,column=1,sticky="w",padx=4); row+=1
        ttk.Label(frm,text="End year (optional):").grid(row=row,column=0,sticky="w")
        ttk.Entry(frm,textvariable=self.end_year,width=10).grid(row=row,column=1,sticky="w",padx=4); row+=1
        ttk.Label(frm,text="Currency label:").grid(row=row,column=0,sticky="w")
        ttk.Entry(frm,textvariable=self.currency,width=10).grid(row=row,column=1,sticky="w",padx=4); row+=1
        ttk.Button(frm,text="Load Columns",command=self.load_columns).grid(row=row,column=0,pady=8,sticky="w")
        pfrm=ttk.Frame(self.tab_port,padding=8); pfrm.pack(fill="both",expand=True)
        ctrl=ttk.Frame(pfrm); ctrl.pack(fill="x",pady=(0,6))
        ttk.Button(ctrl,text="Select All",command=self.select_all).pack(side="left",padx=2)
        ttk.Button(ctrl,text="Clear All",command=self.clear_all).pack(side="left",padx=2)
        ttk.Button(ctrl,text="Equal Weights",command=self.equal_weights).pack(side="left",padx=2)
        self.cols_canvas=tk.Canvas(pfrm,borderwidth=0,highlightthickness=0)
        self.cols_frame=ttk.Frame(self.cols_canvas); vsb=ttk.Scrollbar(pfrm,orient="vertical",command=self.cols_canvas.yview)
        self.cols_canvas.configure(yscrollcommand=vsb.set); vsb.pack(side="right",fill="y"); self.cols_canvas.pack(side="left",fill="both",expand=True)
        self.cols_canvas.create_window((0,0),window=self.cols_frame,anchor="nw")
        self.cols_frame.bind("<Configure>",lambda e:self.cols_canvas.configure(scrollregion=self.cols_canvas.bbox("all")))
        cw=ttk.LabelFrame(pfrm,text="Contributions & Withdrawals",padding=8); cw.pack(fill="x",pady=(8,0))
        ttk.Label(cw,textvariable=self.currency_display,foreground=HOUSE_GREY).grid(row=0,column=0,sticky="w")
        ttk.Label(cw,text="Start value").grid(row=0,column=1,sticky="w")
        ttk.Entry(cw,textvariable=self.start_val,width=12).grid(row=0,column=2,sticky="w",padx=4)
        ttk.Label(cw,text="Annual contribution").grid(row=0,column=3,sticky="w")
        ttk.Entry(cw,textvariable=self.annual_contrib,width=12).grid(row=0,column=4,sticky="w",padx=4)
        ttk.Label(cw,text="Withdraw % (decimal)").grid(row=0,column=5,sticky="w")
        ttk.Entry(cw,textvariable=self.withdraw_pct,width=10).grid(row=0,column=6,sticky="w",padx=4)
        ttk.Label(cw,text="Mode").grid(row=0,column=7,sticky="w")
        ttk.Combobox(cw,textvariable=self.withdraw_mode,values=["initial","current"],width=8,state="readonly").grid(row=0,column=8,sticky="w")
        instructions = (
            "Start value: your initial portfolio balance in the chosen currency.\n"
            "Annual contribution: spread evenly across months and added at the START of each month before returns are applied.\n"
            "Withdraw % (decimal): e.g., 0.04 for a 4% annual withdrawal.\n"
            "Mode:\n"
            "  • initial → same $ amount each month (initial balance × rate / 12).\n"
            "  • current → same % of CURRENT balance each month (rate / 12)."
        )
        ttk.Label(cw, text=instructions, justify="left", wraplength=950).grid(row=1, column=0, columnspan=9, sticky="w", pady=(8,0))
        ofrm=ttk.Frame(self.tab_outputs,padding=8); ofrm.pack(fill="both",expand=True)
        ttk.Label(ofrm,text="Output folder:").grid(row=0,column=0,sticky="w")
        ttk.Entry(ofrm,textvariable=self.out_dir,width=60).grid(row=0,column=1,sticky="we",padx=4)
        ttk.Button(ofrm,text="Browse...",command=self.browse_outdir).grid(row=0,column=2,padx=4)
        ttk.Label(ofrm,text="File prefix:").grid(row=1,column=0,sticky="w")
        ttk.Entry(ofrm,textvariable=self.out_prefix,width=20).grid(row=1,column=1,sticky="w",padx=4)
        ttk.Label(ofrm,text="Chart DPI:").grid(row=2,column=0,sticky="w")
        ttk.Entry(ofrm,textvariable=self.dpi,width=8).grid(row=2,column=1,sticky="w",padx=4)
        ttk.Checkbutton(ofrm,text="Save SVG",variable=self.save_svg).grid(row=2,column=2,sticky="w")
        ttk.Checkbutton(ofrm,text="Log scale for cumulative",variable=self.log_cum).grid(row=2,column=3,sticky="w")
        ttk.Button(ofrm,text="Generate",command=self.generate).grid(row=3,column=0,pady=10,sticky="w")
    def browse_file(self):
        fp=filedialog.askopenfilename(title="Select data file",filetypes=[("Data files","*.csv *.xlsx *.xls"),("All files","*.*")])
        if fp: self.file_path.set(fp)
    def browse_outdir(self):
        d=filedialog.askdirectory(title="Select output folder")
        if d: self.out_dir.set(d)
    def load_columns(self):
        try: df=self.peek_dataframe()
        except Exception as e: messagebox.showerror("Error",f"Failed to load file:\n{e}"); return
        self.columns=[c for c in df.columns]
        for w in self.cols_frame.winfo_children(): w.destroy()
        header=ttk.Frame(self.cols_frame); header.pack(fill="x")
        ttk.Label(header,text="Include",width=8).pack(side="left",padx=4)
        ttk.Label(header,text="Column",width=40).pack(side="left",padx=4)
        ttk.Label(header,text="Weight %",width=10).pack(side="left",padx=4)
        self.include_vars.clear(); self.weight_vars.clear()
        for col in self.columns:
            rowf=ttk.Frame(self.cols_frame); rowf.pack(fill="x",pady=1)
            inc=tk.BooleanVar(value=False); self.include_vars[col]=inc
            ttk.Checkbutton(rowf,variable=inc,width=8).pack(side="left",padx=4)
            ttk.Label(rowf,text=col,width=40,anchor="w").pack(side="left",padx=4)
            wv=tk.DoubleVar(value=0.0); self.weight_vars[col]=wv
            ttk.Entry(rowf,textvariable=wv,width=10).pack(side="left",padx=4)
        messagebox.showinfo("Columns loaded","Columns detected and ready to weight.")
    def select_all(self):
        for v in self.include_vars.values(): v.set(True)
    def clear_all(self):
        for v in self.include_vars.values(): v.set(False)
        for wv in self.weight_vars.values(): wv.set(0.0)
    def equal_weights(self):
        inc=[c for c,v in self.include_vars.items() if v.get()]
        if not inc: messagebox.showwarning("No selection","Select some columns first (Include)."); return
        eq=100.0/len(inc)
        for c in inc: self.weight_vars[c].set(eq)
    def peek_dataframe(self) -> pd.DataFrame:
        path=self.file_path.get().strip()
        if not path: raise ValueError("No input file selected.")
        sheet=self.sheet.get().strip() or None
        is_percent=bool(self.is_percent.get()); cpi_col=self.cpi_col.get().strip() or None
        raw=auto_load_table(path, sheet); date_col=detect_date_column(raw); raw=coerce_dates(raw, date_col)
        if is_percent:
            for c in raw.columns:
                if c==date_col: continue
                if cpi_col and c==cpi_col: continue
                if pd.api.types.is_numeric_dtype(raw[c]): raw[c]=raw[c]/100.0
        raw=raw.set_index(raw[date_col]).drop(columns=[date_col])
        b=self.begin_year.get().strip(); e=self.end_year.get().strip()
        begin=int(b) if b else None; end=int(e) if e else None
        return filter_years(raw, begin, end)
    def generate(self):
        try: df=self.peek_dataframe()
        except Exception as e: messagebox.showerror("Error",f"Failed to load/parse data:\n{e}"); return
        selected=[c for c,v in self.include_vars.items() if v.get()]
        if not selected: messagebox.showwarning("No selection","Select at least one column to include."); return
        weights={c: float(self.weight_vars[c].get()) for c in selected}
        total=sum(weights.values())
        if abs(total-100.0)>1e-6: messagebox.showerror("Weights error",f"Weights must sum to 100, got {total:.2f}."); return
        monthly=df[selected].copy()
        cpi_name=self.cpi_col.get().strip() or None
        if self.want_real.get() and cpi_name:
            if cpi_name not in df.columns: messagebox.showerror("CPI missing",f"CPI column '{cpi_name}' not found in data."); return
            cpi=df[cpi_name].dropna()
            for col in monthly.columns: monthly[col]=to_real_series(monthly[col], cpi)
        w_vec=np.array([weights[c] for c in monthly.columns], dtype=float)/100.0
        portfolio=(monthly*w_vec).sum(axis=1); out_df=pd.DataFrame({"Portfolio": portfolio})
        title=title_from_weights(weights); annual=monthly_to_annual(out_df); cum=(1.0+out_df).cumprod(); dd=drawdown_from_cum(cum)
        out_dir=self.out_dir.get().strip()
        if not out_dir: messagebox.showerror("Output folder","Please choose an output folder."); return
        os.makedirs(out_dir, exist_ok=True); prefix=self.out_prefix.get().strip() or "portfolio"
        dpi=int(self.dpi.get()); svg=bool(self.save_svg.get()); currency=self.currency.get().strip() or "USD"
        excel_path=os.path.join(out_dir, f"{prefix}.xlsx")
        try: export_excel(excel_path, out_df, annual, cum, dd)
        except Exception as e: messagebox.showwarning("Excel export skipped", f"Could not write Excel:\n{e}")
        base=os.path.join(out_dir, prefix)
        try:
            chart_annual(annual, base+"_annual_returns.png", dpi, svg, title)
            chart_cum_currency(cum, base+"_cumulative.png", dpi, svg, title, currency)
            chart_dd(dd, base+"_drawdowns.png", dpi, svg, title)
            chart_contrib(out_df, float(self.start_val.get()), float(self.annual_contrib.get()), base+"_contributions.png", dpi, svg, title, currency)
            chart_withdraw(out_df, float(self.start_val.get()), float(self.withdraw_pct.get()), self.withdraw_mode.get(), base+"_withdrawals.png", dpi, svg, title, currency)
            chart_heatmap(annual, base+"_heatmap.png", dpi, svg, title)
        except Exception as e:
            messagebox.showerror("Chart error", f"Failed to generate charts:\n{e}"); return
        messagebox.showinfo("Done", f"Outputs saved to:\n{out_dir}")

def main():
    app = App(); app.mainloop()

if __name__ == "__main__":
    main()
