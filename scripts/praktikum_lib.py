"""
praktikum_lib - společné pomůcky pro vypocet.py (fyzikální praktikum MFF UK).

Leží v kořeni složky praktika vedle vypocet.py a NEUPRAVUJE SE: co je specifické pro úlohu
(fyzika, tabulky, grafy úlohy), patří do vypocet.py. Pravidla: znalosti 01_python_a_data.md.

    import praktikum_lib as pl
    data = pl.nacti_data()                       # data.txt, při chybě srozumitelně skončí
    p = data.u('p', 'err_p')                     # ufloat s nejistotou a jménem 'p'
    k, f = data.tabulka('REZONATOR_VZDUCH').T    # sloupce tabulky jako numpy pole
    U, u_U = data.sloupec_meridla('VA', 0, 'U')  # hodnoty + nejistoty bodů z [PRESNOST]
    fit = pl.vazeny_fit(pl.primka, k, f, data.hodnota('err_f'), jmeno='fit (vzduch)')
    vystup = pl.Vystup('Rychlost šíření zvuku')
    vystup.vysledek('c', c, 'm/s')               # '(342 ± 1) m/s (relativní nejistota 0,3 %)'
    vystup.uloz()                                # zapíše vysledky_analyzy.txt
"""
import os
import re
import sys
from decimal import Decimal, InvalidOperation
from typing import NamedTuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import correlated_values, ufloat  # ufloat i pro vypocet.py: pl.ufloat(...)

DATA_FILE = 'data.txt'
OUTPUT_FILE = 'vysledky_analyzy.txt'
PROTOKOL_DIR = 'protokol'                       # grafy patří k LaTeX dokumentu

STYL_SOUBOR = 'styl_grafu.json'                 # v kořeni studenta, o složku výš než praktikum


# ==========================================
# STYL GRAFŮ (01, bod 3.7)
# ==========================================
# Každý student má vlastní styl grafů (paleta, písmo, značky, osy) odvozený ze seedu
# v ../styl_grafu.json: ve všech jeho praktikách stejný, u dvou studentů různý. Bez souboru
# platí klasický černobílý styl. Série se vždy liší i tvarem značky a stylem čáry, takže graf
# zůstane čitelný i při černobílém tisku.
#     ax.errorbar(x, y, yerr=u_y, **pl.body(0), label='vzduch (data)')
#     ax.plot(x_fit, fit(x_fit), **pl.cara(0), label='vzduch (vážený fit)')
#     pl.pouzij_styl(seed=7)        # jiný styl jen pro tohle praktikum (na přání studenta)

# Velikost písma a obrázku je pro tisk pevná (v LaTeXu width=0.85\textwidth), styl ji nemění.
_ZAKLAD_RC = {
    'font.size': 13, 'axes.labelsize': 14, 'legend.fontsize': 12,
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'axes.facecolor': 'white', 'figure.facecolor': 'white',
}

KLASICKY_STYL = {
    'seed': None,
    'barvy': ['black'] * 6,
    'znacky': ['o', 's', '^', 'D', 'v', 'p'],
    'dute': [False, True, False, True, False, True],
    'cary': ['-', '--', '-.', ':'],
    'pismo': 'DejaVu Sans', 'matematika': 'dejavusans',
    'ramecek': True, 'ticky': 'out', 'vedlejsi_ticky': False, 'mrizka': None,
    'velikost_znacky': 6.0, 'tloustka_cary': 1.5, 'capsize': 3.0, 'vyska': 4.4,
}

# (rodina písma, sada pro matematiku) - všechny jsou součástí matplotlibu a umí češtinu.
_PISMA = [('DejaVu Sans', 'dejavusans'), ('DejaVu Serif', 'dejavuserif'),
          ('STIXGeneral', 'stix')]


def _lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _srgb(c: float) -> float:
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


_RGB_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]])
_BILA = np.array([0.95047, 1.0, 1.08883])
# Simulace barvosleposti (Machado et al. 2009, plná protanopie a deuteranopie) v lineárním RGB.
_CVD = {
    'protanopie': np.array([[0.152286, 1.052583, -0.204868],
                            [0.114503, 0.786281, 0.099216],
                            [-0.003882, -0.048116, 1.051998]]),
    'deuteranopie': np.array([[0.367322, 0.860646, -0.227968],
                              [0.280085, 0.672501, 0.047413],
                              [-0.011820, 0.042940, 0.968881]]),
}


def _lab_z_lin(rgb_lin) -> np.ndarray:
    t = (_RGB_XYZ @ np.asarray(rgb_lin, dtype=float)) / _BILA
    f = np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def _lin_z_hex(barva: str) -> np.ndarray:
    return np.array([_lin(int(barva[i:i + 2], 16) / 255) for i in (1, 3, 5)])


def lab(barva: str, vada: str | None = None) -> np.ndarray:
    """CIELAB barvy '#rrggbb' (L* 0 = černá, 100 = bílá), případně tak, jak ji vidí člověk
    s protanopií/deuteranopií."""
    rgb = _lin_z_hex(barva)
    if vada:
        rgb = np.clip(_CVD[vada] @ rgb, 0, 1)
    return _lab_z_lin(rgb)


def _hex_z_lch(L: float, C: float, h: float) -> str | None:
    """CIELCh -> '#rrggbb', nebo None, když barva leží mimo sRGB."""
    a, b = C * np.cos(np.radians(h)), C * np.sin(np.radians(h))
    fy = (L + 16) / 116
    f = np.array([fy + a / 500, fy, fy - b / 200])
    t = np.where(f > 6 / 29, f ** 3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * _BILA
    rgb = np.linalg.solve(_RGB_XYZ, t)
    if rgb.min() < -1e-9 or rgb.max() > 1 + 1e-9:
        return None
    return '#' + ''.join(f'{round(255 * _srgb(min(max(c, 0.0), 1.0))):02x}' for c in rgb)


def kontrola_palety(barvy: list[str]) -> list[str]:
    """Vrátí seznam problémů palety (prázdný = paleta je v pořádku pro protokol):
    každá barva dost tmavá na bílém papíře i v černobílém tisku (L* <= 60, kontrast >= 3:1)
    a každé dvě barvy rozlišitelné normálním okem (ΔE >= 20) i při protanopii a
    deuteranopii (ΔE >= 9 mezi prvními čtyřmi, kde bývá většina grafů)."""
    problemy = []
    for b in barvy:
        if lab(b)[0] > 60:
            problemy.append(f'{b} je na bílém papíře příliš světlá')
    for i in range(len(barvy)):
        for j in range(i + 1, len(barvy)):
            if np.linalg.norm(lab(barvy[i]) - lab(barvy[j])) < 20:
                problemy.append(f'{barvy[i]} a {barvy[j]} jsou si příliš podobné')
            if j < 4:
                for vada in _CVD:
                    if np.linalg.norm(lab(barvy[i], vada) - lab(barvy[j], vada)) < 9:
                        problemy.append(f'{barvy[i]} a {barvy[j]} splývají při {vada[:-1]}i')
    return problemy


def _paleta(r, n: int = 6) -> list[str]:
    """Náhodná paleta n barev: odstíny rozložené zlatým úhlem od náhodného počátku, střídavě
    tmavší a světlejší (rozliší se i v odstínech šedi). Kandidát, který neprojde
    kontrola_palety, se zahodí a losuje se znovu."""
    for _ in range(500):
        h0, posun, tmava_prvni = r.uniform(0, 360), r.uniform(0, 1), r.random() < 0.5
        barvy = []
        for k in range(n):
            tmava = (k % 2 == 0) == tmava_prvni
            L = (30 + 10 * posun if tmava else 43 + 10 * posun) + r.uniform(-3, 3)
            h = (h0 + 137.508 * k + r.uniform(-10, 10)) % 360
            C = r.uniform(45, 75)
            while (barva := _hex_z_lch(L, C, h)) is None and C > 25:
                C -= 3
            if barva is None:
                break
            barvy.append(barva)
        if len(barvy) == n and not kontrola_palety(barvy):
            return barvy
    return ['#1b4f9c', '#c0392b', '#1e7b45', '#7d3c98', '#8a5a00', '#2c3e50']  # pojistka


def styl_grafu(seed: int) -> dict:
    """Styl grafů jednoznačně určený seedem (stejný seed = vždy stejný styl)."""
    import random
    r = random.Random(seed)
    pismo, matematika = r.choice(_PISMA)
    znacky = ['o', 's', '^', 'D', 'v', 'p']
    r.shuffle(znacky)
    plne_prvni = r.random() < 0.6
    cary = ['--', '-.', ':']
    r.shuffle(cary)
    ramecek = r.random() < 0.5
    return {
        'seed': seed,
        'barvy': _paleta(r),
        'znacky': znacky,
        'dute': [(i % 2 == 1) == plne_prvni for i in range(len(znacky))],
        'cary': ['-'] + cary,
        'pismo': pismo, 'matematika': matematika,
        'ramecek': ramecek,
        'ticky': r.choice(['in', 'out']),
        'vedlejsi_ticky': r.random() < 0.5,
        'mrizka': r.choice([None, None, ':', '--']),
        'velikost_znacky': round(r.uniform(5.0, 7.0), 1),
        'tloustka_cary': round(r.uniform(1.2, 2.0), 1),
        'capsize': float(r.choice([0, 2, 3, 4])),
        'vyska': round(r.uniform(4.2, 4.6), 2),
    }


def _rc(s: dict) -> dict:
    from cycler import cycler
    vnitrni_ticky_dokola = s['ramecek'] and s['ticky'] == 'in'
    return {
        **_ZAKLAD_RC,
        'figure.figsize': (6.4, s['vyska']),
        'font.family': s['pismo'], 'mathtext.fontset': s['matematika'],
        'axes.prop_cycle': cycler(color=s['barvy']),
        'axes.spines.top': s['ramecek'], 'axes.spines.right': s['ramecek'],
        'xtick.direction': s['ticky'], 'ytick.direction': s['ticky'],
        'xtick.top': vnitrni_ticky_dokola, 'ytick.right': vnitrni_ticky_dokola,
        'xtick.minor.visible': s['vedlejsi_ticky'], 'ytick.minor.visible': s['vedlejsi_ticky'],
        'axes.grid': bool(s['mrizka']), 'grid.linestyle': s['mrizka'] or ':',
        'grid.color': '0.75', 'grid.linewidth': 0.6,
        'lines.markersize': s['velikost_znacky'], 'lines.linewidth': s['tloustka_cary'],
        'errorbar.capsize': s['capsize'],
    }


def _nacti_styl_soubor() -> dict:
    """Obsah ../styl_grafu.json ({"seed": 123, ...případné úpravy}), bez souboru {}."""
    import json
    cesta = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, STYL_SOUBOR)
    try:
        with open(cesta, encoding='utf-8') as f:
            obsah = json.load(f)
        return obsah if isinstance(obsah, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        print(f"UPOZORNĚNÍ: {STYL_SOUBOR} nejde přečíst ({e}), grafy budou černobílé.",
              file=sys.stderr)
        return {}


STYL: dict = dict(KLASICKY_STYL)


def pouzij_styl(seed: int | None = None, **zmeny) -> dict:
    """Nastaví styl grafů: seed=None -> styl studenta z ../styl_grafu.json (bez souboru
    klasický černobílý), jiný seed -> jiný náhodný styl. `zmeny` přepíší jednotlivé položky,
    např. barvy=['#1b4f9c', '#c0392b'] nebo mrizka=':'. Volá se samo při importu."""
    global STYL
    soubor = _nacti_styl_soubor() if seed is None else {}
    if seed is None:
        seed = soubor.get('seed')
    STYL = styl_grafu(int(seed)) if seed is not None else dict(KLASICKY_STYL)
    for klic, hodnota in {**soubor, **zmeny}.items():
        if klic in STYL and klic != 'seed':
            STYL[klic] = hodnota
    plt.rcParams.update(_rc(STYL))
    return STYL


def barva(i: int = 0) -> str:
    return STYL['barvy'][i % len(STYL['barvy'])]


def body(i: int = 0) -> dict:
    """Argumenty pro ax.errorbar naměřených bodů i-té série (barva, tvar a výplň značky)."""
    znacka = STYL['znacky'][i % len(STYL['znacky'])]
    duta = STYL['dute'][i % len(STYL['dute'])]
    return {'marker': znacka, 'linestyle': 'none', 'color': barva(i), 'mec': barva(i),
            'mfc': 'white' if duta else barva(i), 'ms': STYL['velikost_znacky'],
            'capsize': STYL['capsize'], 'elinewidth': 1.0}


def cara(i: int = 0) -> dict:
    """Argumenty pro ax.plot křivky (fit, teorie) i-té série - barva jako body, vlastní styl čáry."""
    return {'color': barva(i), 'linestyle': STYL['cary'][i % len(STYL['cary'])],
            'linewidth': STYL['tloustka_cary']}


pouzij_styl()

# Přepínač studenta „Rezidua a χ²“ (stránka Můj agent, ../styl_grafu.json, výchozí vypnuto):
# vypnuto = vysledky_analyzy.txt neuvádí χ²_red a vypocet.py nekreslí rezidua (01, bod 3.6).
# Přeškálování nejistot fitu sqrt(χ²_red) platí v obou případech, jen se o něm nepíše.
REZIDUA = bool(_nacti_styl_soubor().get('rezidua', False))

# Pojistky analýzy („harness“, ../styl_grafu.json klíč "harness", zapíná web podle
# WEB_HARNESS_UZIVATELE): validace data.txt, sjednocení digitu ve sloupci, sekce VAROVÁNÍ
# ANALÝZY, zákaz nulové nejistoty ve výpisu. Bez klíče se knihovna chová jako dřív.
HARNESS = bool(_nacti_styl_soubor().get('harness', False))


def chyba(zprava: str):
    """Ukončí skript srozumitelnou chybou místo tracebacku (01, bod 3.4)."""
    sys.exit(f"CHYBA: {zprava}")


# ==========================================
# FORMÁTOVÁNÍ ČÍSEL (01, body 3.5 a 3.8)
# ==========================================
def cislo(x: float, format_: str = '.4g') -> str:
    """Číslo bez nejistoty s desetinnou čárkou: cislo(1.5, '.1f') -> '1,5'."""
    return format(x, format_).replace('.', ',')


def fmt(x, jednotka: str = '') -> str:
    """Výsledek s nejistotou v zápisu protokolu.

    Nejistota vždy na 1 platnou cifru, hodnota na stejné desetinné místo, desetinná čárka
    a ±, velké/malé řády vytknuté: fmt(ufloat(1.0921e11, 2.1e9), 'Pa')
    -> '(1,09 ± 0,02)·10^11 Pa'. Nikdy kompaktní tvar 1,09(2).
    """
    n, s = x.n, x.s
    if HARNESS and (not s or not np.isfinite(s) or s <= 1e-12 * abs(n)):
        chyba(f"výsledek {f'{cislo(n)} {jednotka}'.strip()} má nulovou nebo nekonečnou nejistotu - "
              f"ve vstupech chybí nejistota (δ = 0, natvrdo nula, chybějící err_*) a „± 0“ by šlo do "
              f"protokolu. Doplň nejistotu vstupu do data.txt; přesné číslo bez nejistoty vypiš "
              f"jako číslo (pl.cislo, vystup.diskuze(popis, float(x))).")
    if not s or not np.isfinite(s):
        return f"{cislo(n)} {jednotka}".strip()
    # Řád se vytkne u velkých/malých čísel a vždy, když by nejistota měla víc než jednu
    # číslici před čárkou ((3579 ± 16) -> (3,58 ± 0,02)·10^3).
    rad = int(np.floor(np.log10(abs(n)))) if n and (not 1e-3 <= abs(n) < 1e4 or s >= 9.5) else 0
    n, s = n / 10**rad, s / 10**rad
    s = float(f"{s:.0e}")                          # 1 platná cifra (0,016 -> 0,02; 0,096 -> 0,1)
    mista = max(0, -int(np.floor(np.log10(s))))
    txt = f"({n:.{mista}f} ± {s:.{mista}f})".replace('.', ',')
    if rad:
        txt += f"·10^{rad}"
    elif not jednotka:
        txt = txt[1:-1]                            # bezrozměrná bez řádu: bez závorky
    return f"{txt} {jednotka}".strip()


def fmt_tex(x, jednotka: str = '') -> str:
    """Totéž co fmt, ale jako matematický zápis pro popisky/legendu grafu (mathtext) i LaTeX:
    fmt_tex(ufloat(213.74, 0.62), 'Hz') -> '$(213{,}7 \\pm 0{,}6)$ Hz'. Čárka v {,} nedělá mezeru."""
    text = fmt(x, 'J')[:-2] if jednotka else fmt(x)   # se jednotkou se závorkou, bez ní jako fmt
    text = re.sub(r'·10\^(-?\d+)', r' \\cdot 10^{\1}', text)
    text = text.replace(',', '{,}').replace(' ± ', ' \\pm ')
    return f"${text}$ {jednotka}".strip()


def relativni(x) -> str:
    """Relativní nejistota v procentech: '0,3 %'."""
    return f"{cislo(100 * x.s / abs(x.n), '.2g')} %" if x.n else "– %"


class Carka(ticker.ScalarFormatter):
    """Čísla na osách s desetinnou čárkou (locale cs_CZ na serveru není)."""
    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace('.', ',')


def carky_na_osach(ax) -> None:
    ax.xaxis.set_major_formatter(Carka())
    ax.yaxis.set_major_formatter(Carka())


def legenda_nad_grafem(ax, ncol: int | None = None) -> None:
    """Legenda NAD osami, mimo data - nikdy nepřekryje body ani křivku (01, bod 3.7).
    Položky téže série se sloučí do jedné podle popisku bez koncové závorky:
    'vzduch (data)' + 'vzduch (vážený fit)' -> jedna položka 'vzduch' (značka i čára).
    Sloupců je tolik co sérií (krátké popisky), u dlouhých popisků (rovnice) jeden."""
    from matplotlib.legend_handler import HandlerTuple
    skupiny: dict[str, list] = {}
    for h, l in zip(*ax.get_legend_handles_labels()):
        zaklad = l.rsplit(' (', 1)[0] if l.endswith(')') and ' (' in l else l
        skupiny.setdefault(zaklad, []).append(h)
    if ncol is None:
        ncol = len(skupiny) if sum(len(k) for k in skupiny) <= 60 else 1
    ax.legend([tuple(h) for h in skupiny.values()], list(skupiny), ncol=ncol,
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)}, handlelength=3.2,
              loc='lower center', bbox_to_anchor=(0.5, 1.01), frameon=False,
              columnspacing=1.2, borderaxespad=0.0)


def cesta_grafu(nazev: str) -> str:
    """'graf_x.pdf' -> 'protokol/graf_x.pdf' (složku případně založí)."""
    os.makedirs(PROTOKOL_DIR, exist_ok=True)
    return os.path.join(PROTOKOL_DIR, nazev)


# ==========================================
# NAČTENÍ data.txt (formát viz vzory/vzor_data.txt)
# ==========================================
class Data:
    """Obsah data.txt: hlavička (# KLÍČ: hodnota), skaláry/seznamy (klic = ...) a tabulky
    (řádky čísel oddělených čárkou v sekci [NAZEV]). Chybějící hodnota nebo '?' ukončí skript
    srozumitelnou chybou, nikdy se tiše nedosadí."""

    def __init__(self, hlavicka: dict, hodnoty: dict, tabulky: dict, cesta: str,
                 prepnuti: dict | None = None, povolene_radky: set | None = None):
        self.hlavicka, self.hodnoty, self.tabulky, self.cesta = hlavicka, hodnoty, tabulky, cesta
        # Jen pro harness (_zkontroluj_tabulku): komentáře o rozsahu uprostřed tabulky
        # {sekce: [(řádek, text)]} a řádky s únikem `# harness: povoleno – důvod`.
        self.prepnuti, self.povolene_radky = prepnuti or {}, povolene_radky or set()
        self._zkontrolovane: set[str] = set()

    def _text(self, klic: str):
        text = self.hodnoty.get(klic, '').strip()
        return None if text in ('', '?') else text

    def hodnota(self, klic: str, default: float | None = None) -> float:
        """Číselná hodnota `klic = 1.5`; default jen pro nepovinné hodnoty."""
        text = self._text(klic)
        if text is None:
            if default is not None:
                return default
            chyba(f"v {self.cesta} chybí nebo není vyplněná hodnota '{klic}'.")
        try:
            return float(text)
        except ValueError:
            chyba(f"hodnota '{klic}' v {self.cesta} není číslo: {text!r} (desetinná tečka).")

    def u(self, klic: str, err_klic: str, jmeno: str | None = None):
        """ufloat(hodnota, nejistota, jméno) - jméno (default = klíč) použije rozpočet nejistot."""
        nejistota = self.hodnota(err_klic)
        if nejistota <= 0:
            chyba(f"nejistota '{err_klic}' musí být kladná (je {nejistota}).")
        return ufloat(self.hodnota(klic), nejistota, jmeno or klic)

    def seznam(self, klic: str) -> np.ndarray:
        """Seznam `klic = 1.2, 3.4, 5.6` jako numpy pole."""
        text = self._text(klic)
        if text is None:
            chyba(f"v {self.cesta} chybí nebo není vyplněný seznam '{klic}'.")
        try:
            return np.array([float(x) for x in text.split(',') if x.strip()])
        except ValueError:
            chyba(f"seznam '{klic}' v {self.cesta} obsahuje nečíselnou položku: {text!r}.")

    def tabulka(self, sekce: str, sloupcu: int | None = None) -> np.ndarray:
        """Tabulka ze sekce [sekce] jako 2D pole (řádek = měření); `.T` dá sloupce."""
        radky = self.tabulky.get(sekce)
        if not radky:
            chyba(f"tabulka [{sekce}] v {self.cesta} chybí nebo je prázdná - doplň naměřená data.")
        cisla = []
        for cislo_radku, bunky in radky:
            if sloupcu is not None and len(bunky) != sloupcu:
                chyba(f"{self.cesta}, řádek {cislo_radku}: tabulka [{sekce}] má mít {sloupcu} "
                      f"sloupců, řádek jich má {len(bunky)}.")
            try:
                cisla.append([float(b) for b in bunky])
            except ValueError:
                chyba(f"{self.cesta}, řádek {cislo_radku}: v tabulce [{sekce}] je nečíselná "
                      f"nebo nevyplněná hodnota: {', '.join(bunky)}")
        if len({len(r) for r in cisla}) > 1:
            chyba(f"tabulka [{sekce}] v {self.cesta} má řádky s různým počtem sloupců.")
        if HARNESS and sekce not in self._zkontrolovane:
            self._zkontroluj_tabulku(sekce)
            self._zkontrolovane.add(sekce)
        return np.array(cisla)

    def _zkontroluj_tabulku(self, sekce: str) -> None:
        """Harness: jisté chyby zápisu tabulky ukončí skript s číslem řádku (web-1/23, 28).
        - komentář o rozsahu („# range se přepnul na 100 mV“) mezi řádky tabulky: přepnutí
          rozsahu patří do sloupce rozsahu u každého řádku a hodnoty zůstávají v jednotce
          hlavičky,
        - skok ~1000× mezi sousedními řádky sloupce (V -> mV uprostřed série).
        Únik pro skutečně takové hodnoty: `# harness: povoleno – důvod` na řádku."""
        for radek, text in self.prepnuti.get(sekce, []):
            chyba(f"{self.cesta}, řádek {radek}: komentář '{text}' uprostřed tabulky [{sekce}] - "
                  f"přepnutí rozsahu se zapisuje do sloupce rozsahu u každého řádku (např. "
                  f"'rozsah_U / V'), hodnoty zůstávají v jednotce z hlavičky tabulky. Oprav data.txt.")
        radky = self.tabulky[sekce]
        for (_, predchozi), (cislo_radku, bunky) in zip(radky, radky[1:]):
            if cislo_radku in self.povolene_radky:
                continue
            for a, b in zip(predchozi, bunky):
                x, y = abs(float(a)), abs(float(b))
                if not x or not y or min(_platnych_cifer(a), _platnych_cifer(b)) < 2:
                    continue
                if 700 <= max(x, y) / min(x, y) <= 1430:
                    chyba(f"{self.cesta}, řádek {cislo_radku}: v tabulce [{sekce}] skočila hodnota "
                          f"{a} -> {b} asi 1000× - nejsou hodnoty od tohoto řádku v jiné jednotce "
                          f"(V/mV, A/mA)? Všechny hodnoty sloupce musí být v jednotce z hlavičky "
                          f"tabulky, rozsah patří do sloupce rozsahu. Je-li skok skutečný, připiš "
                          f"na řádek '# harness: povoleno – <důvod>'.")

    def presnost(self, velicina: str, rozsah: float | None = None) -> 'Presnost':
        """Přesnost přístroje ze sekce [PRESNOST]:
        `pres_U = 0.5, 2`       multimetr: % z hodnoty, digity,
        `pres_U = 0.015, 0.003%` stolní multimetr: % z hodnoty, % z rozsahu (znak % na konci),
        `pres_I = 1.5`          ručkové měřidlo: třída přesnosti (= % z rozsahu).
        `pres_I@10 = 2, 5` platí jen pro rozsah 10 a má přednost před `pres_I`."""
        klic = f'pres_{velicina}'
        if rozsah is not None:
            for k in self.hodnoty:
                if k.startswith(klic + '@'):
                    try:
                        r = float(k.split('@', 1)[1])
                    except ValueError:
                        chyba(f"rozsah v klíči '{k}' v {self.cesta} není číslo.")
                    if np.isclose(r, rozsah):
                        klic = k
                        break
        text = self._text(klic)
        if text is None:
            chyba(f"v {self.cesta} chybí nebo není vyplněná přesnost '{klic}' (sekce [PRESNOST]: "
                  f"multimetr '{klic} = % z hodnoty, digity' nebo '% z hodnoty, % z rozsahu%', "
                  f"ručkové měřidlo '{klic} = třída').")
        polozky = [x.strip() for x in text.split(',') if x.strip()]
        try:
            cisla = [float(x.rstrip('%')) for x in polozky]
        except ValueError:
            chyba(f"přesnost '{klic}' v {self.cesta} obsahuje nečíselnou položku: {text!r}.")
        if len(cisla) not in (1, 2) or polozky[0].endswith('%') or min(cisla) < 0 or max(cisla) == 0:
            chyba(f"přesnost '{klic}' v {self.cesta} má být 'třída' (ručkové měřidlo), "
                  f"'% z hodnoty, digity' nebo '% z hodnoty, % z rozsahu%' (multimetr), "
                  f"nezáporná a ne celá nulová: {text!r}.")
        if len(cisla) == 1:
            return Presnost(z_rozsahu=cisla[0])
        if polozky[1].endswith('%'):
            return Presnost(z_hodnoty=cisla[0], z_rozsahu=cisla[1])
        return Presnost(z_hodnoty=cisla[0], digity=cisla[1])

    def sloupec_meridla(self, sekce: str, sloupec: int, velicina: str, rozsah=None, oscil=None):
        """Sloupec tabulky a standardní nejistoty jeho bodů z [PRESNOST] (`nejistota_meridla`):
        `U, u_U = data.sloupec_meridla('VA_CHARAKTERISTIKA', 0, 'U')`. `rozsah` = číslo, nebo
        pole s rozsahem u každého řádku (sloupec tabulky); ručkové měřidlo ho potřebuje vždy,
        multimetr s % z rozsahu taky; multimetr s digity jen pro `pres_X@rozsah` a kontrolu
        opsaných nul. Neznámý rozsah při automatické volbě dá `nejmensi_rozsah`.
        `oscil` = polovina rozpětí kolísání údaje v jednotkách sloupce (číslo, nebo pole / sloupec
        tabulky `oscil_X` po řádcích; 0 = údaj nekolísal): nejistota se složí podle `s_kolisanim`."""
        pocet_sloupcu = self.tabulka(sekce).shape[1]
        if not 0 <= sloupec < pocet_sloupcu:
            chyba(f"tabulka [{sekce}] v {self.cesta} nemá sloupec {sloupec} (má {pocet_sloupcu}).")
        cisla_radku = [c for c, _ in self.tabulky[sekce]]
        zapisy = [bunky[sloupec] for _, bunky in self.tabulky[sekce]]
        rozsahy = [None] * len(zapisy) if rozsah is None else np.broadcast_to(rozsah, len(zapisy))
        if HARNESS:
            self._zkontroluj_rozsahy(sekce, velicina, cisla_radku, zapisy, rozsahy)
        # Harness: digit z nejpřesněji opsané hodnoty sloupce na stejném rozsahu ('0.6' vedle
        # '0.59' -> 0,01, ne 0,1; web-1/35, 39). Bez harnessu z každého zápisu zvlášť.
        digity_sloupce = _sjednocene_digity(zapisy, rozsahy) if HARNESS else [digit(z) for z in zapisy]
        hodnoty, nejistoty, digity, malo_mist, sjednoceno = [], [], set(), [], []
        for zapis, r, d in zip(zapisy, rozsahy, digity_sloupce):
            presnost = self.presnost(velicina, r)
            hodnoty.append(float(zapis))
            nejistoty.append(nejistota_meridla(zapis, presnost, r, digit_zapisu=d))
            if presnost.digity:
                digity.add((r, digit(zapis)))
                if d < digit(zapis):
                    sjednoceno.append(zapis)
                # multimetr ukazuje aspoň 1000 digitů rozsahu (3½ místa: 19.99 na rozsahu 20)
                if r is not None and d > r / 1000 * 1.0001:
                    malo_mist.append(f"{zapis} na rozsahu {cislo(r)}")
        hlas = _varuj if HARNESS else _upozorni
        if malo_mist:
            hlas(f"{velicina} v [{sekce}] má málo desetinných míst ({', '.join(malo_mist[:3])}"
                 f"{', …' if len(malo_mist) > 3 else ''}) - jsou hodnoty opsané z displeje "
                 f"i s nulami na konci? Digit multimetru se bere z posledního opsaného místa.")
        if HARNESS and sjednoceno:
            _varuj(f"{velicina} v [{sekce}]: {', '.join(sjednoceno[:3])}{', …' if len(sjednoceno) > 3 else ''} "
                   f"má méně desetinných míst než zbytek sloupce na stejném rozsahu - digit multimetru "
                   f"se vzal z nejpřesněji opsané hodnoty sloupce. Ověř se studentem, že displej "
                   f"ukazoval stejný počet míst (a v data.txt to oprav, např. 0.6 -> 0.60).")
        elif len({r for r, _ in digity}) < len(digity):
            _upozorni(f"{velicina} v [{sekce}] má na stejném rozsahu různý počet desetinných míst "
                      f"- ověř, že hodnoty jsou opsané z displeje i s nulami na konci (digit "
                      f"multimetru se bere z posledního opsaného místa), nebo že se měnil rozsah.")
        return np.array(hodnoty), s_kolisanim(np.array(nejistoty), oscil)

    def _zkontroluj_rozsahy(self, sekce, velicina, cisla_radku, zapisy, rozsahy) -> None:
        """Harness: hodnota nad deklarovaným rozsahem ukončí skript (jistě jiná jednotka nebo
        špatný rozsah; měřidla snesou nejvýš ~20 % přes rozsah). Hodnota pod tisícinou
        rozsahu je VAROVÁNÍ (napětí ve V pod hlavičkou mV, web-1/23)."""
        male = []
        for cislo_radku, zapis, r in zip(cisla_radku, zapisy, rozsahy):
            if r is None or cislo_radku in self.povolene_radky:
                continue
            x = abs(float(zapis))
            if x > 1.25 * r:
                chyba(f"{self.cesta}, řádek {cislo_radku}: {velicina} = {zapis} v tabulce [{sekce}] je "
                      f"nad použitým rozsahem {cislo(r, '.6g')} - hodnota v jiné jednotce než hlavička "
                      f"(V/mV), nebo špatně zapsaný rozsah? Oprav data.txt.")
            if 0 < x < r / 1000:
                male.append(f"{zapis} (řádek {cislo_radku})")
        if male:
            _varuj(f"{velicina} v [{sekce}]: {', '.join(male[:3])}{', …' if len(male) > 3 else ''} je pod "
                   f"tisícinou použitého rozsahu - nejsou hodnoty v jiné jednotce, než uvádí hlavička "
                   f"tabulky (V místo mV)? Ověř se studentem, data sám nepřepočítávej.")

    def u_meridla(self, klic: str, velicina: str, rozsah: float | None = None,
                  jmeno: str | None = None, oscil: float = 0.0):
        """ufloat z hodnoty `klic = 1.200` s nejistotou přístroje z [PRESNOST]
        (`nejistota_meridla`) a případně s kolísáním údaje `oscil` (`s_kolisanim`); jméno
        (default = klíč) použije rozpočet nejistot."""
        self.hodnota(klic)                              # srozumitelná chyba, chybí-li hodnota
        zapis = self._text(klic)
        presnost = self.presnost(velicina, rozsah)
        if HARNESS and rozsah is not None:
            if abs(float(zapis)) > 1.25 * rozsah:
                chyba(f"{klic} = {zapis} v {self.cesta} je nad použitým rozsahem {cislo(rozsah, '.6g')} "
                      f"- hodnota v jiné jednotce, nebo špatně zapsaný rozsah? Oprav data.txt.")
            if presnost.digity and digit(zapis) > rozsah / 1000 * 1.0001:
                _varuj(f"{klic} = {zapis} má málo desetinných míst na rozsahu {cislo(rozsah, '.6g')} - "
                       f"je opsaná z displeje i s nulami na konci (např. 1.00)? Digit multimetru se bere "
                       f"z posledního opsaného místa.")
        u = float(s_kolisanim(nejistota_meridla(zapis, presnost, rozsah), oscil))
        return ufloat(float(zapis), u, jmeno or klic)


def nacti_data(cesta: str = DATA_FILE) -> Data:
    """Přečte data.txt. Komentáře začínají '#'; data.txt se nikdy nepřepisuje."""
    if not os.path.exists(cesta):
        chyba(f"soubor '{cesta}' neexistuje.")
    with open(cesta, encoding='utf-8') as f:
        radky = f.read().splitlines()

    hlavicka, hodnoty, tabulky, sekce = {}, {}, {}, None
    prepnuti, povolene, cekajici = {}, set(), {}
    for cislo_radku, radek in enumerate(radky, start=1):
        if sekce is None and (m := re.match(r'^#\s*([^:#=]+?)\s*:\s*(.*)$', radek)):
            hlavicka[m.group(1).upper()] = m.group(2).strip()
        obsah = radek.split('#', 1)[0].strip()
        komentar = radek.split('#', 1)[1].strip() if '#' in radek else ''
        if _UNIK.search(radek):
            povolene.add(cislo_radku)
            komentar = ''
        # Komentář o rozsahu až za prvním řádkem tabulky (ne v hlavičce sekce) = přepnutí
        # rozsahu uprostřed série; platí, jen když po něm v sekci následují další data.
        o_rozsahu = sekce in tabulky and _KOMENTAR_ROZSAHU.search(komentar)
        if not obsah:
            if o_rozsahu:
                cekajici[sekce] = (cislo_radku, komentar)
            continue
        if m := re.match(r'^\[(.+)\]$', obsah):
            sekce = m.group(1).strip()
            continue
        if '=' in obsah:
            klic, hodnota = (s.strip() for s in obsah.split('=', 1))
            if klic in hodnoty:
                chyba(f"{cesta}, řádek {cislo_radku}: hodnota '{klic}' je v souboru dvakrát.")
            hodnoty[klic] = hodnota
        elif sekce is not None:
            if sekce in cekajici:
                prepnuti.setdefault(sekce, []).append(cekajici.pop(sekce))
            if o_rozsahu:
                prepnuti.setdefault(sekce, []).append((cislo_radku, komentar))
            bunky = [b.strip() for b in re.split(r'[,;\t]', obsah) if b.strip()]
            tabulky.setdefault(sekce, []).append((cislo_radku, bunky))
    return Data(hlavicka, hodnoty, tabulky, cesta, prepnuti, povolene)


# „# range se přepnul na 100 mV“, „# rozsah 2 V“ - slovo rozsah/range a číslo za ním.
_KOMENTAR_ROZSAHU = re.compile(r'\b(?:range|rozsah)\w*\b.*\d', re.IGNORECASE)


def _platnych_cifer(zapis: str) -> int:
    """Počet platných cifer zápisu: '0.0446' -> 3, '120' -> 3, '1.00' -> 3, '0' -> 0."""
    try:
        cifry = Decimal(zapis.strip()).as_tuple().digits
    except InvalidOperation:
        return 0
    return len(''.join(map(str, cifry)).lstrip('0'))


# ==========================================
# NEJISTOTY MĚŘIDEL Z [PRESNOST] (06, bod 2)
# ==========================================
_UPOZORNENI: list[str] = []     # varování z čtení dat; Vystup.uloz je přidá do PŘEDPOKLADŮ


def _upozorni(text: str) -> None:
    if text not in _UPOZORNENI:
        print(f"  [předpoklad] {text}")
        _UPOZORNENI.append(text)


_VAROVANI: list[str] = []       # harness: VAROVÁNÍ ANALÝZY - agent je musí zmínit studentovi


def _varuj(text: str) -> None:
    """Harness: nález analýzy, který není jistá chyba (ta skript ukončí), ale student o něm
    musí vědět. Vystup.uloz ho zapíše do sekce VAROVÁNÍ ANALÝZY, agent ho po spuštění
    vypocet.py dostane k výsledku nástroje."""
    if text not in _VAROVANI:
        print(f"  [!] VAROVÁNÍ ANALÝZY: {text}")
        _VAROVANI.append(text)


def _sjednocene_digity(zapisy: list[str], rozsahy) -> list[float]:
    """Digit každého zápisu sloupce: nejmenší digit mezi zápisy na stejném rozsahu - '0.6'
    vedle '0.59' a '0.60' dostane 0,01. Bez zapsaného rozsahu (automatická volba) se digit
    zjemní nejvýš na tolik platných cifer, kolik jich má nejdelší zápis sloupce, takže '12.3'
    vedle '0.123' nedostane 0,001 (na vyšším rozsahu displej tolik míst neukazuje)."""
    vysledek = [digit(z) for z in zapisy]
    skupiny: dict = {}
    for i, r in enumerate(rozsahy):
        skupiny.setdefault(None if r is None else float(r), []).append(i)
    for r, indexy in skupiny.items():
        nejmensi = min(vysledek[i] for i in indexy)
        platnych = max(_platnych_cifer(zapisy[i]) for i in indexy)
        for i in indexy:
            d = nejmensi
            x = abs(float(zapisy[i]))
            if r is None and x and platnych:
                d = max(d, 10.0 ** (np.floor(np.log10(x)) - (platnych - 1)))
            vysledek[i] = min(vysledek[i], d)
    return vysledek


def digit(zapis: str) -> float:
    """Hodnota posledního opsaného místa: '1.200' -> 0.001, '35' -> 1, '2.50e-3' -> 1e-5."""
    try:
        exponent = Decimal(zapis.strip()).as_tuple().exponent
    except InvalidOperation:
        chyba(f"hodnota {zapis!r} není číslo (desetinná tečka).")
    return 10.0 ** exponent


def s_kolisanim(u_pristroj, oscil=None):
    """Standardní nejistota odečtu rozšířená o kolísání údaje na displeji.

    `oscil` je POLOVINA rozpětí kolísání, (max - min)/2, ve stejných jednotkách jako údaj
    (0 nebo None = údaj nekolísal). Kolísání se bere jako rozpětí ±oscil kolem střední
    hodnoty a převádí se na standardní nejistotu u = oscil/2 (rozhodnutí majitele 27. 10.);
    s nejistotou přístroje se skládá kvadraticky: u = sqrt(u_pristroj² + (oscil/2)²).
    Funguje i na pole (sloupec tabulky). Do PŘEDPOKLADŮ se zapíše věta pro Metody."""
    if oscil is None:
        return u_pristroj
    o = np.asarray(oscil, float)
    if not np.any(o):
        return u_pristroj
    _upozorni("kolísání údaje na displeji (sloupec/klíč oscil_*) je polovina rozpětí kolísání; "
              "převádí se na standardní nejistotu u = oscil/2 a s nejistotou přístroje z [PRESNOST] "
              "se skládá kvadraticky (do Metod protokolu napiš tuto větu).")
    return np.sqrt(np.asarray(u_pristroj, float) ** 2 + (o / 2) ** 2)


class Presnost(NamedTuple):
    """Přesnost přístroje: Δ = z_hodnoty % · |X| + digity · digit + z_rozsahu % · rozsah."""
    z_hodnoty: float = 0.0      # % z měřené hodnoty (multimetr)
    digity: float = 0.0         # počet digitů (ruční multimetr)
    z_rozsahu: float = 0.0      # % z rozsahu (stolní multimetr) = třída (ručkové měřidlo)


def nejistota_meridla(zapis: str, presnost: Presnost, rozsah: float | None = None,
                      digit_zapisu: float | None = None) -> float:
    """
    Standardní nejistota jednoho odečtu z přesnosti přístroje, u = Δ/√3 (rovnoměrné rozdělení),
    Δ = z_hodnoty/100·|X| + digity·digit + z_rozsahu/100·rozsah:
    - digit = poslední opsané místo zápisu, proto se hodnota z multimetru opisuje z displeje
      i s nulami na konci ('1.200' -> 0,001),
    - rozsah (v jednotkách hodnoty) je nutný, je-li z_rozsahu > 0 (ručkové měřidlo, stolní
      multimetr),
    - digit_zapisu: digit určený jinak než ze zápisu (sjednocený ve sloupci, sloupec_meridla).
    """
    if presnost.z_rozsahu and rozsah is None:
        chyba(f"přesnost {cislo(presnost.z_rozsahu)} % z rozsahu (ručkové měřidlo: třída) "
              f"potřebuje použitý rozsah - doplň ho do data.txt.")
    delta = presnost.z_hodnoty / 100 * abs(float(zapis))
    if presnost.digity:
        delta += presnost.digity * (digit(zapis) if digit_zapisu is None else digit_zapisu)
    if presnost.z_rozsahu:
        delta += presnost.z_rozsahu / 100 * rozsah
    return delta / np.sqrt(3)


def u_z_max(delta):
    """Standardní nejistota z maximální chyby Δ (tolerance dekády, přesnost odečtu udaná jako
    „±Δ“): u = Δ/√3 (rovnoměrné rozdělení). Funguje i na pole. Ve vypocet.py se √3 nepíše
    ručně - kontrola skriptu (kontrola_skriptu.py) vlastní dělení √3 odmítne."""
    return np.asarray(delta, float) / np.sqrt(3) if np.ndim(delta) else float(delta) / np.sqrt(3)


def nejmensi_rozsah(x, rozsahy, velicina: str = '') -> np.ndarray:
    """Rozsah, který zvolí automatická volba rozsahu: nejmenší z `rozsahy`, do kterého se |x|
    vejde (nad největší zůstane největší). Jen když student rozsah nezapsal - zapíše se to
    do PŘEDPOKLADŮ K OVĚŘENÍ."""
    rozsahy = np.sort(np.asarray(rozsahy, dtype=float))
    x = np.abs(np.atleast_1d(np.asarray(x, dtype=float)))
    vysledek = rozsahy[np.minimum(np.searchsorted(rozsahy, x), len(rozsahy) - 1)]
    _upozorni(f"rozsah měřidla{' pro ' + velicina if velicina else ''} nezapsán - předpokládá se "
              f"automatická volba nejmenšího rozsahu z {', '.join(cislo(r, '.6g') for r in rozsahy)}, "
              f"do kterého se hodnota vejde.")
    return vysledek


# ==========================================
# REGRESE (01, bod 3.6)
# ==========================================
def primka(x, a, b):
    """Obecná přímka y = a·x + b."""
    return a * x + b


def primka_pocatkem(x, a):
    """Přímka počátkem y = a·x."""
    return a * x


def chi2_red(y, y_fit, sigma, n_param: int) -> float:
    """Redukované chi^2 váženého fitu: sum(((y - y_fit) / sigma)^2) / (n - n_param)."""
    return float(np.sum(((np.asarray(y) - y_fit) / sigma) ** 2) / (len(y) - n_param))


class Fit:
    """Výsledek váženého fitu: `parametry` (korelované ufloaty), `chi2_red`, `rezidua`,
    `u_y` (efektivní nejistoty bodů) a fit(x) = hodnota modelu."""

    def __init__(self, model, x, y, u_y, popt, pcov, jmeno, preskalovat=True):
        self.model, self.x, self.y, self.u_y = model, x, y, u_y
        self.chi2_red = chi2_red(y, model(x, *popt), u_y, len(popt)) if len(y) > len(popt) else float('nan')
        # Při chi2_red > 1 jsou nejistoty bodů podhodnocené nebo model nesedí - nejistoty
        # parametrů se zvětší faktorem sqrt(chi2_red). Při chi2_red < 1 se NIKDY nezmenšují
        # (to dělá absolute_sigma=False a vede to na nereálně malé nejistoty, 01 bod 3.6).
        self.preskalovano = preskalovat and self.chi2_red > 1
        if self.preskalovano:
            pcov = pcov * self.chi2_red
        self.popt, self.pcov = popt, pcov
        # Parametry nesou korelaci z kovarianční matice (např. a a b přímky jsou silně
        # korelované) - výpočty z nich jsou proto správně. V rozpočtu nejistot vystupují
        # společně pod jménem fitu, rozdělit podíl mezi korelované parametry nejde.
        self.parametry = correlated_values(popt, pcov, tags=[jmeno] * len(popt))
        self.jmeno = jmeno
        self.rezidua = y - model(x, *popt)

    def __call__(self, x):
        return self.model(x, *self.popt)

    def sigma_od_nuly(self, index: int) -> float:
        """O kolik sigma se parametr liší od nuly (např. b přímky, která má jít počátkem)."""
        p = self.parametry[index]
        return abs(p.n) / p.s


def vazeny_fit(model, x, y, u_y, u_x=None, p0=None, jmeno='fit', preskalovat=True) -> Fit:
    """
    Vážený fit `curve_fit(model, x, y, sigma=u_y, absolute_sigma=True)` - váhy 1/u_y^2,
    nejistoty parametrů z kovarianční matice. Je-li chi2_red > 1 (a `preskalovat`),
    nejistoty parametrů se zvětší faktorem sqrt(chi2_red); při chi2_red < 1 zůstanou
    (nikdy se nezmenšují). Má-li nezanedbatelnou nejistotu i x (`u_x`),
    použije se efektivní nejistota u_ef^2 = u_y^2 + (f'(x)·u_x)^2 (3 iterace).
    `u_y`/`u_x` mohou být číslo i pole. `jmeno` = položka v rozpočtu nejistot.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    u_y = np.broadcast_to(np.asarray(u_y, float), y.shape).copy()
    if np.any(u_y <= 0):
        chyba(f"{jmeno}: nejistoty bodů musí být kladné.")
    u_ef = u_y
    for _ in range(3 if u_x is not None else 1):
        try:
            popt, pcov = curve_fit(model, x, y, p0=p0, sigma=u_ef, absolute_sigma=True)
        except (RuntimeError, ValueError) as e:
            chyba(f"{jmeno}: fit nekonverguje ({e}) - zkontroluj data nebo počáteční odhad p0.")
        if u_x is None:
            break
        krok = 1e-6 * (np.ptp(x) or 1.0)
        derivace = (model(x + krok, *popt) - model(x - krok, *popt)) / (2 * krok)
        u_ef = np.sqrt(u_y**2 + (derivace * np.asarray(u_x, float))**2)
        p0 = popt
    fit = Fit(model, x, y, u_ef, popt, pcov, jmeno, preskalovat)
    if HARNESS:
        _diagnostika_fitu(fit)
        if u_x is None:
            _FITY_BEZ_UX.append(fit)
            for x_usecek, u_x_usecek in _USECKY_X:
                _zkontroluj_ux(fit, x_usecek, u_x_usecek)
    return fit


# Harness: fity bez u_x a série s vodorovnými úsečkami - chybove_usecky a vazeny_fit se
# navzájem kontrolují, ať nezáleží na pořadí volání ve vypocet.py.
_FITY_BEZ_UX: list = []
_USECKY_X: list = []
_OSY_S_VAROVANIM: list = []
CHI2_MEZE = (0.1, 10.0)


def _diagnostika_fitu(fit: 'Fit') -> None:
    """Harness: χ²_red mimo CHI2_MEZE -> VAROVÁNÍ s DIAGNOSTIKOU FITU (3 největší rezidua
    v násobcích nejistoty bodu). Velké χ² se neřeší zvětšením nejistot bodů ani vyhozením
    bodů bez důvodu - nejdřív data (přepis, jednotky, rozsah) a model."""
    chi2 = fit.chi2_red
    if not np.isfinite(chi2) or CHI2_MEZE[0] <= chi2 <= CHI2_MEZE[1]:
        return
    normovana = fit.rezidua / fit.u_y
    nejvetsi = np.argsort(-np.abs(normovana))[:3]
    body_txt = ', '.join(f"x = {cislo(fit.x[i], '.4g')}: {cislo(normovana[i], '+.1f')}σ" for i in nejvetsi)
    if chi2 > CHI2_MEZE[1]:
        rada = ("body se od modelu liší víc, než dovolují jejich nejistoty - zkontroluj data "
                "(přepis, jednotky, rozsah) a model; nejistoty bodů kvůli tomu nezvětšuj")
    else:
        rada = ("nejistoty bodů jsou nejspíš nadhodnocené (dvakrát započtená složka) nebo obsahují "
                "společnou systematickou složku, která do vah fitu nepatří (01, bod 3.6)")
    _varuj(f"DIAGNOSTIKA FITU {fit.jmeno}: redukované χ² = {cislo(chi2, '.3g')} (bodů {len(fit.x)}) "
           f"je mimo {cislo(CHI2_MEZE[0])}–{cislo(CHI2_MEZE[1])}; {rada}. Největší rezidua: {body_txt}.")


def _zkontroluj_ux(fit: 'Fit', x, u_x) -> None:
    """Harness: fit bez u_x na datech, jejichž nejistota x (přenesená směrnicí modelu) je větší
    než nejistota y -> VAROVÁNÍ (fit by měl dostat u_x)."""
    x = np.asarray(x, float)
    if x.shape != fit.x.shape or not np.allclose(x, fit.x):
        return
    krok = 1e-6 * (np.ptp(x) or 1.0)
    derivace = (fit.model(x + krok, *fit.popt) - fit.model(x - krok, *fit.popt)) / (2 * krok)
    vliv_x = np.abs(derivace) * np.broadcast_to(np.asarray(u_x, float), x.shape)
    if np.median(vliv_x) > np.median(fit.u_y):
        _varuj(f"{fit.jmeno}: nejistota x (přenesená směrnicí, medián {cislo(np.median(vliv_x), '.2g')}) je "
               f"větší než nejistota y (medián {cislo(np.median(fit.u_y), '.2g')}), ale fit ji nebere - "
               f"předej ji: pl.vazeny_fit(..., u_x=u_x).")


def _hodnoty_a_nejistoty(v, u):
    """Pole čísel, nebo ufloatů (pak se nejistota vezme z nich, pokud `u` není zadané)."""
    v = np.atleast_1d(np.asarray(v))
    if v.dtype == object and v.size and hasattr(v.flat[0], 'n'):
        hodnoty = np.array([p.n for p in v], float)
        return hodnoty, (np.array([p.s for p in v], float) if u is None else u)
    return v.astype(float), u


def _podil_osy(ax, hodnoty, u, osa: str) -> np.ndarray:
    """Délka celé chybové úsečky (2u) jako podíl rozsahu osy (i logaritmické)."""
    dolni, horni = ax.get_xlim() if osa == 'x' else ax.get_ylim()
    u = np.broadcast_to(np.asarray(u, float), hodnoty.shape)
    if (ax.get_xscale() if osa == 'x' else ax.get_yscale()) == 'log' and dolni > 0:
        return 2 * u / (np.maximum(np.abs(hodnoty), 1e-300) * np.log(10) * np.log10(horni / dolni))
    return 2 * u / (abs(horni - dolni) or 1.0)


def chybove_usecky(ax, x, y, u_y=None, u_x=None, serie: int = 0, label: str | None = None):
    """
    Naměřené body i-té série s chybovými úsečkami (ax.errorbar s **body(serie)):
    `pl.chybove_usecky(ax, I, U, u_U, u_x=u_I, serie=0, label='série 1 (data)')`.
    x a y mohou být i pole ufloatů (nejistota se pak vezme z nich). Vrací kontejner errorbar.

    Harness (pl.HARNESS) navíc zapíše VAROVÁNÍ ANALÝZY:
    - všechny úsečky jsou kratší než ~1 % rozsahu osy nebo než značka -> do popisku grafu
      patří věta „Chybové úsečky jsou menší než velikost bodů.“ (jinak čtenář neví, že tam jsou),
    - nejistota x je větší než nejistota y (přenesená směrnicí), ale fit se stejnými x ji nebere.
    Volej ji po nastavení měřítka os (set_xscale/set_yscale), kontroluje se aktuální rozsah.
    """
    x, u_x = _hodnoty_a_nejistoty(x, u_x)
    y, u_y = _hodnoty_a_nejistoty(y, u_y)
    kontejner = ax.errorbar(x, y, yerr=u_y, xerr=u_x, **body(serie), label=label)
    if not HARNESS:
        return kontejner
    ax.relim()
    ax.autoscale_view()
    vyska = ax.get_window_extent().height or 1.0
    znacka = STYL['velikost_znacky'] * ax.figure.dpi / 72 / vyska     # průměr značky / výška osy
    mez = max(0.01, znacka)
    neviditelne_y = u_y is None or np.all(_podil_osy(ax, y, u_y, 'y') < mez)
    neviditelne_x = u_x is None or np.all(_podil_osy(ax, x, u_x, 'x') < mez)
    if (neviditelne_y and neviditelne_x and (u_y is not None or u_x is not None)
            and not any(a is ax for a in _OSY_S_VAROVANIM)):
        _OSY_S_VAROVANIM.append(ax)                 # jedna věta do popisku na graf, ne na sérii
        _varuj(f"graf{' se sérií ' + label if label else ''}: chybové úsečky jsou menší než značky bodů "
               f"(pod ~1 % rozsahu osy) - do popisku grafu napiš „Chybové úsečky jsou menší než "
               f"velikost bodů.“")
    if u_x is not None:
        _USECKY_X.append((x, u_x))
        for fit in _FITY_BEZ_UX:
            _zkontroluj_ux(fit, x, u_x)
    return kontejner


def graf_rezidui(ax, fit: Fit, popisek_x: str, popisek_y: str, serie: int = 0) -> None:
    """Rezidua fitu s chybovými úsečkami a nulovou čárou do připravené osy `ax`."""
    ax.errorbar(fit.x, fit.rezidua, yerr=fit.u_y, **body(serie))
    ax.axhline(0, color='0.35', linewidth=0.8)
    ax.set_xlabel(popisek_x)
    ax.set_ylabel(popisek_y)
    carky_na_osach(ax)


# ==========================================
# ROZBOR PŘESNOSTI (01, bod 3.9)
# ==========================================
def rozpocet_nejistot(vysledek) -> dict[str, float]:
    """
    Podíl jednotlivých vstupů na ROZPTYLU výsledku v % (součet 100 %), od největšího.
    Vstupy musí mít v ufloat jméno, např. ufloat(8550, 100, 'ρ'); vstupy se stejným
    jménem (např. odečty d v každém řádku tabulky) se sečtou.
    Vzorec: podíl_i = (∂f/∂x_i · σ_i)² / σ_f².
    """
    rozptyly = {}
    for promenna, slozka in vysledek.error_components().items():
        jmeno = getattr(promenna, 'tag', None) or '?'
        rozptyly[jmeno] = rozptyly.get(jmeno, 0.0) + slozka ** 2
    celkem = sum(rozptyly.values())
    if celkem == 0:
        return {}
    return dict(sorted(((k, 100 * v / celkem) for k, v in rozptyly.items()), key=lambda kv: -kv[1]))


def graf_rozpoctu(rozpocty: dict[str, dict[str, float]], soubor: str) -> None:
    """Sloupcový graf rozpočtu nejistot - jeden panel na výsledek, vstupy od největšího podílu.
    Výška je shora omezená, aby se graf i s mnoha výsledky vešel na stránku protokolu."""
    radku = sum(len(r) for r in rozpocty.values())
    fig, osy = plt.subplots(len(rozpocty), 1, squeeze=False,
                            figsize=(8, min(9.0, 1.2 + 0.45 * radku + 0.6 * len(rozpocty))),
                            gridspec_kw={'height_ratios': [max(1, len(r)) for r in rozpocty.values()]})
    for ax, (vysledek, rozpocet) in zip(osy[:, 0], rozpocty.items()):
        jmena = list(rozpocet)[::-1]                      # největší podíl nahoře
        ax.barh(jmena, [rozpocet[j] for j in jmena], color=barva(0))
        ax.set_xlim(0, 100)
        ax.set_title(vysledek, loc='left', fontsize=13)  # popisek panelu, ne nadpis grafu
        ax.xaxis.set_major_formatter(Carka())
    osy[-1, 0].set_xlabel('Podíl na rozptylu výsledku / %')
    fig.tight_layout()
    fig.savefig(soubor)
    plt.close(fig)


def potrebna_hodnota(model, cil: float, vychozi: float = 0.0, krok: float = 1e-6) -> float:
    """
    Obrácená otázka: jakou hodnotu by musel mít parametr jevu (posun nuly, oteplení plynu...),
    aby výsledek vyšel jako tabulková hodnota `cil`. `model(parametr)` vrací výsledek (float);
    `vychozi` je hodnota, se kterou se počítalo (typicky 0). Lineární odhad z derivace.
    """
    derivace = (model(vychozi + krok) - model(vychozi)) / krok
    return vychozi + (cil - model(vychozi)) / derivace


# ==========================================
# VÝSTUP vysledky_analyzy.txt (01, bod 3.8)
# ==========================================
SEKCE_VAROVANI = "VAROVÁNÍ ANALÝZY"
_UNIK = re.compile(r"#\s*harness:\s*povoleno\s*[-–—:]\s*(\S.*?)\s*$", re.IGNORECASE)


def _povolene_vyjimky() -> list[str]:
    """Výjimky z kontroly skriptu (`# harness: povoleno – důvod` ve vypocet.py) jako věty do
    PŘEDPOKLADŮ K OVĚŘENÍ - student má vidět, kde skript vědomě obchází knihovnu a proč."""
    skript = getattr(sys.modules.get('__main__'), '__file__', None)
    if not skript or not os.path.isfile(skript):
        return []
    try:
        with open(skript, encoding='utf-8') as f:
            radky = f.read().splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    return [f"skript na řádku {i} vědomě obchází praktikum_lib ({radek.split('#', 1)[0].strip()}): "
            f"{m.group(1)} - ověř, že je to oprávněné."
            for i, radek in enumerate(radky, start=1) if (m := _UNIK.search(radek))]


class Vystup:
    """
    Skládá vysledky_analyzy.txt: hlavička, PŘEDPOKLADY K OVĚŘENÍ, sekce s výsledky, rozpočet
    nejistot, obrácená otázka a na konci sanity checky. Předpoklady a varování se vypisují
    i do konzole hned, jak vzniknou.
    """

    def __init__(self, nazev_ulohy: str, cesta: str = OUTPUT_FILE):
        self.nazev, self.cesta = nazev_ulohy, cesta
        self.radky, self.predpoklady, self.kontroly = [], [], []

    # --- obecné řádky ---
    def sekce(self, nadpis: str) -> None:
        self.radky += ['', f"--- {nadpis} ---"]

    def radek(self, text: str = '') -> None:
        self.radky.append(text)

    def vysledek(self, popis: str, x, jednotka: str = '', relativne: bool = True) -> None:
        """'c = (342 ± 1) m/s  (relativní nejistota 0,3 %)'."""
        text = f"{popis} = {fmt(x, jednotka)}"
        if relativne and x.s:
            text += f"  (relativní nejistota {relativni(x)})"
        self.radky.append(text)

    def diskuze(self, popis: str, x, jednotka: str = '') -> None:
        """Číslo pro Diskusi/Závěr, které není jinde ve výpisu (poměr, odhad vlivu, rozdíl...).
        Do Diskuse a Závěru smí jen čísla z `vysledky_analyzy.txt` (kontrola_dat.py) - když
        potřebuješ další, spočti ho ve skriptu a vypiš sem: `vystup.diskuze('|b|/U_H,min', x, '%')`.
        `x` je ufloat (vypíše se s nejistotou) nebo obyčejné číslo."""
        if not any(r.startswith('--- ČÍSLA PRO DISKUZI') for r in self.radky):
            self.sekce("ČÍSLA PRO DISKUZI A ZÁVĚR (odhady a odvozené hodnoty ze skriptu)")
        text = fmt(x, jednotka) if hasattr(x, 'n') else f"{cislo(float(x))} {jednotka}".strip()
        self.radky.append(f"{popis} = {text}")

    def fit(self, popis: str, fit: Fit, jmena: list[str], jednotky: list[str],
            model: str = '', nulovy: int | None = None) -> None:
        """Parametry fitu, chi^2_red (jen při zapnutém REZIDUA) a případně o kolik sigma je
        parametr `nulovy` od nuly."""
        self.radky.append(f"{popis}: {model}".rstrip(': '))
        self.radky.append("  vážený fit (váhy 1/u^2, absolute_sigma=True), nejistoty z kovarianční matice"
                          + (", vynásobené sqrt(chi^2_red)" if fit.preskalovano and REZIDUA else ""))
        for jmeno, p, jednotka in zip(jmena, fit.parametry, jednotky):
            self.radky.append(f"  {jmeno} = {fmt(p, jednotka)}")
        if nulovy is not None:
            self.radky.append(f"  {jmena[nulovy]} se od nuly liší o {cislo(fit.sigma_od_nuly(nulovy), '.1f')} sigma")
        if not REZIDUA:
            return
        self.radky.append(f"  redukované chi^2 = {cislo(fit.chi2_red, '.2f')} (bodů {len(fit.x)})")
        if fit.chi2_red > 3:
            self.predpoklad(f"{popis}: chi^2_red = {cislo(fit.chi2_red, '.1f')} >> 1 - nejistoty bodů "
                            f"jsou podhodnocené nebo model nesedí (01, bod 3.6).")

    # --- povinné části ---
    def predpoklad(self, text: str) -> None:
        """Co skript odhadl/předpokládal - student to má ověřit (01, bod 0)."""
        print(f"  [předpoklad] {text}")
        self.predpoklady.append(text)

    def varovani(self, text: str) -> None:
        """Nález analýzy pro sekci VAROVÁNÍ ANALÝZY (jen se zapnutým harnessem, jinak jde do
        PŘEDPOKLADŮ): co není jistá chyba, ale student o tom musí vědět."""
        if HARNESS:
            _varuj(text)
        else:
            self.predpoklad(text)

    def kontrola(self, podminka: bool, text: str) -> bool:
        """Sanity check (01, bod 3.4): nesplněný se vypíše jako VAROVÁNÍ, splněný jako OK."""
        if not podminka:
            print(f"  [!] VAROVÁNÍ: {text}")
        self.kontroly.append(('OK' if podminka else 'VAROVÁNÍ', text))
        return bool(podminka)

    def rozpocty(self, hlavni: dict, soubor: str | None = None) -> dict:
        """Rozpočet nejistot hlavních výsledků {'E (mosaz)': E, ...} jako sekce textu - do
        protokolu jde slovně v Diskusi (04, bod 1.2). Graf jen na výslovné přání (`soubor`)."""
        rozpocty = {jmeno: rozpocet_nejistot(v) for jmeno, v in hlavni.items()}
        self.sekce("ROZPOČET NEJISTOT (podíl vstupu na rozptylu výsledku)")
        if soubor:
            graf = cesta_grafu(soubor)
            graf_rozpoctu(rozpocty, graf)
            self.radky.append(f"Graf: {graf}")
        for jmeno, rozpocet in rozpocty.items():
            self.radky.append(f"{jmeno} (relativní nejistota {relativni(hlavni[jmeno])}): "
                              + ", ".join(f"{k} {cislo(p, '.0f')} %" for k, p in rozpocet.items()))
        return rozpocty

    def odchylka(self, jmeno: str, x, tabulkova: float, zdroj: str = '', jevy=()) -> float:
        """
        Srovnání s tabulkovou hodnotou v násobcích sigma. Při odchylce > 2 sigma vypíše
        obrácenou otázku pro `jevy` = [(popis jevu, potřebná hodnota, jednotka), ...]
        (potřebnou hodnotu spočítá potrebna_hodnota). Vrací odchylku v sigma.
        """
        sigma = (x.n - tabulkova) / x.s
        if not any(r.startswith('--- SROVNÁNÍ') for r in self.radky):
            self.sekce("SROVNÁNÍ S TABULKOVÝMI HODNOTAMI A OBRÁCENÁ OTÁZKA")
        zdroj = f" ({zdroj})" if zdroj else ''
        self.radky.append(f"{jmeno}: {fmt(x)} vs. tabulkově {cislo(tabulkova)}{zdroj} -> "
                          f"odchylka {cislo(sigma, '+.1f')} sigma")
        if abs(sigma) > 2:
            for popis, potreba, jednotka in jevy:
                self.radky.append(f"  {popis} by musel mít hodnotu {cislo(potreba, '.2g')} {jednotka}".rstrip())
        return sigma

    def uloz(self) -> None:
        hlavicka = ["=" * 50, f"VÝSLEDKY ANALÝZY: {self.nazev}", "=" * 50, '']
        if HARNESS:
            # Varování nahoře: agent je dostane k výsledku spuštění a musí je zmínit studentovi.
            hlavicka += [f"--- {SEKCE_VAROVANI} ---"]
            hlavicka += [f"- {v}" for v in _VAROVANI] or ["- žádná"]
            hlavicka += ['']
        hlavicka += ["--- PŘEDPOKLADY K OVĚŘENÍ ---"]
        predpoklady = self.predpoklady + [u for u in _UPOZORNENI if u not in self.predpoklady]
        if HARNESS:
            predpoklady += [p for p in _povolene_vyjimky() if p not in predpoklady]
        hlavicka += [f"- {p}" for p in predpoklady] or ["- žádné"]
        konec = ['', "--- SANITY CHECKY ---"]
        konec += [f"[{stav}] {text}" for stav, text in self.kontroly] or ["(žádné)"]
        with open(self.cesta, 'w', encoding='utf-8') as f:
            f.write('\n'.join(hlavicka + self.radky + konec) + '\n')
        print(f"Výsledky byly uloženy do souboru: {self.cesta}")
