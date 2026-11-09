# D29 – Arduino ve fyzice

Kód, programy pro Arduino a naměřená data k úloze 29 „Arduino ve fyzice“ fyzikálního
praktika II na MFF UK (měřeno 5.–9. 11. 2026). Protokol se na tento repozitář odkazuje.

## Obsah

| složka | obsah |
|---|---|
| `sketches/` | programy pro Arduino |
| `scripts/` | záznam z Arduina (`pyserial`) a vyhodnocení, jména podle úkolu (`ukolN_*.py`) |
| `data/` | naměřená data (CSV) a souhrny vyhodnocení (JSON) po úkolech |
| `figures/` | grafy v protokolu (PDF), generuje `scripts/protokol_grafy.py` |

## Skripty, úkoly a výstupy v protokolu

| úkol | program pro Arduino | záznam | vyhodnocení | v protokolu |
|---|---|---|---|---|
| 1 | `blink`, `read_voltage` | `ukol1_potenciometr.py` | `ukol1_vysledky.py` | tabulka napětí potenciometru |
| 2 | `va_led` | `ukol2_va_led.py` | `ukol2_kontrola.py`, `ukol2_analyza.py` | `graf_va_led.pdf`, tabulka prahů |
| 3 | `rc_puvodni` | `ukol3_rc.py` | `ukol3_analyza.py`, `ukol3_tau_R.py` | tabulka τ, `graf_srovnani_10k.pdf`, `graf_tau_R.pdf` |
| 4 | `rc_buffer` | `ukol4_rc.py` | `ukol4_analyza.py`, `ukol4_tau_R.py` | tabulka τ, `graf_tau_R.pdf` |
| 4 (odpor pinu) | `rpin` | `ukol4_rpin.py record` | `ukol4_rpin.py analyse` | R_pin v textu |
| 5 | `stream` | `ukol5_linearita.py`, `ukol5_live.py` (míření) | `ukol5_analyza.py` | tabulka signálu, `graf_linearita.pdf` |
| 6 | `stream` | `ukol6a_prumerovani.py` | `ukol6a_srovnani.py` | `graf_sigma_N.pdf`, `graf_spektrum.pdf` |
| 7.1 | `stream` | `ukol6_frekvence.py` | `ukol6_bode.py` | `graf_bode.pdf` |
| 7.2 | `stream` | `ukol6_slaby_signal.py mer` | `ukol6_slaby_signal.py ana` | `graf_slaby_signal.pdf` |

`scripts/protokol_grafy.py` čte jen soubory v `data/` a kreslí všech devět grafů protokolu;
čísla citovaná v protokolu přitom přepočítá a vypíše do `figures/protokol_grafy_vystup.txt`.
Styl grafů (barvy, značky, desetinná čárka) nastavuje `scripts/praktikum_lib.py` podle
`styl_grafu.json`.

## Podíl AI asistenta

Při práci na tomto protokolu jsem pro psaní kódů na Arduino využil AI asistenta Claude. 
