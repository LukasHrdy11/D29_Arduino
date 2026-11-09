# D29 – Arduino ve fyzice

Kód, programy pro Arduino a naměřená data k úloze 29 „Arduino ve fyzice“ fyzikálního
praktika II na MFF UK (měřeno 5.–9. 11. 2026). Protokol se na tento repozitář odkazuje
(verze `v1.0`).

Úkoly 1–3 jsou ze zadání (seznámení s Arduinem, VA charakteristiky LED, RC článek původní
metodou), úkoly 4–7 jsou rozšíření schválená vedoucím: RC článek se vzorkováním do paměti
(buffer, úkol 4), linearita Si fotodiody (5), průměrování a šum (6) a softwarový lock-in (7).
Číslování úkolů odpovídá protokolu; skripty úkolu 6 se jmenují `ukol6a_*`, skripty
lock-inu (úkol 7) `ukol6_*`.

Hardware: Arduino Uno R3 napájené z USB, součástky ze sady praktika, Si fotodioda
s pracovním odporem 1 MΩ. Arduino jen vzorkuje, veškeré výpočty dělá Python na počítači.

## Obsah

| složka | obsah |
|---|---|
| `sketches/` | programy pro Arduino (každý ve složce se stejným jménem, jak vyžaduje Arduino IDE) |
| `scripts/` | záznam z Arduina (`pyserial`) a vyhodnocení, jména podle úkolu (`ukolN_*.py`) |
| `data/` | naměřená data (CSV) a souhrny vyhodnocení (JSON) po úkolech |
| `figures/` | grafy v protokolu (PDF), generuje `scripts/protokol_grafy.py` |

Formáty dat: řádky začínající `#` jsou metadata (`klíč=hodnota`), pak hlavička a data.
Proudová data (`ref,adc`) jsou vzorky 1 kHz: `ref` je stav modulované LED (pin D9), `adc`
kód převodníku 0–1023.

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

V `data/ukol6_lockin/` jsou navíc kontrolní měření elektrického přeslechu
(`slaby_kontrola_*`: LED vypnutá, drátek místo LED, samostatná zem), která v protokolu
nejsou uvedena.

`scripts/protokol_grafy.py` čte jen soubory v `data/` a kreslí všech devět grafů protokolu;
čísla citovaná v protokolu přitom přepočítá a vypíše do `figures/protokol_grafy_vystup.txt`.
Styl grafů (barvy, značky, desetinná čárka) nastavuje `scripts/praktikum_lib.py` podle
`styl_grafu.json`.

## Reprodukce

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python3 scripts/protokol_grafy.py          # grafy do figures/
python3 scripts/ukol4_tau_R.py data/ukol4_rc_buffer/rc_*.json   # např. τ(R) a C z úkolu 4
```

Vyhodnocovací skripty pracují nad `data/` (a vedle dat zapisují náhledy PNG). Záznamové
skripty potřebují Arduino na `/dev/ttyACM0` (jinak `-p <port>`); program se nahraje
v Arduino IDE nebo přes `arduino-cli`:

```bash
arduino-cli compile -b arduino:avr:uno sketches/stream
arduino-cli upload  -b arduino:avr:uno -p /dev/ttyACM0 sketches/stream
```

Sériový monitor v Arduino IDE musí být zavřený, jinak je port obsazený.

## Podíl AI asistenta

Při práci na tomto protokolu jsem pro psaní kódů na Arduino využil AI asistenta Claude. 

## Licence

MIT, viz `LICENSE`.

## Zveřejnění (pro autora)

```bash
git init
git add .
git commit -m "Kód a data k úloze D29 Arduino ve fyzice"
git branch -M main
git remote add origin https://github.com/LukasHrdy11/D29_Arduino.git
git push -u origin main
git tag v1.0
git push origin v1.0
```
