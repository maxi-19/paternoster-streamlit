import matplotlib
matplotlib.use("Agg")
import streamlit as st
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO


# Parameter
kassetten_hoehen = [105, 145, 215, 255, 355, 455, 800]  # mm
max_kassette_gewicht = 3000  # kg
spielraum = 15  # mm
start_hoehe = 400  # mm
max_turmhoehe = 7500  # mm
max_turmgewicht = 100000  # kg
kassette_breite = 840  # mm
kassette_laenge = 6500  # mm

def parse_breite_hoehe(masse_str):
    try:
        teile = masse_str.replace(",", ".").split("x")
        b = float(teile[0].strip())
        h = float(teile[1].strip())
    except Exception as e:
        b = h = 100
    return min(b, h), max(b, h)

def make_empty_turm():
    return {'hoehe': start_hoehe, 'kassetten': [], 'gewicht': 0}

# --- Variante a: Sortenrein pro Kassette ---
def verteile_profile_auf_kassetten(data):
    kassetten = []
    for index, profil in data.iterrows():
        try:
            bezeichnung = str(profil["Bezeichnung"])
            masse = str(profil["Breite x Höhe x Dicke"])
            laenge = float(str(profil["Länge in mm"]).replace(",", "."))
            anzahl = int(profil["Stückzahl"])
            gesamtgewicht = float(str(profil["Gewicht (kg)"]).replace(",", "."))
            gewicht_stk = float(str(profil["Gewicht/stk (kg)"]).replace(",", "."))
            if any([
                pd.isna(bezeichnung) or bezeichnung.strip() == "",
                pd.isna(masse) or masse.strip() == "",
                pd.isna(laenge),
                pd.isna(anzahl),
                pd.isna(gesamtgewicht),
                pd.isna(gewicht_stk),
            ]):
                continue
        except Exception as e:
            st.warning(f"Problem in Zeile {index+2} (Excel): {e}")
            continue

        rest = int(anzahl)
        hoehe, breite = parse_breite_hoehe(masse)
        pro_lange = int(kassette_laenge // laenge)
        pro_breite = int(kassette_breite // breite)
        pro_ebene = max(1, pro_lange * pro_breite)
        kass_nr = 1
        while rest > 0:
            max_stk_by_weight = min(pro_ebene, int(max_kassette_gewicht // gewicht_stk))
            max_stk_by_weight = max(1, max_stk_by_weight)
            max_ebenen_by_height = max(1, max([kh for kh in kassetten_hoehen if kh >= hoehe]) // hoehe)
            max_ebenen = min(max_ebenen_by_height, math.ceil(rest / max_stk_by_weight))
            stk_in_kassette = min(rest, max_stk_by_weight * max_ebenen)
            ebene_n = math.ceil(stk_in_kassette / max_stk_by_weight)
            stapelhoehe = hoehe * ebene_n
            kassettenhoehe = None
            for kh in kassetten_hoehen:
                if stapelhoehe <= kh:
                    kassettenhoehe = kh
                    break
            if not kassettenhoehe:
                kassettenhoehe = kassetten_hoehen[-1]
            kassettengewicht = gewicht_stk * stk_in_kassette
            while (kassettengewicht > max_kassette_gewicht or stk_in_kassette <= 0):
                ebene_n -= 1
                if ebene_n <= 0:
                    ebene_n = 1
                    break
                stk_in_kassette = min(rest, max_stk_by_weight * ebene_n)
                stapelhoehe = hoehe * ebene_n
                for kh in kassetten_hoehen:
                    if stapelhoehe <= kh:
                        kassettenhoehe = kh
                        break
                kassettengewicht = gewicht_stk * stk_in_kassette
            kassetten.append({
                'bezeichnung': bezeichnung,
                'maße': masse,
                'laenge': laenge,
                'anzahl': int(stk_in_kassette),
                'kassettenhoehe': kassettenhoehe,
                'gewicht': round(kassettengewicht, 1),
                'stapelhoehe': stapelhoehe,
                'pro_ebene': max_stk_by_weight,
                'pro_breite': pro_breite,
                'pro_lange': pro_lange,
                'profilbreite': breite,
                'profilhoehe': hoehe,
                'sortenrein': True
            })
            rest -= int(stk_in_kassette)
            kass_nr += 1
    return kassetten


def tuerme_nacheinander_fuellen_sortenrein(kassetten):
    tuerme = []
    akt_turm = make_empty_turm()
    for kas in kassetten:
        kasgewicht = kas["gewicht"]
        kashoehe = kas["kassettenhoehe"]
        next_hoehe = kashoehe + spielraum
        if (
            akt_turm['hoehe'] + next_hoehe <= max_turmhoehe
            and akt_turm['gewicht'] + kasgewicht <= max_turmgewicht
        ):
            akt_turm['kassetten'].append(kas)
            akt_turm['hoehe'] += next_hoehe
            akt_turm['gewicht'] += kasgewicht
        else:
            tuerme.append(akt_turm)
            akt_turm = make_empty_turm()
            akt_turm['kassetten'].append(kas)
            akt_turm['hoehe'] += next_hoehe
            akt_turm['gewicht'] += kasgewicht
    if akt_turm['kassetten']:
        tuerme.append(akt_turm)
    for turm in tuerme:
        turm['kassetten'].sort(key=lambda k: -k['gewicht'])
    # Paare zu Doppeltürmen gruppieren
    tuermeliste = []
    for i in range(0, len(tuerme), 2):
        turm1 = tuerme[i]
        turm2 = tuerme[i+1] if i+1 < len(tuerme) else make_empty_turm()
        tuermeliste.append({'turm1': turm1, 'turm2': turm2})
    return tuermeliste


# --- Variante b: Max. Ausnutzung, gemischte Kassetten ---
def gemischte_kassetten_bin_packing(data):
    """Erstellt möglichst volle Kassetten unabhängig von Sortenreinheit."""
    # Wir packen alle Einzelprofile als "Stück" in eine gemeinsame Liste
    profile_einzeln = []
    for index, profil in data.iterrows():
        try:
            bezeichnung = str(profil["Bezeichnung"])
            masse = str(profil["Breite x Höhe x Dicke"])
            laenge = float(str(profil["Länge in mm"]).replace(",", "."))
            anzahl = int(profil["Stückzahl"])
            gesamtgewicht = float(str(profil["Gewicht (kg)"]).replace(",", "."))
            gewicht_stk = float(str(profil["Gewicht/stk (kg)"]).replace(",", "."))
            if any([
                pd.isna(bezeichnung) or bezeichnung.strip() == "",
                pd.isna(masse) or masse.strip() == "",
                pd.isna(laenge),
                pd.isna(anzahl),
                pd.isna(gesamtgewicht),
                pd.isna(gewicht_stk),
            ]):
                continue
        except Exception as e:
            st.warning(f"Problem in Zeile {index+2} (Excel): {e}")
            continue

        hoehe, breite = parse_breite_hoehe(masse)
        for i in range(anzahl):
            profile_einzeln.append({
                'bezeichnung': bezeichnung,
                'maße': masse,
                'laenge': laenge,
                'gewicht': gewicht_stk,
                'profilbreite': breite,
                'profilhoehe': hoehe
            })
    # Sortiere schwerste Profile zuerst
    profile_einzeln.sort(key=lambda x: -x['gewicht'])

    kassetten = []
    # "Bin Packing": Füge Profile einzeln in Kassetten ein, bis sie voll sind
    for prof in profile_einzeln:
        placed = False
        for kas in kassetten:
            if (
                kas['gewicht'] + prof['gewicht'] <= max_kassette_gewicht
                and kas['laenge'] == prof['laenge']
            ):
                # Breite prüfen: Füge Profil hinzu, falls noch Platz
                # Annahme: Keine Überschneidung in Ebene!
                if (
                    kas['benutzte_breite'] + prof['profilbreite'] <= kassette_breite
                    and kas['benutzte_hoehe'] + prof['profilhoehe'] <= max(kassetten_hoehen)
                ):
                    kas['profile'].append(prof)
                    kas['gewicht'] += prof['gewicht']
                    kas['benutzte_breite'] += prof['profilbreite']
                    kas['benutzte_hoehe'] = max(kas['benutzte_hoehe'], prof['profilhoehe'])
                    placed = True
                    break
        if not placed:
            # Neue Kassette anlegen
            kassetten.append({
                'profile': [prof],
                'laenge': prof['laenge'],
                'gewicht': prof['gewicht'],
                'benutzte_breite': prof['profilbreite'],
                'benutzte_hoehe': prof['profilhoehe'],
            })
    # Kassettendaten in Tabelle verwandeln
    result_kassetten = []
    for i, kas in enumerate(kassetten):
        kassettenhoehe = min([kh for kh in kassetten_hoehen if kh >= kas['benutzte_hoehe']])
        # Zähle Profile pro Typ in Kassette
        profile_counter = {}
        for p in kas['profile']:
            key = (p['bezeichnung'], p['maße'])
            if key not in profile_counter:
                profile_counter[key] = 0
            profile_counter[key] += 1
        profil_liste = []
        for (bezeichnung, masse), count in profile_counter.items():
            profil_liste.append(f"{bezeichnung}: {count} Stk.")
        result_kassetten.append({
            "Kassetten-Nr": i+1,
            "Profil(e)": "; ".join(profil_liste),
            "Länge": kas['laenge'],
            "Kassettenhöhe": kassettenhoehe,
            "Kassettengewicht": round(kas['gewicht'],1),
            "Sortenrein": False if len(profile_counter)>1 else True
        })
    return result_kassetten

def tuerme_nacheinander_fuellen(kassetten):
    tuerme = []
    akt_turm = make_empty_turm()
    for kas in kassetten:
        kasgewicht = kas["Kassettengewicht"]
        kashoehe = kas["Kassettenhöhe"]
        next_hoehe = kashoehe + spielraum
        # Prüfe, ob Kassette noch passt, sonst neuen Turm beginnen
        if (
            akt_turm['hoehe'] + next_hoehe <= max_turmhoehe
            and akt_turm['gewicht'] + kasgewicht <= max_turmgewicht
        ):
            akt_turm['kassetten'].append(kas)
            akt_turm['hoehe'] += next_hoehe
            akt_turm['gewicht'] += kasgewicht
        else:
            tuerme.append(akt_turm)
            akt_turm = make_empty_turm()
            akt_turm['kassetten'].append(kas)
            akt_turm['hoehe'] += next_hoehe
            akt_turm['gewicht'] += kasgewicht
    if akt_turm['kassetten']:
        tuerme.append(akt_turm)
    for turm in tuerme:
        turm['kassetten'].sort(key=lambda k: -k['Kassettengewicht'])
    # Jetzt Paare zu Doppeltürmen bilden
    tuermeliste = []
    i = 0
    while i < len(tuerme):
        turm1 = tuerme[i]
        turm2 = tuerme[i+1] if (i+1) < len(tuerme) else make_empty_turm()  # oder None, je nach Visualisierung
        tuermeliste.append({'turm1': turm1, 'turm2': turm2})
        i += 2
    return tuermeliste





def plot_doppelturm_einzeln(tuermeliste, titel="Seitenansicht Doppeltürme", turmbreite=520):
    figs = []
    for i, dt in enumerate(tuermeliste):
        for tnum, turm in enumerate([dt['turm1'], dt['turm2']]):
            if not turm['kassetten']:
                continue

            fig, ax = plt.subplots(figsize=(7, 12))   # Turm breit, Bild hoch
            ax.set_title(f"Doppelturm {i+1} – Turm {tnum+1}", fontsize=18, pad=15)
            ax.set_xlabel("")
            ax.set_ylabel("Höhe [mm]", fontsize=11)
            ax.set_xlim(0, turmbreite)
            ax.set_ylim(0, max_turmhoehe + 400)
            ax.set_xticks([])
            ax.grid(axis='y', linestyle=":", alpha=0.15)

            # Technik-Bereich unten
            ax.fill_between([0, turmbreite], 0, start_hoehe, color='lightgrey', alpha=0.7, zorder=0)
            ax.text(turmbreite/2, start_hoehe/2, f"Technik\n({start_hoehe:.0f} mm)", va="center", ha="center", fontsize=9)

            y_curr = start_hoehe
            for k_idx, kass in enumerate(turm['kassetten']):
                rect = plt.Rectangle((turmbreite*0.05, y_curr), turmbreite*0.9, kass['kassettenhoehe'],
                                    edgecolor="black", facecolor="white", lw=1.3, zorder=2)
                ax.add_patch(rect)

                labeltext = (
                    f"Kassette {k_idx+1} | "
                    f"{kass.get('bezeichnung','')} | "
                    f"{kass['kassettenhoehe']} mm | "
                    f"Stück: {kass.get('anzahl', kass.get('Stück', ''))} | "
                    f"{kass['gewicht']:.1f} kg"
                )

                ax.text(
                    turmbreite/2, y_curr + kass['kassettenhoehe']/2,
                    labeltext, va="center", ha="center", fontsize=6.5, 
                    wrap=False, clip_on=True, zorder=10,
                )

                y_curr += kass['kassettenhoehe'] + spielraum

            ax.set_xticks([])
            fig.tight_layout()
            figs.append(fig)
    return figs


def plot_doppelturm_einzeln_gemischt(tuermeliste, titel="Seitenansicht Doppeltürme (gemischt)", turmbreite=520):
    figs = []
    for i, dt in enumerate(tuermeliste):
        for tnum, turm in enumerate([dt['turm1'], dt['turm2']]):
            if not turm['kassetten']:
                continue

            fig, ax = plt.subplots(figsize=(7, 12))
            ax.set_title(f"Doppelturm {i+1} – Turm {tnum+1}", fontsize=18, pad=15)
            ax.set_xlabel("")
            ax.set_ylabel("Höhe [mm]", fontsize=11)
            ax.set_xlim(0, turmbreite)
            ax.set_ylim(0, max_turmhoehe + 400)
            ax.set_xticks([])
            ax.grid(axis='y', linestyle=":", alpha=0.15)

            # Technik-Bereich unten
            ax.fill_between([0, turmbreite], 0, start_hoehe, color='lightgrey', alpha=0.7, zorder=0)
            ax.text(turmbreite/2, start_hoehe/2, f"Technik\n({start_hoehe:.0f} mm)", va="center", ha="center", fontsize=9)

            y_curr = start_hoehe
            for k_idx, kass in enumerate(turm['kassetten']):
                rect = plt.Rectangle((turmbreite*0.05, y_curr), turmbreite*0.9, kass['Kassettenhöhe'],
                                    edgecolor="black", facecolor="white", lw=1.3, zorder=2)
                ax.add_patch(rect)

                # Label für gemischte Kassette (alle Profile)
                labeltext = (
                    f"Kassette {k_idx+1} | "
                    f"{kass.get('Profil(e)','')} | "
                    f"{kass['Kassettenhöhe']} mm | "
                    f"{kass['Kassettengewicht']:.1f} kg"
                )

                ax.text(
                    turmbreite/2, y_curr + kass['Kassettenhöhe']/2,
                    labeltext, va="center", ha="center", fontsize=6.5,
                    wrap=False, clip_on=True, zorder=10,
                )

                y_curr += kass['Kassettenhöhe'] + spielraum

            ax.set_xticks([])
            fig.tight_layout()
            figs.append(fig)
    return figs







# --- Streamlit-UI ---
st.set_page_config(page_title="Paternoster Optimierung", layout="wide")
st.title("Paternoster-Systeme Optimieren")

st.write("""
Lade deine Excel-Datei mit den Profilen hoch.
Wähle den Modus:
- **Sortenrein**: Nur ein Profiltyp pro Kassette (Empfehlung für echte Lagerhaltung)
- **Gemischt**: Möglichst wenig Kassetten und Türme, Profile werden gemischt gelagert (theoretisch maximale Raumausnutzung)
""")

modus = st.radio(
    "Modus wählen:",
    [
        "Sortenrein pro Kassette (Standard, a)",
        "Maximale Ausnutzung, gemischte Kassetten (b)"
    ],
    help="Sortenrein = nur ein Profil pro Kassette; Gemischt = alles zusammen, minimale Anzahl Paternoster"
)

uploaded_file = st.file_uploader("Excel-Datei auswählen", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("Datei erfolgreich geladen!")
        st.write("Vorschau deiner Daten:", df.head())

        if "Sortenrein" in modus:
            kassetten = verteile_profile_auf_kassetten(df)
            if kassetten:
                kassetten_df = pd.DataFrame(kassetten)
                gesamtkassetten = len(kassetten_df)
                gesamtgewicht = kassetten_df["gewicht"].sum()
                max_gewicht = kassetten_df["gewicht"].max()
                min_gewicht = kassetten_df["gewicht"].min()
                durchschn_gewicht = kassetten_df["gewicht"].mean()
                max_hoehe = kassetten_df["kassettenhoehe"].max()
                st.markdown("## Zusammenfassung (Kassetten)")
                st.write(f"- **Gesamtkassetten:** {gesamtkassetten}")
                st.write(f"- **Gesamtzuladung:** {gesamtgewicht:.1f} kg")
                st.write(f"- **Durchschnittliche Kassettenbeladung:** {durchschn_gewicht:.1f} kg")
                st.write(f"- **Höchste Kassettenbeladung:** {max_gewicht:.1f} kg")
                st.write(f"- **Leichteste Kassettenbeladung:** {min_gewicht:.1f} kg")
                st.write(f"- **Höchste Kassette:** {max_hoehe} mm")
            tuermeliste = tuerme_nacheinander_fuellen_sortenrein(kassetten)
            anzahl_doppeltuerme = len(tuermeliste)
            figs = plot_doppelturm_einzeln(tuermeliste)

# Spalten für nebeneinander-Darstellung
            cols = st.columns(2)
            for idx, fig in enumerate(figs):
                dt_num = idx // 2 + 1
                turm_num = idx % 2 + 1

                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
                buf.seek(0)
                with cols[idx % 2]:
                    with st.expander(f"Doppelturm {dt_num} – Turm {turm_num}", expanded=False):
                        st.image(buf, width=1100)
                plt.close(fig)


            st.markdown(f"### Benötigte Doppeltürme: **{anzahl_doppeltuerme}** (also **{anzahl_doppeltuerme*2}** einzelne Paternoster)")
            for i, dt in enumerate(tuermeliste, 1):
                for tnum, turm in enumerate([dt['turm1'], dt['turm2']], 1):
                    if not turm['kassetten']:
                        continue
                    with st.expander(f"Doppelturm {i} – Turm {tnum}: Höhe {turm['hoehe']:.0f} mm, Gewicht {turm['gewicht']:.1f} kg, Kassetten: {len(turm['kassetten'])}"):
                        kass_df = pd.DataFrame(turm['kassetten'])
                        kass_df.insert(0, "Kassette", ["Kassette "+str(i+1) for i in range(len(kass_df))])
                        st.dataframe(kass_df)
        else:
            # Variante b: gemischte Kassetten
            kassetten = gemischte_kassetten_bin_packing(df)
            if kassetten:
                kassetten_df = pd.DataFrame(kassetten)
                gesamtkassetten = len(kassetten_df)
                gesamtgewicht = kassetten_df["Kassettengewicht"].sum()
                max_gewicht = kassetten_df["Kassettengewicht"].max()
                min_gewicht = kassetten_df["Kassettengewicht"].min()
                durchschn_gewicht = kassetten_df["Kassettengewicht"].mean()
                max_hoehe = kassetten_df["Kassettenhöhe"].max()
                st.markdown("## Zusammenfassung (Kassetten)")
                st.write(f"- **Gesamtkassetten:** {gesamtkassetten}")
                st.write(f"- **Gesamtzuladung:** {gesamtgewicht:.1f} kg")
                st.write(f"- **Durchschnittliche Kassettenbeladung:** {durchschn_gewicht:.1f} kg")
                st.write(f"- **Höchste Kassettenbeladung:** {max_gewicht:.1f} kg")
                st.write(f"- **Leichteste Kassettenbeladung:** {min_gewicht:.1f} kg")
                st.write(f"- **Höchste Kassette:** {max_hoehe} mm")
            tuermeliste = tuerme_nacheinander_fuellen(kassetten)
            anzahl_doppeltuerme = len(tuermeliste)
            figs = plot_doppelturm_einzeln_gemischt(tuermeliste)
            cols = st.columns(2)
            for idx, fig in enumerate(figs):
                dt_num = idx // 2 + 1
                turm_num = idx % 2 + 1
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
                buf.seek(0)
                with cols[idx % 2]:
                    with st.expander(f"Doppelturm {dt_num} – Turm {turm_num}", expanded=False):
                        st.image(buf, width=1100)
                plt.close(fig)
            st.markdown(f"### Benötigte Doppeltürme: **{anzahl_doppeltuerme}** (also **{anzahl_doppeltuerme*2}** einzelne Paternoster)")
            for i, dt in enumerate(tuermeliste, 1):
                for tnum, turm in enumerate([dt['turm1'], dt['turm2']], 1):
                    if not turm['kassetten']:
                        continue
                    with st.expander(f"Doppelturm {i} – Turm {tnum}: Höhe {turm['hoehe']:.0f} mm, Gewicht {turm['gewicht']:.1f} kg, Kassetten: {len(turm['kassetten'])}"):
                        kass_df = pd.DataFrame(turm['kassetten'])
                        kass_df.insert(0, "Kassette", ["Kassette "+str(i+1) for i in range(len(kass_df))])
                        st.dataframe(kass_df)

        st.success("Optimierung abgeschlossen. Siehe Ergebnisse oben!")
    except Exception as e:

        st.error(f"Fehler beim Einlesen oder Berechnen: {e}")

