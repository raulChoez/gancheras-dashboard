"""
Dashboard ejecutivo — Estimación de tiempos de salida de gancheras en galvanizado
Maestría en Ciencia de Datos · ESPOL · Raúl Alejandro Chóez Vélez

Ejecutar:  streamlit run app.py
Requiere el CSV `Gancheras_Analitico_Corregido_Auditable_v1.csv` en la misma carpeta
(o subirlo desde la barra lateral).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ----------------------------------------------------------------------------------
# Configuración (proviene del notebook Taller_Modelizacion_Gancheras_V3)
# ----------------------------------------------------------------------------------
CSV_NOMBRE = "Gancheras_Analitico_Corregido_Auditable_v1.csv"
SEMILLA = 42
# Variables conservadas tras la eliminación iterativa por VIF (> 5)
F_BASE = ["turno", "numerganchera", "es_primera_ganchera", "lineas_material", "piezas_total", "peso_negro_total_kg",
          "peso_promedio_pieza_kg", "familias_distintas", "granallados_distintos", "normas_distintas", "espesores_distintos",
          "con_area_mayor", "proporcion_piezas_decapado", "proporcion_piezas_granallado_significativo", "dia_semana", "es_fin_semana"]
F_SEC = F_BASE + ["hora_ingreso_dec", "min_desde_inicio_turno", "cola_al_ingreso"]
# Mejores hiperparámetros de RandomizedSearchCV con TimeSeriesSplit (4 pliegues)
HIPER = {
    "RF directo — base (sin secuencia)": dict(feats=F_BASE, tipo="directo", n_estimators=200, min_samples_leaf=20, max_features=0.4, max_depth=None),
    "RF híbrido — base (sin secuencia)": dict(feats=F_BASE, tipo="residual", n_estimators=200, min_samples_leaf=5, max_features=0.6, max_depth=6),
    "RF directo + secuencia": dict(feats=F_SEC, tipo="directo", n_estimators=300, min_samples_leaf=5, max_features=0.85, max_depth=10),
    "RF híbrido + secuencia": dict(feats=F_SEC, tipo="residual", n_estimators=300, min_samples_leaf=5, max_features=1.0, max_depth=16),
}
MEJOR = "RF directo + secuencia"

# Paleta (validada para daltonismo): modelo/real = teal, heurística = terracota, tercera serie = azul
TEAL, TERRA, AZUL, NAVY, GRIS, TINTA, TINTA2 = "#008C82", "#C8602F", "#2F5F9E", "#1F3A5F", "#9AA0A8", "#1E2430", "#5A6270"
NOMBRES = {"numerganchera": "Posición en el turno", "cola_al_ingreso": "Cola al ingreso", "hora_ingreso_dec": "Hora de ingreso",
           "min_desde_inicio_turno": "Min. desde inicio del turno", "piezas_total": "Piezas", "peso_negro_total_kg": "Peso negro (kg)",
           "peso_promedio_pieza_kg": "Peso por pieza", "dia_semana": "Día de la semana", "es_primera_ganchera": "Primera ganchera",
           "es_fin_semana": "Fin de semana", "lineas_material": "Líneas de material", "familias_distintas": "Familias distintas",
           "granallados_distintos": "Granallados distintos", "normas_distintas": "Normas distintas", "espesores_distintos": "Espesores distintos",
           "con_area_mayor": "Con área mayor", "proporcion_piezas_decapado": "Prop. decapado",
           "proporcion_piezas_granallado_significativo": "Prop. granallado", "turno": "Turno"}
DIAS = {1: "Lunes", 2: "Martes", 3: "Miércoles", 4: "Jueves", 5: "Viernes", 6: "Sábado"}
ORDEN_HORA = ["Mañana (6–11)", "Mediodía (11–13)", "Tarde (13–19)", "Noche (19–24)", "Madrugada (0–6)"]

st.set_page_config(page_title="Gancheras · Dashboard ejecutivo", page_icon="⏱️", layout="wide")
st.markdown(f"""
<style>
.block-container {{padding-top: 1.6rem; padding-bottom: 2rem;}}
h1, h2, h3 {{color: {NAVY};}}
.kpi {{background:#F4F6F8; border-radius:10px; padding:14px 18px; height:100%;}}
.kpi .lbl {{color:{TINTA2}; font-size:0.85rem; margin-bottom:2px;}}
.kpi .val {{color:{TINTA}; font-size:1.55rem; font-weight:700; line-height:1.15;}}
.kpi .sub {{color:{TINTA2}; font-size:0.82rem; margin-top:2px;}}
.hall {{background:#F4F6F8; border-radius:10px; padding:14px 18px 6px 18px; margin-bottom:6px;}}
.hall h4 {{color:{NAVY}; margin:0 0 4px 0; font-size:1.08rem;}}
.hall p {{color:{TINTA}; margin:0 0 6px 0; font-size:0.92rem;}}
.tag {{display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px;}}
.bdg {{display:inline-block; font-size:0.72rem; font-weight:600; padding:1px 8px; border-radius:10px; margin-bottom:6px; white-space:nowrap;}}
.bdg.prueba {{background:#E0F2F0; color:#00665F; border:1px solid #9FD3CF;}}
.bdg.total {{background:#E6ECF4; color:#1F3A5F; border:1px solid #B9C7DA;}}
.nota {{color:{TINTA2}; font-size:0.85rem;}}
</style>""", unsafe_allow_html=True)


def fmt(x, d=1):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".").replace("-", "−")


def hhmm(h):
    h = h % 24
    return f"{int(h):02d}:{int(round((h % 1) * 60)) % 60:02d}"


def layout(fig, h=320, titulo=None, leyenda=True):
    fig.update_layout(separators=",.", height=h, margin=dict(l=10, r=10, t=40 if titulo else 10, b=10), plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(family="Calibri, Arial, sans-serif", size=13, color=TINTA), title=dict(text=titulo, font=dict(size=14, color=NAVY)),
                      showlegend=leyenda, legend=(dict(orientation="h", yanchor="top", y=-0.22, x=0) if titulo else dict(orientation="h", yanchor="bottom", y=1.02, x=0)),
                      hoverlabel=dict(bgcolor="white"))
    fig.update_xaxes(showgrid=False, linecolor="#C9D0D8", tickfont=dict(color=TINTA2))
    fig.update_yaxes(gridcolor="#ECEFF2", zeroline=False, tickfont=dict(color=TINTA2))
    return fig


def redondear(t):
    # El ERP guarda algunas horas como 08:04:59.999; se redondea al minuto más cercano
    return pd.Timestamp(t).round("min")


def kpi(col, lbl, val, sub="", conj=None):
    col.markdown(f'<div class="kpi">{badge(conj) if conj else ""}<div class="lbl">{lbl}</div><div class="val">{val}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)


N_INFO = {}


def badge(conj):
    if conj == "prueba":
        return f'<span class="bdg prueba">Conjunto de prueba · {N_INFO.get("prueba", "")}</span><br>'
    return f'<span class="bdg total">Conjunto total · {N_INFO.get("total", "")}</span><br>'


# ----------------------------------------------------------------------------------
# Datos (misma preparación que el notebook)
# ----------------------------------------------------------------------------------
@st.cache_data(show_spinner="Preparando datos…")
def preparar(fuente):
    raw = pd.read_csv(fuente, sep=None, engine="python")
    raw.columns = raw.columns.astype(str).str.replace("﻿", "", regex=False).str.strip()
    raw = raw.dropna(subset=["codigoprodgalv"]).copy()
    for c in ["tiempo_estimado_min", "tiempo_real_min", "error_firmado_min", "error_absoluto_min", "apto_modelado",
              "tiempo_preinmersion_min", "tiempo_inmersion_salida_min"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    for c in ["fecha_produccion", "hora_ingreso", "hora_inicio_inmersion", "hora_salida_real"]:
        raw[c] = pd.to_datetime(raw[c], errors="coerce", format="mixed")
    df = raw[raw.apto_modelado == 1].sort_values(["codigoprodgalv", "hora_ingreso", "numerganchera"]).reset_index(drop=True)
    df["hora_ingreso_dec"] = df.hora_ingreso.dt.hour + df.hora_ingreso.dt.minute / 60
    df["min_desde_inicio_turno"] = (df.hora_ingreso - df.groupby("codigoprodgalv").hora_ingreso.transform("min")).dt.total_seconds() / 60

    def cola(g):
        t, s = g.hora_ingreso.values, g.hora_inicio_inmersion.values
        return pd.Series([int(((t < ti) & (s > ti)).sum()) for ti in t], index=g.index)
    df["cola_al_ingreso"] = df.groupby("codigoprodgalv", group_keys=False).apply(cola)
    df["franja_posicion"] = pd.cut(df.numerganchera, [0, 1, 8, 12, np.inf], labels=["1", "2–8", "9–12", "13+"]).astype(str)
    h = df.hora_ingreso_dec
    df["franja_horaria"] = np.select([h < 6, h < 11, h < 13, h < 19], ["Madrugada (0–6)", "Mañana (6–11)", "Mediodía (11–13)", "Tarde (13–19)"], "Noche (19–24)")
    df["turno_nombre"] = df.turno.map({"D": "Diurno", "N": "Nocturno"})
    df["familia_principal"] = np.where(df.familias_presentes.str.contains(";"), "Mixta", df.familias_presentes.str.split(" ").str[0].str.title())
    return df.sort_values(["fecha_produccion", "hora_ingreso", "codigoprodgalv", "numerganchera"]).reset_index(drop=True)


def preproc(feats):
    num = [c for c in feats if c != "turno"]
    return ColumnTransformer([("num", SimpleImputer(strategy="median"), num),
                              ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), ["turno"])])


@st.cache_data(show_spinner=False)
def mca(df):
    """Análisis de Correspondencias Múltiples (mismo procedimiento que el notebook)."""
    activas = {"turno_nombre": "Turno", "franja_posicion": "Posición", "franja_horaria": "Hora", "familia_principal": "Familia",
               "con_granallado_significativo": "Granallado"}
    D = df[list(activas)].copy()
    D["con_granallado_significativo"] = D["con_granallado_significativo"].map({1: "Sí", 0: "No", 1.0: "Sí", 0.0: "No"}).fillna("No")
    D = D.astype(str).rename(columns=activas)
    Zi = pd.get_dummies(D, prefix_sep=": ").astype(float)
    Q = len(activas)
    P = Zi.values / Zi.values.sum(); r = P.sum(1); c = P.sum(0)
    S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
    U, sig, Vt = np.linalg.svd(S, full_matrices=False)
    lam = sig ** 2
    F = (U * sig) / np.sqrt(r)[:, None]
    G = (Vt.T * sig) / np.sqrt(c)[:, None]
    adj = np.where(lam > 1 / Q, ((Q / (Q - 1)) * (lam - 1 / Q)) ** 2, 0); pct = adj / adj.sum()
    coords = pd.DataFrame(G[:, :2], index=Zi.columns, columns=["Dim1", "Dim2"])
    coords["variable"] = [i.split(": ")[0] for i in coords.index]
    coords["n"] = Zi.sum().values.astype(int)
    sentido = pd.cut(df.error_firmado_min, [-np.inf, -10, 10, np.inf], labels=["Sobreestimada", "Precisa (±10)", "Subestimada"])
    sup = pd.DataFrame({k: F[(sentido == k).values, :2].mean(0) / sig[:2] for k in sentido.cat.categories}, index=["Dim1", "Dim2"]).T
    sup["n"] = sentido.value_counts().reindex(sup.index).values
    return coords, sup, pct[:2]


@st.cache_resource(show_spinner="Entrenando modelos (solo la primera vez)…")
def entrenar(df):
    fechas = np.array(sorted(df.fecha_produccion.unique()))
    corte = fechas[int(len(fechas) * 0.8)]
    train, test = df[df.fecha_produccion < corte].copy(), df[df.fecha_produccion >= corte].copy()
    modelos = {}
    for nombre, cfg in HIPER.items():
        y = train.tiempo_real_min if cfg["tipo"] == "directo" else train.tiempo_real_min - train.tiempo_estimado_min
        rf = RandomForestRegressor(n_estimators=cfg["n_estimators"], min_samples_leaf=cfg["min_samples_leaf"], max_features=cfg["max_features"],
                                   max_depth=cfg["max_depth"], random_state=SEMILLA, n_jobs=-1)
        modelos[nombre] = Pipeline([("pre", preproc(cfg["feats"])), ("model", rf)]).fit(train[cfg["feats"]], y)

    def predecir(nombre, datos):
        cfg = HIPER[nombre]; p = modelos[nombre].predict(datos[cfg["feats"]])
        return p if cfg["tipo"] == "directo" else datos.tiempo_estimado_min.values + p

    med_pos = train.groupby("numerganchera").tiempo_real_min.median()
    pred = {"Heurística ERP": test.tiempo_estimado_min.values,
            "Recalibración por posición": test.numerganchera.map(med_pos).fillna(train.tiempo_real_min.median()).values}
    for k in HIPER:
        pred[k] = predecir(k, test)
    real = test.tiempo_real_min.values
    filas = {}
    for k, v in pred.items():
        e = real - v
        filas[k] = {"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()), "Sesgo": e.mean(), "R²": r2_score(real, v),
                    "% dentro ±10 min": 100 * (np.abs(e) <= 10).mean()}
    met = pd.DataFrame(filas).T.sort_values("MAE")
    met["Mejora vs heurística (%)"] = 100 * (1 - met.MAE / met.loc["Heurística ERP", "MAE"])

    # Explicabilidad del mejor modelo
    pipe = modelos[MEJOR]; pre, rf = pipe.named_steps["pre"], pipe.named_steps["model"]
    nombres_x = [NOMBRES.get(c, c) for c in F_SEC if c != "turno"] + ["Turno: " + v for v in pre.named_transformers_["cat"].named_steps["oh"].categories_[0]]
    explainer = shap.TreeExplainer(rf)
    muestra = test.sample(min(600, len(test)), random_state=SEMILLA)
    Xm = pre.transform(muestra[F_SEC]); Xm = Xm.toarray() if hasattr(Xm, "toarray") else Xm
    sv = explainer.shap_values(Xm)
    shap_imp = pd.Series(np.abs(sv).mean(0), index=nombres_x).sort_values(ascending=False)
    resid = real - pred[MEJOR]
    return dict(modelos=modelos, corte=corte, train=train, test=test.assign(**{"pred_" + k: v for k, v in pred.items()}), met=met, explainer=explainer,
                nombres_x=nombres_x, shap_imp=shap_imp, q_resid=np.percentile(resid, [10, 90]), med_pos=med_pos)


# ----------------------------------------------------------------------------------
# Carga
# ----------------------------------------------------------------------------------
ruta = Path(__file__).parent / CSV_NOMBRE
with st.sidebar:
    st.markdown("### ⏱️ Gancheras")
    st.caption("Prototipo de dashboard ejecutivo · Maestría en Ciencia de Datos · ESPOL")
    subido = st.file_uploader("Dataset (CSV)", type="csv", help=f"Si no lo subes, se usa {CSV_NOMBRE} junto a app.py")
fuente = subido if subido is not None else (ruta if ruta.exists() else None)
if fuente is None:
    st.warning(f"Coloca **{CSV_NOMBRE}** junto a `app.py` o súbelo desde la barra lateral.")
    st.stop()

df = preparar(fuente)
R = entrenar(df)
met, test, train = R["met"], R["test"], R["train"]
H, B, BC = met.loc["Heurística ERP"], met.loc[MEJOR], met.loc["RF directo — base (sin secuencia)"]
pct_espera = 100 * df.tiempo_preinmersion_min.sum() / df.tiempo_real_min.sum()
piso = df.loc[df.numerganchera == 1, 'tiempo_preinmersion_min'].mean()  # primera ganchera: nadie delante
pct_sobre = 100 * (df.error_firmado_min < 0).mean()
miles = lambda v: f"{v:,}".replace(",", ".")
N_INFO.update(prueba=f"{miles(len(test))} gancheras", total=f"{miles(len(df))} gancheras")
pct_test = 100 * len(test) / len(df)

with st.sidebar:
    st.markdown("---")
    st.markdown(f"**Datos:** {miles(len(df))} gancheras aptas")
    st.markdown(f"**Período:** {df.fecha_produccion.min():%d/%m/%Y} – {df.fecha_produccion.max():%d/%m/%Y}")
    st.markdown("**División temporal de los datos**")
    st.markdown(f'{badge("total")}<span class="nota">todo el año: diagnóstico de la heurística</span>', unsafe_allow_html=True)
    st.markdown(f"- Entrenamiento: **{miles(len(train))}** gancheras ({fmt(100 - pct_test)} %) · hasta {pd.Timestamp(R['corte']) - pd.Timedelta(days=1):%d/%m/%Y}\n"
                f"- Prueba: **{miles(len(test))}** gancheras ({fmt(pct_test)} %) · desde {pd.Timestamp(R['corte']):%d/%m/%Y}")
    st.markdown(f'{badge("prueba")}<span class="nota">gancheras que el modelo nunca vio: comparación heurística vs. modelo</span>', unsafe_allow_html=True)
    st.markdown(f"**Modelo principal:** {MEJOR}")
    st.caption("Las cifras se calculan en vivo a partir del CSV; no hay valores precargados.")

st.markdown(f"<h1 style='margin-bottom:0'>¿Cuándo sale realmente una ganchera?</h1>", unsafe_allow_html=True)
st.markdown(f"<p class='nota' style='font-size:1rem'>Diagnóstico de la heurística del ERP y estimación explicable con aprendizaje automático · galvanizado por inmersión en caliente</p>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["① Hallazgos", "② Diagnóstico de la heurística", "③ Simulador de una ganchera", "④ Modelo y validación"])

# ==================================================================================
# 1. HALLAZGOS
# ==================================================================================
with tab1:
    st.markdown(f"### El tiempo de salida lo define la **secuencia** de gancheras, no el baño de zinc")
    st.markdown(f'<p class="nota">Cada indicador y gráfico indica con qué datos se calculó: {badge("total").replace("<br>", "")} = las {miles(len(df))} gancheras del año, '
                f'para describir la heurística · {badge("prueba").replace("<br>", "")} = las {miles(len(test))} gancheras más recientes ({fmt(pct_test, 0)} %), '
                f'que el modelo nunca vio, para comparar heurística y modelo en igualdad de condiciones.</p>', unsafe_allow_html=True)
    c = st.columns(4)
    kpi(c[0], "Error medio de la estimación (MAE)", f"{fmt(H.MAE)} → {fmt(B.MAE)} min", f"heurística → modelo · {fmt(B['Mejora vs heurística (%)'], 0)} % menos error", "prueba")
    kpi(c[1], "Gancheras con error ≤ ±10 min", f"{fmt(H['% dentro ±10 min'], 0)} % → {fmt(B['% dentro ±10 min'], 0)} %", "heurística → modelo", "prueba")
    kpi(c[2], "Gancheras sobreestimadas por la heurística", f"{fmt(pct_sobre)} %", f"sesgo medio {fmt(df.error_firmado_min.mean())} min", "total")
    kpi(c[3], "Del tiempo real ocurre antes del zinc (tinas 1–6)", f"{fmt(pct_espera)} %", f"zinc + enfriamiento (tinas 7–8): ~{fmt(df.tiempo_inmersion_salida_min.median(), 0)} min", "total")
    st.write("")

    a, b = st.columns(2)
    with a:
        st.markdown(f'<div class="hall">{badge("total")}<h4>1 · La heurística promete tarde: sobreestima 3 de cada 4 gancheras</h4>'
                    f'<p>Casi siempre da ~{fmt(df.tiempo_estimado_min.median(), 0)} min, sin importar la carga ni la fila. '
                    f'<b>Para la operación:</b> se comunican salidas más tardías de lo que realmente ocurren.</p></div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_histogram(x=df.tiempo_real_min, xbins=dict(start=0, end=240, size=5), name="Tiempo real", marker_color=TEAL, opacity=0.85,
                          hovertemplate="%{x} min: %{y} gancheras<extra>Real</extra>")
        fig.add_histogram(x=df.tiempo_estimado_min, xbins=dict(start=0, end=240, size=5), name="Estimado (heurística)", marker_color=TERRA, opacity=0.75,
                          hovertemplate="%{x} min: %{y} gancheras<extra>Heurística</extra>")
        fig.update_layout(barmode="overlay"); fig.update_xaxes(title="Minutos desde el ingreso", range=[0, 220]); fig.update_yaxes(title="Gancheras")
        st.plotly_chart(layout(fig, 300), width="stretch")
    with b:
        st.markdown(f'<div class="hall">{badge("total")}<h4>2 · El {fmt(pct_espera, 0)} % del tiempo ocurre antes del zinc</h4>'
                    f'<p>Del ingreso (tina 1) a la inmersión (tina 7): unos {fmt(piso, 0)} min de pretratamiento que tarda incluso la primera ganchera, '
                    f'más la espera por las gancheras que van delante. Zinc y enfriamiento: ~{fmt(df.tiempo_inmersion_salida_min.median(), 0)} min. '
                    f'<b>Para la operación:</b> lo que hace variar el tiempo entre gancheras es la cola, no el zinc.</p></div>', unsafe_allow_html=True)
        esp, inm = df.tiempo_preinmersion_min.mean(), df.tiempo_inmersion_salida_min.mean()
        fig = go.Figure()
        fig.add_bar(y=["Tiempo real"], x=[piso], orientation="h", name="Tinas 1–6 sin nadie delante (1.ª ganchera)", marker_color=NAVY,
                    text=[f"Pretratamiento ≈ {fmt(piso)} min"], textposition="inside", insidetextfont=dict(color="white", size=13),
                    hovertemplate="%{x:.1f} min<extra>Tinas 1–6 sin nadie delante</extra>")
        fig.add_bar(y=["Tiempo real"], x=[esp - piso], orientation="h", name="Espera por la cola", marker_color=AZUL,
                    text=[f"+{fmt(esp - piso)}"], textposition="inside", insidetextfont=dict(color="white", size=13),
                    hovertemplate="%{x:.1f} min<extra>Espera por la cola</extra>")
        fig.add_bar(y=["Tiempo real"], x=[inm], orientation="h", name="Zinc + enfriamiento (tinas 7–8)", marker_color=TERRA,
                    hovertemplate="%{x:.1f} min<extra>Zinc + enfriamiento</extra>")
        fig.update_layout(barmode="stack", bargap=0.45); fig.update_xaxes(title="Minutos promedio por ganchera")
        st.plotly_chart(layout(fig, 300).update_layout(legend_traceorder="normal"), width="stretch")

    a, b = st.columns(2)
    with a:
        st.markdown(f'<div class="hall">{badge("total")}<h4>3 · Las primeras gancheras del turno salen más rápido</h4>'
                    f'<p>El tiempo real depende de la posición en el turno; la heurística es plana. '
                    f'<b>Para la operación:</b> la posición y la cola deben entrar en la estimación.</p></div>', unsafe_allow_html=True)
        pos = df.groupby("numerganchera").agg(n=("tiempo_real_min", "size"), real=("tiempo_real_min", "mean"), est=("tiempo_estimado_min", "mean"))
        pos = pos[pos.n >= 30]
        fig = go.Figure()
        fig.add_scatter(x=pos.index, y=pos.real, name="Tiempo real", mode="lines+markers", line=dict(color=TEAL, width=2), marker=dict(size=8),
                        customdata=pos.n, hovertemplate="Posición %{x}: %{y:.1f} min (n=%{customdata})<extra>Real</extra>")
        fig.add_scatter(x=pos.index, y=pos.est, name="Heurística", mode="lines+markers", line=dict(color=TERRA, width=2), marker=dict(size=8),
                        hovertemplate="Posición %{x}: %{y:.1f} min<extra>Heurística</extra>")
        fig.update_xaxes(title="Posición de la ganchera en el turno (n ≥ 30)"); fig.update_yaxes(title="Minutos promedio")
        st.plotly_chart(layout(fig, 300), width="stretch")
    with b:
        st.markdown(f'<div class="hall">{badge("prueba")}<h4>4 · Con la secuencia, el error baja {fmt(B["Mejora vs heurística (%)"], 0)} %</h4>'
                    f'<p>Agregar cola, hora de ingreso y tiempo desde el inicio del turno es lo que mejora la estimación; la carga apenas aporta. '
                    f'<b>Para la operación:</b> registrar ingreso e inmersión en el momento hace posible usarlo.</p></div>', unsafe_allow_html=True)
        sel = ["Heurística ERP", "Recalibración por posición", "RF directo — base (sin secuencia)", MEJOR]
        etq = ["Heurística ERP", "Recalibración por posición", "RF · base (sin secuencia)", "RF · + secuencia"]
        col = [TERRA, GRIS, AZUL, TEAL]
        fig = go.Figure(go.Bar(x=[met.loc[s, "MAE"] for s in sel], y=etq, orientation="h", marker_color=col,
                               text=[f"{fmt(met.loc[s, 'MAE'])} min" for s in sel], textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}: %{x:.2f} min<extra></extra>"))
        fig.update_xaxes(title="Error absoluto medio en prueba (min)", range=[0, met.MAE.max() * 1.25]); fig.update_yaxes(autorange="reversed")
        st.plotly_chart(layout(fig, 300, leyenda=False), width="stretch")
    st.caption("Las asociaciones no implican causalidad: son hipótesis para validar con el personal de producción.")

# ==================================================================================
# 2. DIAGNÓSTICO
# ==================================================================================
with tab2:
    st.markdown("### ¿Dónde y cuándo se equivoca la heurística?")
    st.markdown(f'{badge("total").replace("<br>", "")} <span class="nota">Esta pestaña evalúa solo la heurística del ERP sobre todo el año (con los filtros que elijas); no usa el modelo.</span>', unsafe_allow_html=True)
    f = st.columns([2, 1.2, 1.6, 2])
    rango = f[0].date_input("Período", (df.fecha_produccion.min().date(), df.fecha_produccion.max().date()),
                            min_value=df.fecha_produccion.min().date(), max_value=df.fecha_produccion.max().date())
    turnos = f[1].multiselect("Turno", ["Diurno", "Nocturno"], default=["Diurno", "Nocturno"])
    fams = sorted(df.familia_principal.unique())
    familias = f[2].multiselect("Familia principal", fams, default=fams)
    franjas = f[3].multiselect("Franja horaria de ingreso", ORDEN_HORA, default=ORDEN_HORA)
    d = df.copy()
    if isinstance(rango, (tuple, list)) and len(rango) == 2:
        d = d[(d.fecha_produccion.dt.date >= rango[0]) & (d.fecha_produccion.dt.date <= rango[1])]
    d = d[d.turno_nombre.isin(turnos) & d.familia_principal.isin(familias) & d.franja_horaria.isin(franjas)]
    if d.empty:
        st.info("No hay gancheras con esos filtros."); st.stop()

    c = st.columns(4)
    kpi(c[0], "Gancheras", f"{len(d):,}".replace(",", "."), f"{d.codigoprodgalv.nunique()} producciones")
    kpi(c[1], "MAE de la heurística", f"{fmt(d.error_absoluto_min.mean())} min", f"global: {fmt(df.error_absoluto_min.mean())} min")
    kpi(c[2], "Sesgo medio (real − estimado)", f"{fmt(d.error_firmado_min.mean())} min", "negativo = promete más tarde")
    kpi(c[3], "Sobreestimadas", f"{fmt(100 * (d.error_firmado_min < 0).mean())} %", f"dentro de ±10 min: {fmt(100 * (d.error_absoluto_min <= 10).mean())} %")
    st.write("")

    a, b = st.columns(2)
    with a:
        hh = d.assign(h=d.hora_ingreso.dt.hour).groupby("h").agg(n=("error_firmado_min", "size"), sesgo=("error_firmado_min", "mean"))
        hh = hh[hh.n >= 15]
        fig = go.Figure(go.Bar(x=hh.index, y=hh.sesgo, marker_color=[TERRA if v > 0 else TEAL for v in hh.sesgo], customdata=hh.n,
                               hovertemplate="%{x}:00 h · sesgo %{y:.1f} min (n=%{customdata})<extra></extra>"))
        fig.add_hline(y=0, line_color=TINTA2, line_width=1)
        fig.update_xaxes(title="Hora de ingreso (horas con n ≥ 15)", dtick=1); fig.update_yaxes(title="Sesgo (min)")
        st.plotly_chart(layout(fig, 330, "Sesgo por hora de ingreso: barras naranjas = la heurística se queda corta", leyenda=False), width="stretch")
        st.markdown('<p class="nota">Cerca del mediodía la heurística pasa a <b>subestimar</b>: coincide con la pausa de almuerzo, que alarga la espera.</p>', unsafe_allow_html=True)
    with b:
        orden_p = ["1", "2–8", "9–12", "13+"]
        ht = d.groupby(["franja_posicion", "franja_horaria"]).agg(n=("error_firmado_min", "size"), s=("error_firmado_min", "mean")).reset_index()
        ht.loc[ht.n < 10, "s"] = np.nan
        piv = ht.pivot(index="franja_posicion", columns="franja_horaria", values="s").reindex(index=orden_p, columns=[o for o in ORDEN_HORA if o in franjas])
        pn = ht.pivot(index="franja_posicion", columns="franja_horaria", values="n").reindex(index=orden_p, columns=piv.columns)
        lim = np.nanmax(np.abs(piv.values)) if np.isfinite(piv.values).any() else 1
        fig = go.Figure(go.Heatmap(z=piv.values, x=[c.split(" (")[0] for c in piv.columns], y=["Pos. " + p for p in orden_p], zmin=-lim, zmax=lim,
                                   colorscale=[[0, TEAL], [0.5, "#EEF0F2"], [1, TERRA]], customdata=pn.values,
                                   text=[[("" if np.isnan(v) else fmt(v, 0)) for v in r] for r in piv.values], texttemplate="%{text}",
                                   hovertemplate="%{y} · %{x}: sesgo %{z:.1f} min (n=%{customdata})<extra></extra>", colorbar=dict(title="min")))
        st.plotly_chart(layout(fig, 330, "Sesgo medio por posición y franja (celdas con n ≥ 10)", leyenda=False), width="stretch")
        st.markdown('<p class="nota">Turquesa = sobreestima (promete tarde) · naranja = subestima. Las primeras posiciones de la mañana concentran la sobreestimación.</p>', unsafe_allow_html=True)

    mens = d.set_index("fecha_produccion").resample("MS").agg(n=("error_absoluto_min", "size"), MAE=("error_absoluto_min", "mean"), sesgo=("error_firmado_min", "mean"))
    mens = mens[mens.n > 0]
    fig = go.Figure()
    fig.add_scatter(x=mens.index, y=mens.MAE, name="MAE", mode="lines+markers", line=dict(color=AZUL, width=2), marker=dict(size=8), hovertemplate="%{x|%b %Y}: %{y:.1f} min<extra>MAE</extra>")
    fig.add_scatter(x=mens.index, y=mens.sesgo, name="Sesgo", mode="lines+markers", line=dict(color=TERRA, width=2), marker=dict(size=8), hovertemplate="%{x|%b %Y}: %{y:.1f} min<extra>Sesgo</extra>")
    fig.add_hline(y=0, line_color=TINTA2, line_width=1); fig.update_yaxes(title="Minutos")
    st.plotly_chart(layout(fig, 280, "Evolución mensual: el error es estable, es estructural"), width="stretch")
    st.markdown("#### Perfiles multivariantes (MCA): ¿qué características aparecen juntas?")
    st.markdown(f'{badge("total").replace("<br>", "")} <span class="nota">Calculado con todas las gancheras del año (no cambia con los filtros). '
                'Categorías <b>cercanas</b> suelen darse juntas; las del <b>centro</b> son comunes y no distinguen perfiles. '
                'Los rombos naranjas (sentido de la desviación) son <b>suplementarios</b>: no construyen el mapa, se ubican después para ver con qué perfiles se asocian.</span>',
                unsafe_allow_html=True)
    coords, sup, pct_mca = mca(df)
    col_var = {"Turno": NAVY, "Posición": TEAL, "Hora": "#6D597A", "Familia": GRIS, "Granallado": "#5B7553"}
    a, b = st.columns([2.2, 1], gap="large")
    with a:
        fig = go.Figure()
        POS_TXT = {"Hora: Mañana (6–11)": "top center", "Familia: Mixta": "bottom center", "Turno: Diurno": "middle right", "Posición: 9–12": "top left",
                   "Hora: Mediodía (11–13)": "bottom center", "Familia: Cañerias": "top right", "Familia: Cañerías": "top right", "Granallado: Sí": "bottom right",
                   "Granallado: No": "bottom left", "Familia: Laminas": "top left", "Familia: Láminas": "top left", "Familia: Placas": "top right",
                   "Hora: Madrugada (0–6)": "top left", "Turno: Nocturno": "top left", "Posición: 13+": "middle right"}
        PREF = {"Posición": "Pos. ", "Granallado": "Granallado ", "Turno": "Turno ", "Hora": "", "Familia": ""}
        for v, g in coords.groupby("variable", sort=False):
            fig.add_scatter(x=g.Dim1, y=g.Dim2, mode="markers+text", name=v, text=[PREF[v] + i.split(": ")[1].lower() if v in ("Turno", "Granallado") else PREF[v] + i.split(": ")[1] for i in g.index],
                            textposition=[POS_TXT.get(i, "top center") for i in g.index],
                            textfont=dict(size=11, color=col_var[v]), marker=dict(size=10, color=col_var[v], line=dict(color="white", width=1.5)),
                            customdata=g.n, hovertemplate="%{fullData.name}: %{text}<br>%{customdata} gancheras<extra></extra>")
        fig.add_scatter(x=sup.Dim1, y=sup.Dim2, mode="markers+text", name="Sentido de la desviación", text=sup.index, textposition=[{"Sobreestimada": "top right", "Precisa (±10)": "middle left", "Subestimada": "bottom center"}[i] for i in sup.index],
                        textfont=dict(size=12, color=TERRA), marker=dict(symbol="diamond", size=15, color=TERRA, line=dict(color="white", width=1.5)),
                        customdata=sup.n, hovertemplate="Desviación: %{text}<br>%{customdata} gancheras<extra>suplementaria</extra>")
        fig.add_hline(y=0, line_color="#C9D0D8", line_dash="dot"); fig.add_vline(x=0, line_color="#C9D0D8", line_dash="dot")
        fig.update_xaxes(title=f"Dim 1 ({fmt(pct_mca[0] * 100)} %): turno diurno ← → nocturno", zeroline=False)
        fig.update_yaxes(title=f"Dim 2 ({fmt(pct_mca[1] * 100)} %): turno avanzado ↓ ↑ inicio del turno", zeroline=False)
        st.plotly_chart(layout(fig, 520), width="stretch")
    with b:
        st.markdown(f"""
<div class="hall"><h4>Cómo leer el mapa</h4>
<p><b>Eje horizontal ({fmt(pct_mca[0] * 100)} %):</b> separa lo diurno (izquierda) de lo nocturno y la madrugada (derecha).</p>
<p><b>Eje vertical ({fmt(pct_mca[1] * 100)} %):</b> arriba el <b>inicio del turno</b> (posición 1, 2–8, mañana); abajo el <b>turno avanzado</b> (posición 13+, mediodía, tarde).</p>
<p><b>Sobreestimada</b> se ubica hacia el inicio del turno; <b>precisa</b> y <b>subestimada</b>, hacia el turno avanzado y el mediodía.</p>
<p><b>Familias y granallado</b> quedan cerca del centro: la carga no distingue perfiles de error.</p>
<p class="nota">Porcentajes con corrección de Benzécri. Los rombos están cerca del centro: la asociación es coherente pero moderada.</p></div>
""", unsafe_allow_html=True)

    with st.expander("Ver y descargar el detalle filtrado"):
        cols = ["fecha_produccion", "codigoprodgalv", "numerganchera", "turno_nombre", "hora_ingreso", "cola_al_ingreso", "familia_principal",
                "tiempo_estimado_min", "tiempo_real_min", "error_firmado_min"]
        st.dataframe(d[cols], width="stretch", hide_index=True)
        st.download_button("Descargar CSV", d[cols].to_csv(index=False).encode("utf-8"), "gancheras_filtradas.csv", "text/csv")

# ==================================================================================
# 3. SIMULADOR
# ==================================================================================
with tab3:
    st.markdown("### ¿Cuánto tardará esta ganchera? Heurística vs. modelo explicable")
    st.markdown(f'<span class="nota">Modelo entrenado con las {miles(len(train))} gancheras de entrenamiento; el rango probable proviene de sus errores en el </span>{badge("prueba").replace("<br>", "")}', unsafe_allow_html=True)
    st.markdown('<p class="nota">Ajusta las condiciones al momento del ingreso. El modelo (Random Forest + secuencia) predice en vivo y explica qué variables suben o bajan el tiempo.</p>', unsafe_allow_html=True)
    izq, der = st.columns([1, 1.6], gap="large")
    med = train[F_SEC].median(numeric_only=True)
    with izq:
        turno = st.radio("Turno", ["Diurno", "Nocturno"], horizontal=True)
        tcod = "D" if turno == "Diurno" else "N"
        posicion = st.slider("Posición en el turno", 1, 30, 5)
        sub = train[(train.turno == tcod) & (train.numerganchera == posicion)]
        ref = sub if len(sub) >= 10 else train[train.turno == tcod]
        opciones = [i / 4 for i in range(96)]
        hora = st.select_slider("Hora de ingreso", opciones, value=float(round(ref.hora_ingreso_dec.median() * 4) / 4), format_func=hhmm,
                                key=f"hora_{tcod}_{posicion}")
        cola_v = st.slider("Gancheras en cola al ingresar", 0, 12, int(ref.cola_al_ingreso.median()), key=f"cola_{tcod}_{posicion}")
        mdi = st.slider("Minutos desde el inicio del turno", 0, 900, int(ref.min_desde_inicio_turno.median()), 5, key=f"mdi_{tcod}_{posicion}")
        with st.expander("Carga y día (influyen poco)"):
            peso = st.number_input("Peso negro (kg)", 1.0, 5000.0, float(round(med.peso_negro_total_kg)), 10.0)
            piezas = st.number_input("Piezas", 1, 2000, int(med.piezas_total))
            dia = st.selectbox("Día", list(DIAS.values()), index=2)
            granallado = st.slider("Proporción de piezas con granallado significativo", 0.0, 1.0, 0.0, 0.05)
    x = pd.DataFrame([med]).copy()
    x["turno"] = tcod; x["numerganchera"] = posicion; x["es_primera_ganchera"] = int(posicion == 1)
    x["hora_ingreso_dec"] = hora; x["cola_al_ingreso"] = cola_v; x["min_desde_inicio_turno"] = mdi
    x["peso_negro_total_kg"] = peso; x["piezas_total"] = piezas; x["peso_promedio_pieza_kg"] = peso / piezas
    dnum = [k for k, v in DIAS.items() if v == dia][0]; x["dia_semana"] = dnum; x["es_fin_semana"] = int(dnum == 6)
    x["proporcion_piezas_granallado_significativo"] = granallado
    modelo = R["modelos"][MEJOR]
    pred = float(modelo.predict(x[F_SEC])[0])
    lo, hi = pred + R["q_resid"][0], pred + R["q_resid"][1]
    heur = float(ref.tiempo_estimado_min.median())
    Xs = modelo.named_steps["pre"].transform(x[F_SEC]); Xs = Xs.toarray() if hasattr(Xs, "toarray") else Xs
    expl = shap.TreeExplainer(modelo.named_steps["model"])
    sv = expl.shap_values(Xs)[0]; base = float(np.ravel(expl.expected_value)[0])
    with der:
        c = st.columns(3)
        kpi(c[0], "Heurística ERP (valor típico)", f"{fmt(heur, 0)} min", f"salida ≈ {hhmm(hora + heur / 60)}")
        kpi(c[1], "Modelo explicable", f"{fmt(pred, 0)} min", f"salida ≈ {hhmm(hora + pred / 60)}")
        kpi(c[2], "Rango probable (80 %)", f"{fmt(max(lo, 0), 0)}–{fmt(hi, 0)} min", f"{hhmm(hora + max(lo, 0) / 60)} – {hhmm(hora + hi / 60)}")
        st.write("")
        contrib = pd.Series(sv, index=R["nombres_x"])
        contrib = contrib.groupby(lambda k: "Turno" if k.startswith("Turno") else k).sum()
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:7][::-1]
        fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color=[TERRA if v > 0 else TEAL for v in top.values],
                               text=[("+" if v > 0 else "") + fmt(v) + " min" for v in top.values], textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}: %{x:+.1f} min<extra></extra>"))
        fig.add_vline(x=0, line_color=TINTA2, line_width=1)
        m = top.abs().max() * 1.35
        fig.update_xaxes(title=f"Minutos que suma (naranja) o resta (turquesa) respecto al promedio de {fmt(base, 0)} min", range=[-m, m])
        st.plotly_chart(layout(fig, 330, "¿Por qué el modelo predice esto? (SHAP)", leyenda=False), width="stretch")
        sube = top[top >= 1].sort_values(ascending=False); baja = top[top <= -1].sort_values()
        frase = []
        if len(sube): frase.append(f"lo que más **alarga** la espera es *{sube.index[0].lower()}* (+{fmt(sube.iloc[0])} min)")
        if len(baja): frase.append(f"lo que más la **acorta** es *{baja.index[0].lower()}* ({fmt(baja.iloc[0])} min)")
        if frase:
            st.markdown("En palabras simples: " + " y ".join(frase) + ".")
        else:
            st.markdown("En palabras simples: ninguna variable mueve la predicción más de 1 minuto respecto al promedio.")

    st.markdown("---")
    st.markdown("#### Revisar una ganchera real del período de prueba")
    st.markdown(f'{badge("prueba").replace("<br>", "")} <span class="nota">Solo gancheras que el modelo no vio al entrenar: su error es un resultado honesto.</span>', unsafe_allow_html=True)
    info_prod = (test.groupby(test.codigoprodgalv.astype(int))
                 .agg(fecha=("fecha_produccion", "min"), turno=("turno_nombre", "first"), n=("numerganchera", "size")))
    prods = info_prod.sort_values(["fecha", "turno"], ascending=[False, True]).index.tolist()
    etiqueta_prod = lambda p: f"{p} · {info_prod.loc[p, 'fecha']:%d/%m/%Y} · {info_prod.loc[p, 'turno']} · {info_prod.loc[p, 'n']} gancheras"
    c1, c2 = st.columns([1, 1])
    prod = c1.selectbox("Producción (id · fecha · turno)", prods, format_func=etiqueta_prod, help="Puedes escribir el id o la fecha para buscar")
    gs = test[test.codigoprodgalv == prod].sort_values("numerganchera")
    g_n = c2.selectbox("Ganchera (posición en el turno)", gs.numerganchera.astype(int).tolist())
    g = gs[gs.numerganchera == g_n].iloc[0]
    c = st.columns(4)
    kpi(c[0], "Ingreso", f"{redondear(g.hora_ingreso):%H:%M}", f"{g.fecha_produccion:%d/%m/%Y} · {g.turno_nombre} · cola {int(g.cola_al_ingreso)}")
    kpi(c[1], "Tiempo real", f"{fmt(g.tiempo_real_min, 0)} min", f"salida {redondear(g.hora_salida_real):%H:%M}")
    kpi(c[2], "Heurística ERP", f"{fmt(g.tiempo_estimado_min, 0)} min", f"error {fmt(g.tiempo_real_min - g.tiempo_estimado_min, 0)} min")
    kpi(c[3], "Modelo explicable", f"{fmt(g['pred_' + MEJOR], 0)} min", f"error {fmt(g.tiempo_real_min - g['pred_' + MEJOR], 0)} min")

# ==================================================================================
# 4. MODELO Y VALIDACIÓN
# ==================================================================================
with tab4:
    st.markdown("### ¿Qué tan confiable es el modelo?")
    st.markdown(f'{badge("prueba").replace("<br>", "")} <span class="nota">Todas las métricas y gráficos de esta pestaña se calculan sobre las mismas {miles(len(test))} gancheras de prueba.</span>', unsafe_allow_html=True)
    a, b = st.columns([1.25, 1], gap="large")
    with a:
        tabla = met.rename(columns={"Mejora vs heurística (%)": "Mejora %", "% dentro ±10 min": "±10 min %"})
        tabla.index.name = "Modelo"
        st.dataframe(tabla.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "Sesgo": "{:.2f}", "R²": "{:.2f}", "±10 min %": "{:.1f}",
                                         "Mejora %": "{:.1f}"}), width="stretch")
        st.markdown(f'<p class="nota">Métricas en minutos sobre las mismas {miles(len(test))} gancheras de prueba (las más recientes, desde {pd.Timestamp(R["corte"]):%d/%m/%Y}). '
                    "MAE = error absoluto medio; sesgo = real − predicho (negativo = sobreestima).</p>", unsafe_allow_html=True)
        st.markdown("#### Cómo se validó")
        st.markdown(f"""
- **Partición temporal 80/20 por fecha:** se entrena con el pasado ({len(train)} gancheras) y se evalúa con el futuro ({len(test)}), como funcionaría en planta.
- **Validación cruzada temporal** (`TimeSeriesSplit`, 4 pliegues por días) y **búsqueda aleatoria de hiperparámetros** optimizando MAE (en el notebook).
- **Variables:** conjunto **base** de 16 variables seleccionadas por VIF (carga, contexto y posición en el turno, ya registradas en el ERP) y conjunto **+ secuencia**, que agrega 3 variables construidas a partir de las horas y disponibles al ingreso: hora de ingreso, minutos desde el inicio del turno y cola al ingreso.
- **Rango probable del simulador:** percentiles 10 y 90 de los residuos del modelo en prueba ({fmt(R['q_resid'][0])} y +{fmt(R['q_resid'][1])} min).
""")
    with b:
        imp = R["shap_imp"].groupby(lambda k: "Turno" if k.startswith("Turno") else k).sum().sort_values(ascending=False)[:8][::-1]
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=AZUL, text=[fmt(v) for v in imp.values],
                               textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:.2f} min<extra></extra>"))
        fig.update_xaxes(title="Aporte medio |SHAP| (min)", range=[0, imp.max() * 1.25])
        st.plotly_chart(layout(fig, 360, f"Variables más influyentes · {MEJOR}", leyenda=False), width="stretch")
        fig = go.Figure()
        fig.add_scatter(x=test.tiempo_real_min, y=test["pred_Heurística ERP"], mode="markers", name="Heurística", marker=dict(color=TERRA, size=8, opacity=0.35),
                        hovertemplate="real %{x:.0f} · estimado %{y:.0f}<extra>Heurística</extra>")
        fig.add_scatter(x=test.tiempo_real_min, y=test["pred_" + MEJOR], mode="markers", name="Modelo", marker=dict(color=TEAL, size=8, opacity=0.45),
                        hovertemplate="real %{x:.0f} · predicho %{y:.0f}<extra>Modelo</extra>")
        lim = float(np.percentile(test.tiempo_real_min, 99.5))
        fig.add_scatter(x=[0, lim], y=[0, lim], mode="lines", line=dict(color=TINTA2, dash="dash", width=1), showlegend=False, hoverinfo="skip")
        fig.update_xaxes(title="Tiempo real (min)", range=[0, lim]); fig.update_yaxes(title="Estimado / predicho (min)", range=[0, lim])
        st.plotly_chart(layout(fig, 340, "Predicho vs. real (la diagonal es el acierto perfecto)"), width="stretch")
    st.markdown("#### ¿Qué tan seguido acierta cada modelo según tu tolerancia? (curva REC)")
    st.markdown('<p class="nota">Para cada margen de error permitido (eje horizontal), el porcentaje de gancheras de prueba cuya estimación quedó dentro de ese margen. '
                'La curva más alta es mejor para cualquier tolerancia. Elige el margen que tu operación considera aceptable.</p>', unsafe_allow_html=True)
    tol = st.slider("Tolerancia aceptable (± minutos)", 1, 30, 10, 1)
    real_t = test.tiempo_real_min.values
    rec_mod = [("Heurística ERP", "Heurística ERP", TERRA, "dash"), ("Recalibración por posición", "Recalibración por posición", GRIS, "dot"),
               ("RF directo — base (sin secuencia)", "RF · base (sin secuencia)", AZUL, "solid"), (MEJOR, "RF · + secuencia", TEAL, "solid")]
    umbrales = np.arange(0, 31)
    cc = st.columns(4)
    fig = go.Figure()
    for i, (k, etq, colr, dash) in enumerate(rec_mod):
        err = np.abs(real_t - test["pred_" + k].values)
        curva = [100 * (err <= u).mean() for u in umbrales]
        fig.add_scatter(x=umbrales, y=curva, name=etq, mode="lines", line=dict(color=colr, width=2.5, dash=dash),
                        hovertemplate="±%{x} min: %{y:.1f} %<extra>" + etq + "</extra>")
        kpi(cc[i], etq, f"{fmt(100 * (err <= tol).mean())} %", f"gancheras con error ≤ ±{tol} min")
    fig.add_vline(x=tol, line_color=TINTA2, line_dash="dot", annotation_text=f"±{tol} min", annotation_position="bottom right")
    fig.update_xaxes(title="Tolerancia: error absoluto permitido (± minutos)", dtick=5, range=[0, 30])
    fig.update_yaxes(title="% de gancheras dentro de la tolerancia", range=[0, 100], ticksuffix=" %")
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(layout(fig, 380), width="stretch")
    st.caption("La recalibración sube en escalones porque predice medianas que, como los tiempos reales, están en múltiplos de 5 min. El umbral de ±10 min usado en los indicadores es un criterio de análisis; "
               "la tolerancia definitiva debe acordarse con producción según el uso de la estimación.")
    st.info("**Limitaciones:** asociación no implica causalidad · los tiempos reales se registran en múltiplos de 5 min · un año de datos de una planta · "
            "las variables de secuencia requieren registrar ingreso e inmersión en el momento en que ocurren.")
