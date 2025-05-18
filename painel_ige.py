import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from wordcloud import WordCloud
from xhtml2pdf import pisa
import base64
import io

# Simulação de dados com base nos dashboards da IGE
def gerar_dados(seed=42, n=500):
    np.random.seed(seed)
    df = pd.DataFrame({
        "Ano": np.random.choice(range(2016, 2024), n),
        "TipoInspecao": np.random.choice(["Ordinária", "Extraordinária"], n),
        "UGE": np.random.choice(["UGE A", "UGE B", "UGE C", "UGE D"], n),
        "FonteFinanciamento": np.random.choice(["AF", "OE", "FSE"], n),
        "ClasseDespesa": np.random.choice(["Pessoal", "Material", "Serviços", "Outros"], n),
        "DiasInspecao": np.random.poisson(5, n),
        "NumeroCorrecoes": np.random.poisson(10, n),
        "EstadoCorrecao": np.random.choice(["Executada", "Em curso", "Não iniciada"], n),
        "DescricaoCorrecao": np.random.choice([
            "Ajustes no controlo interno de materiais",
            "Revisão de contratos administrativos",
            "Melhorias no processo de aquisição",
            "Atualização dos procedimentos contábeis",
            "Revisão das práticas de manutenção",
            "Aperfeiçoamento de sistemas de TI"
        ], n)
    })
    df["Risco"] = ((df["NumeroCorrecoes"] > 12) & (df["EstadoCorrecao"] != "Executada")).astype(int)
    return df

df = gerar_dados()

def treinar_modelo(df):
    x = df[["Ano", "DiasInspecao", "NumeroCorrecoes"]]
    y = df["Risco"]
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(x, y)
    df["RiscoPrevisto"] = modelo.predict(x)
    return modelo, modelo.feature_importances_, df

model, importancias, df = treinar_modelo(df)

def grafico_correcoes(df):
    return px.histogram(df, x="UGE", y="NumeroCorrecoes", color="EstadoCorrecao",
                        title="Correções por UGE", barmode="group")

def grafico_risco(df):
    return px.bar(df.groupby("UGE")["RiscoPrevisto"].mean().reset_index(),
                  x="UGE", y="RiscoPrevisto", title="Risco Médio Previsto por UGE")

def grafico_importancia(importancias):
    return px.bar(
        pd.DataFrame({
            "Variável": ["Ano", "DiasInspecao", "NumeroCorrecoes"],
            "Importância": importancias
        }).sort_values(by="Importância", ascending=False),
        x="Variável", y="Importância", title="Importância das Variáveis no Modelo de Risco"
    )

txt = " ".join(df["DescricaoCorrecao"])
cloud = WordCloud(width=400, height=200, background_color="white").generate(txt)
buf = io.BytesIO()
cloud.to_image().save(buf, format="PNG")
data_uri = base64.b64encode(buf.getvalue()).decode("utf-8")
img_src = f"data:image/png;base64,{data_uri}"

cenarios = pd.DataFrame({
    "Recursos": [10, 20, 30, 40, 50],
    "Capacidade": [15, 30, 45, 55, 60],
    "RiscoAgregado": [0.75, 0.6, 0.5, 0.42, 0.35]
})

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO])
app.title = "Painel IA - IGE"

app.layout = dbc.Container([
    dbc.Row(html.H1("📊 Sistema de Diagnóstico com IA – Inspeção-Geral do Exército", 
                className="text-center text-info fw-bold display-5 mb-4")),

    dbc.Row(html.H2("Painel Estratégico com IA – Inspeção-Geral do Exército", className="text-center text-info fw-bold")),

    dbc.Row([
        dbc.Col(dbc.Button("🛠️ Simular Correções Totais", id="btn-simulador", color="secondary", className="w-100 mb-3")),
        dbc.Col(html.Div(id="resultado-simulador", className="text-light fw-bold"))
    ]),
    dbc.Row([
        dcc.Store(id="store-figuras"),
        dbc.Col(dbc.Button("📄 Exportar Relatório (PDF)", id="btn-pdf-completo", color="info", className="w-100 mb-3")),
        dcc.Download(id="download-relatorio-pdf")
    ]),
    dbc.Row([
        dbc.Col(dbc.Button("🤖 Gerar Diagnóstico com IA", id="btn-gerar-relatorio", color="danger", className="w-100"), width=3),
        dbc.Col(html.Div(id="diagnostico-gerado", className="text-light fw-bold"), width=9)
    ], className="mb-2"),
    dbc.Row([
        dbc.Col(dbc.Button("Exportar Relatório em PDF", id="btn-exportar-pdf", color="light", className="mb-4 w-100"), width=3),
    ], justify="end"),

    dbc.Tooltip(
    "Gera um diagnóstico automático com base nos filtros aplicados",
    target="btn-gerar-relatorio",
    placement="right"
    ),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Total de Inspeções"),
            dbc.CardBody(html.H4(f"{len(df)}", className="text-success text-center fw-bold"))
        ]), md=3),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Total de Correções"),
            dbc.CardBody(html.H4(f"{df['NumeroCorrecoes'].sum()}", className="text-warning text-center fw-bold"))
        ]), md=3),
        dbc.Col(dbc.Card([
            dbc.CardHeader("% Correções Executadas"),
            dbc.CardBody(html.H4(f"{(df[df['EstadoCorrecao']=='Executada'].shape[0]/len(df)*100):.1f}%", className="text-info text-center fw-bold"))
        ]), md=3),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Risco Médio Previsto"),
            dbc.CardBody([
                html.H4(f"{df['RiscoPrevisto'].mean():.2f}", className="text-danger text-center fw-bold"),
                dbc.Alert("⚠️ ALTO RISCO MÉDIO DETECTADO", id="alerta-risco", color="danger", className="text-center fw-bold fs-5")
            ])
        ]), md=3)
    ], className="my-4"),

    dbc.Row([
        dbc.Col([html.Label("Ano"), dcc.Dropdown(sorted(df["Ano"].unique()), multi=True, id="filtro-ano")], md=4),
        dbc.Col([html.Label("UGE"), dcc.Dropdown(sorted(df["UGE"].unique()), multi=True, id="filtro-uge")], md=4),
        dbc.Col([html.Label("Estado da Correção"), dcc.Dropdown(df["EstadoCorrecao"].unique(), multi=True, id="filtro-estado")], md=4)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dbc.Button("Limpar Filtros", id="btn-resetar-filtros", color="secondary", className="w-100"))
    ], className="mb-3"),

    dbc.Row([dbc.Col(dcc.Graph(id="grafico-correcoes"), md=6),
             dbc.Col(dcc.Graph(id="grafico-risco"), md=6)]),
    dbc.Row([dbc.Col(dcc.Graph(id="grafico-velocimetro"), md=12)]),
    dbc.Row([dbc.Col(html.Div(id="lista-uges-risco"), md=12)]),
    dbc.Row([dbc.Col(dcc.Graph(id="grafico-importancia"), md=12)])
    ], fluid=True)
    

@app.callback(
    Output("resultado-simulador", "children"),
    Input("btn-simulador", "n_clicks"),
    prevent_initial_call=True
)
def simular_execucao_total(n):
    df_simulado = df.copy()
    df_simulado["EstadoCorrecao"] = "Executada"
    x_sim = df_simulado[["Ano", "DiasInspecao", "NumeroCorrecoes"]]
    df_simulado["RiscoPrevisto"] = model.predict(x_sim)
    risco_medio_simulado = df_simulado["RiscoPrevisto"].mean()
    return f"✅ Se todas as correções fossem executadas, o risco médio previsto seria reduzido para {risco_medio_simulado:.2f}."


@app.callback(
    Output("filtro-ano", "value"),
    Output("filtro-uge", "value"),
    Output("filtro-estado", "value"),
    Input("btn-resetar-filtros", "n_clicks"),
    prevent_initial_call=True
)
def resetar_filtros(n):
    return None, None, None

@app.callback(
    Output("download-relatorio-pdf", "data"),
    Input("btn-pdf-completo", "n_clicks"),
    State("diagnostico-gerado", "children"),
    State("store-figuras", "data"),
    prevent_initial_call=True
)
def exportar_pdf_completo(n, diagnostico, figs_data):
    from plotly.io import to_image
    from plotly.graph_objects import Figure
    import tempfile

    html_parts = [f"<h2>Diagnóstico IA</h2><p>{diagnostico}</p>"]
    for key in ["fig4", "fig1", "fig2", "fig3"]:
        fig = Figure(figs_data[key])
        img_bytes = to_image(fig, format="png")
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        html_parts.append(f'<img src="data:image/png;base64,{b64}" style="width:100%;">')
    html_content = "".join(html_parts)

    pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pisa.CreatePDF(io.StringIO(html_content), dest=pdf_file)

    return dcc.send_file(pdf_file.name)

@app.callback(
    Output("diagnostico-gerado", "children"),
    Input("btn-gerar-relatorio", "n_clicks"),
    State("filtro-ano", "value"),
    State("filtro-uge", "value"),
    State("filtro-estado", "value")
)
def gerar_relatorio(n, f_ano, f_uge, f_estado):
    if not n:
        return ""
    dff = df.copy()
    if f_ano: dff = dff[dff["Ano"].isin(f_ano)]
    if f_uge: dff = dff[dff["UGE"].isin(f_uge)]
    if f_estado: dff = dff[dff["EstadoCorrecao"].isin(f_estado)]
    media = dff["RiscoPrevisto"].mean() if not dff.empty else 0
    p_exec = dff[dff["EstadoCorrecao"]=="Executada"].shape[0] / dff.shape[0] if not dff.empty else 0
    nota = []
    if media > 0.8:
        nota.append("⚠️ Elevado risco médio detectado. Recomenda-se ação imediata nas UGEs críticas.")
    if p_exec < 0.5:
        nota.append("⚠️ Baixa taxa de execução das correções. Monitorização reforçada é recomendada.")
    if not nota:
        nota.append("✅ Situação sob controlo. Nenhum alerta crítico detectado.")
    return " ".join(nota)

@app.callback(
    Output("grafico-importancia", "figure"),
    Output("grafico-correcoes", "figure"),
    Output("grafico-risco", "figure"),
    Output("grafico-velocimetro", "figure"),
    Output("alerta-risco", "style"),
    Output("lista-uges-risco", "children"),
    Output("store-figuras", "data"),
    Input("filtro-ano", "value"),
    Input("filtro-uge", "value"),
    Input("filtro-estado", "value")
)
def actualizar_graficos(f_ano, f_uge, f_estado):
    dff = df.copy()
    if f_ano: dff = dff[dff["Ano"].isin(f_ano)]
    if f_uge: dff = dff[dff["UGE"].isin(f_uge)]
    if f_estado: dff = dff[dff["EstadoCorrecao"].isin(f_estado)]

    media_risco = dff["RiscoPrevisto"].mean() if not dff.empty else 0
    alerta_style = {"display": "block", "color": "yellow"} if media_risco > 0.8 else {"display": "none"}

    fig1 = grafico_correcoes(dff)
    fig2 = grafico_risco(dff)
    fig3 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=media_risco,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risco Médio (0–1)"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.4], 'color': "lightblue"},
                {'range': [0.4, 0.7], 'color': "orange"},
                {'range': [0.7, 1.0], 'color': "red"}
            ]
        }
    ))

    top_uges = dff.groupby("UGE")["RiscoPrevisto"].mean().reset_index()
    top_uges = top_uges.sort_values(by="RiscoPrevisto", ascending=False).head(5)
    lista_uges = html.Ul([
        html.Li(f"UGE: {row['UGE']} – Risco Médio: {row['RiscoPrevisto']:.2f}") for _, row in top_uges.iterrows()
    ])
    lista_header = html.Div([
        html.H5("Top 5 UGEs com Maior Risco Médio Previsto", className="text-warning fw-bold"),
        lista_uges
    ])

    fig4 = grafico_importancia(importancias)

    figuras_json = {
        "fig1": fig1.to_dict(),
        "fig2": fig2.to_dict(),
        "fig3": fig3.to_dict(),
        "fig4": fig4.to_dict()
    }

    return fig4, fig1, fig2, fig3, alerta_style, lista_header, figuras_json

app.layout.children.append(
    dbc.Row([
        dbc.Col(html.Img(
            src=img_src,
            style={"width": "40%", "border": "1px solid #ccc", "display": "block", "margin": "auto"}
        ))
    ], className="my-4")
)


app.layout.children.append(
    dbc.Row(html.Footer("Protótipo desenvolvido com Dash e IA aplicada ao controlo interno. © IGE 2025", 
                        className="text-center text-muted mt-5 mb-3"), justify="center")
)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host="0.0.0.0", port=port)

#if __name__ == '__main__':
   #app.run(debug=False, host="0.0.0.0", port=7860)


