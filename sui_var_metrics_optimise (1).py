import os
import json
from html import escape
from plotly.utils import PlotlyJSONEncoder
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from verticapy import *
from mlib.pyver.monitoring.profile_resources import profile_resources

class VariableMonitoring():

    def __init__(self):
        self.cnxn = current_connection()
        self.cursor = self.cnxn.cursor()

    def get_quartile_data(self, table_name: str):
        req = f"""
            SELECT IDT_PDE_MES_SCO, COD_PJT, NOM_VAR_MDE, LIB_CAT_STS, COD_TYP_OBJ, NBR_OBJ, VAL_BRN_SPR_QTL, LIB_RGP_SUI_MDE
            FROM {table_name}
            WHERE COD_TYP_OBJ IN ('S', 'P') AND COD_TYP_VAR = 'QNV'
        """
        self.cursor.execute(req)
        data = self.cursor.fetchall()
        
        def parse_p(p):
            p_str = str(p).strip().replace('.0', '')
            return datetime(int(p_str[1:5]), int(p_str[5:7]), 1) if len(p_str) >= 7 else datetime.now()

        def clean_val(val):
            return str(val).strip() if val is not None else ""

        res = [
            [
                clean_val(line[0]), 
                parse_p(line[0]), 
                clean_val(line[1]), 
                clean_val(line[2]), 
                clean_val(line[3]), 
                clean_val(line[4]), 
                int(line[5]) if line[5] is not None else 0,
                float(line[6]) if line[6] is not None else None,
                clean_val(line[7])
            ] 
            for line in data
        ]
        
        df_raw = pd.DataFrame(res, columns=["Periode_Raw", "Date", "Projet", "Variable", "Quartile", "Type_Obj", "Value", "Seuil_Raw", "Scope"])
        
        # Filtre strict sur GLOBAL et CIC
        df_raw = df_raw[df_raw['Scope'].isin(['GLOBAL', 'CIC'])]
        
        # pivot_table pour contrer l'erreur de doublons
        idx = ["Scope", "Projet", "Date", "Variable", "Quartile"]
        df_pivot = df_raw.pivot_table(index=idx, columns="Type_Obj", values="Value", aggfunc="sum").reset_index()
        
        df_pivot['P'] = df_pivot.get('P', 0).fillna(0).astype(int)
        df_pivot['S'] = df_pivot.get('S', 0).fillna(0).astype(int)
        df_pivot["Personnes"] = df_pivot["P"]
        df_pivot["Toppes"] = df_pivot["S"]
        df_pivot["Taux"] = (df_pivot["S"] / df_pivot["P"]).fillna(0)
        
        df_seuils = df_raw.groupby(["Scope", "Projet", "Variable", "Quartile"])["Seuil_Raw"].max().reset_index()
        df_seuils.rename(columns={"Seuil_Raw": "Seuil"}, inplace=True)
        
        df_pivot = pd.merge(df_pivot, df_seuils, on=["Scope", "Projet", "Variable", "Quartile"], how="left")
        
        return df_pivot.sort_values(by=["Scope", "Projet", "Variable", "Quartile", "Date"])

    def generate_chart(self, df_var, scope_name, project_name, var_name):
        quartiles = sorted(df_var['Quartile'].unique())
        metriques = ['Personnes', 'Toppes', 'Taux']
        colors = {'Q1': '#2ecc71', 'Q2': '#3498db', 'Q3': '#f1c40f', 'Q4': '#e67e22', 'Q5': '#e74c3c'}
        
        last_date = df_var['Date'].max()
        
        fig = go.Figure()
        
        for met in metriques:
            for q in quartiles:
                df_q = df_var[df_var['Quartile'] == q]
                
                if met in ['Toppes', 'Taux']:
                    df_q = df_q[df_q['Date'] < last_date]
                
                if df_q.empty:
                    continue
                
                fmt = ".3%" if met == 'Taux' else ",d"
                seuil_val = df_q['Seuil'].iloc[0] if not df_q['Seuil'].empty and pd.notna(df_q['Seuil'].iloc[0]) else "N/A"
                seuil_txt = f"{seuil_val:,.2f}" if isinstance(seuil_val, (int, float)) else "N/A"
                
                trace_name = f"Quartile {q}<br><sub style='font-size: 11px;'>Seuil: {seuil_txt}</sub>"
                
                fig.add_trace(go.Scatter(
                    x=df_q['Date'], 
                    y=df_q[met], 
                    mode='lines+markers', 
                    name=trace_name,
                    meta=met,
                    line=dict(width=2.5, color=colors.get(q, '#95a5a6')), 
                    marker=dict(size=6),
                    hovertemplate=(
                        f"<b>Quartile {q}</b> (Seuil: {seuil_txt})<br>"
                        f"Date: %{{x|%B %Y}}<br>"
                        f"Valeur: %{{y:{fmt}}}"
                        f"<extra></extra>"
                    )
                ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='-apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif', size=12, color="#1e293b"),
            title=dict(text=f"<b>{project_name} ({var_name}) - Périmètre : {scope_name}</b>", x=0.02, y=0.96, xanchor='left', yanchor='top'),
            margin=dict(l=50, r=20, t=50, b=120), height=500,
            legend=dict(
                orientation="h", 
                yanchor="top", 
                y=-0.18, 
                xanchor="center", 
                x=0.5, 
                font=dict(size=11),
                itemclick="toggle",
                itemdoubleclick="toggleothers"
            ), 
            hovermode="x unified"
        )
        fig.update_xaxes(showgrid=True, gridcolor='#e2e8f0', linecolor='#cbd5e1')
        fig.update_yaxes(showgrid=True, gridcolor='#e2e8f0', linecolor='#cbd5e1')
        return fig
    @profile_resources()
    def generate_html_report(self, table_name: str = 'MLINCM.TDSQSCO_MKT_VARIABLES'):
        df = self.get_quartile_data(table_name)
        
        scopes = sorted(df['Scope'].unique())
        projets = sorted(df['Projet'].unique())
        
        s_options = "".join([f'<option value="{idx}">{s}</option>\n' for idx, s in enumerate(scopes)])
        p_options = "".join([f'<option value="{idx}">{p}</option>\n' for idx, p in enumerate(projets)])
        
        # Les figures sont sérialisées, mais une seule sera rendue par Plotly.
        figures = {}
        variable_names = {}
        for s_idx, s in enumerate(scopes):
            for p_idx, p in enumerate(projets):
                df_sp = df[(df['Scope'] == s) & (df['Projet'] == p)]
                names = sorted(df_sp['Variable'].unique())
                key = f"{s_idx}_{p_idx}"
                variable_names[key] = names
                for v_idx, v in enumerate(names):
                    df_var = df_sp[df_sp['Variable'] == v]
                    fig = self.generate_chart(df_var, s, p, v)
                    figures[f"{key}_{v_idx}"] = fig.to_plotly_json()

        # Encodage compact, sans insertion accidentelle de balises HTML.
        figures_json = json.dumps(figures, cls=PlotlyJSONEncoder,
                                  ensure_ascii=True, separators=(',', ':')).replace('<', '\\u003c')
        names_json = json.dumps(variable_names, ensure_ascii=True,
                                separators=(',', ':')).replace('<', '\\u003c')
        now = datetime.now().strftime("%d/%m/%Y à %H:%M")
        
        html_content = f"""<!DOCTYPE html><html lang="fr"><head><meta charset="UTF-8"><title>Suivi Quartiles</title>
        <style>
            :root {{
                --bg-main: #f8fafc; --bg-card: #ffffff; --border-color: #e2e8f0;
                --text-main: #0f172a; --text-muted: #64748b;
                --primary: #2563eb; --primary-light: #eff6ff;
                --success: #10b981; --success-light: #ecfdf5;
            }}
            body {{ background-color: var(--bg-main); font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; margin: 0; padding: 0; color: var(--text-main); display: flex; flex-direction: column; height: 100vh; overflow: hidden; }}
            .top-bar {{ background: var(--bg-card); border-bottom: 1px solid var(--border-color); padding: 14px 30px; display: flex; justify-content: space-between; align-items: center; z-index: 10; flex-shrink: 0; }}
            .left-controls {{ display: flex; align-items: center; gap: 20px; }}
            .project-selector-group {{ display: flex; align-items: center; gap: 10px; }}
            .project-selector-group label {{ font-size: 11px; font-weight: 700; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.5px; }}
            .modern-select {{ min-width: 220px; padding: 8px 12px; font-size: 13px; font-weight: 600; border-radius: 6px; border: 1px solid var(--border-color); background-color: #fff; color: var(--text-main); cursor: pointer; outline: none; }}
            .modern-select:focus {{ border-color: var(--primary); }}
            .info-bar {{ display: flex; gap: 20px; font-size: 12px; color: var(--text-muted); border-left: 1px solid var(--border-color); padding-left: 20px; }}
            .info-label {{ font-weight: 600; color: var(--text-main); }}
            .metric-tabs {{ display: flex; background: #f1f5f9; padding: 4px; border-radius: 8px; gap: 2px; }}
            .met-btn {{ border: none; background: transparent; padding: 6px 14px; font-size: 13px; font-weight: 600; border-radius: 6px; cursor: pointer; color: var(--text-muted); transition: all 0.2s; }}
            .met-btn.active {{ background: var(--bg-card); color: var(--text-main); box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
            .workspace {{ flex: 1; display: flex; overflow: hidden; width: 100%; }}
            .sidebar-variables {{ width: 280px; background: var(--bg-card); border-right: 1px solid var(--border-color); display: flex; flex-direction: column; overflow-y: auto; padding: 16px; flex-shrink: 0; }}
            .sidebar-variables h2 {{ font-size: 11px; font-weight: 700; color: var(--text-muted); margin: 4px 0 12px 4px; letter-spacing: 0.5px; text-transform: uppercase; }}
            .v-btn {{ display: flex; align-items: center; width: 100%; text-align: left; background: transparent; border: none; padding: 6px 12px; margin-bottom: 2px; font-size: 13px; font-weight: 500; border-radius: 6px; cursor: pointer; color: var(--text-main); transition: all 0.15s ease; gap: 8px; }}
            .v-btn:hover {{ background: #f1f5f9; }}
            .v-btn.active {{ background: var(--success-light); color: var(--success); font-weight: 600; }}
            .dashboard-view {{ flex: 1; padding: 30px; overflow-y: auto; }}
            .chart-box {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); }}
            .scope-content {{ display: none; }}
            .scope-content.active-scope {{ display: block; }}
        </style>
        <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
        <script>
            const figures = {figures_json};
            const variableNames = {names_json};
            let currS = '0', currP = '0', currV = 0, currMetric = 'Personnes';
            const metrics = ['Personnes', 'Toppes', 'Taux'];

            function escapeHtml(value) {{
                return String(value).replace(/[&<>"']/g, ch => ({{
                    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
                }})[ch]);
            }}

            function renderVariables() {{
                const names = variableNames[currS + '_' + currP] || [];
                const sidebar = document.getElementById('variableButtons');
                sidebar.innerHTML = names.map((name, idx) =>
                    `<button class="v-btn ${{idx === currV ? 'active' : ''}}" data-index="${{idx}}">${{escapeHtml(name)}}</button>`
                ).join('');
                sidebar.querySelectorAll('button').forEach(btn => {{
                    btn.addEventListener('click', () => {{
                        currV = Number(btn.dataset.index);
                        renderVariables();
                        renderChart();
                    }});
                }});
            }}

            function renderChart() {{
                const target = document.getElementById('activeChart');
                const fig = figures[currS + '_' + currP + '_' + currV];
                if (!fig) {{
                    Plotly.purge(target);
                    target.textContent = 'Aucune donnée pour cette sélection.';
                    return;
                }}
                // N'instancie que les traces correspondant à la métrique choisie.
                const data = fig.data.filter(trace => trace.meta === currMetric);
                const layout = {{...fig.layout, yaxis: {{...fig.layout.yaxis,
                    tickformat: currMetric === 'Taux' ? '.2%' : ''}}}};
                Plotly.react(target, data, layout, {{displayModeBar: false, responsive: true}});
            }}

            function handleFilterChange() {{
                // Conserver le nom de la variable, pas son index (qui varie selon le périmètre).
                const previousNames = variableNames[currS + '_' + currP] || [];
                const selectedName = previousNames[currV];

                currS = document.getElementById('scopeSelect').value;
                currP = document.getElementById('projectSelect').value;

                const nextNames = variableNames[currS + '_' + currP] || [];
                const sameVariableIndex = nextNames.indexOf(selectedName);
                currV = sameVariableIndex >= 0 ? sameVariableIndex : 0;

                renderVariables();
                renderChart();
            }}

            function switchMetric(metricName, btn) {{
                currMetric = metricName;
                document.querySelectorAll('.met-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                renderChart();
            }}

            window.addEventListener('DOMContentLoaded', () => {{
                renderVariables();
                renderChart();
            }});
        </script></head><body>
            <div class="top-bar">
                <div class="left-controls">
                    <div class="project-selector-group">
                        <label>Périmètre :</label>
                        <select id="scopeSelect" class="modern-select" onchange="handleFilterChange()">
                            {s_options}
                        </select>
                    </div>
                    <div class="project-selector-group">
                        <label>Projet :</label>
                        <select id="projectSelect" class="modern-select" onchange="handleFilterChange()">
                            {p_options}
                        </select>
                    </div>
                    <div class="info-bar">
                        <div><span class="info-label">Source :</span> <code>{table_name}</code></div>
                        <div><span class="info-label">Périmètre :</span> GLOBAL / CIC</div>
                        <div><span class="info-label">MàJ :</span> {now}</div>
                    </div>
                </div>
                <div class="metric-tabs">
                    <button class="met-btn active" onclick="switchMetric('Personnes', this)">Effectif</button>
                    <button class="met-btn" onclick="switchMetric('Toppes', this)">Nb SCN</button>
                    <button class="met-btn" onclick="switchMetric('Taux', this)">Taux SCN</button>
                </div>
            </div>
            <div class="workspace">
                <div class="sidebar-variables">
                    <h2>Variables du projet</h2>
                    <div id="variableButtons"></div>
                </div>
                <div class="dashboard-view">
                    <div class="chart-box"><div id="activeChart"></div></div>
                </div>
            </div>
        </body></html>"""
        
        out = f"{os.environ.get('path_out', '.')}/quartile_monitoring.html"
        with open(out, "w", encoding="utf-8") as f: 
            f.write(html_content)
            
        print(f"✅ Rapport HTML complet généré : {out}")
        return html_content
