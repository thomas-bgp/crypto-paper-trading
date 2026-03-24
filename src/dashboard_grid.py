"""
Dashboard de resultados do Grid Search — Visualização completa.
Roda com: python dashboard_grid.py
Abre no browser em http://localhost:8051
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')


def load_results():
    path = os.path.join(RESULTS_DIR, 'grid_search_results.csv')
    df = pd.read_csv(path)
    df = df.dropna(subset=['sharpe', 'cagr'])
    # Clean extreme outliers for visualization
    df = df[df['cagr'] < 500]
    df = df[df['cagr'] > -100]
    return df


def create_dashboard():
    df = load_results()
    yr_cols = sorted([c for c in df.columns if c.startswith('yr_')])

    # ═══════════════════════════════════════════════════════
    # FIGURE 1: Sharpe vs CAGR scatter (colored by direction)
    # ═══════════════════════════════════════════════════════
    fig1 = px.scatter(
        df, x='cagr', y='sharpe',
        color='direction',
        size=df['max_dd'].abs().clip(upper=80),
        hover_data=['lookback_days', 'holding_days', 'top_n', 'universe_top',
                    'trailing_stop_pct', 'regime_filter', 'max_dd', 'calmar'],
        title='Sharpe vs CAGR — All 1,344 Configurations',
        labels={'cagr': 'CAGR (%)', 'sharpe': 'Sharpe Ratio'},
        color_discrete_map={'long_only': '#2196F3', 'long_short': '#FF5722'},
        opacity=0.6,
    )
    fig1.update_layout(height=600, template='plotly_dark')

    # ═══════════════════════════════════════════════════════
    # FIGURE 2: Sharpe distribution by direction
    # ═══════════════════════════════════════════════════════
    fig2 = go.Figure()
    for direction, color in [('long_only', '#2196F3'), ('long_short', '#FF5722')]:
        sub = df[df['direction'] == direction]
        fig2.add_trace(go.Histogram(
            x=sub['sharpe'], name=direction, opacity=0.7,
            marker_color=color, nbinsx=40,
        ))
    fig2.update_layout(
        title='Distribuição de Sharpe Ratio por Direção',
        xaxis_title='Sharpe Ratio', yaxis_title='Count',
        barmode='overlay', height=400, template='plotly_dark',
    )
    fig2.add_vline(x=1.0, line_dash="dash", line_color="yellow",
                   annotation_text="Sharpe=1.0")
    fig2.add_vline(x=0.5, line_dash="dot", line_color="gray",
                   annotation_text="Sharpe=0.5")

    # ═══════════════════════════════════════════════════════
    # FIGURE 3: Heatmap — Lookback vs Holding (mean Sharpe)
    # ═══════════════════════════════════════════════════════
    for direction in ['long_only', 'long_short']:
        sub = df[df['direction'] == direction]
        pivot = sub.pivot_table(values='sharpe', index='lookback_days',
                                columns='holding_days', aggfunc='mean')
        if direction == 'long_only':
            fig3a = go.Figure(go.Heatmap(
                z=pivot.values, x=pivot.columns.astype(str),
                y=pivot.index.astype(str),
                colorscale='RdYlGn', text=pivot.values.round(2),
                texttemplate='%{text}', zmin=0, zmax=2.5,
            ))
            fig3a.update_layout(
                title=f'Mean Sharpe: Lookback × Holding (Long-Only)',
                xaxis_title='Holding Days', yaxis_title='Lookback Days',
                height=400, template='plotly_dark',
            )
        else:
            fig3b = go.Figure(go.Heatmap(
                z=pivot.values, x=pivot.columns.astype(str),
                y=pivot.index.astype(str),
                colorscale='RdYlGn', text=pivot.values.round(2),
                texttemplate='%{text}', zmin=0, zmax=2.5,
            ))
            fig3b.update_layout(
                title=f'Mean Sharpe: Lookback × Holding (Long-Short)',
                xaxis_title='Holding Days', yaxis_title='Lookback Days',
                height=400, template='plotly_dark',
            )

    # ═══════════════════════════════════════════════════════
    # FIGURE 4: Max DD vs Sharpe (risk frontier)
    # ═══════════════════════════════════════════════════════
    fig4 = px.scatter(
        df, x='max_dd', y='sharpe',
        color='direction',
        hover_data=['lookback_days', 'holding_days', 'top_n', 'cagr', 'calmar'],
        title='Efficient Frontier: Max Drawdown vs Sharpe',
        labels={'max_dd': 'Max Drawdown (%)', 'sharpe': 'Sharpe Ratio'},
        color_discrete_map={'long_only': '#2196F3', 'long_short': '#FF5722'},
        opacity=0.6,
    )
    fig4.update_layout(height=500, template='plotly_dark')
    fig4.add_hline(y=1.0, line_dash="dash", line_color="yellow")
    fig4.add_vline(x=-20, line_dash="dash", line_color="yellow",
                   annotation_text="DD=-20%")

    # ═══════════════════════════════════════════════════════
    # FIGURE 5: Box plots by parameter
    # ═══════════════════════════════════════════════════════
    fig5 = make_subplots(rows=2, cols=3,
                         subplot_titles=['Lookback Days', 'Holding Days', 'Top N',
                                        'Universe Top', 'Trailing Stop %', 'Regime Filter'])
    params = ['lookback_days', 'holding_days', 'top_n',
              'universe_top', 'trailing_stop_pct', 'regime_filter']
    for i, param in enumerate(params):
        row, col = divmod(i, 3)
        for direction, color in [('long_only', '#2196F3'), ('long_short', '#FF5722')]:
            sub = df[df['direction'] == direction]
            fig5.add_trace(go.Box(
                x=sub[param].astype(str), y=sub['sharpe'],
                name=direction, marker_color=color, showlegend=(i == 0),
            ), row=row + 1, col=col + 1)
    fig5.update_layout(height=700, template='plotly_dark',
                       title='Sharpe por Parâmetro')

    # ═══════════════════════════════════════════════════════
    # FIGURE 6: Yearly returns heatmap (top 20 configs)
    # ═══════════════════════════════════════════════════════
    top20 = df.nlargest(20, 'sharpe')
    labels = [f"{r['direction'][:2].upper()}_lb{r['lookback_days']}_h{r['holding_days']}_n{r['top_n']}"
              for _, r in top20.iterrows()]
    yr_data = top20[yr_cols].values
    yr_labels = [c.replace('yr_', '') for c in yr_cols]

    fig6 = go.Figure(go.Heatmap(
        z=yr_data, x=yr_labels, y=labels,
        colorscale='RdYlGn', zmid=0,
        text=yr_data.round(1), texttemplate='%{text}%',
    ))
    fig6.update_layout(
        title='Retornos Anuais — Top 20 Configs por Sharpe',
        xaxis_title='Ano', yaxis_title='Config',
        height=600, template='plotly_dark',
    )

    # ═══════════════════════════════════════════════════════
    # FIGURE 7: Calmar (CAGR/MaxDD) distribution
    # ═══════════════════════════════════════════════════════
    df_calmar = df[df['calmar'].abs() < 50]  # filter extreme
    fig7 = px.histogram(
        df_calmar, x='calmar', color='direction',
        nbins=50, barmode='overlay', opacity=0.7,
        title='Distribuição de Calmar Ratio (CAGR / Max DD)',
        labels={'calmar': 'Calmar Ratio'},
        color_discrete_map={'long_only': '#2196F3', 'long_short': '#FF5722'},
    )
    fig7.update_layout(height=400, template='plotly_dark')

    # ═══════════════════════════════════════════════════════
    # FIGURE 8: Summary stats table
    # ═══════════════════════════════════════════════════════
    summary_data = []
    for direction in ['long_only', 'long_short']:
        sub = df[df['direction'] == direction]
        summary_data.append({
            'Direction': direction,
            'Count': len(sub),
            'Mean Sharpe': f"{sub['sharpe'].mean():.2f}",
            'Median Sharpe': f"{sub['sharpe'].median():.2f}",
            'Best Sharpe': f"{sub['sharpe'].max():.2f}",
            'Mean CAGR': f"{sub['cagr'].mean():.0f}%",
            'Best CAGR': f"{sub['cagr'].max():.0f}%",
            'Mean MaxDD': f"{sub['max_dd'].mean():.0f}%",
            'Best MaxDD': f"{sub['max_dd'].max():.0f}%",
            '% Sharpe>1': f"{(sub['sharpe']>1).mean()*100:.0f}%",
            'Mean Calmar': f"{sub['calmar'].mean():.1f}",
        })
    summary_df = pd.DataFrame(summary_data)

    fig8 = go.Figure(go.Table(
        header=dict(values=list(summary_df.columns),
                    fill_color='#1e1e1e', font=dict(color='white', size=13),
                    align='center'),
        cells=dict(values=[summary_df[c] for c in summary_df.columns],
                   fill_color=[['#0d47a1', '#bf360c']],
                   font=dict(color='white', size=12),
                   align='center'),
    ))
    fig8.update_layout(title='Resumo por Direção', height=200, template='plotly_dark')

    # ═══════════════════════════════════════════════════════
    # FIGURE 9: Top 10 configs table
    # ═══════════════════════════════════════════════════════
    top10 = df.nlargest(10, 'sharpe')[
        ['direction', 'lookback_days', 'holding_days', 'top_n',
         'universe_top', 'trailing_stop_pct', 'regime_filter',
         'cagr', 'sharpe', 'max_dd', 'calmar']
    ]
    top10['cagr'] = top10['cagr'].apply(lambda x: f"{x:.0f}%")
    top10['sharpe'] = top10['sharpe'].apply(lambda x: f"{x:.2f}")
    top10['max_dd'] = top10['max_dd'].apply(lambda x: f"{x:.1f}%")
    top10['calmar'] = top10['calmar'].apply(lambda x: f"{x:.1f}")

    fig9 = go.Figure(go.Table(
        header=dict(values=list(top10.columns),
                    fill_color='#1e1e1e', font=dict(color='white', size=12),
                    align='center'),
        cells=dict(values=[top10[c] for c in top10.columns],
                   fill_color='#263238',
                   font=dict(color='white', size=11),
                   align='center'),
    ))
    fig9.update_layout(title='Top 10 Configurações por Sharpe', height=350,
                       template='plotly_dark')

    # ═══════════════════════════════════════════════════════
    # BUILD HTML
    # ═══════════════════════════════════════════════════════
    html_parts = [
        '<html><head>',
        '<title>Crypto Momentum Grid Search Dashboard</title>',
        '<style>',
        'body { background: #121212; color: #e0e0e0; font-family: monospace; margin: 20px; }',
        'h1 { color: #ff9800; text-align: center; }',
        'h2 { color: #4fc3f7; border-bottom: 1px solid #333; padding-bottom: 5px; }',
        '.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }',
        '.full { grid-column: 1 / -1; }',
        '</style></head><body>',
        '<h1>CRYPTO MOMENTUM — GRID SEARCH DASHBOARD</h1>',
        f'<p style="text-align:center;color:#888;">{len(df)} configurations tested | '
        f'{len(df[df["direction"]=="long_only"])} long-only | '
        f'{len(df[df["direction"]=="long_short"])} long-short</p>',
        '<div class="grid">',
        f'<div class="full">{fig8.to_html(full_html=False, include_plotlyjs="cdn")}</div>',
        f'<div class="full">{fig9.to_html(full_html=False, include_plotlyjs=False)}</div>',
        f'<div class="full">{fig1.to_html(full_html=False, include_plotlyjs=False)}</div>',
        f'<div>{fig2.to_html(full_html=False, include_plotlyjs=False)}</div>',
        f'<div>{fig7.to_html(full_html=False, include_plotlyjs=False)}</div>',
        f'<div>{fig3a.to_html(full_html=False, include_plotlyjs=False)}</div>',
        f'<div>{fig3b.to_html(full_html=False, include_plotlyjs=False)}</div>',
        f'<div class="full">{fig4.to_html(full_html=False, include_plotlyjs=False)}</div>',
        f'<div class="full">{fig5.to_html(full_html=False, include_plotlyjs=False)}</div>',
        f'<div class="full">{fig6.to_html(full_html=False, include_plotlyjs=False)}</div>',
        '</div></body></html>',
    ]

    html_path = os.path.join(RESULTS_DIR, 'grid_search_dashboard.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))

    print(f"Dashboard saved to: {html_path}")
    return html_path


if __name__ == '__main__':
    path = create_dashboard()
    # Open in browser
    import webbrowser
    webbrowser.open(f'file:///{os.path.abspath(path)}')
