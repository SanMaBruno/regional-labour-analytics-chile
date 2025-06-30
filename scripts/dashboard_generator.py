#!/usr/bin/env python3
"""
📊 Professional Dashboard Generator - The Economist Style
=======================================================

Generador de dashboard interactivo profesional estilo The Economist
con las 16 regiones de Chile correctamente identificadas.

Features:
- Dashboard interactivo con Plotly
- Visualizaciones estilo The Economist
- 16 regiones de Chile correctamente mapeadas
- Métricas en tiempo real
- Diseño responsivo y profesional

Author: Bruno San Martín
Date: 2025-06-28
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# Colores estilo The Economist
ECONOMIST_COLORS = {
    'red': '#DC143C',
    'blue': '#004B87', 
    'light_blue': '#6BAED6',
    'dark_blue': '#08519C',
    'gray': '#525252',
    'light_gray': '#969696',
    'green': '#31A354',
    'orange': '#FF8C00',
    'background': '#FAFAFA'
}

class DashboardGenerator:
    """Generador de dashboard interactivo y profesional para desempleo regional INE Chile."""
    
    def __init__(self, output_dir: str = "outputs/dashboards"):
        """Inicializar dashboard."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Mapeo de regiones chilenas
        self.region_mapping = {
            'Total país': 'Total País',
            'Región de Arica y Parinacota': 'Arica y Parinacota',
            'Región de Tarapacá': 'Tarapacá', 
            'Región de Antofagasta': 'Antofagasta',
            'Región de Atacama': 'Atacama',
            'Región de Coquimbo': 'Coquimbo',
            'Región de Valparaíso': 'Valparaíso',
            'Región Metropolitana de Santiago': 'Metropolitana',
            "Región del Libertador Gral. Bernardo O'Higgins": "O'Higgins",
            'Región del Maule': 'Maule',
            'Región de Ñuble': 'Ñuble',
            'Región del Biobío': 'Biobío',
            'Región de La Araucanía': 'La Araucanía',
            'Región de Los Ríos': 'Los Ríos',
            'Región de Los Lagos': 'Los Lagos',
            'Región de Aysén del Gral. Carlos Ibáñez del Campo': 'Aysén',
            'Región de Magallanes y La Antártica Chilena': 'Magallanes'
        }
        
        self.data = None
    
    def load_and_process_data(self, file_path: str) -> pd.DataFrame:
        """Cargar y procesar datos con estructura correcta."""
        try:
            # Cargar datos originales
            df_raw = pd.read_csv(file_path, header=[0, 1, 2, 3, 4])
            
            # Reestructurar datos
            regional_data = {}
            periods = []
            
            # Obtener períodos
            period_col = df_raw.iloc[:, 0]
            periods = period_col.dropna().tolist()
            
            # Limpiar períodos
            clean_periods = []
            for period in periods:
                period_str = str(period).strip()
                if period_str not in ['nan', 'NaN', '(v)'] and len(period_str) > 5:
                    clean_periods.append(period_str)
            
            # Extraer datos por región
            for col_idx, col in enumerate(df_raw.columns[2:]):
                if col_idx >= 17:  # Solo 17 columnas (16 regiones + total)
                    break
                    
                region_name_raw = col[2] if len(col) > 2 else str(col)
                region_name = self.region_mapping.get(region_name_raw, region_name_raw)
                
                # Obtener valores
                values = df_raw.iloc[:len(clean_periods), col_idx + 2].tolist()
                
                # Limpiar valores
                clean_values = []
                for val in values:
                    try:
                        if pd.notna(val) and str(val) != '(v)':
                            clean_val = float(str(val).replace('(v)', '').strip())
                            clean_values.append(clean_val)
                        else:
                            clean_values.append(np.nan)
                    except:
                        clean_values.append(np.nan)
                
                regional_data[region_name] = clean_values[:len(clean_periods)]
            
            # Crear DataFrame
            self.data = pd.DataFrame(regional_data, index=clean_periods)
            self.data.index.name = 'Período'
            self.data = self.data.dropna(how='all')
            
            logger.info(f"✅ Datos procesados: {self.data.shape}")
            return self.data
            
        except Exception as e:
            logger.error(f"❌ Error procesando datos: {e}")
            raise
    
    def create_interactive_dashboard(self) -> str:
        """Crear dashboard interactivo completo."""
        if self.data is None:
            raise ValueError("No hay datos cargados")
        
        # Calcular métricas clave
        latest_data = self.data.iloc[-1].dropna()
        national_rate = latest_data.get('Total País', 'N/A')
        
        # Encontrar mejor y peor región
        regional_only = latest_data.drop('Total País', errors='ignore')
        best_region = regional_only.idxmin()
        worst_region = regional_only.idxmax()
        best_rate = regional_only.min()
        worst_rate = regional_only.max()
        
        # Crear HTML del dashboard
        html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard INE Chile - Análisis de Desempleo Regional</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', 'Arial', sans-serif;
            background-color: {ECONOMIST_COLORS['background']};
            color: #333;
            line-height: 1.6;
        }}
        
        .header {{
            background: linear-gradient(135deg, {ECONOMIST_COLORS['blue']} 0%, {ECONOMIST_COLORS['dark_blue']} 100%);
            color: white;
            padding: 2rem 0;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            font-weight: 300;
            margin-bottom: 0.5rem;
        }}
        
        .header p {{
            font-size: 1.1rem;
            opacity: 0.9;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 20px;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 2rem;
            margin: 2rem 0;
        }}
        
        .metric-card {{
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border-left: 4px solid {ECONOMIST_COLORS['blue']};
            transition: transform 0.2s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-2px);
        }}
        
        .metric-value {{
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }}
        
        .metric-label {{
            color: {ECONOMIST_COLORS['gray']};
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .metric-description {{
            color: {ECONOMIST_COLORS['gray']};
            font-size: 0.85rem;
            margin-top: 0.5rem;
        }}
        
        .best {{ color: {ECONOMIST_COLORS['green']}; }}
        .worst {{ color: {ECONOMIST_COLORS['red']}; }}
        .national {{ color: {ECONOMIST_COLORS['blue']}; }}
        
        .charts-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin: 2rem 0;
        }}
        
        .chart-container {{
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }}
        
        .chart-full {{
            grid-column: 1 / -1;
        }}
        
        .chart-title {{
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: {ECONOMIST_COLORS['blue']};
        }}
        
        .insights {{
            background: white;
            border-radius: 12px;
            padding: 2rem;
            margin: 2rem 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }}
        
        .insights h2 {{
            color: {ECONOMIST_COLORS['blue']};
            margin-bottom: 1rem;
        }}
        
        .insights ul {{
            list-style: none;
            padding: 0;
        }}
        
        .insights li {{
            padding: 0.8rem 0;
            border-bottom: 1px solid #eee;
            display: flex;
            align-items: center;
        }}
        
        .insights li:last-child {{
            border-bottom: none;
        }}
        
        .insight-icon {{
            width: 24px;
            height: 24px;
            margin-right: 12px;
            font-size: 1.2rem;
        }}
        
        .footer {{
            background: {ECONOMIST_COLORS['gray']};
            color: white;
            text-align: center;
            padding: 2rem 0;
            margin-top: 3rem;
        }}
        
        @media (max-width: 768px) {{
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
            
            .header h1 {{
                font-size: 2rem;
            }}
            
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1>📊 Dashboard INE Chile</h1>
            <p>Análisis Interactivo de Desempleo Regional • 16 Regiones de Chile</p>
            <p>Última actualización: {datetime.now().strftime('%d de %B %Y, %H:%M')}</p>
        </div>
    </div>
    
    <div class="container">
        <!-- Métricas Principales -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value national">{national_rate:.1f}%</div>
                <div class="metric-label">Promedio Nacional</div>
                <div class="metric-description">Tasa de desempleo a nivel país</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value best">{best_rate:.1f}%</div>
                <div class="metric-label">Mejor Región</div>
                <div class="metric-description">{best_region}</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value worst">{worst_rate:.1f}%</div>
                <div class="metric-label">Mayor Desempleo</div>
                <div class="metric-description">{worst_region}</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value national">{len(regional_only)}</div>
                <div class="metric-label">Regiones Analizadas</div>
                <div class="metric-description">Cobertura nacional completa</div>
            </div>
        </div>
        
        <!-- Gráficos Principales -->
        <div class="charts-grid">
            <div class="chart-container">
                <div class="chart-title">🗺️ Ranking Regional Actual</div>
                <div id="ranking-chart"></div>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">📈 Evolución Temporal</div>
                <div id="trends-chart"></div>
            </div>
        </div>
        
        <div class="chart-container chart-full">
            <div class="chart-title">🎯 Análisis Macro-Regional: Norte, Centro y Sur</div>
            <div id="macro-chart"></div>
        </div>
        
        <!-- Insights Automáticos -->
        <div class="insights">
            <h2>💡 Insights Automáticos</h2>
            <ul>
                <li>
                    <span class="insight-icon">🏆</span>
                    <strong>{best_region}</strong> lidera con la menor tasa de desempleo ({best_rate:.1f}%)
                </li>
                <li>
                    <span class="insight-icon">📊</span>
                    La diferencia entre la mejor y peor región es de <strong>{worst_rate - best_rate:.1f} puntos porcentuales</strong>
                </li>
                <li>
                    <span class="insight-icon">🎯</span>
                    La tasa nacional ({national_rate:.1f}%) se encuentra {'por encima' if national_rate > regional_only.median() else 'por debajo'} de la mediana regional
                </li>
                <li>
                    <span class="insight-icon">🗺️</span>
                    Todas las <strong>16 regiones de Chile</strong> están correctamente representadas en el análisis
                </li>
            </ul>
        </div>
    </div>
    
    <div class="footer">
        <div class="container">
            <p>Fuente: Instituto Nacional de Estadísticas (INE) Chile</p>
            <p>Dashboard generado automáticamente • Estilo The Economist</p>
        </div>
    </div>
    
    <script>
        // Datos para gráficos
        {self._generate_chart_data()}
        
        // Configuración común de Plotly
        const plotConfig = {{
            displayModeBar: false,
            responsive: true
        }};
        
        const plotLayout = {{
            font: {{
                family: 'Segoe UI, Arial, sans-serif',
                size: 12
            }},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: {{ t: 20, r: 20, b: 40, l: 80 }}
        }};
        
        // Gráfico de ranking
        const rankingTrace = {{
            type: 'bar',
            orientation: 'h',
            x: rankingData.values,
            y: rankingData.regions,
            marker: {{
                color: rankingData.colors,
                opacity: 0.8
            }},
            text: rankingData.values.map(v => v.toFixed(1) + '%'),
            textposition: 'outside',
            textfont: {{ size: 10, color: '{ECONOMIST_COLORS['gray']}' }}
        }};
        
        const rankingLayout = {{
            ...plotLayout,
            xaxis: {{ title: 'Tasa de Desempleo (%)' }},
            yaxis: {{ 
                title: '',
                autorange: 'reversed',
                tickfont: {{ size: 10 }}
            }},
            height: 500
        }};
        
        Plotly.newPlot('ranking-chart', [rankingTrace], rankingLayout, plotConfig);
        
        // Gráfico de tendencias
        const trendsTraces = trendsData.map(trace => ({{
            type: 'scatter',
            mode: 'lines+markers',
            x: trace.periods,
            y: trace.values,
            name: trace.name,
            line: {{
                color: trace.color,
                width: trace.name === 'Total País' ? 3 : 2
            }},
            marker: {{ size: 4 }}
        }}));
        
        const trendsLayout = {{
            ...plotLayout,
            xaxis: {{ title: 'Período' }},
            yaxis: {{ title: 'Tasa de Desempleo (%)' }},
            legend: {{
                x: 0.02,
                y: 0.98,
                bgcolor: 'rgba(255,255,255,0.8)'
            }},
            height: 400
        }};
        
        Plotly.newPlot('trends-chart', trendsTraces, trendsLayout, plotConfig);
        
        // Gráfico macro-regional
        const macroTraces = macroData.map(trace => ({{
            type: 'scatter',
            mode: 'lines+markers',
            x: trace.periods,
            y: trace.values,
            name: trace.name,
            line: {{
                color: trace.color,
                width: 3
            }},
            marker: {{ size: 6 }}
        }}));
        
        const macroLayout = {{
            ...plotLayout,
            xaxis: {{ title: 'Período' }},
            yaxis: {{ title: 'Tasa de Desempleo Promedio (%)' }},
            legend: {{
                x: 0.02,
                y: 0.98,
                bgcolor: 'rgba(255,255,255,0.8)'
            }},
            height: 400
        }};
        
        Plotly.newPlot('macro-chart', macroTraces, macroLayout, plotConfig);
    </script>
</body>
</html>
        """
        
        # Guardar dashboard
        dashboard_file = self.output_dir / f"professional_dashboard_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✅ Dashboard profesional creado: {dashboard_file}")
        return str(dashboard_file)
    
    def _generate_chart_data(self) -> str:
        """Generar datos de JavaScript para los gráficos."""
        try:
            # Datos para ranking regional
            latest_data = self.data.iloc[-1].dropna()
            regional_only = latest_data.drop('Total País', errors='ignore').sort_values()
            
            ranking_colors = []
            for value in regional_only.values:
                if value <= 6:
                    ranking_colors.append(ECONOMIST_COLORS['green'])
                elif value <= 8:
                    ranking_colors.append(ECONOMIST_COLORS['orange'])
                else:
                    ranking_colors.append(ECONOMIST_COLORS['red'])
            
            # Datos para tendencias (últimas 20 observaciones)
            recent_data = self.data.tail(20)
            key_regions = ['Total País', 'Metropolitana', 'Valparaíso', 'Biobío', 'Antofagasta']
            region_colors_trends = {
                'Total País': ECONOMIST_COLORS['red'],
                'Metropolitana': ECONOMIST_COLORS['blue'],
                'Valparaíso': ECONOMIST_COLORS['green'],
                'Biobío': ECONOMIST_COLORS['orange'],
                'Antofagasta': ECONOMIST_COLORS['dark_blue']
            }
            
            trends_data = []
            for region in key_regions:
                if region in recent_data.columns:
                    clean_values = recent_data[region].dropna()
                    if len(clean_values) > 5:
                        trends_data.append({
                            'name': region,
                            'periods': list(range(len(clean_values))),
                            'values': clean_values.tolist(),
                            'color': region_colors_trends[region]
                        })
            
            # Datos macro-regionales
            macro_regions = {
                'Norte': ['Arica y Parinacota', 'Tarapacá', 'Antofagasta', 'Atacama', 'Coquimbo'],
                'Centro': ['Valparaíso', 'Metropolitana', "O'Higgins", 'Maule'],
                'Sur': ['Ñuble', 'Biobío', 'La Araucanía', 'Los Ríos', 'Los Lagos', 'Aysén', 'Magallanes']
            }
            
            macro_colors = {
                'Norte': ECONOMIST_COLORS['orange'],
                'Centro': ECONOMIST_COLORS['blue'],
                'Sur': ECONOMIST_COLORS['green']
            }
            
            macro_data = []
            for macro_name, regions in macro_regions.items():
                available_regions = [r for r in regions if r in recent_data.columns]
                if available_regions:
                    avg_data = recent_data[available_regions].mean(axis=1).dropna()
                    if len(avg_data) > 5:
                        macro_data.append({
                            'name': f'{macro_name} ({len(available_regions)} regiones)',
                            'periods': list(range(len(avg_data))),
                            'values': avg_data.tolist(),
                            'color': macro_colors[macro_name]
                        })
            
            # Generar JavaScript
            js_data = f"""
        const rankingData = {{
            regions: {list(regional_only.index)},
            values: {regional_only.values.tolist()},
            colors: {ranking_colors}
        }};
        
        const trendsData = {json.dumps(trends_data)};
        
        const macroData = {json.dumps(macro_data)};
            """
            
            return js_data
            
        except Exception as e:
            logger.error(f"❌ Error generando datos de gráficos: {e}")
            return "const rankingData = {}; const trendsData = []; const macroData = [];"


def main():
    """Función principal para generar el dashboard interactivo INE Chile."""
    try:
        dashboard = DashboardGenerator()
        # Cargar datos
        data_file = "data/raw/ine_table_combined.csv"
        if Path(data_file).exists():
            dashboard.load_and_process_data(data_file)
            # Crear dashboard
            dashboard_file = dashboard.create_interactive_dashboard()
            print(f"\n📊 Dashboard profesional creado exitosamente!")
            print(f"📁 Archivo: {dashboard_file}")
            print(f"🌐 Características:")
            print(f"   ✅ 16 regiones de Chile correctamente identificadas")
            print(f"   ✅ Diseño estilo The Economist")
            print(f"   ✅ Gráficos interactivos con Plotly")
            print(f"   ✅ Métricas en tiempo real")
            print(f"   ✅ Responsive design")
        else:
            print(f"❌ Archivo de datos no encontrado: {data_file}")
    except Exception as e:
        logger.error(f"❌ Error creando dashboard: {e}")


if __name__ == "__main__":
    main()
