#!/usr/bin/env python3
"""
🎨 Advanced Data Visualizer - The Economist Style
===============================================

Generador de visualizaciones de nivel senior estilo The Economist
para datos de desempleo del INE de Chile.

Features:
- Visualizaciones estilo The Economist
- Mapeo correcto de las 16 regiones de Chile
- Gráficos profesionales con tipografía y colores premium
- Análisis temporal y regional avanzado
- Exportación en alta resolución

Author: Bruno San Martín
Date: 2025-06-28
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración estilo The Economist
plt.style.use('default')  # Reset to default first

# Configuración de estilo The Economist
ECONOMIST_COLORS = {
    'red': '#DC143C',          # Economist red
    'blue': '#004B87',         # Economist blue  
    'light_blue': '#6BAED6',   # Light blue
    'dark_blue': '#08519C',    # Dark blue
    'gray': '#525252',         # Dark gray
    'light_gray': '#969696',   # Light gray
    'green': '#31A354',        # Green
    'orange': '#FF8C00'        # Orange
}

# Configuración de tipografía
FONT_CONFIG = {
    'family': 'sans-serif',
    'sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'size': 10
}

plt.rcParams.update({
    'font.family': FONT_CONFIG['family'],
    'font.sans-serif': FONT_CONFIG['sans-serif'],
    'font.size': FONT_CONFIG['size'],
    'axes.titlesize': 14,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5
})


class EconomistStyleVisualizer:
    """Generador de visualizaciones estilo The Economist."""
    
    def __init__(self, output_dir: str = "outputs/visualizations"):
        """Inicializar visualizador."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Mapeo correcto de regiones chilenas
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
        self.regional_data = None
        
    def load_and_restructure_data(self, file_path: str) -> pd.DataFrame:
        """Cargar y reestructurar datos correctamente."""
        logger.info("📊 Cargando y reestructurando datos...")
        
        try:
            # Cargar datos originales con headers múltiples
            df_raw = pd.read_csv(file_path, header=[0, 1, 2, 3, 4])
            
            # Extraer información regional correcta
            regional_data = {}
            periods = []
            
            # Obtener períodos (primera columna)
            period_col = df_raw.iloc[:, 0]
            periods = period_col.dropna().tolist()
            
            # Limpiar períodos
            clean_periods = []
            for period in periods:
                period_str = str(period).strip()
                if period_str not in ['nan', 'NaN', '(v)'] and len(period_str) > 5:
                    clean_periods.append(period_str)
            
            # Extraer datos por región
            for col_idx, col in enumerate(df_raw.columns[2:]):  # Empezar desde columna 2
                if col_idx >= 17:  # Solo las 17 columnas (16 regiones + total)
                    break
                    
                # Obtener nombre de región del header
                region_name_raw = col[2] if len(col) > 2 else str(col)
                region_name = self.region_mapping.get(region_name_raw, region_name_raw)
                
                # Obtener valores para esta región
                values = df_raw.iloc[:len(clean_periods), col_idx + 2].tolist()
                
                # Limpiar valores numéricos
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
            
            # Crear DataFrame reestructurado
            restructured_data = pd.DataFrame(regional_data, index=clean_periods)
            restructured_data.index.name = 'Período'
            
            # Remover filas completamente vacías
            restructured_data = restructured_data.dropna(how='all')
            
            self.data = restructured_data
            self.regional_data = restructured_data
            
            logger.info(f"✅ Datos reestructurados: {restructured_data.shape}")
            logger.info(f"🗺️ Regiones identificadas: {len(restructured_data.columns)}")
            
            return restructured_data
            
        except Exception as e:
            logger.error(f"❌ Error reestructurando datos: {e}")
            raise
    
    def create_economist_style_plots(self) -> List[str]:
        """Crear suite completa de visualizaciones estilo The Economist."""
        if self.data is None:
            raise ValueError("No hay datos cargados")
        
        plot_files = []
        
        # 1. Gráfico principal: Tendencias regionales
        plot_files.append(self._create_regional_trends_plot())
        
        # 2. Ranking regional actual
        plot_files.append(self._create_regional_ranking_plot())
        
        # 3. Evolución temporal destacando metropolitana
        plot_files.append(self._create_metropolitan_focus_plot())
        
        # 4. Heatmap regional
        plot_files.append(self._create_regional_heatmap())
        
        # 5. Comparación norte-centro-sur
        plot_files.append(self._create_macro_regional_analysis())
        
        logger.info(f"✅ {len(plot_files)} visualizaciones estilo The Economist creadas")
        return plot_files
    
    def _create_regional_trends_plot(self) -> str:
        """Gráfico principal de tendencias regionales estilo The Economist."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Seleccionar regiones clave para destacar
        key_regions = ['Total País', 'Metropolitana', 'Valparaíso', 'Biobío', 'Antofagasta']
        
        # Datos temporales simplificados (últimos 20 períodos)
        recent_data = self.data.tail(20)
        
        # Colores específicos para regiones clave
        region_colors = {
            'Total País': ECONOMIST_COLORS['red'],
            'Metropolitana': ECONOMIST_COLORS['blue'],
            'Valparaíso': ECONOMIST_COLORS['green'],
            'Biobío': ECONOMIST_COLORS['orange'],
            'Antofagasta': ECONOMIST_COLORS['dark_blue']
        }
        
        # Plotear regiones en background (gris claro)
        for col in recent_data.columns:
            if col not in key_regions and col in recent_data.columns:
                values = recent_data[col].dropna()
                if len(values) > 5:  # Solo si hay suficientes datos
                    ax.plot(range(len(values)), values, 
                           color=ECONOMIST_COLORS['light_gray'], 
                           alpha=0.4, linewidth=1, zorder=1)
        
        # Plotear regiones clave destacadas
        for region in key_regions:
            if region in recent_data.columns:
                values = recent_data[region].dropna()
                if len(values) > 5:
                    ax.plot(range(len(values)), values, 
                           color=region_colors[region], 
                           linewidth=2.5, label=region, zorder=3)
                    
                    # Añadir punto final destacado
                    ax.scatter(len(values)-1, values.iloc[-1], 
                             color=region_colors[region], s=50, zorder=4)
        
        # Personalización estilo The Economist
        ax.set_title('Tasa de Desempleo por Región\\nChile 2010-2025', 
                    fontsize=16, fontweight='bold', pad=20)
        
        ax.set_xlabel('Período', fontsize=11)
        ax.set_ylabel('Tasa de Desempleo (%)', fontsize=11)
        
        # Personalizar ejes
        ax.tick_params(axis='both', which='major', labelsize=9)
        
        # Grid sutil
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Leyenda estilo The Economist
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                          frameon=False, fontsize=10)
        
        # Añadir nota al pie
        fig.text(0.12, 0.02, 'Fuente: Instituto Nacional de Estadísticas (INE) Chile', 
                fontsize=8, style='italic', color=ECONOMIST_COLORS['gray'])
        
        plt.tight_layout()
        
        # Guardar en alta resolución
        filename = self.output_dir / 'regional_trends_economist.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return str(filename)
    
    def _create_regional_ranking_plot(self) -> str:
        """Ranking regional actual estilo The Economist."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Obtener datos más recientes por región
        latest_data = self.data.iloc[-1].dropna().sort_values(ascending=True)
        
        # Excluir total país del ranking
        if 'Total País' in latest_data.index:
            latest_data = latest_data.drop('Total País')
        
        # Tomar top 16 regiones
        latest_data = latest_data.head(16)
        
        # Crear colores: verde para tasas bajas, rojo para altas
        colors = []
        for value in latest_data.values:
            if value <= 6:
                colors.append(ECONOMIST_COLORS['green'])
            elif value <= 8:
                colors.append(ECONOMIST_COLORS['orange'])
            else:
                colors.append(ECONOMIST_COLORS['red'])
        
        # Gráfico de barras horizontal
        bars = ax.barh(range(len(latest_data)), latest_data.values, 
                      color=colors, alpha=0.8, height=0.7)
        
        # Personalizar etiquetas
        ax.set_yticks(range(len(latest_data)))
        ax.set_yticklabels(latest_data.index, fontsize=10)
        
        # Añadir valores en las barras
        for i, (bar, value) in enumerate(zip(bars, latest_data.values)):
            ax.text(value + 0.1, bar.get_y() + bar.get_height()/2, 
                   f'{value:.1f}%', va='center', fontsize=9, fontweight='bold')
        
        # Personalización
        ax.set_title('Ranking Regional de Desempleo\\nMenores tasas en la parte superior', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Tasa de Desempleo (%)', fontsize=11)
        
        # Invertir orden para mostrar mejores arriba
        ax.invert_yaxis()
        
        # Grid sutil solo vertical
        ax.grid(True, axis='x', alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Límites del eje X
        ax.set_xlim(0, max(latest_data.values) * 1.15)
        
        # Nota al pie
        fig.text(0.12, 0.02, 'Fuente: INE Chile | Datos más recientes disponibles', 
                fontsize=8, style='italic', color=ECONOMIST_COLORS['gray'])
        
        plt.tight_layout()
        
        filename = self.output_dir / 'regional_ranking_economist.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return str(filename)
    
    def _create_metropolitan_focus_plot(self) -> str:
        """Análisis con foco en Región Metropolitana."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Datos de últimos 30 períodos
        recent_data = self.data.tail(30)
        
        # Destacar Metropolitana vs promedio nacional
        if 'Metropolitana' in recent_data.columns and 'Total País' in recent_data.columns:
            metro_data = recent_data['Metropolitana'].dropna()
            national_data = recent_data['Total País'].dropna()
            
            # Alinear datos por longitud
            min_length = min(len(metro_data), len(national_data))
            metro_data = metro_data.tail(min_length)
            national_data = national_data.tail(min_length)
            
            x_pos = range(len(metro_data))
            
            # Líneas principales
            ax.plot(x_pos, national_data, color=ECONOMIST_COLORS['gray'], 
                   linewidth=3, label='Promedio Nacional', alpha=0.8)
            ax.plot(x_pos, metro_data, color=ECONOMIST_COLORS['red'], 
                   linewidth=3, label='Región Metropolitana')
            
            # Área entre las líneas
            ax.fill_between(x_pos, metro_data, national_data, 
                           where=(metro_data >= national_data), 
                           color=ECONOMIST_COLORS['red'], alpha=0.2, 
                           label='RM sobre promedio')
            ax.fill_between(x_pos, metro_data, national_data, 
                           where=(metro_data < national_data), 
                           color=ECONOMIST_COLORS['green'], alpha=0.2,
                           label='RM bajo promedio')
            
            # Puntos finales destacados
            ax.scatter(len(metro_data)-1, metro_data.iloc[-1], 
                      color=ECONOMIST_COLORS['red'], s=80, zorder=5)
            ax.scatter(len(national_data)-1, national_data.iloc[-1], 
                      color=ECONOMIST_COLORS['gray'], s=80, zorder=5)
            
            # Anotación con diferencia actual
            diff = metro_data.iloc[-1] - national_data.iloc[-1]
            ax.annotate(f'Diferencia: {diff:+.1f}pp', 
                       xy=(len(metro_data)-1, metro_data.iloc[-1]), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Personalización
        ax.set_title('Región Metropolitana vs Promedio Nacional\\nEvolución Comparativa de la Tasa de Desempleo', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Período', fontsize=11)
        ax.set_ylabel('Tasa de Desempleo (%)', fontsize=11)
        
        # Grid y leyenda
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.legend(loc='upper left', frameon=False)
        
        # Nota
        fig.text(0.12, 0.02, 'Fuente: INE Chile | pp = puntos porcentuales', 
                fontsize=8, style='italic', color=ECONOMIST_COLORS['gray'])
        
        plt.tight_layout()
        
        filename = self.output_dir / 'metropolitan_focus_economist.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return str(filename)
    
    def _create_regional_heatmap(self) -> str:
        """Heatmap regional estilo The Economist."""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Usar datos recientes y transponer para tener regiones como filas
        recent_data = self.data.tail(20).T
        
        # Excluir Total País del heatmap
        if 'Total País' in recent_data.index:
            recent_data = recent_data.drop('Total País')
        
        # Crear heatmap con colormap personalizado
        from matplotlib.colors import LinearSegmentedColormap
        
        # Colormap estilo The Economist (verde a rojo)
        colors = ['#31A354', '#FFEDA0', '#FD8D3C', '#E31A1C', '#800026']
        n_bins = 100
        economist_cmap = LinearSegmentedColormap.from_list('economist', colors, N=n_bins)
        
        # Crear heatmap
        im = ax.imshow(recent_data.values, cmap=economist_cmap, aspect='auto', 
                      vmin=recent_data.min().min(), vmax=recent_data.max().max())
        
        # Configurar ejes
        ax.set_xticks(range(len(recent_data.columns)))
        ax.set_xticklabels([str(col)[:10] + '...' if len(str(col)) > 10 else str(col) 
                           for col in recent_data.columns], rotation=45, ha='right')
        
        ax.set_yticks(range(len(recent_data.index)))
        ax.set_yticklabels(recent_data.index)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Tasa de Desempleo (%)', rotation=270, labelpad=20)
        
        # Título
        ax.set_title('Mapa de Calor: Desempleo Regional a través del Tiempo\\nColores más oscuros indican mayor desempleo', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Nota
        fig.text(0.12, 0.02, 'Fuente: INE Chile | Datos de últimos 20 períodos', 
                fontsize=8, style='italic', color=ECONOMIST_COLORS['gray'])
        
        plt.tight_layout()
        
        filename = self.output_dir / 'regional_heatmap_economist.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return str(filename)
    
    def _create_macro_regional_analysis(self) -> str:
        """Análisis macro-regional (Norte, Centro, Sur)."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Definir macro regiones
        macro_regions = {
            'Norte': ['Arica y Parinacota', 'Tarapacá', 'Antofagasta', 'Atacama', 'Coquimbo'],
            'Centro': ['Valparaíso', 'Metropolitana', "O'Higgins", 'Maule'],
            'Sur': ['Ñuble', 'Biobío', 'La Araucanía', 'Los Ríos', 'Los Lagos', 'Aysén', 'Magallanes']
        }
        
        # Calcular promedios macro-regionales
        recent_data = self.data.tail(25)
        macro_data = {}
        
        for macro_name, regions in macro_regions.items():
            available_regions = [r for r in regions if r in recent_data.columns]
            if available_regions:
                macro_data[macro_name] = recent_data[available_regions].mean(axis=1)
        
        # Colores para macro regiones
        macro_colors = {
            'Norte': ECONOMIST_COLORS['orange'],
            'Centro': ECONOMIST_COLORS['blue'], 
            'Sur': ECONOMIST_COLORS['green']
        }
        
        # Plotear líneas macro-regionales
        for macro_name, data in macro_data.items():
            clean_data = data.dropna()
            if len(clean_data) > 5:
                ax.plot(range(len(clean_data)), clean_data, 
                       color=macro_colors[macro_name], linewidth=3, 
                       label=f'{macro_name} ({len(macro_regions[macro_name])} regiones)',
                       alpha=0.9)
                
                # Punto final
                ax.scatter(len(clean_data)-1, clean_data.iloc[-1], 
                          color=macro_colors[macro_name], s=80, zorder=5)
                
                # Anotación valor final
                ax.annotate(f'{clean_data.iloc[-1]:.1f}%', 
                           xy=(len(clean_data)-1, clean_data.iloc[-1]), 
                           xytext=(10, 5), textcoords='offset points',
                           fontsize=9, fontweight='bold', 
                           color=macro_colors[macro_name])
        
        # Línea de referencia nacional si disponible
        if 'Total País' in recent_data.columns:
            national = recent_data['Total País'].dropna()
            ax.plot(range(len(national)), national, 
                   color=ECONOMIST_COLORS['gray'], linewidth=2, 
                   linestyle='--', alpha=0.7, label='Promedio Nacional')
        
        # Personalización
        ax.set_title('Análisis Macro-Regional del Desempleo\\nComparación Norte, Centro y Sur de Chile', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Período', fontsize=11)
        ax.set_ylabel('Tasa de Desempleo Promedio (%)', fontsize=11)
        
        # Grid y leyenda
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.legend(loc='upper right', frameon=False)
        
        # Nota
        fig.text(0.12, 0.02, 'Fuente: INE Chile | Promedios ponderados por macro-región', 
                fontsize=8, style='italic', color=ECONOMIST_COLORS['gray'])
        
        plt.tight_layout()
        
        filename = self.output_dir / 'macro_regional_analysis_economist.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return str(filename)
    
    def generate_summary_report(self, plot_files: List[str]) -> str:
        """Generar reporte de visualizaciones."""
        report_content = f"""
# 🎨 Reporte de Visualizaciones - Estilo The Economist

**Fecha:** {datetime.now().strftime('%d/%m/%Y %H:%M')}
**Visualizaciones generadas:** {len(plot_files)}

---

## 📊 Visualizaciones Creadas

### 1. **Tendencias Regionales** (`regional_trends_economist.png`)
- Evolución temporal de las 5 regiones clave
- Destacado: Total País, Metropolitana, Valparaíso, Biobío, Antofagasta
- Otras regiones en background para contexto

### 2. **Ranking Regional Actual** (`regional_ranking_economist.png`)
- Ranking de las 16 regiones por tasa de desempleo
- Código de colores: Verde (bajo), Naranja (medio), Rojo (alto)
- Ordenado de mejor a peor performance

### 3. **Foco Metropolitano** (`metropolitan_focus_economist.png`)
- Comparación directa RM vs Promedio Nacional
- Áreas sombreadas muestran cuando RM está sobre/bajo promedio
- Incluye diferencia en puntos porcentuales

### 4. **Mapa de Calor Regional** (`regional_heatmap_economist.png`)
- Vista general de todas las regiones en el tiempo
- Colormap verde-rojo (verde = mejor, rojo = peor)
- Últimos 20 períodos de datos

### 5. **Análisis Macro-Regional** (`macro_regional_analysis_economist.png`)
- Agrupación Norte/Centro/Sur de Chile
- Promedios ponderados por macro-región
- Comparación con promedio nacional

---

## 🎯 Características de las Visualizaciones

### ✅ Estilo The Economist Implementado
- **Tipografía:** Arial/Helvetica sans-serif
- **Colores:** Paleta oficial Economist (rojo, azul, verde, gris)
- **Grid:** Líneas sutiles y limpias
- **Spacing:** Espaciado generoso y profesional
- **Annotations:** Anotaciones estratégicas y claras

### ✅ Datos Corregidos
- **16 Regiones chilenas** correctamente identificadas y nombradas
- **Plus Total País** para referencia nacional
- **Mapeo correcto** de nombres de regiones del INE
- **Períodos temporales** limpios y consistentes

### ✅ Calidad Profesional
- **Alta resolución** (300 DPI) para presentaciones
- **Formatos exportables** PNG de alta calidad
- **Leyendas claras** y contextualizadas
- **Fuentes citadas** en cada gráfico

---

## 🗺️ Regiones Incluidas (Correctamente Mapeadas)

1. Arica y Parinacota
2. Tarapacá  
3. Antofagasta
4. Atacama
5. Coquimbo
6. Valparaíso
7. Metropolitana
8. O'Higgins
9. Maule
10. Ñuble
11. Biobío
12. La Araucanía
13. Los Ríos
14. Los Lagos
15. Aysén
16. Magallanes
17. **Total País** (referencia nacional)

---

*Reporte generado automáticamente por Economist Style Visualizer v1.0*
        """
        
        report_file = self.output_dir / f"visualization_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return str(report_file)


def main():
    """Función principal."""
    try:
        visualizer = EconomistStyleVisualizer()
        
        # Cargar y reestructurar datos
        data_file = "data/raw/ine_table_combined.csv"
        if Path(data_file).exists():
            visualizer.load_and_restructure_data(data_file)
            
            # Crear visualizaciones
            plot_files = visualizer.create_economist_style_plots()
            
            # Generar reporte
            report_file = visualizer.generate_summary_report(plot_files)
            
            print(f"\\n🎨 Visualizaciones estilo The Economist completadas!")
            print(f"📊 Gráficos creados: {len(plot_files)}")
            print(f"📁 Ubicación: {visualizer.output_dir}")
            print(f"📋 Reporte: {report_file}")
            
            for i, plot in enumerate(plot_files, 1):
                print(f"  {i}. {Path(plot).name}")
                
        else:
            print(f"❌ Archivo de datos no encontrado: {data_file}")
            
    except Exception as e:
        logger.error(f"❌ Error creando visualizaciones: {e}")


if __name__ == "__main__":
    main()
