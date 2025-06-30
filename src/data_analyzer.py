"""
📈 Data Analytics Module - INE Data Scraper
===========================================

Este módulo contiene las funciones para análisis estadístico y 
generación de visualizaciones de los datos del INE.

Author: Bruno San Martín
Date: 2025-06-28
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import warnings
from datetime import datetime

# Configurar estilo y warnings
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAnalyzer:
    """
    Clase principal para análisis de datos del INE.
    """
    
    def __init__(self, output_dir: str = "outputs/visualizations"):
        """
        Inicializa el analizador de datos.
        
        Args:
            output_dir (str): Directorio para guardar visualizaciones
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Carga datos para análisis.
        
        Args:
            file_path (str): Ruta al archivo de datos
            
        Returns:
            pd.DataFrame: Datos cargados
        """
        try:
            self.data = pd.read_csv(file_path, encoding='utf-8-sig')
            logger.info(f"✅ Datos cargados para análisis: {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"❌ Error cargando datos: {e}")
            return pd.DataFrame()
    
    def generate_summary_statistics(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Genera estadísticas descriptivas completas.
        
        Args:
            df (pd.DataFrame): Datos a analizar
            
        Returns:
            Dict: Estadísticas descriptivas
        """
        if df is None:
            df = self.data
        
        if df is None or df.empty:
            logger.error("❌ No hay datos para analizar")
            return {}
        
        stats = {
            'basic_info': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(df.select_dtypes(include=['object']).columns)
            },
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # Estadísticas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        # Estadísticas categóricas
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            stats['categorical_summary'][col] = {
                'unique_values': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                'frequency_count': df[col].value_counts().head().to_dict()
            }
        
        return stats
    
    def create_correlation_matrix(self, df: Optional[pd.DataFrame] = None, 
                                 save: bool = True) -> None:
        """
        Crea matriz de correlación para variables numéricas.
        
        Args:
            df (pd.DataFrame): Datos a analizar
            save (bool): Si guardar la visualización
        """
        if df is None:
            df = self.data
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            logger.warning("⚠️ Insuficientes columnas numéricas para correlación")
            return
        
        # Crear figura
        plt.figure(figsize=(12, 8))
        correlation_matrix = df[numeric_cols].corr()
        
        # Crear heatmap
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={'label': 'Coeficiente de Correlación'})
        
        plt.title('Matriz de Correlación - Variables Numéricas', 
                 fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'correlation_matrix.png', 
                       dpi=300, bbox_inches='tight')
            logger.info("💾 Matriz de correlación guardada")
        
        plt.show()
    
    def create_distribution_plots(self, df: Optional[pd.DataFrame] = None, 
                                save: bool = True) -> None:
        """
        Crea gráficos de distribución para variables numéricas.
        
        Args:
            df (pd.DataFrame): Datos a analizar
            save (bool): Si guardar las visualizaciones
        """
        if df is None:
            df = self.data
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            logger.warning("⚠️ No hay columnas numéricas para distribución")
            return
        
        # Calcular filas y columnas para subplots
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            ax = axes[i] if len(numeric_cols) > 1 else axes
            
            # Histogram con KDE
            sns.histplot(data=df, x=col, kde=True, ax=ax)
            ax.set_title(f'Distribución de {col}', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Ocultar subplots vacíos
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'distribution_plots.png', 
                       dpi=300, bbox_inches='tight')
            logger.info("💾 Gráficos de distribución guardados")
        
        plt.show()
    
    def create_interactive_dashboard(self, df: Optional[pd.DataFrame] = None, 
                                   save: bool = True) -> None:
        """
        Crea un dashboard interactivo con Plotly.
        
        Args:
            df (pd.DataFrame): Datos a analizar
            save (bool): Si guardar el dashboard
        """
        if df is None:
            df = self.data
        
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distribución de Variables', 'Correlaciones', 
                          'Valores por Categoría', 'Tendencias Temporales'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Plot 1: Distribución
        if len(numeric_cols) > 0:
            first_numeric = numeric_cols[0]
            fig.add_trace(
                go.Histogram(x=df[first_numeric], name=first_numeric),
                row=1, col=1
            )
        
        # Plot 2: Correlación (heatmap)
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values, 
                          x=corr_matrix.columns, 
                          y=corr_matrix.columns,
                          colorscale='RdBu',
                          name="Correlación"),
                row=1, col=2
            )
        
        # Plot 3: Categorical analysis
        if len(categorical_cols) > 0:
            first_cat = categorical_cols[0]
            value_counts = df[first_cat].value_counts().head(10)
            fig.add_trace(
                go.Bar(x=value_counts.index, y=value_counts.values, 
                      name=first_cat),
                row=2, col=1
            )
        
        # Plot 4: Time series (si hay columnas de fecha)
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0 and len(numeric_cols) > 0:
            date_col = date_cols[0]
            numeric_col = numeric_cols[0]
            fig.add_trace(
                go.Scatter(x=df[date_col], y=df[numeric_col], 
                          mode='lines+markers', name='Tendencia'),
                row=2, col=2
            )
        
        # Actualizar layout
        fig.update_layout(
            title_text="Dashboard Analítico - Datos INE",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        if save:
            fig.write_html(self.output_dir / 'interactive_dashboard.html')
            logger.info("💾 Dashboard interactivo guardado")
        
        fig.show()
    
    def generate_comprehensive_report(self, df: Optional[pd.DataFrame] = None) -> str:
        """
        Genera un reporte comprensivo del análisis.
        
        Args:
            df (pd.DataFrame): Datos a analizar
            
        Returns:
            str: Ruta del reporte generado
        """
        if df is None:
            df = self.data
        
        # Generar todas las visualizaciones
        logger.info("📊 Generando reporte comprensivo...")
        
        self.create_correlation_matrix(df)
        self.create_distribution_plots(df)
        self.create_interactive_dashboard(df)
        
        # Generar estadísticas
        stats = self.generate_summary_statistics(df)
        
        # Crear reporte en texto
        report_path = self.output_dir / 'analysis_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("📊 REPORTE DE ANÁLISIS - DATOS INE\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("📋 INFORMACIÓN GENERAL:\n")
            f.write(f"   • Total de filas: {stats['basic_info']['total_rows']:,}\n")
            f.write(f"   • Total de columnas: {stats['basic_info']['total_columns']}\n")
            f.write(f"   • Columnas numéricas: {stats['basic_info']['numeric_columns']}\n")
            f.write(f"   • Columnas categóricas: {stats['basic_info']['categorical_columns']}\n\n")
            
            f.write("📈 RESUMEN ESTADÍSTICO:\n")
            for col, data in stats.get('numeric_summary', {}).items():
                f.write(f"\n   {col}:\n")
                f.write(f"     - Media: {data.get('mean', 0):.2f}\n")
                f.write(f"     - Mediana: {data.get('50%', 0):.2f}\n")
                f.write(f"     - Desviación estándar: {data.get('std', 0):.2f}\n")
            
            f.write("\n📊 ANÁLISIS CATEGÓRICO:\n")
            for col, data in stats.get('categorical_summary', {}).items():
                f.write(f"\n   {col}:\n")
                f.write(f"     - Valores únicos: {data['unique_values']}\n")
                f.write(f"     - Más frecuente: {data['most_frequent']}\n")
        
        logger.info(f"✅ Reporte generado: {report_path}")
        return str(report_path)

    def time_series_analysis(self) -> dict:
        """Análisis de series temporales."""
        if self.data is None:
            raise ValueError("No hay datos cargados")
        
        try:
            # Identificar columna de tiempo
            time_col = 'periodo' if 'periodo' in self.data.columns else self.data.columns[0]
            
            # Análisis de tendencias
            trends = {}
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols[:5]:  # Primeras 5 columnas numéricas
                values = self.data[col].dropna()
                if len(values) > 2:
                    # Calcular tendencia
                    x = np.arange(len(values))
                    coeffs = np.polyfit(x, values, 1)
                    trends[col] = {
                        'slope': coeffs[0],
                        'intercept': coeffs[1],
                        'trend': 'creciente' if coeffs[0] > 0 else 'decreciente'
                    }
            
            return {
                'trends': trends,
                'total_periods': len(self.data),
                'analysis_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error en análisis temporal: {e}")
            return {}
    
    def regional_analysis(self) -> dict:
        """Análisis por regiones."""
        if self.data is None:
            raise ValueError("No hay datos cargados")
        
        try:
            # Buscar columnas que contengan información regional
            region_cols = [col for col in self.data.columns 
                          if any(term in col.lower() for term in ['region', 'zona', 'área'])]
            
            if not region_cols:
                return {'message': 'No se encontraron datos regionales'}
            
            # Análisis por región (usando primera columna regional)
            region_col = region_cols[0]
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            
            regional_stats = {}
            for region in self.data[region_col].unique()[:10]:  # Top 10 regiones
                region_data = self.data[self.data[region_col] == region]
                stats = {}
                for col in numeric_cols[:3]:  # Primeras 3 métricas
                    values = region_data[col].dropna()
                    if len(values) > 0:
                        stats[col] = {
                            'promedio': float(values.mean()),
                            'maximo': float(values.max()),
                            'minimo': float(values.min())
                        }
                regional_stats[str(region)] = stats
            
            return {
                'regional_statistics': regional_stats,
                'total_regions': len(self.data[region_col].unique()),
                'analysis_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error en análisis regional: {e}")
            return {}

    def create_visualizations(self) -> list:
        """Crear visualizaciones avanzadas."""
        if self.data is None:
            raise ValueError("No hay datos cargados")
        
        viz_files = []
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Configurar estilo
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # 1. Gráfico de tendencias temporales
            fig, ax = plt.subplots(figsize=(12, 8))
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:5]
            
            for col in numeric_cols:
                values = self.data[col].dropna()
                if len(values) > 1:
                    ax.plot(range(len(values)), values, marker='o', label=col[:30])
            
            ax.set_title('Tendencias Temporales - Datos INE', fontsize=16, fontweight='bold')
            ax.set_xlabel('Período')
            ax.set_ylabel('Valor')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            trends_file = self.output_dir / 'tendencias_temporales.png'
            plt.savefig(trends_file, dpi=300, bbox_inches='tight')
            plt.close()
            viz_files.append(str(trends_file))
            
            # 2. Distribución de valores
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols[:4]):
                values = self.data[col].dropna()
                if len(values) > 0:
                    axes[i].hist(values, bins=20, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'Distribución: {col[:30]}')
                    axes[i].set_xlabel('Valor')
                    axes[i].set_ylabel('Frecuencia')
                    axes[i].grid(True, alpha=0.3)
            
            plt.suptitle('Distribuciones de Variables Clave', fontsize=16, fontweight='bold')
            plt.tight_layout()
            dist_file = self.output_dir / 'distribuciones.png'
            plt.savefig(dist_file, dpi=300, bbox_inches='tight')
            plt.close()
            viz_files.append(str(dist_file))
            
            # 3. Mapa de calor de correlaciones
            numeric_data = self.data.select_dtypes(include=[np.number]).iloc[:, :10]
            if len(numeric_data.columns) > 1:
                fig, ax = plt.subplots(figsize=(12, 8))
                corr_matrix = numeric_data.corr()
                
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                           square=True, ax=ax, cbar_kws={'shrink': 0.8})
                ax.set_title('Matriz de Correlaciones', fontsize=16, fontweight='bold')
                
                plt.tight_layout()
                corr_file = self.output_dir / 'correlaciones.png'
                plt.savefig(corr_file, dpi=300, bbox_inches='tight')
                plt.close()
                viz_files.append(str(corr_file))
            
            logger.info(f"✅ Visualizaciones creadas: {len(viz_files)}")
            return viz_files
            
        except Exception as e:
            logger.error(f"❌ Error creando visualizaciones: {e}")
            return viz_files

    def descriptive_analysis(self) -> dict:
        """Análisis estadístico descriptivo."""
        if self.data is None:
            raise ValueError("No hay datos cargados")
        
        try:
            # Análisis básico
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            stats = {}
            
            for col in numeric_cols[:5]:  # Primeras 5 columnas numéricas
                values = self.data[col].dropna()
                if len(values) > 0:
                    stats[col] = {
                        'count': len(values),
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'median': float(values.median())
                    }
            
            return {
                'descriptive_statistics': stats,
                'total_variables': len(numeric_cols),
                'analysis_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error en análisis descriptivo: {e}")
            return {}

def main():
    """Función principal para testing del módulo."""
    analyzer = DataAnalyzer()
    
    # Crear datos de ejemplo
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'Region': np.random.choice(['Metropolitana', 'Valparaíso', 'Biobío'], 1000),
        'Valor': np.random.normal(100, 20, 1000),
        'Indicador': np.random.choice(['A', 'B', 'C'], 1000),
        'Año': np.random.choice([2020, 2021, 2022, 2023], 1000)
    })
    
    print("🧪 Testing Data Analyzer...")
    
    # Generar estadísticas
    stats = analyzer.generate_summary_statistics(sample_data)
    print("📊 Estadísticas generadas:")
    print(f"   Total filas: {stats['basic_info']['total_rows']}")
    
    # Generar visualizaciones
    analyzer.data = sample_data
    analyzer.create_correlation_matrix()
    analyzer.create_distribution_plots()


if __name__ == "__main__":
    main()
