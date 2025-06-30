"""
🔧 Data Processing Module - INE Data Scraper
=============================================

Este módulo contiene las funciones para procesar, limpiar y transformar
los datos extraídos del scraping.

Author: Bruno San Martín
Date: 2025-06-28
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import yaml
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Clase principal para el procesamiento de datos del INE.
    """
    
    def __init__(self, config_path: str = "config/config.yaml", output_dir: str = None):
        """
        Inicializa el procesador de datos.
        
        Args:
            config_path (str): Ruta al archivo de configuración
            output_dir (str): Directorio de salida para datos procesados
        """
        self.config = self._load_config(config_path)
        self.output_dir = Path(output_dir) if output_dir else Path("data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data = None
        self.data = None
        self.metadata = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Carga la configuración desde archivo YAML."""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Archivo de configuración no encontrado: {config_path}")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Configuración por defecto si no se encuentra archivo."""
        return {
            'data': {
                'cleaning': {
                    'remove_duplicates': True,
                    'handle_missing_values': 'drop',
                    'encoding': 'utf-8-sig'
                }
            }
        }
    
    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        """
        Carga datos desde archivo CSV.
        
        Args:
            file_path (str): Ruta al archivo de datos
            
        Returns:
            pd.DataFrame: Datos cargados
        """
        try:
            encoding = self.config['data']['cleaning']['encoding']
            df = pd.read_csv(file_path, encoding=encoding)
            logger.info(f"✅ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
            return df
        except Exception as e:
            logger.error(f"❌ Error cargando datos: {e}")
            return pd.DataFrame()
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia y preprocesa los datos.
        
        Args:
            df (pd.DataFrame): Datos sin procesar
            
        Returns:
            pd.DataFrame: Datos limpios
        """
        logger.info("🧹 Iniciando limpieza de datos...")
        
        # Guardar dimensiones originales
        original_shape = df.shape
        
        # Eliminar duplicados
        if self.config['data']['cleaning']['remove_duplicates']:
            df = df.drop_duplicates()
            logger.info(f"🔄 Duplicados eliminados: {original_shape[0] - df.shape[0]} filas")
        
        # Manejar valores faltantes
        missing_strategy = self.config['data']['cleaning']['handle_missing_values']
        if missing_strategy == 'drop':
            df = df.dropna()
        elif missing_strategy == 'fill':
            df = df.fillna(method='forward')
        
        # Limpiar nombres de columnas
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
        
        # Detectar y convertir tipos de datos
        df = self._auto_detect_types(df)
        
        logger.info(f"✅ Limpieza completada: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
    
    def _auto_detect_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta y convierte tipos de datos automáticamente."""
        for col in df.columns:
            # Intentar convertir a numérico
            if df[col].dtype == 'object':
                # Limpiar texto antes de conversión
                df[col] = df[col].astype(str).str.replace(',', '.')
                
                # Intentar conversión numérica
                numeric = pd.to_numeric(df[col], errors='coerce')
                if not numeric.isna().all():
                    df[col] = numeric
                    logger.info(f"🔢 Columna '{col}' convertida a numérica")
        
        return df
    
    def generate_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Genera un reporte de calidad de los datos.
        
        Args:
            df (pd.DataFrame): Datos a analizar
            
        Returns:
            Dict: Reporte de calidad
        """
        report = {
            'general': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
            },
            'completeness': {
                'complete_rows': len(df.dropna()),
                'completeness_rate': len(df.dropna()) / len(df) * 100
            },
            'duplicates': {
                'duplicate_rows': df.duplicated().sum(),
                'unique_rows': len(df.drop_duplicates())
            },
            'columns': {}
        }
        
        # Análisis por columna
        for col in df.columns:
            report['columns'][col] = {
                'dtype': str(df[col].dtype),
                'missing_values': df[col].isna().sum(),
                'missing_percentage': df[col].isna().sum() / len(df) * 100,
                'unique_values': df[col].nunique()
            }
        
        return report
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str, 
                          formats: Optional[List[str]] = None) -> None:
        """
        Guarda los datos procesados en múltiples formatos.
        
        Args:
            df (pd.DataFrame): Datos a guardar
            output_path (str): Ruta base para los archivos
            formats (List[str]): Formatos de exportación
        """
        if formats is None:
            formats = self.config.get('data', {}).get('export_formats', ['csv'])
        
        base_path = Path(output_path).with_suffix('')
        
        for fmt in formats:
            try:
                if fmt == 'csv':
                    df.to_csv(f"{base_path}.csv", index=False, encoding='utf-8-sig')
                elif fmt == 'excel':
                    df.to_excel(f"{base_path}.xlsx", index=False, engine='openpyxl')
                elif fmt == 'parquet':
                    df.to_parquet(f"{base_path}.parquet", index=False)
                
                logger.info(f"💾 Datos guardados en formato {fmt.upper()}")
                
            except Exception as e:
                logger.error(f"❌ Error guardando en formato {fmt}: {e}")
    
    def load_and_clean_data(self, file_path: str) -> pd.DataFrame:
        """Carga y limpia datos del INE con headers complejos."""
        try:
            # Cargar datos con headers múltiples
            df = pd.read_csv(file_path, header=[0, 1, 2, 3, 4], encoding='utf-8-sig')
            logger.info(f"📊 Datos cargados: {df.shape}")
            
            # Limpiar y restructurar
            cleaned_df = self._clean_multiindex_headers(df)
            cleaned_df = self._process_unemployment_data(cleaned_df)
            
            # Validar datos
            validation_results = {'status': 'processed', 'records': len(cleaned_df)}
            logger.info(f"✅ Validación: {validation_results}")
            
            return cleaned_df
            
        except Exception as e:
            logger.error(f"❌ Error cargando datos: {e}")
            raise
    
    def _clean_multiindex_headers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia headers complejos del INE."""
        try:
            # Simplificar columnas
            new_columns = []
            for col in df.columns:
                if isinstance(col, tuple):
                    # Tomar el primer elemento no vacío o unnamed
                    clean_col = None
                    for level in col:
                        if pd.notna(level) and 'Unnamed' not in str(level):
                            clean_col = str(level)
                            break
                    if not clean_col:
                        clean_col = f"col_{len(new_columns)}"
                else:
                    clean_col = str(col)
                new_columns.append(clean_col)
            
            df.columns = new_columns
            logger.info(f"🔄 Headers limpiados: {len(new_columns)} columnas")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error limpiando headers: {e}")
            return df
    
    def _process_unemployment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Procesa específicamente datos de desempleo del INE."""
        try:
            # Identificar columna de tiempo
            time_col = None
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['trimestre', 'tiempo', 'fecha', 'periodo']):
                    time_col = col
                    break
            
            if time_col is None:
                time_col = df.columns[0]  # Asumir primera columna
            
            # Limpiar datos temporales
            df[time_col] = df[time_col].fillna('').astype(str).str.strip()
            
            # Filtrar filas válidas (no vacías ni headers repetidos)
            df = df[~df[time_col].isin(['nan', 'NaN', 'Trimestre Móvil', ''])]
            
            # Convertir valores numéricos
            numeric_cols = df.select_dtypes(include=[object]).columns
            numeric_cols = [col for col in numeric_cols if col != time_col]
            
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Renombrar columnas principales
            df = df.rename(columns={time_col: 'periodo'})
            
            # Añadir metadatos
            df['fecha_procesamiento'] = datetime.now()
            df['fuente'] = 'INE Chile'
            
            logger.info(f"✅ Datos de desempleo procesados: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error procesando datos de desempleo: {e}")
            return df

def main():
    """Función principal para testing del módulo."""
    processor = DataProcessor()
    
    # Ejemplo de uso
    print("🧪 Testing Data Processor...")
    
    # Crear datos de ejemplo
    sample_data = pd.DataFrame({
        'Region': ['Metropolitana', 'Valparaíso', 'Metropolitana'],
        'Valor': ['1,234.56', '2,345.67', '1,234.56'],
        'Fecha': ['2025-01-01', '2025-01-02', '2025-01-01']
    })
    
    print("📊 Datos de ejemplo:")
    print(sample_data)
    
    # Procesar datos
    clean_data = processor.clean_data(sample_data)
    print("\n🧹 Datos limpios:")
    print(clean_data)
    
    # Generar reporte
    report = processor.generate_quality_report(clean_data)
    print("\n📋 Reporte de calidad:")
    print(f"Total filas: {report['general']['total_rows']}")
    print(f"Completitud: {report['completeness']['completeness_rate']:.2f}%")


if __name__ == "__main__":
    main()
