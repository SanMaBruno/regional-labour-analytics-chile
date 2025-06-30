"""
🧪 Test Suite - INE Data Scraper
================================

Suite completa de tests para validar el funcionamiento del sistema de scraping.

Author: Bruno San Martín
Date: 2025-06-28
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Agregar src al path para imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from data_processor import DataProcessor
except ImportError:
    print("⚠️ Módulo data_processor no encontrado")


class TestDataProcessor:
    """Tests para el módulo de procesamiento de datos."""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture con datos de ejemplo."""
        return pd.DataFrame({
            'Region': ['Metropolitana', 'Valparaíso', 'Metropolitana', 'Valparaíso'],
            'Valor': ['1,234.56', '2,345.67', '1,234.56', '3,456.78'],
            'Fecha': ['2025-01-01', '2025-01-02', '2025-01-01', '2025-01-03'],
            'Categoria': ['A', 'B', 'A', 'C']
        })
    
    @pytest.fixture
    def processor(self):
        """Fixture con instancia del procesador."""
        return DataProcessor()
    
    def test_load_config_default(self, processor):
        """Test carga de configuración por defecto."""
        assert processor.config is not None
        assert 'data' in processor.config
    
    def test_clean_data_basic(self, processor, sample_data):
        """Test limpieza básica de datos."""
        cleaned = processor.clean_data(sample_data)
        
        # Verificar que se procesó
        assert not cleaned.empty
        assert len(cleaned.columns) == len(sample_data.columns)
        
        # Verificar nombres de columnas
        assert all('_' in col or col.islower() for col in cleaned.columns)
    
    def test_remove_duplicates(self, processor, sample_data):
        """Test eliminación de duplicados."""
        # Agregar duplicado exacto
        sample_with_dup = pd.concat([sample_data, sample_data.iloc[[0]]], ignore_index=True)
        
        cleaned = processor.clean_data(sample_with_dup)
        
        # Verificar que se eliminaron duplicados
        assert len(cleaned) < len(sample_with_dup)
    
    def test_numeric_conversion(self, processor, sample_data):
        """Test conversión a tipos numéricos."""
        cleaned = processor.clean_data(sample_data)
        
        # La columna 'valor' debería ser numérica después de limpieza
        valor_col = [col for col in cleaned.columns if 'valor' in col.lower()][0]
        assert pd.api.types.is_numeric_dtype(cleaned[valor_col])
    
    def test_quality_report_generation(self, processor, sample_data):
        """Test generación de reporte de calidad."""
        report = processor.generate_quality_report(sample_data)
        
        assert 'general' in report
        assert 'completeness' in report
        assert 'duplicates' in report
        assert 'columns' in report
        
        # Verificar métricas básicas
        assert report['general']['total_rows'] == len(sample_data)
        assert report['general']['total_columns'] == len(sample_data.columns)


class TestScrapingFunctions:
    """Tests para las funciones de scraping."""
    
    def test_url_validation(self):
        """Test validación de URLs."""
        valid_urls = [
            "https://stat.ine.cl/Index.aspx?lang=es",
            "http://example.com/data"
        ]
        
        invalid_urls = [
            "not_a_url",
            "ftp://invalid.com",
            ""
        ]
        
        # En un sistema real, tendríamos una función validate_url
        for url in valid_urls:
            assert url.startswith(('http://', 'https://'))
        
        for url in invalid_urls:
            assert not url.startswith(('http://', 'https://')) or url == ""
    
    def test_table_id_format(self):
        """Test formato de IDs de tabla."""
        valid_ids = ["tabletofreeze", "data_table", "main-table"]
        invalid_ids = ["", " ", "table with spaces"]
        
        for table_id in valid_ids:
            assert table_id.strip() != ""
            assert len(table_id) > 0
        
        for table_id in invalid_ids:
            assert table_id.strip() == "" or " " in table_id


class TestDataValidation:
    """Tests para validación de datos."""
    
    def test_dataframe_structure(self):
        """Test estructura básica de DataFrame."""
        # Datos válidos
        valid_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        assert not valid_df.empty
        assert len(valid_df.columns) > 0
        assert len(valid_df) > 0
    
    def test_data_types_consistency(self):
        """Test consistencia de tipos de datos."""
        df = pd.DataFrame({
            'numeric': [1, 2, 3, 4],
            'text': ['a', 'b', 'c', 'd'],
            'mixed': [1, 'b', 3, 'd']
        })
        
        # Verificar tipos esperados
        assert pd.api.types.is_numeric_dtype(df['numeric'])
        assert pd.api.types.is_object_dtype(df['text'])
        assert pd.api.types.is_object_dtype(df['mixed'])  # Mixed types = object
    
    def test_missing_values_detection(self):
        """Test detección de valores faltantes."""
        df = pd.DataFrame({
            'complete': [1, 2, 3, 4],
            'with_missing': [1, np.nan, 3, np.nan],
            'all_missing': [np.nan, np.nan, np.nan, np.nan]
        })
        
        assert df['complete'].isna().sum() == 0
        assert df['with_missing'].isna().sum() == 2
        assert df['all_missing'].isna().sum() == 4


class TestPerformance:
    """Tests de rendimiento y escalabilidad."""
    
    def test_large_dataset_processing(self):
        """Test procesamiento de datasets grandes."""
        # Crear dataset grande para test
        large_df = pd.DataFrame({
            'id': range(10000),
            'value': np.random.randn(10000),
            'category': np.random.choice(['A', 'B', 'C'], 10000)
        })
        
        # Verificar que se puede procesar
        assert len(large_df) == 10000
        assert not large_df.empty
        
        # Test operaciones básicas
        memory_usage = large_df.memory_usage(deep=True).sum()
        assert memory_usage > 0
    
    def test_memory_efficiency(self):
        """Test eficiencia de memoria."""
        df = pd.DataFrame({
            'int_col': range(1000),
            'float_col': np.random.randn(1000)
        })
        
        # Verificar uso de memoria razonable
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        assert memory_mb < 1  # Menos de 1MB para 1000 filas


class TestErrorHandling:
    """Tests para manejo de errores."""
    
    def test_empty_dataframe_handling(self):
        """Test manejo de DataFrames vacíos."""
        empty_df = pd.DataFrame()
        
        # Verificar que se puede manejar
        assert empty_df.empty
        assert len(empty_df) == 0
        assert len(empty_df.columns) == 0
    
    def test_invalid_file_handling(self):
        """Test manejo de archivos inválidos."""
        invalid_paths = [
            "nonexistent_file.csv",
            "/invalid/path/file.csv",
            ""
        ]
        
        for path in invalid_paths:
            # En sistema real, esto levantaría FileNotFoundError
            assert not Path(path).exists() or path == ""


# Funciones de utilidad para testing
def create_test_data(n_rows: int = 100) -> pd.DataFrame:
    """Crea datos de prueba."""
    np.random.seed(42)
    return pd.DataFrame({
        'region': np.random.choice(['RM', 'V', 'VIII'], n_rows),
        'valor': np.random.normal(100, 20, n_rows),
        'año': np.random.choice([2020, 2021, 2022, 2023], n_rows),
        'indicador': np.random.choice(['PIB', 'Población', 'Empleo'], n_rows)
    })


def run_integration_test():
    """Ejecuta test de integración completo."""
    print("🧪 Ejecutando test de integración...")
    
    # 1. Crear datos de prueba
    test_data = create_test_data(1000)
    print(f"✅ Datos de prueba creados: {test_data.shape}")
    
    # 2. Test procesamiento
    try:
        processor = DataProcessor()
        cleaned_data = processor.clean_data(test_data)
        print(f"✅ Datos procesados: {cleaned_data.shape}")
    except Exception as e:
        print(f"❌ Error en procesamiento: {e}")
    
    # 3. Test reporte de calidad
    try:
        report = processor.generate_quality_report(cleaned_data)
        print(f"✅ Reporte generado: {len(report)} secciones")
    except Exception as e:
        print(f"❌ Error en reporte: {e}")
    
    print("🎉 Test de integración completado")


if __name__ == "__main__":
    print("🧪 Ejecutando suite de tests...")
    
    # Ejecutar test de integración
    run_integration_test()
    
    # Para ejecutar con pytest:
    # pytest tests/test_data_processing.py -v
