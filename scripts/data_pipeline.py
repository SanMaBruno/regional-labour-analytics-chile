#!/usr/bin/env python3
"""
🔄 Data Pipeline - INE Unemployment Data Processing
=================================================

Pipeline automatizado completo para procesamiento, análisis y reporte
de datos de desempleo del INE de Chile.

Features:
- Limpieza y transformación automática de datos
- Análisis estadístico y temporal
- Generación de visualizaciones profesionales
- Reportes ejecutivos y técnicos
- Monitoreo de calidad de datos
- Logging completo del proceso

Usage:
    python scripts/data_pipeline.py --mode full
    python scripts/data_pipeline.py --mode analysis
    python scripts/data_pipeline.py --mode reports

Author: Bruno San Martín
Date: 2025-06-28
"""

import argparse
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import yaml
import sys
import os

# Añadir src al path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_processor import DataProcessor
from data_analyzer import DataAnalyzer
from report_generator import ReportGenerator

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class INEDataPipeline:
    """Pipeline completo para datos de desempleo del INE."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Inicializar pipeline con configuración."""
        self.config = self._load_config(config_path)
        self.processor = DataProcessor(
            output_dir=self.config['data']['paths']['processed']
        )
        self.analyzer = DataAnalyzer(
            output_dir='outputs/visualizations'
        )
        self.reporter = ReportGenerator(
            output_dir='outputs/reports'
        )
        
        # Crear directorios necesarios
        for dir_path in self.config['data']['paths'].values():
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Crear directorios de outputs
        for subdir in ['reports', 'visualizations', 'dashboards']:
            Path(f'outputs/{subdir}').mkdir(parents=True, exist_ok=True)
            
    def _load_config(self, config_path: str) -> dict:
        """Cargar configuración desde YAML."""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
            return self._default_config()
    
    def _default_config(self) -> dict:
        """Configuración por defecto."""
        return {
            'data': {
                'paths': {
                    'raw': 'data/raw',
                    'processed': 'data/processed',
                    'outputs': 'outputs',
                    'logs': 'logs'
                }
            },
            'data_sources': {
                'ine_unemployment': 'data/raw/ine_table_combined.csv'
            },
            'analysis': {
                'time_periods': ['quarterly', 'annual'],
                'regions': ['all', 'metropolitan'],
                'demographics': ['total', 'by_gender']
            }
        }
    
    def run_data_processing(self) -> str:
        """Ejecutar limpieza y procesamiento de datos."""
        logger.info("🔄 Iniciando procesamiento de datos...")
        
        try:
            # Cargar datos crudos - usar archivo directo si config falla
            raw_file = 'data/raw/ine_table_combined.csv'
            if 'data_sources' in self.config:
                raw_file = self.config['data_sources'].get('ine_unemployment', raw_file)
            
            if not Path(raw_file).exists():
                raise FileNotFoundError(f"Archivo no encontrado: {raw_file}")
            
            # Procesar datos
            processed_data = self.processor.load_and_clean_data(raw_file)
            
            # Guardar datos procesados
            output_file = Path(self.config['data']['paths']['processed']) / 'unemployment_data_cleaned.csv'
            processed_data.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            logger.info(f"✅ Datos procesados guardados: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"❌ Error en procesamiento: {e}")
            raise
    
    def run_analysis(self, processed_file: str) -> dict:
        """Ejecutar análisis completo de datos."""
        logger.info("📊 Iniciando análisis de datos...")
        
        try:
            # Cargar datos procesados
            self.analyzer.load_data(processed_file)
            
            # Análisis descriptivo
            stats = self.analyzer.descriptive_analysis()
            
            # Análisis temporal
            trends = self.analyzer.time_series_analysis()
            
            # Análisis regional
            regional = self.analyzer.regional_analysis()
            
            # Generar visualizaciones
            viz_files = self.analyzer.create_visualizations()
            
            results = {
                'statistics': stats,
                'trends': trends,
                'regional': regional,
                'visualizations': viz_files
            }
            
            logger.info(f"✅ Análisis completado. Visualizaciones: {len(viz_files)}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en análisis: {e}")
            raise
    
    def run_reporting(self, processed_file: str, analysis_results: dict) -> dict:
        """Generar reportes ejecutivos y técnicos."""
        logger.info("📋 Generando reportes...")
        
        try:
            # Cargar datos para reportes
            self.reporter.load_data(processed_file)
            
            # Reporte ejecutivo
            exec_report = self.reporter.generate_executive_report(
                analysis_results=analysis_results
            )
            
            # Reporte técnico
            tech_report = self.reporter.generate_technical_report(
                analysis_results=analysis_results
            )
            
            # Dashboard interactivo
            dashboard = self.reporter.create_interactive_dashboard()
            
            reports = {
                'executive': exec_report,
                'technical': tech_report,
                'dashboard': dashboard
            }
            
            logger.info("✅ Reportes generados exitosamente")
            return reports
            
        except Exception as e:
            logger.error(f"❌ Error generando reportes: {e}")
            raise
    
    def run_full_pipeline(self) -> dict:
        """Ejecutar pipeline completo."""
        logger.info("🚀 Iniciando pipeline completo de Data Science...")
        
        start_time = datetime.now()
        
        try:
            # Paso 1: Procesamiento
            processed_file = self.run_data_processing()
            
            # Paso 2: Análisis
            analysis_results = self.run_analysis(processed_file)
            
            # Paso 3: Reportes
            reports = self.run_reporting(processed_file, analysis_results)
            
            # Resumen final
            end_time = datetime.now()
            duration = end_time - start_time
            
            summary = {
                'processed_data': processed_file,
                'analysis': analysis_results,
                'reports': reports,
                'execution_time': str(duration),
                'status': 'completed',
                'timestamp': end_time.isoformat()
            }
            
            logger.info(f"🎉 Pipeline completado en {duration}")
            
            # Guardar resumen
            summary_file = Path('outputs/reports') / 'pipeline_summary.json'
            import json
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Pipeline falló: {e}")
            raise


def main():
    """Función principal con argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Pipeline de Data Science para datos del INE')
    parser.add_argument(
        '--mode', 
        choices=['full', 'process', 'analysis', 'reports'],
        default='full',
        help='Modo de ejecución del pipeline'
    )
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Archivo de configuración'
    )
    parser.add_argument(
        '--input',
        help='Archivo de entrada (para modos específicos)'
    )
    
    args = parser.parse_args()
    
    # Crear directorio de logs
    Path('logs').mkdir(exist_ok=True)
    
    try:
        pipeline = INEDataPipeline(args.config)
        
        if args.mode == 'full':
            result = pipeline.run_full_pipeline()
            print(f"\n🎉 Pipeline completado exitosamente!")
            print(f"📁 Resultados en: outputs/")
            
        elif args.mode == 'process':
            result = pipeline.run_data_processing()
            print(f"✅ Datos procesados: {result}")
            
        elif args.mode == 'analysis':
            if not args.input:
                args.input = 'data/processed/unemployment_data_cleaned.csv'
            result = pipeline.run_analysis(args.input)
            print(f"📊 Análisis completado: {len(result['visualizations'])} visualizaciones")
            
        elif args.mode == 'reports':
            if not args.input:
                args.input = 'data/processed/unemployment_data_cleaned.csv'
            # Para reportes necesitamos también cargar análisis previo
            analysis_results = pipeline.run_analysis(args.input)
            result = pipeline.run_reporting(args.input, analysis_results)
            print(f"📋 Reportes generados: {list(result.keys())}")
        
    except Exception as e:
        logger.error(f"❌ Error ejecutando pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
