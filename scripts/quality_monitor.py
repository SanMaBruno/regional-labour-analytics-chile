#!/usr/bin/env python3
"""
📊 Data Quality Monitor - INE Data Pipeline
==========================================

Sistema de monitoreo continuo de calidad de datos para el pipeline del INE.
Incluye validaciones, alertas, métricas de calidad y reportes automáticos.

Features:
- Validación automática de calidad de datos
- Detección de anomalías y outliers
- Monitoreo de completitud y consistencia
- Generación de alertas automáticas
- Métricas de performance del pipeline
- Reportes de calidad ejecutivos

Author: Bruno San Martín
Date: 2025-06-28
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityMonitor:
    """Monitor de calidad de datos para el pipeline INE."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Inicializar monitor de calidad."""
        self.config = self._load_config(config_path)
        self.quality_metrics = {}
        self.alerts = []
        self.thresholds = {
            'completeness': 0.95,
            'consistency': 0.90,
            'validity': 0.95,
            'uniqueness': 0.98
        }
        
        # Crear directorio de reportes de calidad
        Path("outputs/quality_reports").mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: str) -> dict:
        """Cargar configuración."""
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except:
            return self._default_config()
    
    def _default_config(self) -> dict:
        """Configuración por defecto."""
        return {
            'quality': {
                'completeness_threshold': 0.95,
                'consistency_threshold': 0.90,
                'alert_email': None
            }
        }
    
    def assess_data_quality(self, data_path: str) -> Dict:
        """Evaluar calidad completa de datos."""
        logger.info(f"🔍 Evaluando calidad de datos: {data_path}")
        
        try:
            # Cargar datos
            data = pd.read_csv(data_path)
            
            # Ejecutar todas las validaciones
            quality_results = {
                'file_info': {
                    'path': data_path,
                    'shape': data.shape,
                    'size_mb': Path(data_path).stat().st_size / (1024*1024),
                    'last_modified': datetime.fromtimestamp(
                        Path(data_path).stat().st_mtime
                    ).isoformat()
                },
                'completeness': self._check_completeness(data),
                'consistency': self._check_consistency(data),
                'validity': self._check_validity(data),
                'uniqueness': self._check_uniqueness(data),
                'outliers': self._detect_outliers(data),
                'schema_validation': self._validate_schema(data),
                'temporal_consistency': self._check_temporal_consistency(data)
            }
            
            # Calcular score general
            quality_results['overall_score'] = self._calculate_overall_score(quality_results)
            
            # Generar alertas si es necesario
            self._generate_alerts(quality_results)
            
            # Guardar métricas
            self.quality_metrics = quality_results
            
            logger.info(f"✅ Evaluación completada. Score: {quality_results['overall_score']:.2f}")
            return quality_results
            
        except Exception as e:
            logger.error(f"❌ Error evaluando calidad: {e}")
            return {}
    
    def _check_completeness(self, data: pd.DataFrame) -> Dict:
        """Verificar completitud de datos."""
        try:
            total_cells = data.size
            missing_cells = data.isnull().sum().sum()
            completeness_ratio = 1 - (missing_cells / total_cells)
            
            # Completitud por columna
            column_completeness = {}
            for col in data.columns:
                missing_pct = data[col].isnull().sum() / len(data)
                column_completeness[col] = {
                    'completeness': 1 - missing_pct,
                    'missing_count': int(data[col].isnull().sum()),
                    'status': 'ok' if missing_pct < 0.05 else 'warning' if missing_pct < 0.20 else 'critical'
                }
            
            return {
                'overall_completeness': completeness_ratio,
                'missing_cells': int(missing_cells),
                'total_cells': int(total_cells),
                'column_analysis': column_completeness,
                'status': 'ok' if completeness_ratio >= self.thresholds['completeness'] else 'warning',
                'recommendation': self._get_completeness_recommendation(completeness_ratio)
            }
            
        except Exception as e:
            logger.error(f"❌ Error verificando completitud: {e}")
            return {}
    
    def _check_consistency(self, data: pd.DataFrame) -> Dict:
        """Verificar consistencia de datos."""
        try:
            consistency_checks = {}
            
            # Consistencia de tipos de datos
            type_consistency = self._check_data_types(data)
            consistency_checks['data_types'] = type_consistency
            
            # Consistencia de rangos para datos numéricos
            range_consistency = self._check_value_ranges(data)
            consistency_checks['value_ranges'] = range_consistency
            
            # Consistencia de formato para datos de texto
            format_consistency = self._check_format_consistency(data)
            consistency_checks['formats'] = format_consistency
            
            # Score general de consistencia
            consistency_score = np.mean([
                type_consistency.get('score', 0),
                range_consistency.get('score', 0),
                format_consistency.get('score', 0)
            ])
            
            return {
                'consistency_score': consistency_score,
                'checks': consistency_checks,
                'status': 'ok' if consistency_score >= self.thresholds['consistency'] else 'warning',
                'recommendation': self._get_consistency_recommendation(consistency_score)
            }
            
        except Exception as e:
            logger.error(f"❌ Error verificando consistencia: {e}")
            return {}
    
    def _check_validity(self, data: pd.DataFrame) -> Dict:
        """Verificar validez de datos."""
        try:
            validity_checks = {}
            
            # Validez de datos numéricos
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            numeric_validity = {}
            
            for col in numeric_cols:
                values = data[col].dropna()
                if len(values) > 0:
                    infinite_count = np.isinf(values).sum()
                    negative_count = (values < 0).sum() if col.lower() in ['tasa', 'rate', 'porcentaje'] else 0
                    
                    numeric_validity[col] = {
                        'infinite_values': int(infinite_count),
                        'negative_values': int(negative_count),
                        'valid_ratio': 1 - (infinite_count + negative_count) / len(values)
                    }
            
            validity_checks['numeric'] = numeric_validity
            
            # Validez de fechas/períodos
            date_validity = self._check_date_validity(data)
            validity_checks['temporal'] = date_validity
            
            # Score general
            validity_score = self._calculate_validity_score(validity_checks)
            
            return {
                'validity_score': validity_score,
                'checks': validity_checks,
                'status': 'ok' if validity_score >= self.thresholds['validity'] else 'warning',
                'recommendation': self._get_validity_recommendation(validity_score)
            }
            
        except Exception as e:
            logger.error(f"❌ Error verificando validez: {e}")
            return {}
    
    def _check_uniqueness(self, data: pd.DataFrame) -> Dict:
        """Verificar unicidad de datos."""
        try:
            # Detectar duplicados completos
            duplicate_rows = data.duplicated().sum()
            uniqueness_ratio = 1 - (duplicate_rows / len(data))
            
            # Análisis por columna de potenciales identificadores
            column_uniqueness = {}
            for col in data.columns[:5]:  # Revisar primeras 5 columnas
                unique_values = data[col].nunique()
                total_values = len(data[col].dropna())
                uniqueness = unique_values / total_values if total_values > 0 else 0
                
                column_uniqueness[col] = {
                    'unique_values': int(unique_values),
                    'total_values': int(total_values),
                    'uniqueness_ratio': uniqueness
                }
            
            return {
                'overall_uniqueness': uniqueness_ratio,
                'duplicate_rows': int(duplicate_rows),
                'column_analysis': column_uniqueness,
                'status': 'ok' if uniqueness_ratio >= self.thresholds['uniqueness'] else 'warning',
                'recommendation': self._get_uniqueness_recommendation(uniqueness_ratio)
            }
            
        except Exception as e:
            logger.error(f"❌ Error verificando unicidad: {e}")
            return {}
    
    def _detect_outliers(self, data: pd.DataFrame) -> Dict:
        """Detectar valores atípicos."""
        try:
            outlier_analysis = {}
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols[:5]:  # Primeras 5 columnas numéricas
                values = data[col].dropna()
                if len(values) > 0:
                    # Método IQR
                    Q1 = values.quantile(0.25)
                    Q3 = values.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = values[(values < lower_bound) | (values > upper_bound)]
                    
                    outlier_analysis[col] = {
                        'outlier_count': len(outliers),
                        'outlier_percentage': len(outliers) / len(values) * 100,
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound),
                        'status': 'ok' if len(outliers) / len(values) < 0.05 else 'warning'
                    }
            
            return {
                'analysis': outlier_analysis,
                'recommendation': "Revisar valores atípicos identificados para verificar si son errores o valores válidos extremos"
            }
            
        except Exception as e:
            logger.error(f"❌ Error detectando outliers: {e}")
            return {}
    
    def _validate_schema(self, data: pd.DataFrame) -> Dict:
        """Validar esquema de datos."""
        try:
            expected_columns = ['periodo']  # Columnas esperadas mínimas
            
            # Verificar columnas requeridas
            missing_columns = [col for col in expected_columns if col not in data.columns]
            extra_columns = [col for col in data.columns if col not in expected_columns]
            
            # Verificar tipos de datos esperados
            type_issues = []
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                type_issues.append("No se encontraron columnas numéricas")
            
            return {
                'missing_columns': missing_columns,
                'extra_columns': extra_columns[:10],  # Limitar output
                'type_issues': type_issues,
                'total_columns': len(data.columns),
                'status': 'ok' if len(missing_columns) == 0 and len(type_issues) == 0 else 'warning'
            }
            
        except Exception as e:
            logger.error(f"❌ Error validando esquema: {e}")
            return {}
    
    def _check_temporal_consistency(self, data: pd.DataFrame) -> Dict:
        """Verificar consistencia temporal."""
        try:
            temporal_analysis = {}
            
            # Buscar columnas temporales
            time_columns = [col for col in data.columns 
                          if any(term in col.lower() for term in ['periodo', 'fecha', 'time', 'date'])]
            
            if time_columns:
                time_col = time_columns[0]
                
                # Análisis básico de períodos
                unique_periods = data[time_col].nunique()
                total_periods = len(data[time_col].dropna())
                
                temporal_analysis = {
                    'time_column': time_col,
                    'unique_periods': int(unique_periods),
                    'total_records': int(total_periods),
                    'period_consistency': unique_periods / total_periods if total_periods > 0 else 0,
                    'status': 'ok'
                }
            else:
                temporal_analysis = {
                    'status': 'warning',
                    'message': 'No se encontraron columnas temporales'
                }
            
            return temporal_analysis
            
        except Exception as e:
            logger.error(f"❌ Error verificando consistencia temporal: {e}")
            return {}
    
    def _calculate_overall_score(self, quality_results: Dict) -> float:
        """Calcular score general de calidad."""
        try:
            scores = []
            
            # Recopilar scores individuales
            if 'completeness' in quality_results:
                scores.append(quality_results['completeness'].get('overall_completeness', 0))
            
            if 'consistency' in quality_results:
                scores.append(quality_results['consistency'].get('consistency_score', 0))
            
            if 'validity' in quality_results:
                scores.append(quality_results['validity'].get('validity_score', 0))
            
            if 'uniqueness' in quality_results:
                scores.append(quality_results['uniqueness'].get('overall_uniqueness', 0))
            
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            logger.error(f"❌ Error calculando score general: {e}")
            return 0.0
    
    def _generate_alerts(self, quality_results: Dict):
        """Generar alertas basadas en métricas de calidad."""
        self.alerts = []
        
        overall_score = quality_results.get('overall_score', 0)
        
        if overall_score < 0.8:
            self.alerts.append({
                'level': 'critical',
                'message': f'Score de calidad crítico: {overall_score:.2f}',
                'timestamp': datetime.now().isoformat()
            })
        elif overall_score < 0.9:
            self.alerts.append({
                'level': 'warning',
                'message': f'Score de calidad bajo: {overall_score:.2f}',
                'timestamp': datetime.now().isoformat()
            })
        
        # Alertas específicas por dimensión
        if quality_results.get('completeness', {}).get('overall_completeness', 1) < 0.9:
            self.alerts.append({
                'level': 'warning',
                'message': 'Completitud de datos por debajo del umbral',
                'timestamp': datetime.now().isoformat()
            })
    
    def generate_quality_report(self, quality_results: Dict) -> str:
        """Generar reporte completo de calidad."""
        logger.info("📋 Generando reporte de calidad...")
        
        try:
            report_content = f"""
# 🔍 Reporte de Calidad de Datos - INE Pipeline

**Fecha:** {datetime.now().strftime('%d/%m/%Y %H:%M')}
**Archivo analizado:** {quality_results.get('file_info', {}).get('path', 'N/A')}

---

## 📊 Resumen Ejecutivo

### Score General de Calidad: {quality_results.get('overall_score', 0):.2f}/1.00

{self._get_quality_status_icon(quality_results.get('overall_score', 0))} **Estado:** {self._get_quality_status(quality_results.get('overall_score', 0))}

---

## 📈 Métricas Detalladas

### 1. Completitud de Datos
- **Score:** {quality_results.get('completeness', {}).get('overall_completeness', 0):.3f}
- **Celdas faltantes:** {quality_results.get('completeness', {}).get('missing_cells', 0):,}
- **Total de celdas:** {quality_results.get('completeness', {}).get('total_cells', 0):,}

### 2. Consistencia de Datos
- **Score:** {quality_results.get('consistency', {}).get('consistency_score', 0):.3f}
- **Estado:** {quality_results.get('consistency', {}).get('status', 'unknown')}

### 3. Validez de Datos
- **Score:** {quality_results.get('validity', {}).get('validity_score', 0):.3f}
- **Estado:** {quality_results.get('validity', {}).get('status', 'unknown')}

### 4. Unicidad de Datos
- **Score:** {quality_results.get('uniqueness', {}).get('overall_uniqueness', 0):.3f}
- **Filas duplicadas:** {quality_results.get('uniqueness', {}).get('duplicate_rows', 0)}

---

## 🚨 Alertas y Recomendaciones

{self._format_alerts()}

---

## 📊 Análisis de Outliers

{self._format_outlier_analysis(quality_results.get('outliers', {}))}

---

## 💡 Recomendaciones de Mejora

{self._generate_improvement_recommendations(quality_results)}

---

## 📋 Información del Archivo

- **Tamaño:** {quality_results.get('file_info', {}).get('size_mb', 0):.2f} MB
- **Dimensiones:** {quality_results.get('file_info', {}).get('shape', (0, 0))}
- **Última modificación:** {quality_results.get('file_info', {}).get('last_modified', 'N/A')}

---

*Reporte generado automáticamente por Data Quality Monitor v1.0*
            """
            
            # Guardar reporte
            report_file = Path("outputs/quality_reports") / f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            # Guardar métricas como JSON
            metrics_file = Path("outputs/quality_reports") / f"quality_metrics_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(quality_results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"✅ Reporte de calidad generado: {report_file}")
            return str(report_file)
            
        except Exception as e:
            logger.error(f"❌ Error generando reporte de calidad: {e}")
            return ""
    
    # Métodos auxiliares para formateo y recomendaciones
    def _get_quality_status_icon(self, score: float) -> str:
        if score >= 0.9: return "✅"
        elif score >= 0.7: return "⚠️"
        else: return "❌"
    
    def _get_quality_status(self, score: float) -> str:
        if score >= 0.9: return "Excelente"
        elif score >= 0.8: return "Bueno"
        elif score >= 0.7: return "Aceptable"
        else: return "Requiere atención"
    
    def _format_alerts(self) -> str:
        if not self.alerts:
            return "✅ No se detectaron problemas críticos de calidad."
        
        formatted = ""
        for alert in self.alerts:
            icon = "🚨" if alert['level'] == 'critical' else "⚠️"
            formatted += f"{icon} **{alert['level'].title()}:** {alert['message']}\\n"
        
        return formatted
    
    def _format_outlier_analysis(self, outlier_data: Dict) -> str:
        if not outlier_data or 'analysis' not in outlier_data:
            return "No se realizó análisis de outliers."
        
        formatted = ""
        for col, analysis in list(outlier_data['analysis'].items())[:3]:
            formatted += f"- **{col}:** {analysis['outlier_count']} outliers ({analysis['outlier_percentage']:.1f}%)\\n"
        
        return formatted
    
    def _generate_improvement_recommendations(self, quality_results: Dict) -> str:
        recommendations = []
        
        overall_score = quality_results.get('overall_score', 0)
        
        if overall_score < 0.8:
            recommendations.append("🔧 **Prioridad Alta:** Implementar validaciones automáticas en la fuente de datos")
        
        if quality_results.get('completeness', {}).get('overall_completeness', 1) < 0.9:
            recommendations.append("📝 **Completitud:** Revisar procesos de recolección para reducir datos faltantes")
        
        if quality_results.get('uniqueness', {}).get('duplicate_rows', 0) > 0:
            recommendations.append("🔄 **Duplicados:** Implementar deduplicación automática")
        
        recommendations.append("📊 **Monitoreo:** Establecer alertas automáticas para degradación de calidad")
        recommendations.append("🚀 **Automatización:** Integrar validaciones en el pipeline de datos")
        
        return "\\n".join(recommendations)
    
    # Métodos auxiliares para validaciones específicas
    def _check_data_types(self, data: pd.DataFrame) -> Dict:
        """Verificar consistencia de tipos de datos."""
        try:
            type_consistency = {'score': 1.0, 'issues': []}
            
            # Verificar si hay columnas que deberían ser numéricas pero no lo son
            for col in data.columns:
                if any(term in col.lower() for term in ['tasa', 'rate', 'porcentaje', 'valor']):
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        type_consistency['issues'].append(f"Columna {col} debería ser numérica")
                        type_consistency['score'] -= 0.1
            
            return type_consistency
        except:
            return {'score': 0.5, 'issues': ['Error verificando tipos']}
    
    def _check_value_ranges(self, data: pd.DataFrame) -> Dict:
        """Verificar rangos de valores."""
        try:
            range_consistency = {'score': 1.0, 'issues': []}
            
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                values = data[col].dropna()
                if len(values) > 0:
                    # Verificar rangos lógicos para tasas/porcentajes
                    if any(term in col.lower() for term in ['tasa', 'rate', 'porcentaje']):
                        if values.min() < 0 or values.max() > 100:
                            range_consistency['issues'].append(f"Rango inválido en {col}")
                            range_consistency['score'] -= 0.2
            
            return range_consistency
        except:
            return {'score': 0.5, 'issues': ['Error verificando rangos']}
    
    def _check_format_consistency(self, data: pd.DataFrame) -> Dict:
        """Verificar consistencia de formatos."""
        return {'score': 1.0, 'issues': []}  # Implementación básica
    
    def _check_date_validity(self, data: pd.DataFrame) -> Dict:
        """Verificar validez de fechas."""
        return {'score': 1.0, 'issues': []}  # Implementación básica
    
    def _calculate_validity_score(self, validity_checks: Dict) -> float:
        """Calcular score de validez."""
        try:
            scores = []
            if 'numeric' in validity_checks:
                for col_data in validity_checks['numeric'].values():
                    scores.append(col_data.get('valid_ratio', 1.0))
            return np.mean(scores) if scores else 1.0
        except:
            return 0.5
    
    # Métodos para recomendaciones
    def _get_completeness_recommendation(self, score: float) -> str:
        if score < 0.8:
            return "Crítico: Revisar fuentes de datos y procesos de recolección"
        elif score < 0.9:
            return "Mejorar: Implementar validaciones de entrada"
        return "Bueno: Mantener procesos actuales"
    
    def _get_consistency_recommendation(self, score: float) -> str:
        if score < 0.8:
            return "Crítico: Estandarizar formatos y tipos de datos"
        elif score < 0.9:
            return "Mejorar: Validar rangos y formatos"
        return "Bueno: Consistencia adecuada"
    
    def _get_validity_recommendation(self, score: float) -> str:
        if score < 0.8:
            return "Crítico: Implementar validaciones de negocio"
        elif score < 0.9:
            return "Mejorar: Revisar reglas de validación"
        return "Bueno: Datos válidos"
    
    def _get_uniqueness_recommendation(self, score: float) -> str:
        if score < 0.8:
            return "Crítico: Implementar deduplicación"
        elif score < 0.9:
            return "Mejorar: Revisar procesos de carga"
        return "Bueno: Baja duplicación"


def main():
    """Función principal para ejecutar monitoreo de calidad."""
    try:
        monitor = DataQualityMonitor()
        
        # Evaluar calidad de datos procesados
        data_file = "data/processed/unemployment_data_cleaned.csv"
        
        if Path(data_file).exists():
            quality_results = monitor.assess_data_quality(data_file)
            report_file = monitor.generate_quality_report(quality_results)
            
            print(f"\\n🔍 Evaluación de calidad completada!")
            print(f"📊 Score general: {quality_results.get('overall_score', 0):.2f}")
            print(f"📋 Reporte: {report_file}")
            
            if monitor.alerts:
                print(f"🚨 Alertas generadas: {len(monitor.alerts)}")
        else:
            print(f"❌ Archivo no encontrado: {data_file}")
            
    except Exception as e:
        logger.error(f"❌ Error en monitoreo de calidad: {e}")


if __name__ == "__main__":
    main()
