# Dashboard de Desempleo Regional INE Chile

Pipeline profesional de Data Science para el análisis y visualización interactiva del desempleo regional en Chile, usando datos oficiales del INE.

## Estructura del Proyecto

- `data/` — Datos crudos y procesados.
- `scripts/` — Scripts de scraping, procesamiento y visualización.
- `outputs/dashboards/` — Dashboard final en HTML.
- `notebooks/` — Notebooks exploratorios (opcional, no requeridos para producción).
- `src/` — Código fuente modular.
- `requirements.txt` — Dependencias del proyecto.

```
scraping/
├── 📂 config/                     # Configuración general del pipeline
│   └── config.yaml
├── 📂 data/
│   ├── 📂 raw/                    # Datos extraídos sin procesar (scraping)
│   └── 📂 processed/              # Datos limpios y transformados
├── 📂 logs/                       # Logs de ejecución y auditoría
│   └── data_pipeline.log
├── 📂 notebooks/                  # Notebooks exploratorios y pipeline
│   └── Scraping_INE_Tables.ipynb # Pipeline principal automatizado
├── 📂 scripts/                    # Scripts de scraping, procesamiento, visualización y ML
│   ├── data_pipeline.py
│   ├── economist_visualizer.py
│   ├── ml_pipeline.py
│   ├── dashboard_generator.py         # Generador del dashboard interactivo final
│   ├── project_analyzer.py
│   ├── quality_monitor.py
│   └── scraping.py
├── 📂 src/                        # Código fuente modular y utilidades
│   ├── data_analyzer.py
│   ├── data_processor.py
│   └── report_generator.py
├── 📂 tests/                      # Tests unitarios y de integración
│   └── test_data_processing.py
├── � requirements.txt            # Dependencias del proyecto
├── � .gitignore                  # Archivos ignorados por Git
├── � Makefile                    # Automatización de tareas
├── 📄 setup.py                    # Instalación como paquete (opcional)
└── 📄 README.md                   # Documentación principal
```
## Uso Rápido

1. Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Ejecuta el pipeline principal:
   ```bash
   python scripts/dashboard_generator.py
   ```
3. El dashboard final estará en `outputs/dashboards/dashboard_ine_chile.html`.

## Autor

<strong>Bruno San Martín Navarro</strong><br>
<a href="https://www.linkedin.com/in/brunosanmartin/" target="_blank">LinkedIn</a> | <a href="https://github.com/brunosanmartin" target="_blank">GitHub</a>

## Licencia

MIT License.

## 🚀 Instalación y Configuración

### Prerrequisitos

- Python 3.8 o superior
- Google Chrome instalado
- Git (para clonar el repositorio)

### 1. Clonación del Repositorio

```bash
git clone https://github.com/tu-usuario/scraping-ine.git
cd scraping-ine
```

### 2. Configuración del Entorno Virtual

```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
source venv/bin/activate  # En macOS/Linux
# venv\Scripts\activate   # En Windows

# Verificar activación
which python  # Debe mostrar la ruta del venv
```

### 3. Instalación de Dependencias

```bash
# Instalar dependencias
pip install -r requirements.txt

# Verificar instalación
pip list
```

### 4. Configuración de ChromeDriver

El proyecto usa `webdriver-manager` que descarga automáticamente ChromeDriver. No requiere configuración manual.

## 💻 Uso del Sistema

### Opción 1: Notebook Interactivo (Recomendado)

```bash
# Activar entorno
source venv/bin/activate

# Lanzar Jupyter
jupyter notebook

# Abrir: notebooks/Scraping_INE_Tables.ipynb
```

### Opción 2: Script de Línea de Comandos

```bash
# Activar entorno
source venv/bin/activate

# Ejecutar script
python scripts/scraping.py
```

### Configuración Personalizada

Edita los parámetros en el notebook o script:

```python
# Configuración principal
URL_TARGET = "https://stat.ine.cl/Index.aspx?lang=es&SubSessionId=..."
TABLE_ID = "tabletofreeze"
OUTPUT_PREFIX = "ine_table"
HEADLESS_MODE = False  # True para modo silencioso
```

## 🛠️ Funcionalidades Principales

### ✨ Características Técnicas

| Característica | Descripción |
|----------------|-------------|
| **🔍 Detección Automática** | Identifica automáticamente el número de páginas disponibles |
| **📄 Procesamiento Paralelo** | Extrae datos de múltiples páginas secuencialmente |
| **🛡️ Manejo de Errores** | Recuperación automática ante fallos de conexión |
| **💾 Múltiples Formatos** | Exporta datos en CSV con encoding UTF-8 |
| **🎛️ Configuración Flexible** | Parámetros ajustables para diferentes sitios |
| **📊 Validación de Datos** | Verificación automática de integridad |

### 🔧 Componentes del Sistema

#### 1. **Motor de Scraping** (`scrape_all_pages_automatically`)
- Navegación automática entre páginas
- Detección inteligente de elementos
- Extracción robusta de tablas HTML

#### 2. **Gestión de Navegador** (`init_driver`)
- Configuración optimizada de Chrome
- Manejo de opciones de seguridad
- Gestión automática de ChromeDriver

#### 3. **Procesamiento de Datos** (`scrape_table`)
- Conversión HTML a DataFrame
- Validación de estructura de datos
- Debug automático de elementos

## 📊 Análisis de Datos

### Estructura de Datos Extraídos

Los datos extraídos siguen la estructura de las tablas INE:

```python
# Ejemplo de estructura
DataFrame columns: ['Region', 'Comuna', 'Indicador', 'Valor', 'Periodo']
Shape: (n_rows, n_columns)
```

### Métricas de Calidad

- **Completitud**: % de campos no nulos
- **Consistencia**: Validación de tipos de datos
- **Duplicados**: Detección automática de registros repetidos

## 🧪 Testing y Validación

### Casos de Prueba

```bash
# Ejecutar tests (cuando estén implementados)
python -m pytest tests/

# Validación manual
python scripts/validate_data.py
```

### Monitoreo de Performance

- **Tiempo de ejecución**: ~2-5 minutos para datasets típicos
- **Uso de memoria**: <500MB para datasets de 10k+ filas
- **Tasa de éxito**: >95% en condiciones normales de red

## 🚨 Troubleshooting

### Problemas Comunes

| Error | Causa | Solución |
|-------|-------|----------|
| `ChromeDriver not found` | Driver no instalado | Ejecutar `pip install webdriver-manager` |
| `Table not found` | ID de tabla incorrecto | Verificar `TABLE_ID` en el navegador |
| `Timeout error` | Conexión lenta | Aumentar `wait_time` parámetro |
| `Empty DataFrame` | Página no cargada | Verificar URL y disponibilidad del sitio |

### Logs y Debug

```python
# Activar modo debug
HEADLESS_MODE = False  # Ver navegador
logging.basicConfig(level=logging.DEBUG)
```

## 🔮 Roadmap y Mejoras Futuras

### Version 2.0 (Planificado)

- [ ] **API REST**: Endpoint para scraping programático
- [ ] **Dashboard**: Interfaz web para monitoreo
- [ ] **Scheduler**: Ejecuciones automáticas programadas
- [ ] **Data Pipeline**: Integración con bases de datos
- [ ] **Machine Learning**: Detección automática de anomalías

### Version 1.1 (En desarrollo)

- [ ] **Tests automatizados**: Cobertura completa
- [ ] **Docker**: Containerización del proyecto
- [ ] **CI/CD**: Pipeline de integración continua
- [ ] **Documentación API**: Swagger/OpenAPI

## 🤝 Contribuciones

### Guía para Colaboradores

1. **Fork** el repositorio
2. **Crear** feature branch (`git checkout -b feature/nueva-funcionalidad`)
3. **Commit** cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. **Push** a la branch (`git push origin feature/nueva-funcionalidad`)
5. **Crear** Pull Request

### Estándares de Código

- **Style Guide**: PEP 8
- **Formatter**: Black
- **Linter**: Flake8
- **Type Hints**: mypy

```bash
# Formatear código
black scripts/ notebooks/

# Linting
flake8 scripts/
```

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

## 👨‍💻 Autor

**Bruno San Martín**
- GitHub: [@brunosanmartin](https://github.com/SanMaBruno)
- LinkedIn: [bruno-sanmartin](https://www.linkedin.com/in/sanmabruno/)
- Email: bruno.sanmartin@email.com

## 🙏 Agradecimientos

- **Instituto Nacional de Estadísticas (INE) Chile** por proporcionar datos públicos
- **Selenium Community** por la excelente documentación
- **Pandas Development Team** por la librería de manipulación de datos

---

<div align="center">

**⭐ Si este proyecto te resultó útil, considera darle una estrella ⭐**

</div>
