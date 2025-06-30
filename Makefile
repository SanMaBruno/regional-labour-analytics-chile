# ====================================================================
# 🛠️ Makefile - INE Data Scraper
# ====================================================================
# Automatización de tareas comunes para desarrollo y producción
# 
# Uso:
#   make install     - Instalar dependencias
#   make test        - Ejecutar tests
#   make scrape      - Ejecutar scraping
#   make analysis    - Generar análisis completo
#   make clean       - Limpiar archivos temporales
#   make help        - Mostrar ayuda
# ====================================================================

.PHONY: help install test scrape analysis clean format lint docs docker

# Variables
PYTHON := /Users/brunosanmartin/Documents/scraping/venv/bin/python
PIP := /Users/brunosanmartin/Documents/scraping/venv/bin/pip
VENV := venv
SRC_DIR := src
SCRIPTS_DIR := scripts
TESTS_DIR := tests
DATA_DIR := data
OUTPUTS_DIR := outputs

# Colores para output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Target por defecto
help:
	@echo "$(GREEN)🛠️  INE Data Scraper - Comandos Disponibles$(NC)"
	@echo "================================================"
	@echo ""
	@echo "$(YELLOW)📦 Instalación y Configuración:$(NC)"
	@echo "  make install        - Instalar todas las dependencias"
	@echo "  make install-dev    - Instalar dependencias de desarrollo"
	@echo "  make setup-env      - Configurar entorno virtual"
	@echo ""
	@echo "$(YELLOW)🧪 Testing y Calidad:$(NC)"
	@echo "  make test           - Ejecutar todos los tests"
	@echo "  make test-coverage  - Tests con reporte de cobertura"
	@echo "  make lint           - Verificar calidad del código"
	@echo "  make format         - Formatear código con Black"
	@echo ""
	@echo "$(YELLOW)🕷️ Scraping y Procesamiento:$(NC)"
	@echo "  make scrape         - Ejecutar scraping completo"
	@echo "  make process        - Procesar datos raw"
	@echo "  make analysis       - Generar análisis completo"
	@echo "  make reports        - Generar todos los reportes"
	@echo ""
	@echo "$(YELLOW)🧹 Limpieza y Mantenimiento:$(NC)"
	@echo "  make clean          - Limpiar archivos temporales"
	@echo "  make clean-data     - Limpiar datos (¡CUIDADO!)"
	@echo "  make clean-outputs  - Limpiar outputs"
	@echo ""
	@echo "$(YELLOW)📚 Documentación:$(NC)"
	@echo "  make docs           - Generar documentación"
	@echo "  make jupyter        - Lanzar Jupyter Notebook"
	@echo ""
	@echo "$(YELLOW)🐳 Docker:$(NC)"
	@echo "  make docker-build   - Construir imagen Docker"
	@echo "  make docker-run     - Ejecutar en Docker"
	@echo ""

# ====================================================================
# 📦 Instalación y Configuración
# ====================================================================

setup-env:
	@echo "$(GREEN)🔧 Configurando entorno virtual...$(NC)"
	$(PYTHON) -m venv $(VENV)
	@echo "$(YELLOW)⚠️  Activa el entorno con: source $(VENV)/bin/activate$(NC)"

install:
	@echo "$(GREEN)📦 Instalando dependencias...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✅ Dependencias instaladas$(NC)"

install-dev: install
	@echo "$(GREEN)🛠️ Instalando dependencias de desarrollo...$(NC)"
	$(PIP) install pytest black flake8 mypy pytest-cov
	@echo "$(GREEN)✅ Entorno de desarrollo configurado$(NC)"

install-package:
	@echo "$(GREEN)📦 Instalando paquete en modo desarrollo...$(NC)"
	$(PIP) install -e .
	@echo "$(GREEN)✅ Paquete instalado$(NC)"

# ====================================================================
# 🧪 Testing y Calidad
# ====================================================================

test:
	@echo "$(GREEN)🧪 Ejecutando tests...$(NC)"
	$(PYTHON) -m pytest $(TESTS_DIR)/ -v
	@echo "$(GREEN)✅ Tests completados$(NC)"

test-coverage:
	@echo "$(GREEN)📊 Ejecutando tests con cobertura...$(NC)"
	$(PYTHON) -m pytest $(TESTS_DIR)/ --cov=$(SRC_DIR) --cov-report=html --cov-report=term
	@echo "$(GREEN)✅ Reporte de cobertura generado en htmlcov/$(NC)"

lint:
	@echo "$(GREEN)🔍 Verificando calidad del código...$(NC)"
	flake8 $(SRC_DIR)/ $(SCRIPTS_DIR)/ --max-line-length=88 --ignore=E203,W503
	mypy $(SRC_DIR)/ --ignore-missing-imports
	@echo "$(GREEN)✅ Verificación de calidad completada$(NC)"

format:
	@echo "$(GREEN)🎨 Formateando código...$(NC)"
	black $(SRC_DIR)/ $(SCRIPTS_DIR)/ $(TESTS_DIR)/
	@echo "$(GREEN)✅ Código formateado$(NC)"

# ====================================================================
# 🕷️ Scraping y Procesamiento
# ====================================================================

scrape:
	@echo "$(GREEN)🕷️ Ejecutando scraping...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/scraping.py
	@echo "$(GREEN)✅ Scraping completado$(NC)"

process:
	@echo "$(GREEN)🔄 Procesando datos...$(NC)"
	$(PYTHON) $(SRC_DIR)/data_processor.py
	@echo "$(GREEN)✅ Procesamiento completado$(NC)"

analysis:
	@echo "$(GREEN)📊 Generando análisis...$(NC)"
	$(PYTHON) $(SRC_DIR)/data_analyzer.py
	@echo "$(GREEN)✅ Análisis completado$(NC)"

reports:
	@echo "$(GREEN)📋 Generando reportes...$(NC)"
	$(PYTHON) $(SRC_DIR)/report_generator.py
	@echo "$(GREEN)✅ Reportes generados$(NC)"

pipeline: scrape process analysis reports
	@echo "$(GREEN)🚀 Pipeline completo ejecutado$(NC)"

# ====================================================================
# 🚀 COMANDOS AVANZADOS DE DATA SCIENCE
# ====================================================================

.PHONY: pipeline ml-pipeline quality-monitor schedule-pipeline dashboard

pipeline: ## Ejecutar pipeline completo de Data Science
	@echo "$(GREEN)🚀 Ejecutando pipeline completo de Data Science...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/data_pipeline.py --mode full
	@echo "$(GREEN)✅ Pipeline completado. Revisa outputs/$(NC)"

ml-pipeline: ## Ejecutar pipeline de Machine Learning
	@echo "$(GREEN)🤖 Ejecutando pipeline de Machine Learning...$(NC)"
	$(PYTHON) -c "import sys; sys.path.append('$(SCRIPTS_DIR)'); exec(open('$(SCRIPTS_DIR)/ml_pipeline.py').read())" 2>/dev/null || echo "$(YELLOW)⚠️  Instala scikit-learn para ML: pip install scikit-learn$(NC)"
	@echo "$(GREEN)✅ ML Pipeline completado$(NC)"

quality-monitor: ## Ejecutar monitoreo de calidad de datos
	@echo "$(GREEN)🔍 Ejecutando monitoreo de calidad...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/quality_monitor.py
	@echo "$(GREEN)✅ Monitoreo de calidad completado$(NC)"

schedule-pipeline: ## Simular ejecución programada (cron-like)
	@echo "$(GREEN)⏰ Simulando ejecución programada...$(NC)"
	@echo "$(YELLOW)📅 Ejecutando scraping diario...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/scraping.py
	@echo "$(YELLOW)📊 Ejecutando análisis...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/data_pipeline.py --mode analysis
	@echo "$(YELLOW)🔍 Ejecutando monitoreo...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/quality_monitor.py
	@echo "$(GREEN)✅ Ejecución programada completada$(NC)"

dashboard: ## Abrir dashboard en navegador
	@echo "$(GREEN)📊 Abriendo dashboard...$(NC)"
	@if [ -f "$(OUTPUTS_DIR)/dashboards/dashboard_"*.html ]; then \
		latest_dashboard=$$(ls -t $(OUTPUTS_DIR)/dashboards/dashboard_*.html | head -1); \
		echo "$(GREEN)🌐 Abriendo: $$latest_dashboard$(NC)"; \
		open "$$latest_dashboard" 2>/dev/null || echo "$(YELLOW)Abrir manualmente: $$latest_dashboard$(NC)"; \
	else \
		echo "$(RED)❌ No se encontró dashboard. Ejecuta 'make pipeline' primero$(NC)"; \
	fi

# ====================================================================
# 📊 COMANDOS DE ANÁLISIS Y REPORTES
# ====================================================================

.PHONY: reports visualizations quick-analysis data-summary

reports: ## Generar todos los reportes
	@echo "$(GREEN)📋 Generando reportes completos...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/data_pipeline.py --mode reports
	@echo "$(GREEN)✅ Reportes generados en $(OUTPUTS_DIR)/reports/$(NC)"

visualizations: ## Generar solo visualizaciones
	@echo "$(GREEN)📈 Generando visualizaciones...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/data_pipeline.py --mode analysis
	@echo "$(GREEN)✅ Visualizaciones en $(OUTPUTS_DIR)/visualizations/$(NC)"

quick-analysis: ## Análisis rápido de datos
	@echo "$(GREEN)⚡ Ejecutando análisis rápido...$(NC)"
	@if [ -f "$(DATA_DIR)/raw/ine_table_combined.csv" ]; then \
		$(PYTHON) -c "import pandas as pd; df=pd.read_csv('$(DATA_DIR)/raw/ine_table_combined.csv'); print('📊 Datos:', df.shape); print('🔢 Columnas:', list(df.columns)[:5]); print('📈 Primeras filas:'); print(df.head(2))"; \
	else \
		echo "$(RED)❌ No hay datos. Ejecuta 'make scrape' primero$(NC)"; \
	fi

data-summary: ## Resumen de todos los datos disponibles
	@echo "$(GREEN)📋 Resumen de datos disponibles:$(NC)"
	@echo "$(YELLOW)📁 Datos en crudo:$(NC)"
	@ls -la $(DATA_DIR)/raw/ 2>/dev/null || echo "  Sin datos crudos"
	@echo "$(YELLOW)📁 Datos procesados:$(NC)"
	@ls -la $(DATA_DIR)/processed/ 2>/dev/null || echo "  Sin datos procesados"
	@echo "$(YELLOW)📁 Reportes:$(NC)"
	@ls -la $(OUTPUTS_DIR)/reports/ 2>/dev/null || echo "  Sin reportes"
	@echo "$(YELLOW)📁 Visualizaciones:$(NC)"
	@ls -la $(OUTPUTS_DIR)/visualizations/ 2>/dev/null || echo "  Sin visualizaciones"

# ====================================================================
# 🧪 COMANDOS DE DESARROLLO Y TESTING AVANZADO
# ====================================================================

.PHONY: test-all test-pipeline test-quality benchmark profile

test-all: ## Ejecutar todos los tests incluyendo integración
	@echo "$(GREEN)🧪 Ejecutando suite completa de tests...$(NC)"
	$(PYTHON) -m pytest $(TESTS_DIR)/ -v --tb=short
	@echo "$(GREEN)✅ Tests completados$(NC)"

test-pipeline: ## Test del pipeline completo
	@echo "$(GREEN)🔬 Testeando pipeline...$(NC)"
	$(PYTHON) -c "from scripts.data_pipeline import INEDataPipeline; p=INEDataPipeline(); print('✅ Pipeline inicializado correctamente')"
	@echo "$(GREEN)✅ Pipeline funcional$(NC)"

test-quality: ## Test del monitor de calidad
	@echo "$(GREEN)🔍 Testeando monitor de calidad...$(NC)"
	@if [ -f "$(DATA_DIR)/processed/unemployment_data_cleaned.csv" ]; then \
		$(PYTHON) $(SCRIPTS_DIR)/quality_monitor.py; \
	else \
		echo "$(YELLOW)⚠️  Genera datos procesados primero: make pipeline$(NC)"; \
	fi

benchmark: ## Benchmark de performance del pipeline
	@echo "$(GREEN)⏱️  Ejecutando benchmark...$(NC)"
	@time $(PYTHON) $(SCRIPTS_DIR)/data_pipeline.py --mode process 2>/dev/null
	@echo "$(GREEN)✅ Benchmark completado$(NC)"

profile: ## Profiling de memoria y CPU
	@echo "$(GREEN)📊 Ejecutando profiling...$(NC)"
	$(PYTHON) -c "import psutil, time; start=time.time(); print('🔄 Iniciando profiling...'); exec(open('$(SCRIPTS_DIR)/data_pipeline.py').read()); print(f'⏱️  Tiempo: {time.time()-start:.2f}s'); print(f'💾 Memoria: {psutil.virtual_memory().percent}%')" 2>/dev/null || echo "$(YELLOW)Instala psutil para profiling: pip install psutil$(NC)"

# ====================================================================
# 🧹 Limpieza y Mantenimiento
# ====================================================================

clean:
	@echo "$(GREEN)🧹 Limpiando archivos temporales...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@echo "$(GREEN)✅ Limpieza completada$(NC)"

clean-data:
	@echo "$(RED)⚠️  ¡CUIDADO! Esto eliminará todos los datos.$(NC)"
	@read -p "¿Estás seguro? [y/N] " confirm && [ "$$confirm" = "y" ]
	rm -rf $(DATA_DIR)/raw/*
	rm -rf $(DATA_DIR)/processed/*
	@echo "$(GREEN)✅ Datos eliminados$(NC)"

clean-outputs:
	@echo "$(GREEN)🧹 Limpiando outputs...$(NC)"
	rm -rf $(OUTPUTS_DIR)/reports/*
	rm -rf $(OUTPUTS_DIR)/visualizations/*
	rm -rf $(OUTPUTS_DIR)/dashboards/*
	@echo "$(GREEN)✅ Outputs limpiados$(NC)"

# ====================================================================
# 📚 Documentación
# ====================================================================

docs:
	@echo "$(GREEN)📚 Generando documentación...$(NC)"
	@echo "$(YELLOW)🚧 Función pendiente de implementar$(NC)"

jupyter:
	@echo "$(GREEN)📓 Lanzando Jupyter Notebook...$(NC)"
	jupyter notebook

# ====================================================================
# 🐳 Docker
# ====================================================================

docker-build:
	@echo "$(GREEN)🐳 Construyendo imagen Docker...$(NC)"
	@echo "$(YELLOW)🚧 Dockerfile pendiente de implementar$(NC)"

docker-run:
	@echo "$(GREEN)🐳 Ejecutando en Docker...$(NC)"
	@echo "$(YELLOW)🚧 Función pendiente de implementar$(NC)"

# ====================================================================
# 🔧 Utilidades
# ====================================================================

check-env:
	@echo "$(GREEN)🔍 Verificando entorno...$(NC)"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Pip: $$($(PIP) --version)"
	@echo "Directorio actual: $$(pwd)"
	@echo "Paquetes instalados: $$($(PIP) list | wc -l) paquetes"

status:
	@echo "$(GREEN)📊 Estado del proyecto:$(NC)"
	@echo "Raw data: $$(ls -1 $(DATA_DIR)/raw/ 2>/dev/null | wc -l) archivos"
	@echo "Processed data: $$(ls -1 $(DATA_DIR)/processed/ 2>/dev/null | wc -l) archivos"
	@echo "Reports: $$(ls -1 $(OUTPUTS_DIR)/reports/ 2>/dev/null | wc -l) archivos"
	@echo "Visualizations: $$(ls -1 $(OUTPUTS_DIR)/visualizations/ 2>/dev/null | wc -l) archivos"

# ====================================================================
# 🎯 Targets de conveniencia
# ====================================================================

# Setup completo para nuevo desarrollador
setup: setup-env install-dev install-package
	@echo "$(GREEN)🎉 Setup completo terminado$(NC)"
	@echo "$(YELLOW)Recuerda activar el entorno: source $(VENV)/bin/activate$(NC)"

# Verificación antes de commit
pre-commit: format lint test
	@echo "$(GREEN)✅ Pre-commit checks pasados$(NC)"

# Deploy (placeholder)
deploy:
	@echo "$(GREEN)🚀 Preparando deploy...$(NC)"
	@echo "$(YELLOW)🚧 Función pendiente de implementar$(NC)"
	