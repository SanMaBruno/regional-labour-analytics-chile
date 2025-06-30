# 📌 Setup inicial y librerías
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# 📌 1. Inicializar driver
def init_driver(headless: bool = True) -> webdriver.Chrome:
    options = Options()
    if headless:
        options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    return driver

# 📌 2. Función para scrapear tabla por ID
def scrape_table(driver, wait, table_id):
    try:
        table = wait.until(EC.presence_of_element_located((By.ID, table_id)))
        html = table.get_attribute('outerHTML')
        df = pd.read_html(html)[0]
        print(f"✅ Tabla extraída con ID '{table_id}'.")
        return df
    except Exception as e:
        print(f"❌ No se encontró la tabla con ID '{table_id}': {e}")
        # DEBUG: listar IDs de tablas disponibles
        tables = driver.find_elements(By.TAG_NAME, "table")
        ids = [t.get_attribute("id") for t in tables]
        print(f"🔎 IDs de tablas encontrados en la página: {ids}")
        return pd.DataFrame()

# 📌 3. Función para detectar número total de páginas
def get_total_pages(driver, wait):
    try:
        # Buscar elementos de paginación
        pages = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "PageNumber")))
        total_pages = len(pages)
        print(f"🔍 Se detectaron {total_pages} páginas en total")
        return total_pages
    except Exception as e:
        print(f"⚠️ No se pudo detectar paginación: {e}")
        return 1  # Asume 1 página si no hay paginación

# 📌 4. Función para cambiar de página
def go_to_page(driver, wait, page_number):
    try:
        pages = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "PageNumber")))
        if page_number < len(pages):
            pages[page_number].click()
            print(f"✅ Cambiado a la página {page_number + 1}")
            time.sleep(5)  # espera tras el cambio de página
            return True
        else:
            print(f"⚠️ Página {page_number + 1} no existe")
            return False
    except Exception as e:
        print(f"❌ Error al cambiar a la página {page_number + 1}: {e}")
        return False

# 📌 5. Función principal de scraping automático
def scrape_all_pages_automatically(url, table_id, output_prefix="table", headless=True):
    """
    Función principal para hacer scraping automático de todas las páginas disponibles.
    
    Args:
        url (str): URL objetivo para el scraping
        table_id (str): ID de la tabla a extraer
        output_prefix (str): Prefijo para los archivos de salida
        headless (bool): Ejecutar en modo headless o no
    
    Returns:
        pd.DataFrame: DataFrame combinado con todos los datos
    """
    print(f"🚀 Iniciando scraping automático de: {url}")
    print(f"🎯 Buscando tabla con ID: {table_id}")
    
    # Inicializar driver
    driver = init_driver(headless=headless)
    wait = WebDriverWait(driver, 40)
    
    try:
        # Navegar a la URL
        driver.get(url)
        time.sleep(5)
        
        # Detectar número total de páginas
        total_pages = get_total_pages(driver, wait)
        
        # Lista para almacenar todos los DataFrames
        all_dataframes = []
        
        # Iterar por todas las páginas
        for page_num in range(total_pages):
            print(f"\n� Procesando página {page_num + 1} de {total_pages}...")
            
            # Si no es la primera página, navegar a la página específica
            if page_num > 0:
                success = go_to_page(driver, wait, page_num)
                if not success:
                    print(f"⚠️ Saltando página {page_num + 1}")
                    continue
            
            # Extraer tabla de la página actual
            df = scrape_table(driver, wait, table_id)
            
            if not df.empty:
                # Guardar página individual
                output_path = f"data/raw/{output_prefix}_page{page_num + 1}.csv"
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"✅ Página {page_num + 1} exportada: {len(df)} filas")
                
                # Añadir a la lista para combinación
                all_dataframes.append(df)
            else:
                print(f"⚠️ Página {page_num + 1} está vacía")
        
        # Combinar todas las páginas
        if all_dataframes:
            df_combined = pd.concat(all_dataframes, ignore_index=True)
            combined_path = f"data/raw/{output_prefix}_combined.csv"
            df_combined.to_csv(combined_path, index=False, encoding='utf-8-sig')
            
            print(f"\n🎉 SCRAPING COMPLETADO:")
            print(f"   📊 Total de páginas procesadas: {len(all_dataframes)}")
            print(f"   📊 Total de filas extraídas: {len(df_combined)}")
            print(f"   📁 Archivo combinado: {combined_path}")
            
            return df_combined
        else:
            print("❌ No se pudieron extraer datos de ninguna página")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Error durante el scraping: {e}")
        return pd.DataFrame()
    
    finally:
        driver.quit()
        print("🏁 Driver cerrado")

# 📌 6. Ejecución principal - CONFIGURACIÓN AQUÍ
if __name__ == "__main__":
    # 🔧 CONFIGURACIÓN PRINCIPAL - Cambia estos valores según tu necesidad
    URL_TARGET = "https://stat.ine.cl/Index.aspx?lang=es&SubSessionId=78e0518e-d028-4bf8-8d80-444b7277907c"
    TABLE_ID = "tabletofreeze"
    OUTPUT_PREFIX = "ine_table"  # Los archivos se llamarán: ine_table_page1.csv, ine_table_page2.csv, etc.
    HEADLESS_MODE = False  # True para modo silencioso, False para ver el navegador
    
    # Ejecutar scraping automático
    result_df = scrape_all_pages_automatically(
        url=URL_TARGET,
        table_id=TABLE_ID,
        output_prefix=OUTPUT_PREFIX,
        headless=HEADLESS_MODE
    )
    
    # Mostrar resumen final
    if not result_df.empty:
        print(f"\n📋 RESUMEN FINAL:")
        print(f"   🔢 Columnas: {list(result_df.columns)}")
        print(f"   📏 Dimensiones: {result_df.shape}")
        print(f"   📄 Primeras 3 filas:")
        print(result_df.head(3))
    else:
        print("❌ No se obtuvieron datos")