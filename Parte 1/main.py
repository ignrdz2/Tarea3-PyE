import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def cargar_datos(archivo):
    """
    Carga los datos de la ECH desde un archivo CSV.

    Args:
        archivo (str): Ruta al archivo CSV

    Returns:
        pd.DataFrame: DataFrame con los datos cargados
    """
    try:
        df = pd.read_csv(archivo)
        print(f"Datos cargados exitosamente. Dimensiones: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {archivo}")
        return None
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        return None


def calcular_ingreso_per_capita(df):
    """
    Calcula el ingreso per cápita para cada hogar.

    Args:
        df (pd.DataFrame): DataFrame con los datos de hogares

    Returns:
        pd.DataFrame: DataFrame con la columna de ingreso per cápita agregada
    """
    # Mostrar las columnas disponibles para debugging
    print("Columnas disponibles en el dataset:")
    print(df.columns.tolist())
    print(f"\nPrimeras 5 filas del dataset:")
    print(df.head())

    # Verificar que las columnas necesarias existen
    columnas_necesarias = ['ingreso', 'personas_hogar', 'departamento']

    for col in columnas_necesarias:
        if col not in df.columns:
            print(f"Error: No se encontró la columna '{col}'")
            print("Columnas disponibles:", df.columns.tolist())
            return None

    print(f"Usando columna de ingreso: 'ingreso'")
    print(f"Usando columna de habitantes: 'personas_hogar'")
    print(f"Usando columna de departamento: 'departamento'")

    # Calcular ingreso per cápita
    df['Ingreso_per_capita'] = df['ingreso'] / df['personas_hogar']

    # Remover valores nulos o inválidos
    df = df.dropna(subset=['Ingreso_per_capita', 'departamento'])
    df = df[df['Ingreso_per_capita'] > 0]

    print(f"Ingreso per cápita calculado. Registros válidos: {len(df)}")
    return df


def identificar_quintil_superior(df):
    """
    Identifica el quintil superior (20% con mayor ingreso per cápita).

    Args:
        df (pd.DataFrame): DataFrame con ingreso per cápita

    Returns:
        pd.DataFrame: DataFrame con columna indicadora del quintil superior
    """
    # Calcular el percentil 80 (quintil superior)
    percentil_80 = df['Ingreso_per_capita'].quantile(0.8)

    # Crear variable indicadora
    df['Quintil_superior'] = (df['Ingreso_per_capita'] >= percentil_80).astype(int)

    print(f"Umbral quintil superior: ${percentil_80:,.2f}")
    print(f"Hogares en quintil superior: {df['Quintil_superior'].sum()}")

    return df


def analizar_distribucion_por_departamento(df):
    """
    Analiza la distribución de hogares del quintil superior por departamento.

    Args:
        df (pd.DataFrame): DataFrame con datos procesados

    Returns:
        tuple: (tabla_contingencia, estadisticas_descriptivas)
    """
    # Crear tabla de contingencia
    tabla_contingencia = pd.crosstab(df['departamento'], df['Quintil_superior'])

    # Calcular estadísticas descriptivas
    stats_dept = df.groupby('departamento').agg({
        'Quintil_superior': ['count', 'sum', 'mean'],
        'Ingreso_per_capita': ['mean', 'median', 'std']
    }).round(2)

    # Flatten column names
    stats_dept.columns = ['_'.join(col).strip() for col in stats_dept.columns.values]

    print("\nDistribución de hogares por departamento:")
    print(tabla_contingencia)
    print(f"\nTotal hogares quintil superior: {df['Quintil_superior'].sum()}")

    return tabla_contingencia, stats_dept


def prueba_chi_cuadrado(tabla_contingencia):
    """
    Realiza la prueba de bondad de ajuste chi-cuadrado MANUALMENTE.

    Args:
        tabla_contingencia (pd.DataFrame): Tabla de contingencia

    Returns:
        tuple: (chi2_stat, p_value, grados_libertad, frecuencias_esperadas)
    """
    # Extraer solo la columna del quintil superior (columna 1)
    frecuencias_observadas = tabla_contingencia.iloc[:, 1].values

    # Calcular frecuencias esperadas bajo hipótesis nula (distribución uniforme)
    total_quintil_superior = frecuencias_observadas.sum()
    num_departamentos = len(frecuencias_observadas)

    # CORRECCIÓN: Frecuencias esperadas uniformes
    frecuencias_esperadas = np.full(num_departamentos, total_quintil_superior / num_departamentos)

    # CORRECCIÓN: Calcular estadístico chi-cuadrado MANUALMENTE
    # χ² = Σ((Oi - Ei)² / Ei)
    chi2_stat = np.sum((frecuencias_observadas - frecuencias_esperadas) ** 2 / frecuencias_esperadas)

    # Grados de libertad
    grados_libertad = num_departamentos - 1

    # CORRECCIÓN: Calcular p-valor usando distribución chi-cuadrado
    p_value = 1 - stats.chi2.cdf(chi2_stat, grados_libertad)

    # Mostrar cálculos detallados
    print("\n" + "=" * 50)
    print("PRUEBA CHI-CUADRADO DE BONDAD DE AJUSTE (MANUAL)")
    print("=" * 50)
    print(f"H0: Los hogares del quintil superior se distribuyen uniformemente entre departamentos")
    print(f"H1: Los hogares del quintil superior NO se distribuyen uniformemente entre departamentos")

    print(f"\nCálculos detallados:")
    print(f"Total hogares quintil superior: {total_quintil_superior}")
    print(f"Número de departamentos: {num_departamentos}")
    print(f"Frecuencia esperada por departamento: {total_quintil_superior / num_departamentos:.2f}")

    # Mostrar tabla de cálculos
    print(f"\nTabla de cálculos:")
    print(f"{'Dept':<4} {'Observado':<10} {'Esperado':<10} {'(O-E)²/E':<10}")
    print("-" * 40)
    for i, (obs, esp) in enumerate(zip(frecuencias_observadas, frecuencias_esperadas)):
        contribucion = (obs - esp) ** 2 / esp
        print(f"{i + 1:<4} {obs:<10.0f} {esp:<10.2f} {contribucion:<10.4f}")

    print(f"\nEstadístico chi-cuadrado: {chi2_stat:.4f}")
    print(f"Grados de libertad: {grados_libertad}")
    print(f"p-valor: {p_value:.6f}")

    # Valor crítico
    valor_critico = stats.chi2.ppf(0.95, grados_libertad)
    print(f"Valor crítico (α=0.05): {valor_critico:.4f}")

    return chi2_stat, p_value, grados_libertad, frecuencias_esperadas


def interpretar_resultados(chi2_stat, p_value, grados_libertad, alpha=0.05):
    """
    Interpreta los resultados de la prueba chi-cuadrado.

    Args:
        chi2_stat (float): Estadístico chi-cuadrado
        p_value (float): p-valor
        grados_libertad (int): Grados de libertad
        alpha (float): Nivel de significancia
    """
    print(f"\n" + "=" * 50)
    print("INTERPRETACIÓN DE RESULTADOS")
    print("=" * 50)
    print(f"Nivel de significancia (α): {alpha}")
    print(f"Grados de libertad: {grados_libertad}")

    # Valor crítico
    valor_critico = stats.chi2.ppf(1 - alpha, grados_libertad)
    print(f"Valor crítico: {valor_critico:.4f}")

    # Decisión basada en estadístico vs valor crítico
    if chi2_stat > valor_critico:
        print(f"✓ χ² ({chi2_stat:.4f}) > χ²crítico ({valor_critico:.4f})")
        print("DECISIÓN: Se rechaza la hipótesis nula")
        decision = "rechazar"
    else:
        print(f"✗ χ² ({chi2_stat:.4f}) ≤ χ²crítico ({valor_critico:.4f})")
        print("DECISIÓN: No se rechaza la hipótesis nula")
        decision = "no rechazar"

    # Confirmación con p-valor
    if p_value < alpha:
        print(f"✓ p-valor ({p_value:.6f}) < α ({alpha})")
        print("CONFIRMACIÓN: Se rechaza la hipótesis nula")
    else:
        print(f"✗ p-valor ({p_value:.6f}) ≥ α ({alpha})")
        print("CONFIRMACIÓN: No se rechaza la hipótesis nula")

    # Conclusión
    if decision == "rechazar":
        print("\nCONCLUSIÓN: Existe evidencia estadística significativa de que los hogares")
        print("del quintil superior NO se distribuyen uniformemente entre departamentos.")
        print("Algunos departamentos concentran una mayor proporción de hogares ricos.")
    else:
        print("\nCONCLUSIÓN: No hay evidencia estadística suficiente para afirmar")
        print("que existe una distribución no uniforme de hogares ricos entre departamentos.")


def analizar_departamentos_destacados(df, stats_dept):
    """
    Identifica y analiza los departamentos con mayor y menor concentración de hogares ricos.

    Args:
        df (pd.DataFrame): DataFrame con datos procesados
        stats_dept (pd.DataFrame): Estadísticas por departamento
    """
    print(f"\n" + "=" * 50)
    print("ANÁLISIS POR DEPARTAMENTOS")
    print("=" * 50)

    # Calcular proporción de hogares del quintil superior por departamento
    prop_quintil = stats_dept['Quintil_superior_mean']

    # Departamentos con mayor concentración
    top_3 = prop_quintil.nlargest(3)
    print("Departamentos con MAYOR concentración de hogares ricos:")
    for dept, prop in top_3.items():
        print(f"  {dept}: {prop:.1%}")

    # Departamentos con menor concentración
    bottom_3 = prop_quintil.nsmallest(3)
    print("\nDepartamentos con MENOR concentración de hogares ricos:")
    for dept, prop in bottom_3.items():
        print(f"  {dept}: {prop:.1%}")

    # Promedio nacional
    promedio_nacional = df['Quintil_superior'].mean()
    print(f"\nPromedio nacional: {promedio_nacional:.1%}")



def crear_visualizaciones(df, tabla_contingencia):
    """
    Crea visualizaciones para el análisis.

    Args:
        df (pd.DataFrame): DataFrame con datos procesados
        tabla_contingencia (pd.DataFrame): Tabla de contingencia
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Gráfico 1: Distribución de hogares del quintil superior por departamento
    quintil_por_dept = tabla_contingencia.iloc[:, 1]
    quintil_por_dept.plot(kind='bar', ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Distribución de Hogares del Quintil Superior por Departamento')
    axes[0, 0].set_xlabel('Departamento')
    axes[0, 0].set_ylabel('Cantidad de Hogares')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Gráfico 2: Proporción del quintil superior por departamento
    prop_por_dept = df.groupby('departamento')['Quintil_superior'].mean()
    prop_por_dept.plot(kind='bar', ax=axes[0, 1], color='lightcoral')
    axes[0, 1].set_title('Proporción del Quintil Superior por Departamento')
    axes[0, 1].set_xlabel('Departamento')
    axes[0, 1].set_ylabel('Proporción')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].axhline(y=0.2, color='red', linestyle='--', label='Promedio Nacional (20%)')
    axes[0, 1].legend()

    # Gráfico 3: Histograma de ingreso per cápita
    axes[1, 0].hist(df['Ingreso_per_capita'], bins=50, alpha=0.7, color='green')
    axes[1, 0].axvline(df['Ingreso_per_capita'].quantile(0.8), color='red', linestyle='--',
                       label='Umbral Quintil Superior')
    axes[1, 0].set_title('Distribución del Ingreso Per Cápita')
    axes[1, 0].set_xlabel('Ingreso Per Cápita')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].legend()

    # Gráfico 4: Boxplot de ingreso per cápita por quintil
    df.boxplot(column='Ingreso_per_capita', by='Quintil_superior', ax=axes[1, 1])
    axes[1, 1].set_title('Ingreso Per Cápita por Quintil')
    axes[1, 1].set_xlabel('Quintil (0=Otros, 1=Superior)')
    axes[1, 1].set_ylabel('Ingreso Per Cápita')

    plt.tight_layout()
    plt.show()


def conclusiones_finales(p_value, alpha=0.05):
    """
    Presenta las conclusiones finales del análisis.

    Args:
        p_value (float): p-valor de la prueba
        alpha (float): Nivel de significancia
    """
    print(f"\n" + "=" * 60)
    print("CONCLUSIONES FINALES SOBRE EQUIDAD TERRITORIAL EN URUGUAY")
    print("=" * 60)

    if p_value < alpha:
        print("1. DISTRIBUCIÓN NO UNIFORME:")
        print("   - Los hogares con mayores ingresos NO se distribuyen uniformemente")
        print("   - Existe concentración territorial de la riqueza")

        print("\n2. IMPLICACIONES PARA POLÍTICAS PÚBLICAS:")
        print("   - Se requieren políticas diferenciadas por departamento")
        print("   - Necesidad de estrategias de desarrollo territorial")
        print("   - Importancia de la descentralización económica")

        print("\n3. EQUIDAD TERRITORIAL:")
        print("   - Existe inequidad en la distribución territorial del ingreso")
        print("   - Algunos departamentos concentran desproporcionadamente la riqueza")
        print("   - Se sugiere investigar factores explicativos de estas diferencias")
    else:
        print("1. DISTRIBUCIÓN RELATIVAMENTE UNIFORME:")
        print("   - No se encontró evidencia de concentración territorial significativa")
        print("   - La distribución de hogares ricos es relativamente homogénea")

        print("\n2. EQUIDAD TERRITORIAL:")
        print("   - Los resultados sugieren mayor equidad territorial de lo esperado")
        print("   - No se observan patrones claros de concentración departamental")


def main():
    """
    Función principal que ejecuta todo el análisis.
    """
    print("ANÁLISIS DE DISTRIBUCIÓN DEL INGRESO POR DEPARTAMENTO")
    print("Encuesta Continua de Hogares (ECH) 2009-2019")
    print("=" * 60)

    # 1. Cargar datos
    df = cargar_datos('muestra_ech.csv')
    if df is None:
        return

    # 2. Calcular ingreso per cápita
    df = calcular_ingreso_per_capita(df)
    if df is None:
        return

    # 3. Identificar quintil superior
    df = identificar_quintil_superior(df)

    # 4. Analizar distribución por departamento
    tabla_contingencia, stats_dept = analizar_distribucion_por_departamento(df)

    # 5. Realizar prueba chi-cuadrado MANUAL
    chi2_stat, p_value, grados_libertad, freq_esperadas = prueba_chi_cuadrado(tabla_contingencia)

    # 6. Interpretar resultados
    interpretar_resultados(chi2_stat, p_value, grados_libertad)

    # 7. Analizar departamentos destacados
    analizar_departamentos_destacados(df, stats_dept)

    # 8. Crear visualizaciones
    crear_visualizaciones(df, tabla_contingencia)

    # 9. Conclusiones finales
    conclusiones_finales(p_value)


if __name__ == "__main__":
    main()