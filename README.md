# Tarea 3 - Probabilidad y Estadística Aplicada

Este repositorio contiene el desarrollo completo de la Tarea 3 de la materia Probabilidad y Estadística Aplicada, realizada por el Equipo 3 (UCU, 2025).

## Contenido

La tarea se divide en dos partes que abordan pruebas de hipótesis fundamentales en estadística aplicada:

### 1. Distribución del Ingreso por Departamento

**Script:** [Parte 1/main.py](Parte%201/main.py)  
**Datos:** [Parte 1/muestra_ech.csv](Parte%201/muestra_ech.csv)

**Descripción:** Análisis de la distribución territorial del ingreso en Uruguay utilizando datos de la Encuesta Continua de Hogares (ECH) para determinar si los hogares con mayores ingresos per cápita se distribuyen uniformemente entre departamentos.

**Aspectos cubiertos:**

- Cálculo del ingreso per cápita por hogar
- Clasificación de hogares en quintiles según ingreso per cápita
- Identificación del quintil superior (20% con mayor ingreso)
- Construcción de tabla de frecuencias observadas por departamento
- **Implementación manual de la prueba chi-cuadrado de bondad de ajuste:**
  - Cálculo de frecuencias esperadas bajo hipótesis de distribución uniforme
  - Cálculo manual del estadístico χ² = Σ((Oi - Ei)² / Ei)
  - Determinación del p-valor y decisión con α = 0.05
- Análisis de departamentos con mayor/menor concentración de hogares ricos
- Visualizaciones: gráficos de barras, histogramas y boxplots
- Conclusiones sobre equidad territorial en Uruguay

### 2. Comparación de Velocidades de Internet entre Edificios UCU

**Script:** [Parte 2/main.py](Parte%202/main.py)  
**Datos:** [Parte 2/velocidad_internet_ucu (1)](<Parte%202/velocidad_internet_ucu%20(1)>)

**Descripción:** Prueba de hipótesis para comparar las velocidades promedio de internet entre los edificios Central y Semprún de la Universidad Católica del Uruguay.

**Aspectos cubiertos:**

- Análisis exploratorio de datos por edificio
- Cálculo de estadísticas descriptivas (media, desviación estándar, tamaño muestral)
- **Implementación manual de la prueba t de Welch:**
  - Cálculo manual del estadístico t para varianzas desiguales
  - Aplicación de la corrección de Welch-Satterthwaite para grados de libertad
  - Cálculo del p-valor bilateral
- Interpretación de resultados con α = 0.05
- Conclusiones sobre diferencias significativas entre edificios

## Fundamento Teórico

### Prueba Chi-cuadrado de Bondad de Ajuste

- **H₀:** Los hogares del quintil superior se distribuyen uniformemente entre departamentos
- **H₁:** Los hogares del quintil superior NO se distribuyen uniformemente entre departamentos
- **Estadístico:** χ² = Σ((Oi - Ei)² / Ei), donde Oi son frecuencias observadas y Ei son frecuencias esperadas
- **Distribución:** χ² con k-1 grados de libertad (k = número de departamentos)

### Prueba t de Welch

- **H₀:** μ₁ = μ₂ (las medias poblacionales son iguales)
- **H₁:** μ₁ ≠ μ₂ (las medias poblacionales son diferentes)
- **Estadístico:** t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂)
- **Grados de libertad:** Corrección de Welch-Satterthwaite

## Características Técnicas

- **Implementación manual:** Ambas pruebas se implementan sin usar funciones automáticas de pruebas de hipótesis
- **Cálculos transparentes:** Se muestran todos los pasos intermedios y tablas de cálculo
- **Visualizaciones:** Gráficos explicativos para facilitar la interpretación
- **Código modular:** Organizado en funciones específicas para cada tarea

## Cómo usar

Clonar el repositorio y ejecutar cada archivo de manera individual desde un compilador o la terminal. Es importante asegurarse de tener todas las librerías necesarias instaladas.

```bash
pip install pandas numpy scipy matplotlib seaborn  # Necesario solo la primera vez
cd "Parte 1"
python main.py  # Ejecutar análisis de distribución del ingreso
cd "../Parte 2"
python main.py  # Ejecutar comparación de velocidades de internet
```

## Estructura del Repositorio

```
Tarea3-PyE/
├── README.md
├── .idea/                        # Configuración del IDE
├── Parte 1/
│   ├── main.py                   # Análisis chi-cuadrado
│   └── muestra_ech.csv          # Datos ECH
└── Parte 2/
    ├── main.py                   # Prueba t de Welch
    └── velocidad_internet_ucu (1) # Datos de velocidades
```

## Resultados Esperados

- **Parte 1:** Determinación estadística sobre la equidad territorial en la distribución del ingreso
- **Parte 2:** Conclusión sobre diferencias significativas en velocidades de internet entre edificios
- **Ambas partes:** Decisiones fundamentadas en evidencia estadística con nivel de significancia α = 0.05
