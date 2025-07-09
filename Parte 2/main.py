import pandas as pd
from scipy import stats
import numpy as np

# 1. Leer el archivo CSV
df = pd.read_csv('velocidad_internet_ucu (1)')

# 2. Filtrar observaciones de UCU (Central) y UCU (Semprún)
df_filtrado = df[df['Edificio'].isin(['Central', 'Semprún'])]

# 3. Calcular media, desviación estándar y tamaño muestral para cada edificio
stats_por_edificio = df_filtrado.groupby('Edificio')['Velocidad Mb/s'].agg(['mean', 'std', 'count'])
print("Estadísticas por edificio:\n", stats_por_edificio, "\n")

# 4. Extraer velocidades por edificio
central = df_filtrado[df_filtrado['Edificio'] == 'Central']['Velocidad Mb/s']
semprun = df_filtrado[df_filtrado['Edificio'] == 'Semprún']['Velocidad Mb/s']

# 5. Calcular estadístico t manualmente (varianzas desiguales)
mean1, mean2 = central.mean(), semprun.mean()
std1, std2 = central.std(ddof=1), semprun.std(ddof=1)
n1, n2 = central.count(), semprun.count()

se = np.sqrt(std1**2/n1 + std2**2/n2)
t_stat = (mean1 - mean2) / se

# Grados de libertad de Welch-Satterthwaite
df_num = (std1**2/n1 + std2**2/n2)**2
df_den = ((std1**2/n1)**2)/(n1-1) + ((std2**2/n2)**2)/(n2-1)
df_welch = df_num / df_den

# 6. Calcular p-valor bilateral usando la t de Student
p_value = 2 * stats.t.sf(np.abs(t_stat), df=df_welch)

print(f"Estadístico t: {t_stat:.4f}")
print(f"Grados de libertad (Welch): {df_welch:.2f}")
print(f"p-valor: {p_value:.4f}")

# 7. Determinar si se rechaza la hipótesis nula con α = 0.05
alpha = 0.05
if p_value < alpha:
    print("Se rechaza la hipótesis nula: hay diferencia significativa entre las medias.")
else:
    print("No se rechaza la hipótesis nula: no hay diferencia significativa entre las medias.")

