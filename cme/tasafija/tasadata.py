# Analiza el archivo de precios y lista los activos con mayor cantidad de datos
# Agrega el ranking de activos por cantidad de datos como una nueva hoja en el Excel
def ranking_activos_por_datos(path_excel):
	with pd.ExcelFile(path_excel) as xls:
		df = pd.read_excel(xls, xls.sheet_names[0], index_col=0)
	conteo = df.notnull().sum().sort_values(ascending=False)
	# Escribe el ranking en una nueva hoja del mismo archivo
	with pd.ExcelWriter(path_excel, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
		conteo.to_frame('Cantidad de datos').to_excel(writer, sheet_name='ranking_activos')
	print('Ranking de activos guardado en hoja ranking_activos')
	return conteo

# Ejemplo de uso:
# ranking_activos_por_datos('precios_por_fecha.xlsx')
import pandas as pd

def transformar_excel(input_path, output_path, columna_valor='CIERRE'):
	"""
	Transforma un archivo Excel de formato largo a ancho.
	- input_path: ruta del archivo de entrada
	- output_path: ruta del archivo de salida
	- columna_valor: columna de valores a pivotear (por defecto 'CIERRE')
	"""
	df = pd.read_excel(input_path)
	df['FECHA'] = pd.to_datetime(df['FECHA'])
	# Agrupa por FECHA y SIMBOLO y toma el último valor de cierre si hay duplicados
	df_agrupado = df.groupby(['FECHA', 'SIMBOLO'], as_index=False)[columna_valor].last()
	# Pivot: filas=FECHA, columnas=SIMBOLO, valores=CIERRE
	tabla = df_agrupado.pivot(index='FECHA', columns='SIMBOLO', values=columna_valor)
	tabla = tabla.sort_index()
	tabla.to_excel(output_path)

# Ejemplo de uso:
transformar_excel('data.xlsx', 'precios_por_fecha.xlsx')
ranking_activos_por_datos('precios_por_fecha.xlsx')
