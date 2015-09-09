"""Script to getting, analyzing and visualizating data of Titanic."""
import sys
import numpy as np
import pandas as pd
import italo.functions as italo
import hugo.functions as hugo

def get_data(file):
	data = pd.read_csv(file, header=0)
	return data

def main():

	# 1 - Obtencion de datos
	# ==================================================================
	 
	data = get_data("../data/train.csv")
	print(data.head(5))
	print("")
	print("-----------------------------------------------------------")
	print("")
	print(data.tail(5))
	print("")
	print("-----------------------------------------------------------")
	print("")
	print(data.dtypes)
	print("")
	print("-----------------------------------------------------------")
	print("")
	print(type(data))
	print("")
	print("-----------------------------------------------------------")
	print("")
	print(data.info())
	print("")
	print("-----------------------------------------------------------")
	print("")
	print(data.describe())
	print("")
	print("-----------------------------------------------------------")
	print("")
	
	# 2 - Limpieza de datos
	# ================================================================== 
	
	####################################################################
	#																   #
	# 				Limpieza de datos propuesta por Hugo			   #
	#																   #
	####################################################################
	
	data_clean_hugo = hugo.cleanData(data)
	
	####################################################################
	#																   #
	# 			Cierre Limpieza de datos propuesta por Hugo			   #
	#																   #
	####################################################################
	
	####################################################################
	#																   #
	# 				Limpieza de datos propuesta por Italo			   #
	#																   #
	####################################################################
	
	data_clean_italo = italo.cleanData(data)
	
	####################################################################
	#																   #
	# 			Cierre Limpieza de datos propuesta por Italo		   #
	#																   #
	####################################################################
	
	# 3 - Analisis exploratorio de los datos
	# ==================================================================
	
	####################################################################
	#																   #
	# 			Analisis exploratorio propuesto por Hugo			   #
	#																   #
	####################################################################
	
	hugo.exploratoryAnalysis(data_clean_hugo)
	
	####################################################################
	#																   #
	# 			  Cierre analisis exploratorio por Hugo			       #
	#																   #
	####################################################################
	
	####################################################################
	#																   #
	# 			Analisis exploratorio propuesto por Italo			   #
	#																   #
	####################################################################
	
	italo.exploratoryAnalysis(data_clean_italo)
	
	####################################################################
	#																   #
	# 			  Cierre analisis exploratorio por Italo		       #
	#																   #
	####################################################################
	
	# 4 - Obtencion de nuevas caracteristicas
	# ==================================================================
	
	####################################################################
	#																   #
	# 			Nuevas caracteristicas propuestas por Hugo			   #
	#																   #
	####################################################################
	
	new_data_hugo = hugo.newFeatures(data_clean_hugo)
	
	####################################################################
	#																   #
	# 		Cierre nuevas caracteristicas propuestas por Hugo		   #
	#																   #
	####################################################################
	
	####################################################################
	#																   #
	# 			Nuevas caracteristicas propuestas por Italo			   #
	#																   #
	####################################################################
	
	new_data_italo = italo.newFeatures(data_clean_italo)
	
	####################################################################
	#																   #
	# 		Cierre nuevas caracteristicas propuestas por Italo		   #
	#																   #
	####################################################################
	
	# 5 - Preparar caracteristicas para modelo predictivo
	# ==================================================================
	
	####################################################################
	#																   #
	# 			Preparar caracteristicas para predecir Hugo			   #
	#																   #
	####################################################################
	
	predictors = hugo.selectFeatures(new_data_hugo)
	
	####################################################################
	#																   #
	# 		Cierre preparar caracteristicas para predecir Hugo		   #
	#																   #
	####################################################################
	
	####################################################################
	#																   #
	# 		Preparar caracteristicas para predecir Italo			   #
	#																   #
	####################################################################
	
	predictors = italo.selectFeatures(new_data_italo)
	
	####################################################################
	#																   #
	# 		Cierre preparar caracteristicas para predecir Italo		   #
	#																   #
	####################################################################

	# 6 - Algoritmo predictivo
	# ==================================================================
	
	####################################################################
	#																   #
	# 				  Algoritmo de predictivo Hugo			           #
	#																   #
	####################################################################
	
	
	####################################################################
	#																   #
	# 		         Cierre algoritmo de predictivo Hugo		       #
	#																   #
	####################################################################
	
	####################################################################
	#																   #
	# 		            Algoritmo de predictivo Italo			       #
	#																   #
	####################################################################
	
	
	####################################################################
	#																   #
	# 		       Cierre algoritmo de predictivo Italo		           #
	#																   #
	####################################################################
	
	# 7 - Comprobar la calidad de nuestro modelo predictivo
	# ==================================================================
	
	####################################################################
	#																   #
	# 			Calidad algoritmo de predictivo Hugo			       #
	#																   #
	####################################################################
	
	
	####################################################################
	#																   #
	# 		       Calidad algoritmo de predictivo Hugo		           #
	#																   #
	####################################################################
	
	####################################################################
	#																   #
	# 		            Algoritmo de predictivo Italo			       #
	#																   #
	####################################################################
	
	
	####################################################################
	#																   #
	# 		   Cierre calidad algoritmo de predictivo Italo		       #
	#																   #
	####################################################################
	
	# 8 - Interpretar y visualizar resultados
	# ==================================================================
	
	####################################################################
	#																   #
	# 			Interpretar y visualizar resultados Hugo			   #
	#																   #
	####################################################################
	
	
	####################################################################
	#																   #
	# 		       Interpretar y visualizar resultados Hugo		       #
	#																   #
	####################################################################
	
	####################################################################
	#																   #
	# 		     Interpretar y visualizar resultados Italo			   #
	#																   #
	####################################################################
	
	
	####################################################################
	#																   #
	# 		 Cierre interpretar y visualizar resultados Italo		   #
	#																   #
	####################################################################

if __name__ == '__main__':
	sys.exit(main())