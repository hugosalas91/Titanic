import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##################################################################################
# 																				 #
# 							   function cleanData							 	 #
# 																				 #
##################################################################################

def cleanData(data):

	data2 = data
	
	"""
    Creamos una nueva columna Gender con los valores de la columna Sex 
    convertidos en enteros binario.
    female = 0
    male = 1
    """
	
	data2["Gender"] = data2["Sex"].map( {'female': 0, 'male': 1} ).astype(int)
    
	# mostramos los valores únicos de emarked
	print(data2["Embarked"].unique())
	print("")
	print("-----------------------------------------------------------")
    
	"""
    como vemos, al mostrar estos valores de Embarked nos encontramos con
    valores nan, por lo que tenemos que solucionar primero estos valores perdidos.
    Vamos a ver cuantas filas tienen valores Embarked = nan.
	"""
    
	print("Numero de filas con valores nulos de Embarked: " + str(len(data2[data2["Embarked"].isnull()])))
	print("")
	print("-----------------------------------------------------------")
    
	"""
    como hay muy pocas filas con valores de Embarked nulos (sólo 2) las eliminamos.
	"""
    
	data2 = data2.drop(data2.index[data2["Embarked"].isnull()])
    
	"""
    Creamos una nueva columna Embarked2 con los valores de la columna Embarked 
    convertidos en enteros binario.
    S = 0
    C = 1
    Q = 2
	"""
    
	data2["Embarked2"] = data2["Embarked"].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
	"""
    Al igual que en Embarked en la columna Age también nos encontramos con 
    valores nan.
    Vamos a ver cuantas filas tienen valores Age = nan.
	"""
    
	print("Numero de filas con valores nulos de Age: " + str(len(data2[data2["Age"].isnull()])))
	print("")
	print("-----------------------------------------------------------")
    
	"""
    Vemos que existen muchos valores con Age = nan, por lo que, en lugar de borrar 
    las filas que los contienen vamos a cambiar de estrategia y rellenar esas casillas 
    con un valor obtenido de forma inteligente.
    Vamos a obtener el valor mediano para cada clase y género que existe en el Titanic, y con esos 
    valores medianos rellenaremos todas las casillas perdidas.
	"""
    
	median_ages = np.zeros((2,3))
    
	for i in range(0, 2):
		for j in range(0, 3):
			median_ages[i,j] = data2[(data2['Gender'] == i) & (data2['Pclass'] == j+1)]['Age'].dropna().median()
        	
	data2['AgeFill'] = data2['Age']
    
	for i in range(0, 2):
		for j in range(0, 3):
			data2.loc[(data2.Age.isnull()) & (data2.Gender == i) & (data2.Pclass == j+1), 'AgeFill'] = median_ages[i,j]
    
	print(data2.head(5))
	print("")
	print("-----------------------------------------------------------")
	print(data2.describe())
	print("")
	print("-----------------------------------------------------------")

	return data2

##################################################################################
# 																				 #
# 						function exploratoryAnalysis							 #
# 																				 #
##################################################################################
    
def exploratoryAnalysis(data):
    """
    En primer lugar voy a graficar el numero de supervivientes
    """
    print(data.head(5))
    
    data["Survived"].value_counts().plot(kind="bar", alpha=0.55)
    plt.xticks(data["Survived"].unique(), ["Fallecidos", "Supervivientes"], size = 'medium', color = 'black', rotation = 0)
    plt.title("Víctimas del naufragio del Titanic")
    plt.show()
    
    """
    Ahora voy a ver el numero de supervivientes por sexo
    """
    data[data["Gender"] == 1]["Survived"].value_counts().plot(kind="barh", color="blue", label="hombres")
    data[data["Gender"] == 0]["Survived"].value_counts().plot(kind="barh", color="#FA2379", label="mujeres")
    plt.yticks(data["Survived"].unique(), ["Fallecidos", "Supervivientes"], size = 'medium', color = 'black', rotation = 0)
    plt.title("Supervivencia con respecto al género")
    plt.legend(loc='best')
    plt.show() 
    
    """
    Vamos a ver el numero de supervivientes por sexo en proporcion
    """
    proportion_men = data[data["Gender"] == 1]["Survived"].value_counts() / len(data[data["Gender"] == 1]["Survived"])
    proportion_men.plot(kind="barh", color="blue", label="hombres")
    proportion_women = data[data["Gender"] == 0]["Survived"].value_counts() / len(data[data["Gender"] == 0]["Survived"])
    proportion_women.plot(kind="barh", color="#FA2379", label="mujeres", alpha=0.55)
    plt.yticks(data["Survived"].unique(), ["Fallecidos", "Supervivientes"], size = 'medium', color = 'black', rotation = 0)
    plt.title("Proporción de supervivencia por género")
    plt.legend(loc="best")
    plt.show()
    
    """
    En promedio las mujeres tuvieron mas probabilidad de supervivencia que los hombres.
    Ahora vamos a analizar cuantas mujeres de cada edad viajaban en el Titanic
    """
    bins = [0, 10, 20, 30, 40, 50, 60, 70]
    out = pd.cut(data[data["Gender"] == 0]["AgeFill"], bins)
    counts = pd.value_counts(out).sort_index()
    print(counts)
    counts.plot(kind="bar")
    plt.title("Número de mujeres de cada edad que viajaban en el Titanic")
    plt.show()
    
    """
    Vemos que viajaban mas mujeres entre 20 y 30 años de edad.
    Vamos cuántas mujeres sobrevivieron por edades.
    """
    
    out = pd.cut(data[(data["Survived"] == 1) & (data["Gender"] == 0)]["AgeFill"], bins)
    counts = pd.value_counts(out).sort_index()
    print(counts)
    counts.plot(kind="bar")
    plt.title("Número de mujeres supervivientes por edad en el Titanic")
    plt.show()
    
    """
    Vemos en los datos que siguen la logica y tambien sobrevivieron mas mujeres de entre 20 y 30 años.
    Pero ahora vamos a verlo de otra manera, vamos a ver que porcentaje de mujeres en 
    los rangos de edades definidos sobrevivieron al naufragio.
    """
    
    out = pd.cut(data[(data["Survived"] == 1) & (data["Gender"] == 0)]["AgeFill"], bins)
    out2 = pd.cut(data[data["Gender"] == 0]["AgeFill"], bins)
    counts = pd.value_counts(out).sort_index() / pd.value_counts(out2).sort_index()
    print(counts)
    counts.plot(kind="bar")
    plt.title("Porcentaje de mujeres supervivientes respecto a la edad en el Titanic")
    plt.show()
    
    """
    Vemos una cosa curiosa y es que cuanto mas edad tenian las mujeres mayor grado de supervivencia
    tuvieron. Esto podría deberse a que las mujeres de mayor edad pertenecieran a la clase alta, por 
    lo que vamos a ver la supervivencia de las mujeres por clases.
    Primero vamos a ver cuantas mujeres viajaban de cada clase.
    """
    
    data[data["Gender"] == 0]["Pclass"].value_counts().sort_index().plot(kind="pie", labels = ["1ª clase", "2ª clase", "3ª clase"], autopct='%1.1f%%')
    plt.title("Número de mujeres por clases en el Titanic")
    plt.show()
    
    """
    Ahora nos fijamos en cuantas mujeres de cada clase sobrevivieron.
    """
    
    data[(data["Survived"] == 1) & (data["Gender"] == 0)]["Pclass"].value_counts().sort_index().plot(kind="pie", labels = ["1ª clase", "2ª clase", "3ª clase"], autopct='%1.1f%%')
    plt.title("Número de mujeres supervivientes por clases en el Titanic")
    plt.show()
    
    """
    Como vemos sobrevivieron muchas más mujeres de 1ª clase que de 3ª clase, a pesar de 
    que había muchas más de 3ª clase.
    Vamos a ver el porcentaje de mujeres supervivientes respecto a cada clase.
    """
    
    out = data[(data["Survived"] == 1) & (data["Gender"] == 0)]["Pclass"].value_counts().sort_index()
    out2 = data[data["Gender"] == 0]["Pclass"].value_counts().sort_index()
    counts = out / out2
    counts.plot(kind="pie", labels = ["1ª clase", "2ª clase", "3ª clase"], autopct='%1.1f%%')
    plt.title("Porcentaje de mujeres supervivientes respecto a cada clase clase en el Titanic")
    plt.show()
    
    """
    Aqui podemos ver que hubo muchas mas mujeres supervivientes de 1ª y 2ª clase que de 3ª.
    Vamos a calcular el numero de mujeres a bordo del Titanic por edades y clases
    """
    graphic = pd.DataFrame()
    out = pd.cut(data[(data["Gender"] == 0) & (data["Pclass"] == 1)]["AgeFill"], bins)
    counts = pd.value_counts(out).sort_index()
    graphic["1ª clase"] = counts.values
    graphic.index = counts.index
    out = pd.cut(data[(data["Gender"] == 0) & (data["Pclass"] == 2)]["AgeFill"], bins)
    counts = pd.value_counts(out).sort_index()
    graphic["2ª clase"] = counts.values
    out = pd.cut(data[(data["Gender"] == 0) & (data["Pclass"] == 3)]["AgeFill"], bins)
    counts = pd.value_counts(out).sort_index()
    graphic["3ª clase"] = counts.values
    graphic.plot(kind="bar")
    plt.title("Número de mujeres a bordo del Titanic")
    plt.show()
    
    """
    Vamos a calcular el numero de mujeres supervivientes por edades y clases
    """
    
    graphic = pd.DataFrame()
    out = pd.cut(data[(data["Survived"] == 1) & (data["Gender"] == 0) & (data["Pclass"] == 1)]["AgeFill"], bins)
    counts = pd.value_counts(out).sort_index()
    graphic["1ª clase"] = counts.values
    graphic.index = counts.index
    out = pd.cut(data[(data["Survived"] == 1) & (data["Gender"] == 0) & (data["Pclass"] == 2)]["AgeFill"], bins)
    counts = pd.value_counts(out).sort_index()
    graphic["2ª clase"] = counts.values
    out = pd.cut(data[(data["Survived"] == 1) & (data["Gender"] == 0) & (data["Pclass"] == 3)]["AgeFill"], bins)
    counts = pd.value_counts(out).sort_index()
    graphic["3ª clase"] = counts.values
    graphic.plot(kind="bar")
    plt.title("Número de mujeres supervivientes al naufragio del Titanic")
    plt.show()
    
    """
    Vamos a calcular la proporcion de mujeres supervivientes por edades y clases
    """
    
    graphic = pd.DataFrame()
    out = pd.cut(data[(data["Survived"] == 1) & (data["Gender"] == 0) & (data["Pclass"] == 1)]["AgeFill"], bins)
    counts = pd.value_counts(out).sort_index()
    out2 = pd.cut(data[(data["Gender"] == 0) & (data["Pclass"] == 1)]["AgeFill"], bins)
    counts2 = pd.value_counts(out2).sort_index()
    proportion = counts / counts2
    graphic["1ª clase"] = proportion.values
    graphic.index = proportion.index
    out = pd.cut(data[(data["Survived"] == 1) & (data["Gender"] == 0) & (data["Pclass"] == 2)]["AgeFill"], bins)
    counts = pd.value_counts(out).sort_index()
    out2 = pd.cut(data[(data["Gender"] == 0) & (data["Pclass"] == 2)]["AgeFill"], bins)
    counts2 = pd.value_counts(out2).sort_index()
    proportion = counts / counts2
    graphic["2ª clase"] = proportion.values
    out = pd.cut(data[(data["Survived"] == 1) & (data["Gender"] == 0) & (data["Pclass"] == 3)]["AgeFill"], bins)
    counts = pd.value_counts(out).sort_index()
    out2 = pd.cut(data[(data["Gender"] == 0) & (data["Pclass"] == 3)]["AgeFill"], bins)
    counts2 = pd.value_counts(out2).sort_index()
    proportion = counts / counts2
    graphic["3ª clase"] = proportion.values
    graphic.plot(kind="bar")
    plt.title("Proporción de mujeres supervivientes al naufragio del Titanic")
    plt.show()
    
    """
    Vamos a ver el numero de personas en el titanic con x familiares
    """
    data['FamilySize'] = data['SibSp'] + data['Parch']
    
    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax0, ax1, ax2, ax3 = axes.flat
    
    graphic = pd.DataFrame()
    out = pd.cut(data[(data["Gender"] == 0) & (data["Pclass"] == 1)]["AgeFill"], bins)
    counts = pd.value_counts(out).sort_index()
    graphic["1ª clase"] = counts.values
    graphic.index = counts.index
    out = pd.cut(data[(data["Gender"] == 0) & (data["Pclass"] == 2)]["AgeFill"], bins)
    counts = pd.value_counts(out).sort_index()
    graphic["2ª clase"] = counts.values
    out = pd.cut(data[(data["Gender"] == 0) & (data["Pclass"] == 3)]["AgeFill"], bins)
    counts = pd.value_counts(out).sort_index()
    graphic["3ª clase"] = counts.values
    graphic.plot(kind="bar")
    ax0.title("Número de mujeres a bordo del Titanic")
    plt.show()
	

##################################################################################
# 																				 #
# 							function newFeatures							 	 #
# 																				 #
##################################################################################    
    
def newFeatures(data):
    data['FamilySize'] = data['SibSp'] + data['Parch']
    data['Age*Class'] = data.AgeFill * data.Pclass
    return data


##################################################################################
# 																				 #
# 							function selectFeatures							 	 #
# 																				 #
################################################################################## 

def selectFeatures(data):
    pass 