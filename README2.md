# Datos Masivos - Unidad 2

 - **Tecnológico Nacional de México**
 - **Instituto Tecnológico de Tijuana**
 - **Subdirección académica**
 - **Departamento de Sistemas y Computación**
 - **Semestre**: AGOSTO- DICIEMBRE 2019
 - **Ingeniería en Tecnologías de la Información y Comunicaciones**
 - **Materia**: Datos Masivos
 - **Unidad**: 2
 - **Nombre**: Vite Hernández José Omar
 - **No. Control**: 15211706
 - **Docente**: Dr. Jose Christian Romero Hernandez

# Evaluación 
**Examen 1**

**Introducción**
En el presente documento se podrá observar el conocimiento y procedimiento que se realizó en la 1ra evaluación de la unidad 3, el cual fue programado en el lenguaje Scala y se utilizaron librerías de Machine Learning de Spark en 4.4

**Explicación**
Se desarrolló en el lenguaje Scala, las siguientes instrucciones utilizando las librerías de Machine Learning
Se cargo el dataset iris.csv y se realizó una limpieza de los datos necesaria para después ser procesado con el algoritmo Multilayer perceptron.
1. Se cargaron los datos en la variable "data"
2. Se imprimió el esquema de los datos
3. Se eliminaron los campos null
4. Se declarar un vector que transformara los datos a la variable "features"
5. Se transformaron los features usando el dataframe
6. Se declaró un "StringIndexer" que transformada los datos en "species" en datos numéricos
7. Se ajustó las especies indexadas con el vector features
8. Con la variable "splits" se hizo un corte de forma aleatoria
9. Se declaro la variable "train" la cual tendrá el 60% de los datos
10. Se declaro la variable "test" la cual tendrá el 40% de los datos
11. Se establece la configuración de las capas para el modelo de redes neuronales artificiales:
1.  Entrada: 4, al tener 4 variables en el dataset
2.  Intermedias 2
3.  Salida 3, al tener 3 tipos de clases
12. Se configuro el entrenador del algoritmo Multilayer con sus respectivos parámetros
13. Se entreno el modelo con los datos de entrenamiento
14. Se prueban los datos de prueba ya entrenado el modelo
15. Se seleccionó la predicción y la etiqueta que serán guardado en la variable
16. Se muestran algunos datos
17. Se ejecutó la estimación de la precisión del modelo
18. Se imprime el error del modelo
D) Explique detalladamente la función matemática de entrenamiento que utilizo con sus propias palabras
E) Explique la función de error que utilizó para el resultado final

**Código comentado** 
```scala
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.linalg.Vectors
// Se cargan los datos en la variable "data"
val data = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("iris.csv")
// Se imprime el esquema de los datos
data.printSchema()
// Se eliminan los campos null
val dataClean = data.na.drop()
// Se declara un vector que transformara los datos a la variable "features"
val vectorFeatures = (new VectorAssembler().setInputCols(Array("sepal_length","sepal_width", "petal_length","petal_width")).setOutputCol("features"))
// Se transforman los features usando el dataframe
val features = vectorFeatures.transform(dataClean)
// Se declara un "StringIndexer" que transformada los datos en "species" en datos numericos
val speciesIndexer = new StringIndexer().setInputCol("species").setOutputCol("label")
// Ajustamos las especies indexadas con el vector features
val dataIndexed = speciesIndexer.fit(features).transform(features)
// Con la variable "splits" hacemos un corte de forma aleatoria
val splits = dataIndexed.randomSplit(Array(0.6, 0.4), seed = 1234L)
// Se declara la variable "train" la cual tendra el 60% de los datos
val train = splits(0)
// Se declara la variable "test" la cual tendra el 40% de los datos
val test = splits(1)
// Se establece la configuracion de las capas para el modelo de redes neuronales artificiales
// De entrada: 4
// Intermedias 2
// Salida 3, al tener 3 tipos de clases
val layers = Array[Int](4, 5, 4, 3)
// Se configura el entrenador del algoritmo Multilayer con sus respectivos parametros
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
// Se entrena el modelo con los datos de entrenamiento
val model = trainer.fit(train)
// Se prueban ya entrenado el modelo
val result = model.transform(test)
// Se selecciona la prediccion y la etiqueta que seran guardado en la variable
val predictionAndLabels = result.select("prediction", "label")
// Se muestran algunos datos
predictionAndLabels.show()
// Se ejecuta la estimacion de la precision del modelo
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictionAndLabels)
// Se imprime el error del modelo
println(s"Test Error = ${(1.0 - accuracy)}")
//D).Explique detalladamente la funcion matematica de entrenamiento que utilizo con sus propias palabras
/*
La funcion de entrenamiento es el conjunto de datos que se divide en una parte utilizada para entrenar el modelo (60%)
y otra parte para las prueba (40%).
Esta funcion mediante un array hacemos las pruebas de entrenamiento mediante un ramdom
asi de esta manera se entrena el algoritmo y se costruye el modelo
*/
//E).Explique la funcion de error que utilizo para el resultado final
/*
Esta funcion nos sirve para calcular el error de prueba nos ayuda a medir la precisión del modelo utilizando el evaluador.
y asi imprimir con exactitud el error de nuestro problema.
*/
```

# Practicas
**Practica 1**

**Introducción**
La siguiente práctica es el resultado de la exposición 1 impartida por mis compañeros en la unidad 2 de la materia de datos masivos.

**Explicación**

**Correlation**
El coeficiente de correlación de Pearson es una prueba que mide la relación estadística entre dos variables continuas. Si la asociación entre los elementos no es lineal, entonces el coeficiente no se encuentra representado adecuadamente.

El coeficiente de correlación de Pearson puede tomar un rango de valores de +1 a -1. Un valor de 0 indica que no hay asociación entre las dos variables. Un valor mayor que 0 indica una asociación positiva. Es decir, a medida que aumenta el valor de una variable, también lo hace el valor de la otra. Un valor menor que 0 indica una asociación negativa; es decir, a medida que aumenta el valor de una variable, el valor de la otra disminuye.

La fórmula del coeficiente de correlación de Pearson es la siguiente:

![](https://lh5.googleusercontent.com/lT5LR_8yWobCOVQ14jjz3ftPPzkFlKKz5mkGnqrtbGhVT9mpGFA_xaA6LXgxEEG8TkQ3jJgwUeaNIDRS4fks1NYD3_ZklDJiR04EcY-0PFoyNvC52tDrl8QJ7_e4F8SSo0pLmidt)

Donde:

“x” es igual a la variable número uno, “y” pertenece a la variable número dos, “zx” es la desviación estándar de laCorrelation variable uno, “zy” es la desviación estándar de la variable dos y “N” es es número de datos.

**Hypothesis testing**  
Las estadísticas tienen que ver con los datos. Los datos por sí solos no son interesantes. Es la interpretación de los datoHypothesis testings lo que nos interesa. Utilizando las pruebas de hipótesis , tratamos de interpretar o sacar conclusiones sobre la población utilizando datos de muestra. Una prueba de hipótesis evalúa dos afirmaciones mutuamente excluyentes sobre una población para determinar qué afirmación es mejor respaldada por los datos de la muestra. Siempre que queramos hacer afirmaciones sobre la distribución de datos o si un conjunto de resultados es diferente de otro conjunto de resultados en el aprendizaje automático aplicado, debemos confiar en las pruebas de hipótesis estadísticas.

**Summarizer**

Proporciona estadística en resúmenes de vectores en el uso de grandes volúmenes de datos, las métricas son:

1.  Máximo
2.  Mínimo
3.  Promedio
4.  Varianza
5.  Número de no ceros 
6.  Así como resultados totales

**Código comentado** 
**Correlation**
```scala
// Se declaran las librerias necesarias
import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.Row
// Se declaran en la variable "df" los vectores donde tendran los datos a procesar
val data = Seq(
Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))),
Vectors.dense(4.0, 5.0, 0.0, 3.0),
Vectors.dense(6.0, 7.0, 0.0, 8.0),
Vectors.sparse(4, Seq((0, 9.0), (3, 1.0)))
)
// Se tranforma la variable "df" a dataframe donde los datos seran agregados a la columna "features"
val df = data.map(Tuple1.apply).toDF("features")
// Se declara la funcion de correlacion y se empieza a trabajar con ellos
val Row(coeff1: Matrix) = Correlation.corr(df, "features").head
println(s"Pearson correlation matrix:\n  $coeff1")
// Se manda a imprimir la correlacion
val Row(coeff2: Matrix) = Correlation.corr(df, "features", "spearman").head
println(s"Spearman correlation matrix:\n  $coeff2")
```
**Hypothesis testing**
```scala
// Se declaran las librerias necesarias
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.stat.ChiSquareTest
// Se declaran en la variable "data" los vectores donde tendran los datos a procesar
val data = Seq(
(0.0, Vectors.dense(0.5, 10.0)),
(0.0, Vectors.dense(1.5, 20.0)),
(1.0, Vectors.dense(1.5, 30.0)),
(0.0, Vectors.dense(3.5, 30.0)),
(0.0, Vectors.dense(3.5, 40.0)),
(1.0, Vectors.dense(3.5, 40.0))
)
// Se tranforma la variable "data" a dataframe ("df"), este mismo tendra dos columnas llamadas "label", "features"
val df = data.toDF("label", "features")
// Se declara el modelo y sus se agregan sus respectivos parametros
val chi = ChiSquareTest.test(df, "features", "label").head
println(s"pValues = ${chi.getAs[Vector](0)}")
println(s"degreesOfFreedom ${chi.getSeq[Int](1).mkString("[", ",", "]")}")
println(s"statistics ${chi.getAs[Vector](2)}")
```
**Summarizer**
```scala
// Se importan las librerias necesarias
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.stat.Summarizer
// Se crear una variable llamada "data" la cual tendra los datos a procesar
val data = Seq(
(Vectors.dense(2.0, 3.0, 5.0), 1.0),
(Vectors.dense(4.0, 6.0, 7.0), 2.0)
)
// Se transforma la variable "data" en DataFrame con las columnas "features" , "weight"
val df = data.toDF("features", "weight")
df.show()
//Creamos una sumatoria utilizando los datos del DataFrame
val (meanVal, varianceVal) = df.select(Summarizer.metrics("mean", "variance").summary($"features", $"weight").as("summary")).select("summary.mean", "summary.variance").as[(Vector, Vector)].first()
// Se imprime el resultado
println(s"with weight: mean = ${meanVal}, variance = ${varianceVal}")
//Creamos una sumatoria
val (meanVal2, varianceVal2) = df.select(Summarizer.mean($"features"),Summarizer.variance($"features")).as[(Vector, Vector)].first()
// Se imprime el resultado
println(s"without weight: mean = ${meanVal2}, sum = ${varianceVal2}")
```
**Practica 2**

**Introducción** *
La siguiente práctica es el resultado de la exposición 2 impartida por mis compañeros en la unidad 2 de la materia de datos masivos.

**Explicación**
Un árbol de decisión es una serie de nodos, un gráfico direccional que comienza en la base con un solo nodo y se extiende a los muchos nodos hoja que representan las categorías que el árbol puede clasificar. Otra forma de pensar en un árbol de decisión es como un diagrama de flujo, donde el flujo comienza en el nodo raíz y termina con una decisión tomada en las hojas. Es una herramienta de apoyo a la decisión. Utiliza un gráfico en forma de árbol para mostrar las predicciones que resultan de una serie de divisiones basadas en características.
Aquí hay algunos términos útiles para describir un árbol de decisión:

  

![](https://lh5.googleusercontent.com/hRQ2NI-TF_qVmKmdlp6K-XpqWXb2IurIvjBM0KVJCA-1nJtguMfc_jOlj5oRCrhOIuANn6OMao7Lx2DdCjFzQqGTODVs8YtaqrRf3WjyyeLK7D8TmLts12NFThnR576zPX1MxPGS)
Nodo raíz: un nodo raíz está al comienzo de un árbol. Representa a toda la población que se analiza. Desde el nodo raíz, la población se divide de acuerdo con varias características, y esos subgrupos se dividen a su vez en cada nodo de decisión debajo del nodo raíz.
División: es un proceso de división de un nodo en dos o más subnodos.
Nodo de decisión: cuando un subnodo se divide en subnodos adicionales, es un nodo de decisión.
Nodo hoja o nodo terminal: los nodos que no se dividen se denominan nodos hoja o terminal.
Poda: la eliminación de los subnodos de un nodo primario se denomina poda. Un árbol se cultiva mediante la división y se contrae mediante la poda.
Rama o subárbol: una subsección del árbol de decisión se denomina rama o subárbol, del mismo modo que una parte de un gráfico se denomina subgrafo.
Nodo padre y nodo hijo: estos son términos relativos. Cualquier nodo que se encuentre debajo de otro nodo es un nodo secundario o subnodo, y cualquier nodo que preceda a esos nodos secundarios se llama nodo primario.

  

![](https://lh5.googleusercontent.com/dUsfp5izTGE2YwZLb5tn2H7pKXR0v03lk5U2qKa9dVGn8AMSXNIBtXqkxOEWMJ4owxOMZOHLDB9tdsiXavQ5pjk7QPdvtNx1BI_Fo_dLafgy6-Aue_HvVMpbIyhtFmqfFvFo_xFr)

**Código comentado**
```scala

```
**Practica 3**

**Introducción**
En este presente documento se deriva de investigación de la sucesión de Fibonacci en el Instituto Tecnológico de Tijuana de la materia Minería de Datos lo cual pude desarrollar mis habilidades y aplicar mis conocimientos aprendidos durante el periodo

**Instrucciones**
1. Programar los 5 algoritmos de la sucesión de Fibonacci

**Explicación**
Basicamente, la sucesion de Fibonacci se realiza sumando siempre los ultimos 2 numeros (Todos los numeros presentes en la sucesion se llaman numeros de Fibonacci) de los siguientes 5 algoritmos matematicos:

**Código comentado** 
```scala

```
**Practica 4**

**Introducción**
En este presente documento se deriva de la explicación de diferentes clases en el Instituto Tecnológico de Tijuana de la materia Minería de Datos lo cual pude desarrollar mis habilidades y aplicar mis conocimientos aprendidos durante el periodo.

**Explicación**
Se declaro una variable “df” de la cual tomará los datos de un documento en formato “csv”, se declararon diferentes funciones para la variable.

**Instrucciones**
1. Agregar 20 funciones básicas para el la variable "df".

**Código comentado** 
```scala

```
**Practica 5**

**Introducción**
En este presente documento se deriva de la explicación de diferentes clases en el Instituto Tecnológico de Tijuana de la materia Minería de Datos lo cual pude desarrollar mis habilidades y aplicar mis conocimientos aprendidos durante el periodo

**Explicación**
Se utilizaron diferentes funciones más comunes para el uso de la variable “df”

**Instrucciones**
1. Agregar 5 funciones 

**Código comentado**
```scala

```
**Practica 6**

**Introducción**
En este presente documento se deriva de la explicación de diferentes clases en el Instituto Tecnológico de Tijuana de la materia Minería de Datos lo cual pude desarrollar mis habilidades y aplicar mis conocimientos aprendidos durante el periodo

**Explicación**
Se declaro una variable “df” de la cual tomará los datos de un documento “csv” y se buscó utilizar la sintaxis de Scala y SparkSQL en la consulta de datos

**Instrucciones**
1. Agregar 5 funciones utilizando el sintaxis de Sparksql y Scala 

**Código comentado**
```scala

```
**Practica 7**

**Introducción**
En este presente documento se deriva de la explicación de diferentes clases en el Instituto Tecnológico de Tijuana de la materia Minería de Datos lo cual pude desarrollar mis habilidades y aplicar mis conocimientos aprendidos durante el periodo

**Explicación**
Se declaro una variable “df” de la cual tomará los datos de un documento “csv” y se buscó utilizar la sintaxis de Scala y SparkSQL en la consulta de datos

**Instrucciones**
1. Agregar 5 funciones utilizando el sintaxis de Sparksql y Scala 

**Código comentado**
```scala

```
**Practica 8**

**Introducción**
En este presente documento se deriva de la explicación de diferentes clases en el Instituto Tecnológico de Tijuana de la materia Minería de Datos lo cual pude desarrollar mis habilidades y aplicar mis conocimientos aprendidos durante el periodo

**Explicación**
Se declaro una variable “df” de la cual tomará los datos de un documento “csv” y se buscó utilizar la sintaxis de Scala y SparkSQL en la consulta de datos

**Instrucciones**
1. Agregar 5 funciones utilizando el sintaxis de Sparksql y Scala 

**Código comentado**
```scala

```
```
