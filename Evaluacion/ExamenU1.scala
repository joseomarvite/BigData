val puntajes = Array(3,4,21,36,10,28,35,5,24,42)

// Se agrego un funcion que recibira un arreglo de numeros enteros (Int)
def breakingRecords (scores:Array[Int]):=
{
     // Se agrego el score, que sera el contador que se ira moviento en los espacios del arreglo
     var score = 0
     // La funcion (.head) nos da la cabeza del arreglo
     var max = scores.head
     var min = scores.head
     // Esta variable llevaran el conteo de puntaje alto 
     var higherScore = 0
     // Esta variable llevaran el conteo de puntaje bajo
     var lowerScore = 0
     // Se crea un (foreach) en el arreglo, que ira comparando la variable (score)
     // en cada espacio del arreglo
     scores.foreach(score => {
          // Si el score es mayor a la cabeza del arreglo, entonces ese tomara el valor 
          // y se agregara al puntaje mas alto, el cual se ira sumando conforme
          // pase todos los elementos del arreglo
          if (score > max) {
               max = score
               higherScore = higherScore + 1
          }
          // Si el score es menor a la cabeza del arreglo, entonces ese tomara el valor 
          // y se agregara al puntaje mas alto, el cual se ira sumando conforme
          // pase todos los elementos del arreglo
          if (score < min) {
               min = score
               lowerScore = lowerScore + 1
          } 
     })
     // Las suma de los puntajes seran enviados a la pantalla
     println(higherScore + " " + lowerScore)
     
}

breakingRecords(scores)
