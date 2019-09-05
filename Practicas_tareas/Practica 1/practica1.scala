//Practica 1

//1. Desarrollar un algoritmo en scala que calcule el radio de un circulo
print("Ingrese la circunferencia del CIRCULO: ")
val circunferencia: Double = scala.io.StdIn.readLine.toInt
val radio: Double = circunferencia/(2*3.1416)
println("El radio del circulo es: " + radio)

//2. Desarrollar un algoritmo en scala que me diga si un numero es primo
def esprimo2(i :Int) : Boolean = {
    if (i <= 1)
      false
     else if (i == 2)
       true
     else
      !(2 to (i-1)).exists(x => i % x == 0)
  }

print("Ingrese un numero: ")
val numero1: Int = scala.io.StdIn.readLine.toInt
print("Ingrese un numero: ")
val numero2: Int = scala.io.StdIn.readLine.toInt
  (numero1 to numero2).foreach(i => if (esprimo2(i)) println("%d es primo.".format(i))) 

//3. Dada la variable bird = "tweet", utiliza interpolacion de string para imprimir "Estoy ecribiendo un tweet"
var bird = "tweet"

//4. Dada la variable mensaje = "Hola Luke yo soy tu padre!" utiliza slilce para extraer la secuencia "Luke"
var variable = "Hola Luke soy tu padre!"
variable.slice(5,9)

//5. Cual es la diferencia en value y una variable en scala?

// Respuesta: Value(val) se le asigna un valor definido y no puede ser cambiado, en cambio una variable(var) puede 
// ser cambiado en cualquier momento

//6. Dada la tupla (2,4,5,1,2,3,3.1416,23) regresa el numero 3.1416 
 var x = (2,4,5,1,2,3,3.1416,23)
println(x._7)
