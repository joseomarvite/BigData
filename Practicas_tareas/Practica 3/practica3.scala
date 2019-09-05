//Primer Algoritmo
def funcion (n: Int): Int = 
{
    if (n<2)
    {
        return n

    }
    else
    {
        return funcion(n-1) + funcion(n-2)
    }
}

funcion(5)

//Segundo Algoritmo
def funcion1(n: Double): Double =
{
    if (n<2)
    {
        return n
    }
    else
    {
        var p = ((1+(Math.sqrt(5)))/2)
        var a = Math.pow(p,n)
        var b = Math.pow((1-p),n)
        var c = Math.sqrt(5)
        var j = ((a-(b)))/(c)
        return(j)
    }

}
funcion1(2)

//Tercer Algoritmo
def funcion2(n: Int): Int =
{
var a = 0
var b = 1
var c = 0
    for (k <- 1 to n)
    {
        c = b + a
        a = b
        b = c
    }
    return(a)
}
funcion2(9)

//Cuarto Algoritmo
def funcion3(n: Int): Int =
{
    var a = 0
    var b = 1
    for(k <- 1 to n)
        {

            b = b + a
            a = b - a
        }
        return(a)
}
funcion3(9)

//Quinto Algoritmo

//Sexto Algoritmo
def funcion5 (n: Double): Double = 
{
    if (n<=0)
    {
        return (n)
    }
    else
    {
        var i: Double = n - 1
        var auxOne: Double = 0
        var auxTwo: Double  = 1 
        var a: Double  = auxTwo
        var b: Double = auxOne
        var c: Double  = auxOne
        var d: Double  = auxTwo
        var prime: Boolean = false
        while (i > 0)
        {
            if (prime == (n % 2 == 0))
            {
                if(prime == true)
                {
                    auxOne = (d*b) + (c*a)
                    auxTwo = ((d+(b*a)) + (c*b))
                    a = auxOne
                    b = auxTwo  
                }
                else
                {
                    var pow1 = Math.pow(c,2)
                    var pow2 = Math.pow(d,2)
                    auxOne = pow1 + pow2
                    auxTwo = (d*((2*(c)) + d))
                    c = auxOne
                    d = auxTwo 
                }
            }
            i = (i / 2)   
        }
        return(a+b)
    }
}








    
