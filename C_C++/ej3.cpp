#include <iostream>
#include <cmath> // <cmath>
#include <float.h>
#include <iomanip>

// Simulación de la función del paracaidista
double funcion(double tiempo, double masa, double coef){
    // Ejemplo simple: velocidad terminal aproximada
    return (masa * 9.8 / coef) * (1 - exp(-coef / masa * tiempo));
}

int main(int argc, char** argv){

// Resolviendo el problema del paracaidista
    double veloant = 0.0;
    double velo = 0.0;
    double tiempo = 0.0;
 velo = funcion(tiempo, 68.1, 12.5);
tiempo=2;
std::cout << std::setprecision(16) << "t = 0.0 \t" << "vel = " << velo << std::endl;
    do {
        veloant = velo;
        velo = funcion(tiempo, 68.1, 12.5);
        std::cout << "t = " << tiempo << " vel = " << velo << std::endl;
        tiempo += 2.0;
    } while ( ( velo - veloant) > 0.0001 ); // Usamos esta relación para parar el ciclo cuando
                                  // en 4 cifras significativas sea igual el resultado 
    return 0;
}
