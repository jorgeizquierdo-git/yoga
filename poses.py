"""
Carga de cada pose (funciones, lambdas, etc). Los archivos están en otro directorio en .json con la siguiente estructura

{
  "postura": {
    "nombre": "Postura del árbol",
    "mantener_s": 10,
    "angulos": {
      "rodilla_der":  {"puntos": [24, 26, 28], "objetivo": 180, "tol": 15},
      "rodilla_izq":  {"puntos": [23, 25, 27], "objetivo": 70,  "tol": 20},
      "codo_der":     {"puntos": [12, 14, 16], "objetivo": 170, "tol": 20}
    }
  }
}

Cada punto es un putuno del esqueleto del programa. Buscamos calcular el ángulo respecto estos puntos para saber si la postura es correcta. 
Objetivo es el ángulo y tol es la tolerancia de fallo. Puede ser que ponderemos dichas tolerancias con una constante en función del ruido del esqueleto.

Preferimos calcular angulos frente a resolver el problema de matching porque solo nos interesa las posiciones relativas entre cada punto y no en su totalidad. Además,
es menos costoso los ángulos que una matriz de matching con softassign

"""