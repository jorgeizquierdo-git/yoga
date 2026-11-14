import numpy as np

def teta(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    vetice_1, vetice_2 = a - b, a - c #a es el vertice del ángulo
    coseno = np.dot(vetice_1, vetice_2) / (np.linalg.norm(vetice_1) * np.linalg.norm(vetice_2) + 1e-10)
    return np.degrees(np.arccos(np.clip(coseno, -1.0, 1.0)))

def matriz_de_putitos(landmarks):
    return np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float64)

def medir_agulos(puntos, esperado):
    return {clave: teta(*puntos[valor["puntos"]]) for clave, valor in esperado.items()}

def puntuar(medidos, esperados):
    aciertos, total = 0, len(esperados["angulos"])
    for clave, valor in esperados["angulos"].items():
        if abs(medidos[clave] - valor["objetivo"]) <= valor["tolerancia"]:
            aciertos += 1

    return aciertos, total
