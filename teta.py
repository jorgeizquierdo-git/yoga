import numpy as np

def teta(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    vetice_1, vetice_2 = a - b, a - c #a es el vertice del ángulo
    coseno = np.dot(vetice_1, vetice_2) / (np.linalg.norm(vetice_1) * np.linalg.norm(vetice_2) + 1e-10)
    return np.degrees(np.arccos(np.clip(coseno, -1.0, 1.0)))