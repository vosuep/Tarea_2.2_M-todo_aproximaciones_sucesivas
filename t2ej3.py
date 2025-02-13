import numpy as np
import matplotlib.pyplot as plt

# Definir la función g(x) para el método de punto fijo
def g(x):
    return np.cos(x)  # Función reescrita de la ecuación cos(x) - x = 0

# Derivada de g(x)
def g_prime(x):
    return -np.sin(x)

# Error absoluto
def error_absoluto(x_new, x_old):
    return abs(x_new - x_old)

# Error relativo
def error_relativo(x_new, x_old):
    return abs((x_new - x_old) / x_new)

# Error cuadrático
def error_cuadratico(x_new, x_old):
    return (x_new - x_old)**2

# Método de punto fijo
def punto_fijo(x0, tol=1e-5, max_iter=100):
    iteraciones = []
    errores_abs = []
    errores_rel = []
    errores_cuad = []

    x_old = x0
    for i in range(max_iter):
        x_new = g(x_old)
        e_abs = error_absoluto(x_new, x_old)
        e_rel = error_relativo(x_new, x_old)
        e_cuad = error_cuadratico(x_new, x_old)

        iteraciones.append((i+1, x_new, e_abs, e_rel, e_cuad))
        errores_abs.append(e_abs)
        errores_rel.append(e_rel)
        errores_cuad.append(e_cuad)

        if e_abs < tol:
            break

        x_old = x_new

    return iteraciones, errores_abs, errores_rel, errores_cuad

# Parámetros iniciales
x0 = 0.5
iteraciones, errores_abs, errores_rel, errores_cuad = punto_fijo(x0)

# Imprimir tabla de iteraciones
print("Iteración | x_n      | Error absoluto | Error relativo | Error cuadrático")
print("-----------------------------------------------------------------------")
for it in iteraciones:
    print(f"{it[0]:9d} | {it[1]:.6f} | {it[2]:.6e} | {it[3]:.6e} | {it[4]:.6e}")

# Graficar la convergencia
x_vals = np.linspace(0, 2, 100)
y_vals = g(x_vals)

plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_vals, label=r"$g(x) = \cos(x)$", color="blue")
plt.plot(x_vals, x_vals, linestyle="dashed", color="red", label="y = x")

# Graficar iteraciones
x_points = [it[1] for it in iteraciones]
y_points = [g(x) for x in x_points]
plt.scatter(x_points, y_points, color="black", zorder=3)
plt.plot(x_points, y_points, linestyle="dotted", color="black", label="Iteraciones")

plt.xlabel("x")
plt.ylabel("g(x)")
plt.legend()
plt.grid(True)
plt.title("Método de Punto Fijo para $cos(x) - x = 0$")
plt.savefig("punto_fijo_convergencia_cos.png")
plt.show()

# Graficar errores
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(errores_abs) + 1), errores_abs, marker="o", label="Error absoluto")
plt.plot(range(1, len(errores_rel) + 1), errores_rel, marker="s", label="Error relativo")
plt.plot(range(1, len(errores_cuad) + 1), errores_cuad, marker="^", label="Error cuadrático")

plt.xlabel("Iteración")
plt.ylabel("Error")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.title("Evolución de los Errores")
plt.savefig("errores_punto_fijo_cos.png")
plt.show()

# Evaluar |g'(x)| en el intervalo de estudio
x_vals_prime = np.linspace(0, 2, 100)
g_prime_vals = np.abs(g_prime(x_vals_prime))

# Graficar |g'(x)|
plt.figure(figsize=(8, 5))
plt.plot(x_vals_prime, g_prime_vals, label=r"$|g'(x)| = |\sin(x)|$", color="green")
plt.axhline(1, color="red", linestyle="dashed", label="Umbral de convergencia: 1")
plt.xlabel("x")
plt.ylabel(r"$|g'(x)|$")
plt.legend()
plt.grid(True)
plt.title("Evaluación de $|g'(x)|$ en el intervalo de estudio")
plt.savefig("g_prime_cos.png")
plt.show()
