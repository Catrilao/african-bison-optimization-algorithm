# Dominios basados en las limitantes máximas de anuncios
domains = {
    "TV_TARDE": list(range(0, 16)),
    "TV_NOCHE": list(range(0, 11)),
    "DIARIO": list(range(0, 26)),
    "REVISTA": list(range(0, 5)),
    "RADIO": list(range(0, 31)),
}

# Restricciones basadas en el presupuesto
costos = {
    "TV_TARDE": (160, 200),
    "TV_NOCHE": (300, 350),
    "DIARIO": (40, 80),
    "REVISTA": (100, 120),
    "RADIO": (10, 20),
}

# Definir restricciones de presupuesto usando costos máximos para dominio más amplio
constraints = {
    # Restricción 1: No más de 3800 um en anuncios de televisión (usando costos máximos)
    ("TV_TARDE", "TV_NOCHE"): lambda tv_t, tv_n: (
        tv_t * costos["TV_TARDE"][1] + tv_n * costos["TV_NOCHE"][1] <= 3800
    ),
    ("TV_NOCHE", "TV_TARDE"): lambda tv_n, tv_t: (
        tv_t * costos["TV_TARDE"][1] + tv_n * costos["TV_NOCHE"][1] <= 3800
    ),
    # Restricción 2: No más de 2800 um en anuncios de diario o revista (usando costos máximos)
    ("DIARIO", "REVISTA"): lambda d, r: (
        d * costos["DIARIO"][1] + r * costos["REVISTA"][1] <= 2800
    ),
    ("REVISTA", "DIARIO"): lambda r, d: (
        d * costos["DIARIO"][1] + r * costos["REVISTA"][1] <= 2800
    ),
    # Restricción 3: No más de 3500 um en diario y radio (usando costos máximos)
    ("DIARIO", "RADIO"): lambda d, ra: (
        d * costos["DIARIO"][1] + ra * costos["RADIO"][1] <= 3500
    ),
    ("RADIO", "DIARIO"): lambda ra, d: (
        d * costos["DIARIO"][1] + ra * costos["RADIO"][1] <= 3500
    ),
}


def revise(x, y):
    """
    Función para revisar y reducir el dominio de la variable x
    basándose en las restricciones con la variable y
    """
    revised = False
    x_domain = domains[x].copy()
    y_domain = domains[y]

    relevant_constraints = [
        constraints[constraint]
        for constraint in constraints
        if constraint[0] == x and constraint[1] == y
    ]

    for x_value in x_domain:
        satisfies = False

        for y_value in y_domain:
            constraint_satisfied = True
            for constraint_func in relevant_constraints:
                if not constraint_func(x_value, y_value):
                    constraint_satisfied = False
                    break
            if constraint_satisfied:
                satisfies = True
                break

        if not satisfies:
            domains[x].remove(x_value)
            revised = True

    return revised


def ac3(arcs):
    """
    Algoritmo AC3 para consistencia de arco
    """
    queue = arcs[:]
    iterations = 0

    while queue:
        (x, y) = queue.pop(0)
        iterations += 1
        print(f"Iteración {iterations}: Revisando arco ({x}, {y})")

        revised = revise(x, y)
        if revised:
            print(f"  Dominio de {x} reducido a: {domains[x]}")
            if not domains[x]:
                print(f"  ¡Error! Dominio de {x} está vacío. No hay solución.")
                return False

            neighbors = [(z, x) for (z, w) in arcs if w == x and z != y]
            queue.extend(neighbors)
        else:
            print(f"  No se modificó el dominio de {x}")

    print(f"\nAC3 completado en {iterations} iteraciones")
    return True


arcs = [
    ("TV_TARDE", "TV_NOCHE"),
    ("TV_NOCHE", "TV_TARDE"),
    ("DIARIO", "REVISTA"),
    ("REVISTA", "DIARIO"),
    ("DIARIO", "RADIO"),
    ("RADIO", "DIARIO"),
]

print("=== PROBLEMA DE CAMPAÑA PUBLICITARIA ABC ===")
print("Dominios iniciales:")
for var, domain in domains.items():
    print(f"{var}: {len(domain)} valores (de {min(domain)} a {max(domain)})")

print("\nRestricciones de presupuesto (usando costos máximos):")
print("1. TV (tarde + noche) ≤ 3800 um  [200*TV_tarde + 350*TV_noche ≤ 3800]")
print("2. Diario + Revista ≤ 2800 um    [80*Diario + 120*Revista ≤ 2800]")
print("3. Diario + Radio ≤ 3500 um      [80*Diario + 20*Radio ≤ 3500]")

print("\nEjecutando AC3...")
print("=" * 50)

success = ac3(arcs)

print("=" * 50)
print("RESULTADOS FINALES:")
if success:
    print("AC3 exitoso - Dominios consistentes:")
    total_combinations = 1
    for var, domain in domains.items():
        print(
            f"{var}: {len(domain)} valores - {domain[:10]}{'...' if len(domain) > 10 else ''}"
        )
        total_combinations *= len(domain)
    print(
        f"\nEspacio de búsqueda reducido a {total_combinations:,} combinaciones posibles"
    )
else:
    print("AC3 falló - No existe solución factible")

print("\nANÁLISIS DE COSTOS MÁXIMOS UTILIZADOS:")
print("TV_TARDE: 200 um/anuncio")
print("TV_NOCHE: 350 um/anuncio")
print("DIARIO: 80 um/anuncio")
print("REVISTA: 120 um/anuncio")
print("RADIO: 20 um/anuncio")

print("\nANÁLISIS DE REDUCCIÓN:")
initial_space = 16 * 11 * 26 * 5 * 31
final_space = 1
for domain in domains.values():
    final_space *= len(domain)

reduction_percentage = ((initial_space - final_space) / initial_space) * 100
print(f"Espacio inicial: {initial_space:,} combinaciones")
print(f"Espacio final: {final_space:,} combinaciones")
print(f"Reducción: {reduction_percentage:.2f}%")
