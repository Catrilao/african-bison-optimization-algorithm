import math
import random
import sys
from typing import List, Tuple, Optional


class Problem:
    def __init__(self):
        self.calidad_min = [65, 90, 40, 60, 20]
        self.calidad_max = [85, 95, 60, 80, 30]
        self.costo_min = [160, 300, 40, 100, 10]
        self.costo_max = [200, 350, 80, 120, 20]
        self.max_anuncios = [15, 10, 25, 4, 30]

        # Restricciones de presupuesto
        self.presupuesto_tv = 3800
        self.presupuesto_diario_revista = 2800
        self.presupuesto_diario_radio = 3500

        # 5 tipos de anuncios
        self.dim = 5

    def costo_por_calidad(self, calidad: int, i: int) -> float:
        qmin = self.calidad_min[i]
        qmax = self.calidad_max[i]
        cmin = self.costo_min[i]
        cmax = self.costo_max[i]

        return cmin + ((cmax - cmin) / (qmax - qmin)) * (calidad - qmin)

    def check(self, x: List[int], q: List[int]) -> bool:
        for i in range(self.dim):
            if x[i] > self.max_anuncios[i]:
                return False

        costos = [self.costo_por_calidad(q[i], i) for i in range(self.dim)]

        if costos[0] * x[0] + costos[1] * x[1] > self.presupuesto_tv:
            return False
        if costos[2] * x[2] + costos[3] * x[3] > self.presupuesto_diario_revista:
            return False
        if costos[2] * x[2] + costos[4] * x[4] > self.presupuesto_diario_radio:
            return False

        return True

    def evaluate_objectives(self, x: List[int], q: List[int]) -> Tuple[int, float]:
        calidad_total = sum(x[i] * q[i] for i in range(self.dim))
        costo_total = 0.0
        for i in range(self.dim):
            costo_total += x[i] * self.costo_por_calidad(q[i], i)

        return (calidad_total, costo_total)

    def fit(self, x: List[int], q: List[int]) -> int:
        return sum(x[i] * q[i] for i in range(self.dim))

    def keep_domain(self, x: List[int]) -> List[int]:
        domain = []
        for i in range(self.dim):
            val = max(0, min(self.max_anuncios[i], round(x[i])))
            domain.append(val)
        return domain


class ParetoSolution:
    def __init__(self, x: List[int], q: List[int], problem: Problem):
        self.x = x.copy()
        self.q = q.copy()
        self.problem = problem
        self.objectives = problem.evaluate_objectives(x, q)
        self.calidad_total = self.objectives[0]
        self.costo_total = self.objectives[1]
        self.is_feasible = problem.check(x, q)

    def dominates(self, other: "ParetoSolution") -> bool:
        if not self.is_feasible:
            return False
        if not other.is_feasible:
            return True

        better_quality = self.calidad_total >= other.calidad_total
        better_cost = self.costo_total <= other.costo_total

        strictly_better = (self.calidad_total > other.calidad_total) or (
            self.costo_total < other.costo_total
        )

        return better_quality and better_cost and strictly_better


class ParetoFront:
    def __init__(self):
        self.solutions: List[ParetoSolution] = []

    def add_solution(self, solution: ParetoSolution) -> None:
        if not solution.is_feasible:
            return

        for existing in self.solutions:
            if existing.dominates(solution):
                return

        self.solutions = [sol for sol in self.solutions if not solution.dominates(sol)]
        self.solutions.append(solution)

    def size(self) -> int:
        return len(self.solutions)

    def get_range_objectives(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        if not self.solutions:
            return ((0, 0), (0, 0))

        calidades = [solution.calidad_total for solution in self.solutions]
        costos = [solution.costo_total for solution in self.solutions]

        return ((min(calidades), max(calidades)), (min(costos), max(costos)))


class EpsilonConstraint:
    def __init__(self, pareto_front: ParetoFront):
        self.pareto_front = pareto_front
        self.problem = Problem()

    def generate_epsilon_values(self, num_points: int = 10) -> List[float]:
        if self.pareto_front.size() == 0:
            return []

        calidad_range, costo_range = self.pareto_front.get_range_objectives()
        costo_min, costo_max = costo_range

        epsilon_values = []
        step = (costo_max - costo_min) / (num_points - 1) if num_points > 1 else 0

        for i in range(num_points):
            epsilon = costo_min + i * step
            epsilon_values.append(epsilon)

        return epsilon_values

    def solve_epsilon_constraint(
        self, epsilon_costo: float
    ) -> Optional[ParetoSolution]:
        best_solution = None
        best_calidad = -1

        for solution in self.pareto_front.solutions:
            if solution.is_feasible and solution.costo_total <= epsilon_costo:
                if solution.calidad_total > best_calidad:
                    best_calidad = solution.calidad_total
                    best_solution = solution

        return best_solution

    def solve_multiple_epsilon(
        self, epsilon_values: List[float]
    ) -> List[Tuple[float, Optional[ParetoSolution]]]:
        results = []

        for epsilon in epsilon_values:
            solution = self.solve_epsilon_constraint(epsilon)
            results.append((epsilon, solution))

        return results

    def analyze_tradeoffs(self, epsilon_values: List[float]) -> None:
        log("\n" + "=" * 80, level=1)
        log("ANÁLISIS ε-CONSTRAINT", level=1)
        log("=" * 80, level=1)
        log(
            f"{'Epsilon (Costo Max)':>20} {'Calidad Máxima':>20} {'Costo Real':>15} {'Factible':>10}",
            level=1,
        )
        log("-" * 80, level=1)

        results = self.solve_multiple_epsilon(epsilon_values)

        for epsilon, solution in results:
            if solution:
                log(
                    f"{epsilon:>20.2f} {solution.calidad_total:>20.2f} {solution.costo_total:>15.2f} {'✓':>10}",
                    level=1,
                )
            else:
                log(
                    f"{epsilon:>20.2f} {'No factible':>20} {'---':>15} {'✗':>10}",
                    level=1,
                )

        log("\n" + "=" * 80, level=1)
        log("DETALLE DE SOLUCIONES ÓPTIMAS", level=1)
        log("=" * 80, level=1)

        for i, (epsilon, solution) in enumerate(results):
            if solution:
                log(f"\nSolución {i + 1} (ε = {epsilon:.2f}):", level=1)
                log(f"  Calidad total: {solution.calidad_total:.2f}", level=1)
                log(f"  Costo total: {solution.costo_total:.2f}", level=1)
                log(f"  Configuración: {solution.x}", level=1)
                log(f"  Calidades: {solution.q}", level=1)

                costos = [
                    self.problem.costo_por_calidad(solution.q[j], j)
                    for j in range(self.problem.dim)
                ]
                tv_cost = costos[0] * solution.x[0] + costos[1] * solution.x[1]
                dr_cost = costos[2] * solution.x[2] + costos[3] * solution.x[3]
                d_radio_cost = costos[2] * solution.x[2] + costos[4] * solution.x[4]

                log("  Restricciones:", level=1)
                log(
                    f"    TV: {tv_cost:.2f} ≤ {self.problem.presupuesto_tv} ({'✓' if tv_cost <= self.problem.presupuesto_tv else '✗'})",
                    level=1,
                )
                log(
                    f"    Diario+Revista: {dr_cost:.2f} ≤ {self.problem.presupuesto_diario_revista} ({'✓' if dr_cost <= self.problem.presupuesto_diario_revista else '✗'})",
                    level=1,
                )
                log(
                    f"    Diario+Radio: {d_radio_cost:.2f} ≤ {self.problem.presupuesto_diario_radio} ({'✓' if d_radio_cost <= self.problem.presupuesto_diario_radio else '✗'})",
                    level=1,
                )


class Individual:
    def __init__(self, problem: Problem):
        self.p = problem
        self.dimension = self.p.dim
        self.x: List[int] = []
        self.q: List[int] = []

        self.initialize_solution()
        self.fitness_value = self.fitness()

    def initialize_solution(self) -> None:
        self.x = [
            random.randint(0, self.p.max_anuncios[i]) for i in range(self.dimension)
        ]
        self.q = [
            random.randint(self.p.calidad_min[i], self.p.calidad_max[i])
            for i in range(self.dimension)
        ]

    def is_feasible(self) -> bool:
        return self.p.check(self.x, self.q)

    def fitness(self) -> int:
        return self.p.fit(self.x, self.q)

    def is_better_than(self, other: "Individual") -> bool:
        return self.fitness() > other.fitness()

    def move(self, x_next: List[int], global_best: "Individual") -> None:
        max_attempts = 100

        for attempt in range(max_attempts):
            new_x = self.p.keep_domain(x_next)
            new_q = [
                random.randint(self.p.calidad_min[i], self.p.calidad_max[i])
                for i in range(self.dimension)
            ]

            if not self.p.check(new_x, new_q):
                continue

            new_fitness = self.p.fit(new_x, new_q)
            if new_fitness > self.fitness_value or (self != global_best):
                self.x = new_x
                self.q = new_q
                self.fitness_value = new_fitness
            return

    def copy(self, other: "Individual") -> None:
        if isinstance(other, Individual):
            self.x = other.x.copy()
            self.q = other.q.copy()
            self.fitness_value = other.fitness_value

    def __str__(self) -> str:
        return f"x: {self.x}, fitness {self.fitness()}"


class ABO:
    def __init__(self, pop_size=50, max_iter=100):
        self.problem = Problem()
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.threshold_1 = 0.5
        self.threshold_2 = 0.6

        self.population = []
        self.male_group = []
        self.female_group = []
        self.global_best = None
        self.male_best = None
        self.female_best = None

        self.pareto_front = ParetoFront()

    def random(self) -> None:
        for _ in range(self.pop_size):
            feasible = False
            while not feasible:
                ind = Individual(self.problem)
                feasible = ind.is_feasible()
            self.population.append(ind)

        pareto_sol = ParetoSolution(ind.x, ind.q, self.problem)
        self.pareto_front.add_solution(pareto_sol)

        self.global_best = self.population[0]
        for i in range(1, self.pop_size):
            if self.population[i].is_better_than(self.global_best):
                self.global_best.copy(self.population[i])

        mid = self.pop_size // 2
        self.male_group = self.population[:mid]
        self.female_group = self.population[mid:]
        self.update_best_individuals()

        self.show_results(0)

    def _update_best(self, attr_name: str, group: List) -> None:
        current_best = max(group, key=lambda x: x.fitness_value)
        best = getattr(self, attr_name)

        if best is None or current_best.is_better_than(best):
            new_best = Individual(self.problem)
            new_best.copy(current_best)
            setattr(self, attr_name, new_best)

    def update_best_individuals(self) -> None:
        self._update_best("global_best", self.population)
        self._update_best("male_best", self.male_group)
        self._update_best("female_best", self.female_group)

    def satiety(self, t: int) -> float:
        S = 2 * math.sin((3 * t / self.max_iter) * math.pi)
        return S

    def temperature(self, t: int) -> float:
        Q = math.cos(-((self.max_iter - t) / self.max_iter) * (math.pi / 2))
        return Q

    def foraging_behavior(self, S: float, individual: Individual) -> List[float]:
        x_rand = random.choice(self.population)
        f_rand = x_rand.fitness()
        f_current = individual.fitness()
        if f_current == 0:
            f_current = 1e-10

        x_next = []

        for j in range(individual.dimension):
            R_1 = random.uniform(-1, 1)
            R_2 = random.uniform(-1, 1)
            R_3 = random.uniform(-5, 5)

            A = math.cos((f_rand / f_current) * math.pi) * R_3

            x_min = 0
            x_max = self.problem.max_anuncios[j]

            xj = (x_rand.x[j] * S) + A * ((x_max - x_min) * R_1) * R_2**3

            x_next.append(xj)

        return x_next

    def bathing_behavior(self, Q: float, individual: Individual) -> List[float]:
        x_next = []

        for j in range(individual.dimension):
            R_4 = random.uniform(0, 2)
            R_5 = random.uniform(0, 2)

            xj = 2 * (self.global_best.x[j] - individual.x[j]) * R_4 + math.exp(
                R_5 * (Q**5)
            ) * math.cos(Q * 2 * math.pi)

            x_next.append(xj)

        return x_next

    def jousting_behavior(
        self, individual: Individual, group_best: Individual
    ) -> List[float]:
        x_next = []

        for j in range(individual.dimension):
            R_6 = random.uniform(-0.01, 0.01)
            R_7 = random.uniform(0, 2)

            x_sbest = group_best.x[j]
            x_current = individual.x[j]

            J = x_sbest - x_sbest * ((random.random() * x_current) / 2) * (
                math.cos(x_current) + math.sin(x_current)
            ) * (R_6**5)
            xj = x_sbest - J * R_7

            x_next.append(xj)

        return x_next

    def mating_behavior_male(self, male: Individual, female: Individual) -> List[float]:
        x_next = []

        for j in range(male.dimension):
            R_8 = random.random()

            f_female = female.fitness_value
            f_male = male.fitness_value
            if f_male == 0:
                f_male = 1e10

            MM = math.cos(f_female / f_male)

            xj = male.x[j] + math.sin(2 * math.pi * R_8) * MM * (
                male.x[j] - female.x[j]
            )

            x_next.append(xj)

        return x_next

    def mating_behavior_female(
        self, male: Individual, female: Individual
    ) -> List[float]:
        x_next = []

        for j in range(female.dimension):
            R_9 = random.random()

            f_male = male.fitness_value
            f_female = female.fitness_value
            if f_female == 0:
                f_female = 1e10

            MF = math.sin(f_male / f_female)

            xj = male.x[j] + math.cos(2 * math.pi * R_9) * MF * (
                female.x[j] - male.x[j]
            )

            x_next.append(xj)

        return x_next

    def eliminating_behavior(self, individual: Individual) -> List[float]:
        x_worst = []

        for j in range(individual.dimension):
            R_10 = random.random()

            x_min = 0
            x_max = self.problem.max_anuncios[j]

            xj = x_min + R_10 * (x_max - x_min)

            x_worst.append(xj)

        return x_worst

    def evolve(self) -> None:
        for t in range(1, self.max_iter + 1):
            S = self.satiety(t)
            Q = self.temperature(t)

            for male in self.male_group:
                x_next: Optional[List[float]] = None

                if abs(S) < self.threshold_1:
                    x_next = self.foraging_behavior(S, male)
                else:
                    if Q < self.threshold_2:
                        x_next = self.bathing_behavior(Q, male)
                    else:
                        r = random.random()
                        if r >= 0.6:
                            x_next = self.jousting_behavior(male, self.male_best)
                        elif 0.1 < r < 0.6:
                            female = random.choice(self.female_group)
                            x_next = self.mating_behavior_male(male, female)
                        else:
                            x_next = self.eliminating_behavior(male)

                male.move(x_next, self.global_best)
                pareto_sol = ParetoSolution(male.x, male.q, self.problem)
                self.pareto_front.add_solution(pareto_sol)

            for female in self.female_group:
                x_next = None

                if abs(S) < self.threshold_1:
                    x_next = self.foraging_behavior(S, female)
                else:
                    if Q < self.threshold_2:
                        x_next = self.bathing_behavior(Q, female)
                    else:
                        r = random.random()
                        if r >= 0.6:
                            x_next = self.jousting_behavior(female, self.female_best)
                        elif 0.1 < r < 0.6:
                            male = random.choice(self.male_group)
                            x_next = self.mating_behavior_female(female, male)
                        else:
                            x_next = self.eliminating_behavior(female)

                female.move(x_next, self.global_best)
                pareto_sol = ParetoSolution(female.x, female.q, self.problem)
                self.pareto_front.add_solution(pareto_sol)

            self.update_best_individuals()
            self.show_results(t)

    def show_results(self, t: int) -> None:
        current_best = max(self.population, key=lambda x: x.fitness_value)
        log(
            f"Iteración {t}: Mejor actual = {current_best.fitness_value:.2f}, "
            f"Mejor global = {self.global_best.fitness_value:.2f}, "
            f"Frente Pareto = {self.pareto_front.size()} soluciones",
            level=2,
        )

    def mostrar_resultado_final(self) -> None:
        log("\n" + "=" * 60, level=0)
        log("RESULTADO FINAL ABO", level=0)
        log("=" * 60, level=0)
        log(f"Mejor solución encontrada:\n  {self.global_best}", level=0)

        tipos_anuncios = ["TV Tarde", "TV Noche", "Diario", "Revista", "Radio"]
        costos = [
            self.problem.costo_por_calidad(self.global_best.q[i], i)
            for i in range(self.problem.dim)
        ]

        log("\nDetalle de la solución:\n", level=0)
        log(
            f"{'Tipo':<12} {'Cantidad':>8} {'Calidad':>10} {'Costo Unitario':>16}",
            level=0,
        )
        log("-" * 50, level=0)
        for i in range(self.problem.dim):
            log(
                f"{tipos_anuncios[i]:<12} {self.global_best.x[i]:>8} "
                f"{self.global_best.q[i]:>10} {costos[i]:>16.2f}",
                level=0,
            )

        total_cost = sum(
            costos[i] * self.global_best.x[i] for i in range(self.problem.dim)
        )
        log(f"\nCosto total de la campaña: {total_cost:.2f}", level=0)
        log(
            f"Calidad total de la campaña: {self.global_best.fitness_value:.2f}",
            level=0,
        )
        log(f"Tamaño del frente de Pareto: {self.pareto_front.size()}", level=0)

    def apply_epsilon_constraint(self) -> None:
        if self.pareto_front.size() == 0:
            print("No hay soluciones en la frontera de Pareto")
            return

        epsilon_method = EpsilonConstraint(self.pareto_front)
        epsilon_values = epsilon_method.generate_epsilon_values(num_points=8)
        epsilon_method.analyze_tradeoffs(epsilon_values)

    def optimizer(self) -> None:
        self.random()
        self.evolve()
        self.mostrar_resultado_final()
        self.apply_epsilon_constraint()


VERBOSITY = 0


def log(msg: str, level: int = 0) -> None:
    if VERBOSITY >= level:
        print(msg)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            VERBOSITY = int(sys.argv[1])
        except ValueError:
            log("Nivel de verbosidad inválido, usando valor por defecto (0).", level=0)

    abo = ABO(pop_size=30, max_iter=500)
    abo.optimizer()
