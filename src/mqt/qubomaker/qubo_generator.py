"""Provides a base class for QUBO generators that can be extended for different problem classes."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
import qiskit.circuit
import sympy as sp

import qiskit

from mqt.qubomaker.device import Calibration
#from qiskit.primitives import BaseSamplerV2, StatevectorSampler
#from qiskit_algorithms import QAOA
#from qiskit_algorithms.optimizers import COBYLA, Optimizer
#from qiskit_algorithms.utils import algorithm_globals
#from qiskit_optimization import QuadraticProgram
#from qiskit_optimization.converters import QuadraticProgramToQubo

#if TYPE_CHECKING:
#    from qiskit.quantum_info import SparsePauliOp 

class EmbeddedSlackChainAssignment:
    chains: list[list[str]]
    slack_dict: dict[str, tuple[str, str]]
    indices: dict[str, int]

    def __init__(self) -> None:
        self.chains = []
        self.indices = {}
        self.slack_dict = {}

    def add_slack_variable(self, new_slack: str, replace_1: str, replace_2: str) -> None:
        self.slack_dict[new_slack] = (replace_1, replace_2)
        parts = new_slack.split("_")
        chain_index = int(parts[1]) - 1
        while chain_index >= len(self.chains):
            self.chains.append([])
        self.chains[chain_index].append(new_slack)

    def compute_indices(self, expr: sp.Expr, offset: int = 0) -> None:
        symbols = [str(s) for s in expr.free_symbols]
        prime_symbols = [s for s in symbols if s.endswith("'")]
        encoding_symbols = [s for s in symbols if not s.endswith("'") and s.startswith("x_")]
        slack_symbols = [s for s in symbols if not s.endswith("'") and s.startswith("y_")]
        self.indices.clear()
        for enc in encoding_symbols:
            self.indices[enc] = int(enc[2:]) - 1 + offset
        slack_symbols.sort(key=lambda x: tuple([int(i) for i in x[2:].split("_")]))
        for slack in slack_symbols:
            self.indices[slack] = len(self.indices)
        prime_symbols.sort(key=lambda x: tuple([x.count("'")] + [int(i) for i in x.replace("'", "")[2:].split("_")]))
        for prime in prime_symbols:
            self.indices[prime] = len(self.indices)
        

        


class QUBOGenerator:
    """A base class for QUBO generators that can be extended for different problem classes.

    Collects constraints and penalties and provides methods for constructing the QUBO representation of a problem.

    Attributes:
        objective_function (sp.Expr | None): The objective function of the problem.
        penalties (list[tuple[sp.Expr, float | None]]): The constraints and corresponding penalties.
    """

    objective_function: sp.Expr | None

    penalties: list[tuple[sp.Expr, float | None]]

    expansion_cache: sp.Expr | None = None

    auxiliary_cache: dict[sp.Expr, sp.Expr]

    smart_slack: bool = True # TODO improve

    disable_caching: bool = False  # If set to True, the caching of the expansion will be disabled. This is useful for debugging.

    def __init__(self, objective_function: sp.Expr | None) -> None:
        """Initializes a new QUBOGenerator instance.

        Args:
            objective_function (sp.Expr | None): The objective function to be used by the QUBO generator.
        """
        self.objective_function = objective_function
        self.penalties = []
        self.auxiliary_cache = {}

    def add_penalty(self, penalty_function: sp.Expr, lam: float | None = None) -> None:
        """Adds a cost function for a constraint to the problem instance.

        A penalty factor can be specified to scale the penalty function. Otherwise, a fitting penalty factor will be
        estimated automatically.

        Args:
            penalty_function (sp.Expr): A cost function that represents a constraint.
            lam (int | None, optional): The penalty scaling factor. Defaults to None.
        """
        self.expansion_cache = None
        self.auxiliary_cache = {}
        self.penalties.append((penalty_function, lam))

    def construct(self) -> sp.Expr:
        """Constructs a mathematical representation of the QUBO formulation.

        This representation is in its simplest form, including sum and product terms.

        Returns:
            sp.Expr: The mathematical representation of the QUBO formulation.
        """
        return cast(
            "sp.Expr",
            functools.reduce(
                lambda current, new: current + new[1] * new[0],
                self._select_lambdas(),
                self.objective_function if self.objective_function is not None else 0,
            ),
        )

    def construct_expansion(self, include_slack_information: bool = False, for_embedding: bool = True) -> sp.Expr | tuple[sp.Expr, EmbeddedSlackChainAssignment]:
        """Constructs a mathematical representation of the QUBO formulation and expands it.

        This will expand sum and product terms into full sums and products of each of their elements.
        The final result will be a sum of terms where each term is a product of variables and scalars.

        Raises:
            TypeError: If the constructed QUBO formulation is not a sympy expression.

        Returns:
            sp.Expr: A mathematical representation of the QUBO formulation in expanded form.
        """
        if self.expansion_cache is not None and not include_slack_information and not self.disable_caching:
            return self.expansion_cache
        expression = self.construct().expand().doit().doit()
        if isinstance(expression, sp.Expr):
            expression = self._construct_expansion(expression).expand()
        expression = expression.doit().expand()
        embedded_assignment = EmbeddedSlackChainAssignment()
        if for_embedding:
            unifying_substitution = {
                var: sp.Symbol(f"x_{i}") for (var, i) in self._get_encoding_variables()
            }
            expression = expression.subs(unifying_substitution)
            expression = self.expand_higher_order_terms(expression, embedded_assignment)
        else:
            expression = self.expand_higher_order_terms_greedy_minimization(expression)
        self.expansion_cache = expression

        if include_slack_information:
            embedded_assignment.compute_indices(expression)
            return expression, embedded_assignment
        return expression
    
    def expand_higher_order_terms(self, expression: sp.Expr, assignment: EmbeddedSlackChainAssignment, chain_index: int = 1) -> sp.Expr:
        assert isinstance(expression, sp.Add) or isinstance(expression, sp.Mul), f"We expect a sum of products or a single product as input but got {expression}"
        problematic_terms = self.__get_problematic_terms(expression.args) if isinstance(expression, sp.Add) else self.__get_problematic_terms([expression])
        
        if not problematic_terms:
            return expression
        
        counts: dict[tuple[sp.Symbol | sp.Function, sp.Symbol | sp.Function], int] = {}
        highest_key: tuple[sp.Symbol | sp.Function, sp.Symbol | sp.Function] | None = None
        for term in problematic_terms:
            symbols = sorted(term.free_symbols, key=lambda s: str(s), reverse=True)
            for s1 in symbols:
                for s2 in symbols:
                    if s1 == s2:
                        continue
                    key = (s1, s2)
                    if key not in counts:
                        counts[key] = 0
                    counts[key] += 1
                    if highest_key is None or counts[key] > counts[highest_key]:
                        highest_key = key

        selected_variables = [highest_key[0], highest_key[1]]
        equality_penality = 0
        if any(str(selected_variables[0]) in x for x in assignment.slack_dict.values()):
            old_var = selected_variables[0]
            current = selected_variables[0]
            while any(str(current) in x for x in assignment.slack_dict.values()):
                current = sp.Symbol(f"{current}'")
            selected_variables[0] = current
            equality_penality += old_var + selected_variables[0] - old_var * selected_variables[0] # introduce (x - x')^2 as penalty.
        if any(str(selected_variables[1]) in x for x in assignment.slack_dict.values()):
            old_var = selected_variables[1]
            current = selected_variables[1]
            while any(str(current) in x for x in assignment.slack_dict.values()):
                current = sp.Symbol(f"{current}'")
            selected_variables[1] = current
            equality_penality += old_var + selected_variables[1] - old_var * selected_variables[1] # introduce (x - x')^2 as penalty.
        
        y = sp.Symbol(f"y_{chain_index}_1")
        assignment.add_slack_variable(str(y), str(selected_variables[0]), str(selected_variables[1]))
        new_expr = expression.subs({highest_key[0] * highest_key[1]: y}) + self.__get_slack_penalty(selected_variables[0], selected_variables[1], y)
        new_expr += equality_penality
        return self.__continue_slack_chain(new_expr, y, assignment, chain_index, 1)
    
    def __get_slack_penalty(self, x1: sp.Expr, x2: sp.Expr, y: sp.Symbol) -> sp.Expr:
        """Computes the slack penalty for a given pair of variables and an auxiliary variable.

        Args:
            x1 (sp.Expr): The first variable.
            x2 (sp.Expr): The second variable.
            y (sp.Symbol): The auxiliary variable.

        Returns:
            sp.Expr: The slack penalty expression.
        """
        return x1 * x2 - 2 * y * x1 - 2 * y * x2 + 3 * y
    
    def __get_problematic_terms(self, terms: list[sp.Mul]) -> list[sp.Mul]:
        problematic_terms: list[sp.Mul] = []
        for term in terms:
            unpowered = self.__unpower(term)
            order = self.__get_order(unpowered)
            if order <= 2:
                continue
            assert isinstance(unpowered, sp.Mul), "Only products of variables can be left from sums of products."
            problematic_terms.append(unpowered)
        return problematic_terms

    def __continue_slack_chain(self, expression: sp.Add, last_slack: sp.Symbol, assignment: EmbeddedSlackChainAssignment, chain_index: int, index: int) -> sp.Add:
        problematic_terms = self.__get_problematic_terms(expression.args)

        if not problematic_terms:
            return expression

        counts: dict[sp.Symbol | sp.Function, int] = {}
        highest_key: sp.Symbol | sp.Function | None = None
        for term in problematic_terms:
            symbols = sorted(term.free_symbols, key=lambda s: str(s), reverse=True)
            if last_slack not in symbols:
                continue
            for s in symbols:
                if s == last_slack:
                    continue
                if s not in counts:
                    counts[s] = 0
                counts[s] += 1
                if highest_key is None or counts[s] > counts[highest_key]:
                    highest_key = s
        
        if highest_key is None:
            return self.expand_higher_order_terms(expression, assignment, chain_index + 1)
        selected_replacement = highest_key
        equality_penality = 0
        if any(str(selected_replacement) in x for x in assignment.slack_dict.values()):
            while any(str(selected_replacement) in x for x in assignment.slack_dict.values()):
                selected_replacement = sp.Symbol(f"{selected_replacement}'")
            equality_penality += highest_key + selected_replacement - highest_key * selected_replacement  # introduce (x - x')^2 as penalty.
        y = sp.Symbol(f"y_{chain_index}_{index + 1}")
        assignment.add_slack_variable(str(y), str(last_slack), str(selected_replacement))
        new_expr = expression.subs({last_slack * highest_key: y}) + self.__get_slack_penalty(last_slack, selected_replacement, y)
        new_expr += equality_penality
        return self.__continue_slack_chain(new_expr, y, assignment, chain_index, index + 1)
        
    def expand_higher_order_terms_greedy_minimization(self, expression: sp.Expr) -> sp.Expr:
        assert isinstance(expression, sp.Add) or isinstance(expression, sp.Mul), f"We expect a sum of products or a single product as input but got {expression}"
        slack_count = 0
        
        while True:
            problematic_terms = self.__get_problematic_terms(expression.args) if isinstance(expression, sp.Add) else self.__get_problematic_terms([expression])
            
            if not problematic_terms:
                return expression
            
            counts: dict[tuple[sp.Symbol | sp.Function, sp.Symbol | sp.Function], int] = {}
            highest_key: tuple[sp.Symbol | sp.Function, sp.Symbol | sp.Function] | None = None
            for term in problematic_terms:
                symbols = sorted(term.free_symbols, key=lambda s: str(s))
                #print("Symbols:", symbols)
                for s1 in symbols:
                    for s2 in symbols:
                        if s1 == s2:
                            continue
                        key = (s1, s2)
                        if key not in counts:
                            counts[key] = 0
                        counts[key] += 1
                        if highest_key is None or counts[key] > counts[highest_key]:
                            highest_key = key

            selected_variables = [highest_key[0], highest_key[1]]
            #print("Selected:", selected_variables)
            
            slack_count += 1
            y = sp.Symbol(f"y_{slack_count}")
            expression = expression.subs({highest_key[0] * highest_key[1]: y}) + self.__get_slack_penalty(selected_variables[0], selected_variables[1], y)
            #print("New Expression:", expression)

    def expand_higher_order_terms_no_embedding(self, expression: sp.Expr) -> sp.Expr:
        """Expands a mathematical QUBO expression.

        Terms of order 3 or higher will be transformed into quadratic terms by adding auxiliary variables recursively until
        the order is 2.

        Args:
            expression (sp.Expr): The expression to transform.

        Returns:
            sp.Expr: The transformed expression.
        """
        result: sp.Expr | float = 0
        auxiliary_dict: dict[sp.Expr, sp.Expr] = {}
        coeffs = expression.as_coefficients_dict()
        for term in coeffs:
            unpowered = self.__unpower(term)
            unpowered = self.__simplify_auxiliary_variables(unpowered, auxiliary_dict)
            order = self.__get_order(unpowered)
            if order <= 2:
                result += unpowered * coeffs[term]
                continue
            new_term = self.__decrease_order(unpowered, auxiliary_dict)
            result += new_term * coeffs[term]
        self.auxiliary_cache = auxiliary_dict
        return cast("sp.Expr", result)

    @staticmethod
    def __simplify_auxiliary_variables(expression: sp.Expr, auxiliary_dict: dict[sp.Expr, sp.Expr]) -> sp.Expr:
        """Minimizes the number of requires auxiliary variables by removing products that have already been transformed in previous steps.

        Args:
            expression (sp.Expr): The expression to optimize
            auxiliary_dict (dict[sp.Expr, sp.Expr]): A dictionary mapping existing products of variables to their resulting auxiliary variable.

        Returns:
            sp.Expr: The optimized expression.
        """
        if not isinstance(expression, sp.Mul):
            return expression
        used_auxiliaries = {term for term in expression.args if term in auxiliary_dict.values()}
        redundant_variables = {term for term in auxiliary_dict if auxiliary_dict[term] in used_auxiliaries}
        remaining_variables = [arg for arg in expression.args if arg not in redundant_variables]
        return sp.Mul(*remaining_variables) if len(remaining_variables) > 1 else remaining_variables[0]

    def __optimal_decomposition(
        self, terms: tuple[sp.Expr, ...], auxiliary_dict: dict[sp.Expr, sp.Expr]
    ) -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
        """Computes the optimal decomposition of a product of variables into terms of order 2.

        Args:
            terms (tuple[sp.Expr, ...]): The terms of the product.
            auxiliary_dict (dict[sp.Expr, sp.Expr]): The previously used auxiliary variables.

        Returns:
            tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]: A tuple containing the two variables that are multiplied, the auxiliary variable used for them, and the remaining expression.
        """
        for x1 in terms:
            for x2 in terms:
                if x1 == x2:
                    continue
                if (x1 * x2) not in auxiliary_dict:
                    continue
                return x1, x2, auxiliary_dict[x1 * x2], sp.Mul(*[term for term in terms if term not in {x1, x2}])
        if self.smart_slack:
            x1 = terms[-2]
            x2 = terms[-1]
            rest = sp.Mul(*terms[:-2])
        else:
            x1 = terms[0]
            x2 = terms[1]
            rest = sp.Mul(*terms[2:])
        y: sp.Symbol = sp.Symbol(f"y_{len(auxiliary_dict) + 1}")
        auxiliary_dict[x1 * x2] = y
        return x1, x2, y, rest

    def __decrease_order(self, expression: sp.Expr, auxiliary_dict: dict[sp.Expr, sp.Expr]) -> sp.Expr:
        """Decreases the order of a product of variables by adding auxiliary variables.

        Args:
            expression (sp.Expr): The expression to transform.
            auxiliary_dict (dict[sp.Expr, sp.Expr]): A dictionary of previously used auxiliary variables.

        Returns:
            sp.Expr: The new expression with lower order.
        """
        (x1, x2, y, rest) = self.__optimal_decomposition(expression.args, auxiliary_dict)
        auxiliary_penalty = x1 * x2 - 2 * y * x1 - 2 * y * x2 + 3 * y
        rest *= y
        if self.__get_order(rest) > 2:
            rest = self.__decrease_order(rest, auxiliary_dict)
        return auxiliary_penalty + rest

    def __get_order(self, expression: sp.Expr) -> int:
        """Computes the order of a product of variables.

        Args:
            expression (sp.Expr): The expression to check.

        Returns:
            int: The order of the expression. If the expression is not a product, 1 is returned.
        """
        if isinstance(expression, sp.Mul):
            return sum(self.__get_order(arg) for arg in expression.args)
        if isinstance(expression, sp.Symbol) or isinstance(expression, sp.Function):
            return 1
        return 0

    def __unpower(self, expression: sp.Expr) -> sp.Expr:
        """Removes exponentiation from an expression.

        This matters, because when using binary variables, x^2 = x always holds. This allows us to compute the order of the
        term more easily.

        Args:
            expression (sp.Expr): The expression to transform.

        Returns:
            sp.Expr: The transformed expression.
        """
        if isinstance(expression, sp.Pow):
            return expression.args[0]
        if isinstance(expression, sp.Mul):
            return sp.Mul(*[self.__unpower(arg) for arg in expression.args])
        return expression

    def __get_auxiliary_variables(self, expression: sp.Expr) -> list[sp.Symbol]:
        """Returns a list of all auxiliary variables used in an expression.

        Auxiliary variables will start with "y_" by definition.

        Args:
            expression (sp.Expr): The expression to check.

        Returns:
            list[sp.Symbol]: The list of employed auxiliary variables.
        """
        if isinstance(expression, sp.Mul):
            return list({var for arg in expression.args for var in self.__get_auxiliary_variables(arg)})
        if isinstance(expression, sp.Symbol) and (str(expression).startswith("y_") or str(expression).endswith("'")):
            return [expression]
        return []

    def _construct_expansion(self, expression: sp.Expr) -> sp.Expr:  # noqa: PLR6301
        """A method that can be extended by classes that inherit from QUBOGenerator to transform the QUBO formulation into expanded form, if that process requires additional steps.

        Args:
            expression (sp.Expr): The expression to transform.

        Returns:
            sp.Expr: The transformed expression.
        """
        return expression

    def construct_qubo_matrix(self, for_embedding: bool = False) -> npt.NDArray[np.int_ | np.float64]:
        """Constructs the matrix representation of the QUBO problem.

        This is achieved by first creating the expanded QUBO formula, and then taking the coefficients of each term.

        Returns:
            npt.NDArray[np.int_ | np.float64]: The matrix representation of the QUBO problem.
        """
        if not for_embedding:
            coefficients = dict(self.construct_expansion(for_embedding=for_embedding).expand().as_coefficients_dict())
            auxiliary_variables = list({var for arg in coefficients for var in self.__get_auxiliary_variables(arg)})
            auxiliary_variables.sort(key=lambda var: tuple([0 if str(var).startswith("y") else 1] + [int(x) for x in str(var)[2:].replace("'", "").split("_")]))
            result = np.zeros((
                self.get_encoding_variable_count() + len(auxiliary_variables),
                self.get_encoding_variable_count() + len(auxiliary_variables),
            ))

            all_variables = dict(self._get_encoding_variables())

            def get_index(variable: sp.Expr) -> int:
                if variable in all_variables:
                    return all_variables[variable] - 1
                return auxiliary_variables.index(cast("sp.Symbol", variable)) + self.get_encoding_variable_count()
        else:
            expr, assignment = self.construct_expansion(include_slack_information=True, for_embedding=for_embedding)
            num_variables = len(assignment.indices)
            result = np.zeros((
                num_variables,
                num_variables,
            ))
            coefficients = dict(expr.as_coefficients_dict())
            def get_index(variable: sp.Expr) -> int:
                return assignment.indices[str(variable)]
        
        for term, value in coefficients.items():
                if isinstance(term, sp.Mul):
                    index1 = get_index(term.args[0])
                    index2 = get_index(term.args[1])
                    if index1 > index2:
                        index1, index2 = index2, index1
                    result[index1][index2] = value
                elif isinstance(term, (sp.Symbol, sp.Function)):
                    index = get_index(term)
                    result[index][index] = value

        return result



    def get_cost(self, assignment: list[int]) -> float:
        """Given an assignment, computes the total cost value of the corresponding cost function evaluation.

        The assignment is given as a binary list, and can either contain assignments for just encoding variables
        or encoding + auxiliary variables. In the former case, the auxiliary values are computed automatically.

        Args:
            assignment (list[int]): The assignment for each variable (either 0 or 1).

        Returns:
            float: The cost value for the assignment.
        """
        if any(x not in {0, 1} for x in assignment):
            msg = "Provided values are not binary (1/0)"
            raise ValueError(msg)

        expansion = self.construct_expansion()
        auxiliary_assignment: dict[sp.Expr, int] = {}

        if len(assignment) == self.get_encoding_variable_count():
            auxiliary_assignment = self.__get_auxiliary_assignment(assignment)
        elif len(assignment) != self.count_required_variables():
            msg = "Invalid assignment length."
            raise ValueError(msg)

        variable_assignment = {item[0]: assignment[item[1] - 1] for item in self._get_encoding_variables()}
        variable_assignment.update(auxiliary_assignment)
        return expansion.subs(variable_assignment).evalf()

    def __get_auxiliary_assignment(self, assignment: list[int]) -> dict[sp.Expr, int]:
        """Generates the assignment of auxiliary variables based on a given encoding variable assignment.

        Every auxiliary variable is defined like `y_k = x_i * x_j`, `y_k = x_i * y_j` or `y_k = y_i * y_j`.
        These definitions are stored in the `auxiliary_cache` dictionary. As there are no circular references,
        a given set of assignments for each `x_i` is enough to compute the values of all `y_k`. This can be used to
        compute the cost for a given QUBO encoding without requiring the user to define all auxiliary variables.

        Args:
            assignment (list[int]): The assignment for the encoding variables `x_i`.

        Returns:
            dict[sp.Expr, int]: An assignment dictionary, mappingeach `y_k` to its value in {0, 1}.
        """
        auxiliary_values: dict[sp.Expr, int] = {}
        encoding_variables = dict(self._get_encoding_variables())
        remaining_variables = set(self.auxiliary_cache.keys())

        def get_var_value(var: sp.Expr) -> int | None:
            if var in encoding_variables:
                return assignment[encoding_variables[var] - 1]
            if var in auxiliary_values:
                return auxiliary_values[var]
            return None

        while remaining_variables:
            to_remove = []
            for aux in remaining_variables:
                (left, right) = aux.args[0], aux.args[1]
                left_val = get_var_value(left)
                right_val = get_var_value(right)
                if left_val is not None and right_val is not None:
                    auxiliary_values[self.auxiliary_cache[aux]] = left_val * right_val
                    to_remove.append(aux)
            for aux in to_remove:
                remaining_variables.remove(aux)

        return auxiliary_values

    def _get_encoding_variables(self) -> list[tuple[sp.Expr, int]]:
        """Returns all non-auxiliary variables used in the QUBO formulation.

        Returns:
            list[tuple[sp.Expr, int]]: A list of tuples containing the variable and its index.
        """
        all_expresions = [self.objective_function] + [penalty[0] for penalty in self.penalties]
        variables = set()
        for expr in all_expresions:
            variables |= expr.free_symbols
        l = sorted(list(variables), key=lambda var: int(str(var)[2:]))
        return [(var, i + 1) for i, var in enumerate(l)]

    def _select_lambdas(self) -> list[tuple[sp.Expr, float]]:
        """Computes the penalty factors for each constraint. May be extended by subclasses.

        Returns:
            list[tuple[sp.Expr, float]]: A list of tuples containing the individual cost functions and their constraints.
        """
        return [(expr, weight) if weight is not None else (expr, 1.0) for (expr, weight) in self.penalties]

    def count_required_variables(self) -> int:
        """Returns the total number of variables required to represent the QUBO problem.

        Returns:
            int: The number of required variables.
        """
        coefficients = dict(self.construct_expansion().as_coefficients_dict())
        auxiliary_variables = list({var for arg in coefficients for var in self.__get_auxiliary_variables(arg)})
        return len(self._get_encoding_variables()) + len(auxiliary_variables)

    def get_encoding_variable_count(self) -> int:
        """Returns the number of non-auxiliary binary variables required to represent the QUBO problem.

        Returns:
            int: The number of required binary variables.
        """
        return len(self._get_encoding_variables())

    def get_variable_index(self, variable: sp.Expr) -> int:
        """For a given variable, returns its index in the QUBO matrix.

        Args:
            _variable (sp.Expr): The variable to investigate.

        Returns:
            int: The index of the variable.
        """
        encoding_variables = [x[0] for x in self._get_encoding_variables()]
        if variable in encoding_variables:
            return encoding_variables.index(variable) + 1
        return -1

    def decode_bit_array(self, _array: list[int]) -> Any:  # noqa: PLR6301, ANN401
        """Given an assignment, decodes it into a meaningful result. May be extended by subclasses.

        Args:
            _array (list[int]): The binary assignment.

        Returns:
            Any: The decoded result.
        """
        return ""

    def construct_qaoa_circuit(self, n_qubits: int = -1, do_reuse: bool = True) -> qiskit.QuantumCircuit:
        include_barriers = False

        interactions = self.construct_interaction_graph(for_embedding=False)
        qubits = max(max(i, j) for i, j in interactions) + 1
        circuit = qiskit.QuantumCircuit(n_qubits if n_qubits != -1 else qubits, qubits)
        gamma = qiskit.circuit.Parameter("gamma")
        beta = qiskit.circuit.Parameter("beta")
        
        

        if not do_reuse:
            for i in range(qubits):
                circuit.h(i)
            if include_barriers:
                circuit.barrier()

            for i, j in interactions:
                circuit.rzz(gamma, i, j)
            if include_barriers:
                circuit.barrier()

            for i in range(qubits):
                circuit.rx(beta, i)
            if include_barriers:
                circuit.barrier()
            for i in range(qubits):
                circuit.measure(i, i)
        else:
            used_qubits = set()
            free_qubits = list(range(qubits))
            substitution: dict[int, int] = {}
            outgoing = {i: set() for i in range(qubits)}
            for i, j in interactions:
                outgoing[i].add(j)
                outgoing[j].add(i)

            def add_new_qubit(qubit: int) -> None:
                substitution[qubit] = free_qubits.pop(0)
                circuit.h(substitution[qubit])
                used_qubits.add(substitution[qubit])

            total_outgoing = set()
            remaining = list(range(qubits))
            covered = set()
            while remaining:
                best_size = qubits + 1
                next_qubit = -1
                for q in remaining:
                    size = len(total_outgoing.union(outgoing[q]))
                    if size < best_size:
                        best_size = size
                        next_qubit = q
                remaining.remove(next_qubit)
                total_outgoing = total_outgoing.union(outgoing[next_qubit])
                add_new_qubit(next_qubit)

                for other in outgoing[next_qubit]:
                    if other in covered:
                        continue
                    if other not in substitution:
                        add_new_qubit(other)
                    circuit.rzz(gamma, substitution[next_qubit], substitution[other])
                covered.add(next_qubit)
                circuit.rx(beta, substitution[next_qubit])
                circuit.measure(substitution[next_qubit], next_qubit)
                free_qubits.insert(0, substitution[next_qubit])

        if do_reuse and len(used_qubits) < len(circuit.qubits):
            return self.construct_qaoa_circuit(n_qubits=len(used_qubits))
        return circuit
    
    def construct_interaction_graph(self, offset: int = 0, for_embedding: bool = False) -> list[tuple[int, int]]:
        """Constructs the interaction graph of the QUBO problem.

        Returns:
            list[tuple[int, int]]: The interaction graph of the QUBO problem.
        """
        qubo = self.construct_qubo_matrix(for_embedding=for_embedding)
        return [(i + offset, j + offset) for i in range(len(qubo)) for j in range(i + 1, len(qubo)) if qubo[i][j] != 0]

    def construct_embedded_qaoa_circuit(self, device: Calibration) -> qiskit.QuantumCircuit:
        expression, assignment = self.construct_expansion(include_slack_information=True, for_embedding=True)
        coefficients = dict(expression.as_coefficients_dict())
        auxiliary_variables = list({var for arg in coefficients for var in self.__get_auxiliary_variables(arg)})
        auxiliary_variables.sort(key=lambda var: tuple([0 if str(var).startswith("y") else 1] + [int(x) for x in str(var)[2:].replace("'", "").split("_")]))

        variable_indices = assignment.indices
        slack_dict = assignment.slack_dict
        indices_to_variables = {i: v for v, i in variable_indices.items()}                
        
        chains = assignment.chains
        interactions = self.construct_interaction_graph(offset=0, for_embedding=True)
        covered_interactions: set[tuple[int, int]]= set()

        substitution: dict[str, int] = {}
        full_nn_chain: list[int] = device.get_connected_qubit_chain()
        nn_qubit_queue: list[int] = device.get_connected_qubit_chain()

        qc = qiskit.QuantumCircuit(device.num_qubits, device.num_qubits)
        beta = qiskit.circuit.Parameter("beta")
        gamma = qiskit.circuit.Parameter("gamma")

        def add_rzz(q1: str, q2: str) -> None:
            nonlocal qc, gamma, substitution
            qc.rzz(gamma, substitution[q1], substitution[q2])
        def add_swap(q1: str, q2: str, decomposed: bool) -> None:
            nonlocal qc, gamma, substitution
            if decomposed:
                qc.sx(substitution[q1])
                qc.sx(substitution[q2])
                qc.cz(substitution[q1], substitution[q2])
                qc.sx(substitution[q1])
                qc.sx(substitution[q2])
                qc.cz(substitution[q1], substitution[q2])
                qc.sx(substitution[q1])
                qc.sx(substitution[q2])
                qc.cz(substitution[q1], substitution[q2])
            else:
                qc.swap(substitution[q1], substitution[q2])
        def get_next_substitution(q: str) -> int:
            nonlocal substitution
            nonlocal nn_qubit_queue
            if q in substitution:
                #print(f"Variable {q} already substituted, using {substitution[q]}")
                return substitution[q]
            substitution[q] = nn_qubit_queue.pop(0)
            return substitution[q]


        chain_groups: list[list[tuple[str, str]]] = []
        for chain in chains:
            first_var = chain[0]
            first_pair = slack_dict[first_var]
            get_next_substitution(first_pair[0])
            get_next_substitution(first_pair[1])
            groups: list[tuple[str, str]] = [] # A group always consists of a slack var and its subsequent encoding var, or the first two encoding variables
            groups.append(first_pair)
            for slack in chain[1:]:
                predecessors = slack_dict[slack]
                last_slack = predecessors[0]
                shared_var = predecessors[1]
                get_next_substitution(last_slack)
                get_next_substitution(shared_var)
                groups.append((last_slack, shared_var))
            substitution[chain[-1]] = nn_qubit_queue.pop(0)
            groups.append((chain[-1], ""))
            chain_groups.append(groups)
        
        #for i in range(qc.num_qubits):
        #    qc.h(i)

        for groups in chain_groups:
            for group in groups:
                if not group[1]:
                    continue
                add_rzz(group[0], group[1])
                covered_interactions.add((variable_indices[group[0]], variable_indices[group[1]]))
        for groups in chain_groups:
            for g1, g2 in zip(groups[:-1], groups[1:]):
                add_rzz(g1[1], g2[0])
                covered_interactions.add((variable_indices[g1[1]], variable_indices[g2[0]]))
                covered_interactions.add((variable_indices[g1[0]], variable_indices[g2[0]])) # will be added in for loop below but needs to be done now already for next step
            
        for i, j in interactions:
            if (i, j) in covered_interactions or (j, i) in covered_interactions:
                continue
            v_i = indices_to_variables[i]
            v_j = indices_to_variables[j]
            if v_i not in substitution:
                substitution[v_i] = nn_qubit_queue.pop(0)
            if v_j not in substitution:
                substitution[v_j] = nn_qubit_queue.pop(0)
            add_rzz(v_i, v_j)

        for groups in chain_groups:
            for g1, g2 in zip(groups[::2], groups[1::2]):
                add_swap(g1[0], g1[1], False)
                add_rzz(g1[1], g2[0])
                add_swap(g1[0], g1[1], False)
            for g1, g2 in zip(groups[1::2], groups[2::2]):
                add_swap(g1[0], g1[1], False)
                add_rzz(g1[1], g2[0])
            
            #for g1, g2 in zip(groups[:-1], groups[1:]):
            #    add_swap(g1[0], g1[1])
            #    add_rzz(g1[1], g2[0])

        qc_front = qiskit.QuantumCircuit(device.num_qubits, device.num_qubits)
        for i in range(len(full_nn_chain) - len(nn_qubit_queue)):
            q = full_nn_chain[i]
            
            qc_front.h(q)
            qc.rx(beta, q)
            qc.measure(q, q)
        qc = qc_front.compose(qc, front=False)

        return qc






    def construct_embedded_qaoa_circuit_old(self, device: Calibration) -> qiskit.QuantumCircuit:
        interactions = self.construct_interaction_graph(offset=1)

        num_encoding_variables = self.get_encoding_variable_count()
        num_variables = max(max(i, j) for (i, j) in interactions)

        # The heavy chain gives a sequence of heavy nodes in the heavyhex topology that can be
        # consumed to find the next physical qubit to use for junctions in the QUBO.
        heavy_chain = device.get_heavy_chain()

        substitution: dict[int, int] = {}

        # The first two variables need special treatment as they are the only encoding variables
        # that are directly connected (without a slack junction).
        next_heavy_node = heavy_chain.pop(0)
        connected_encoding_variables = [(i, j) for i, j in interactions if i <= num_encoding_variables and j <= num_encoding_variables]
        assert len(connected_encoding_variables) == 1
        x1, x2 = connected_encoding_variables[0]
        substitution[x1] = next_heavy_node
        substitution[x2] = [x for x in device.heavy[next_heavy_node] if x in device.heavy[heavy_chain[0]]][0]
        next_heavy_node = heavy_chain.pop(0)

        # The remaining variables are connected through slack junctions.
        slack_interactions = {}
        for (i, j) in interactions:
            if i > num_encoding_variables:
                if i not in slack_interactions:
                    slack_interactions[i] = []
                slack_interactions[i].append(j)
            if j > num_encoding_variables:
                if j not in slack_interactions:
                    slack_interactions[j] = []
                slack_interactions[j].append(i)
                
        current_slack = num_encoding_variables + 1
        while current_slack < num_variables:
            substitution[current_slack] = next_heavy_node
            next_slack_variable = [x for x in slack_interactions[current_slack] if x > num_encoding_variables and x > current_slack][0] # always exactly one, only current_slack == num_variables has 0
            related_encoding_variable = [x for x in slack_interactions[current_slack] if x <= num_encoding_variables and x not in substitution][0]
            substitution[related_encoding_variable] = [x for x in device.heavy[next_heavy_node] if x in device.heavy[heavy_chain[0]]][0]
            next_heavy_node = heavy_chain.pop(0)
            current_slack = next_slack_variable
        substitution[current_slack] = next_heavy_node
        for remaining in slack_interactions[current_slack]:
            if remaining not in substitution:
                for neighbor in device.heavy[next_heavy_node]:
                    if neighbor not in substitution.values():
                        substitution[remaining] = neighbor
                        break
        
        # From the generated interactions and substitution, we can now construct the QAOA circuit.
        gamma = qiskit.circuit.Parameter("gamma")
        beta = qiskit.circuit.Parameter("beta")
        qc = qiskit.QuantumCircuit(device.num_qubits, num_variables)

        for i in range(1, num_variables + 1):
            qc.h(substitution[i])

        for i, j in interactions:
            mapped_i = substitution[i]
            mapped_j = substitution[j]
            if (mapped_i, mapped_j) in device.two_qubit:
                qc.rzz(gamma, mapped_i, mapped_j)
            elif (mapped_j, mapped_i) in device.two_qubit:
                qc.rzz(gamma, mapped_j, mapped_i)
            else:
                shared_neighbor = device.get_shared_neighbor(mapped_i, mapped_j)
                assert shared_neighbor != -1
                if (mapped_i, shared_neighbor) in device.two_qubit:
                    qc.swap(mapped_i, shared_neighbor)
                else:
                    qc.swap(shared_neighbor, mapped_i)
                if (shared_neighbor, mapped_j) in device.two_qubit:
                    qc.rzz(gamma, shared_neighbor, mapped_j)
                else:
                    qc.rzz(gamma, mapped_j, shared_neighbor)
                if (mapped_i, shared_neighbor) in device.two_qubit:
                    qc.swap(mapped_i, shared_neighbor)
                else:
                    qc.swap(shared_neighbor, mapped_i)

        for i in range(1, num_variables + 1):
            qc.rx(beta, substitution[i])

        for i in range(1, num_variables + 1):
            qc.measure(substitution[i], i - 1)

        return qc

