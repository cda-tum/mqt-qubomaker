from __future__ import annotations

from dataclasses import dataclass, field

@dataclass
class Calibration:
    num_qubits: int

    one_qubit: dict[int, float]
    two_qubit: dict[tuple[int, int], float]
    measurement_confidences: dict[int, float]
    basis_gates: list[str]

    connections_dict: dict[int, list[int]] = field(init=False)
    heavy: dict[int, list[int]] = field(init=False)
    heavy_children: dict[int, list[int]] = field(init=False)


    t1: dict[int, float]
    t2: dict[int, float]

    def __post_init__(self) -> None:
        self.connections_dict = {}
        for i, j in self.two_qubit:
            if i not in self.connections_dict:
                self.connections_dict[i] = []
            if j not in self.connections_dict:
                self.connections_dict[j] = []
            self.connections_dict[i].append(j)
            self.connections_dict[j].append(i)
        self.heavy = {}
        self.heavy_children = {}
        for qubit, connections in self.connections_dict.items():
            if len(connections) > 2:
                self.heavy[qubit] = connections
                for connection in connections:
                    if connection not in self.heavy_children:
                        self.heavy_children[connection] = []
                    self.heavy_children[connection].append(qubit)


    @classmethod
    def from_dict(cls, data: dict, basis_gates: list[str]) -> Calibration:
        """Create a Calibration object from a dictionary."""
        return cls(
            num_qubits=data["num_qubits"],
            one_qubit=data["one_qubit"],
            two_qubit=data["two_qubit"],
            measurement_confidences=data["measurement_confidences"],
            basis_gates=basis_gates,
            t1=data["t1"],
            t2=data["t2"],
        )

    def get_shared_neighbor(self, q1: int, q2: int) -> int:
        for x in self.connections_dict[q1]:
            if x in self.connections_dict[q2]:
                return x
        return -1
    
    def get_connected_qubit_chain(self) -> list[int]:
        potential_starts = [x for x in self.connections_dict if len(self.connections_dict[x]) == 1]
        assert len(potential_starts) == 2, "There should be exactly two potential starts for the connected qubit chain."
        start = potential_starts[0]
        path = [start]
        current = start
        while True:
            possible_successors = [x for x in self.connections_dict[current] if (len(path) == 1 or x != path[-2])]
            successor_distances = sorted([(abs(s - current), s) for s in possible_successors])
            if len(successor_distances) == 0:
                break
            current = successor_distances[0][1]
            path.append(current)
        return path
    
    def get_heavy_chain(self) -> list[int]:
        heavy_graph = {}
        for x, children in self.heavy.items():
            heavy_graph[x] = []
            for child in children:
                for other in self.heavy_children[child]:
                    if other != x:
                        heavy_graph[x].append(other)
        potential_starts = [x for x in heavy_graph if len(heavy_graph[x]) == 1]
        potential_starts = [x for x in potential_starts if len(heavy_graph[heavy_graph[x][0]]) == 2]
        
        heavy_chain = [potential_starts[0]]
        while True:
            current = heavy_chain[-1]
            possible_successors = [x for x in heavy_graph[current] if x not in heavy_chain]
            if len(possible_successors) == 0:
                break
            if len(possible_successors) == 1:
                heavy_chain.append(possible_successors[0])
                continue
            assert len(possible_successors) == 2, "Heavy chain must have exactly two successors."
            [s_a, s_b] = possible_successors
            d_a = abs(s_a - current)
            d_b = abs(s_b - current)
            if d_a < d_b:
                heavy_chain.append(s_a)
            else:
                heavy_chain.append(s_b)
        return heavy_chain
    
TEST_500 = Calibration(
    num_qubits=500,
    one_qubit={i: 0.99 for i in range(500)},
    two_qubit={(i, i + 1): 0.99 for i in range(499)},
    measurement_confidences={i: 0.99 for i in range(500)},
    basis_gates=["cz", "id", "rx", "rz", "rzz", "sx", "x"],
    t1={i: 100e-6 for i in range(500)},
    t2={i: 200e-6 for i in range(500)},
)
    