from utils import * 
from graphs import * 


class RandomGraphCircuit:

    def __init__(self, graph_state: GraphState, total_qubits: int, avg_gates_per_layer: float):
        """
        Initialize a brick-wall circuit with periodic boundary conditions.
        
        Args:
            total_qubits (int): Total number of qubits in the circuit (N)
            graph_qubits (int): Number of qubits in each graph state unit (n)
                              Must divide total_qubits evenly (N = n * x)
        """
        self.graph_state = graph_state
        self.graph_qubits = graph_state.num_qubits
        self.avg_gates_per_layer = avg_gates_per_layer
        self.total_qubits = total_qubits
        self.num_blocks = total_qubits // self.graph_qubits
        
        self.circuit = stim.Circuit()
        self.circuit.append('I', [self.total_qubits-1])

        self.simulator = stim.TableauSimulator()
        self.simulator.do_circuit(self.circuit)
        tableau = self.simulator.current_inverse_tableau()
        self.tab_forward = tableau.inverse()

        self.max_gate = int(self.avg_gates_per_layer)
        
        # Cap at theoretical maximum
        max_possible = self.total_qubits // self.graph_qubits
        self.max_gate = min(self.max_gate, max_possible)        

    def generate_random_configs(self, max_attempts:int=100):
        
        # all_groups = [list(range(i, i + self.graph_qubits)) for i in range(self.total_qubits - self.graph_qubits + 1)] # OBC
        
        # PBC

        all_groups = [
            [(i + j) % self.total_qubits for j in range(self.graph_qubits)]
            for i in range(self.total_qubits)
        ]
        valid_configs = []
        
        # Randomly select k non-overlapping gates
        attempts = 0 
        while attempts < max_attempts:
            np.random.shuffle(all_groups)
            selected = []
            used_qubits = set()
            for group in all_groups:
                if not any(q in used_qubits for q in group):
                    selected.append(group)
                    used_qubits.update(group)
                    if len(selected) == self.max_gate:
                        break
            if len(selected) == self.max_gate:
                valid_configs.append(selected)
                break
            attempts += 1
        
        return valid_configs[0] if valid_configs else [np.random.choice(all_groups)]    
    

    def evolve_circuit(self) -> stim.Circuit:

        current_circuit = stim.Circuit()
        current_circuit.append('I', [self.total_qubits-1])

        get_config = self.generate_random_configs()
   
        for config in get_config:
            self.graph_state.apply_to_circuit(current_circuit, config, random_order=False)

        current_simulator = stim.TableauSimulator()
        current_simulator.do_circuit(current_circuit)
        current_tableau = current_simulator.current_inverse_tableau()
        current_tab_forward = current_tableau.inverse()
        
        
        self.tab_forward = current_tab_forward * self.tab_forward
        self.circuit = self.circuit + current_circuit

    
    def entanglement_evolution(self, layers: int):
        
        E = np.zeros(layers, dtype = float)
        
        for layer in range(layers):
            self.evolve_circuit()
            E[layer] = self.entanglement(self.tab_forward)
                
        return E


    def weight_evolution(self, graph_state: GraphState, layers: int, avg_gates_per_layer, max_attempts = 1000 ):
        
        weight = np.zeros(layers, dtype = int)
        
        mid_qubit = int(self.total_qubits//2) - 1
        
        pauli_str = ["I"] * self.total_qubits
        pauli_str[mid_qubit] = "Z"
        pauli_str = "".join(pauli_str)  # "IIZII"
        
        Zi = stim.PauliString(pauli_str)
                
        for layer in range(layers):
            
            self.evolve_circuit(graph_state, avg_gates_per_layer, max_attempts)
            evolved_pauli = Zi.after(self.circuit)
            weight[layer] = evolved_pauli.weight
            
        return weight
        

    def binaryMatrix(self, zStabilizers):
        
        N = len(zStabilizers)
        Na = len(zStabilizers[0])
        binaryMatrix = np.zeros((N,2*Na))
        r = 0 # Row number
        for row in zStabilizers:
            c = 0 # Column number
            for i in row:
                if i == 3: # Pauli Z
                    binaryMatrix[r,Na + c] = 1
                if i == 2: # Pauli Y
                    binaryMatrix[r,Na + c] = 1
                    binaryMatrix[r,c] = 1
                if i == 1: # Pauli X
                    binaryMatrix[r,c] = 1
                c += 1
            r += 1

        return binaryMatrix


    def gf2_rank(self, mat):
        """Compute rank of a binary matrix over GF(2) using Gaussian elimination."""
        mat = mat.copy()
        rank = 0
        n_rows, n_cols = mat.shape
        
        for col in range(n_cols):
            pivot = -1
            # Find pivot row
            for row in range(rank, n_rows):
                if mat[row, col]:
                    pivot = row
                    break
            if pivot == -1:
                continue
                
            # Swap current row with pivot row
            mat[[rank, pivot]] = mat[[pivot, rank]]
            
            # Eliminate this column in other rows
            for row in range(n_rows):
                if row != rank and mat[row, col]:
                    mat[row] = (mat[row] + mat[rank]) % 2
            rank += 1
        
        return rank




    def entanglement(self, tab_forward, sysA = None) -> float:
            
        stabilizers = tab_forward.to_stabilizers()

        if sysA:
            sysB = [x for x in range(self.total_qubits) if x not in sysA]
        else:
            sysA = [i for i in range(int(self.total_qubits/2))]
            sysB = [x for x in range(int(self.total_qubits/2),self.total_qubits,1)]

        gA = [stim.PauliString([s[q] for q in sysA]) for s in stabilizers]

        na = len(sysA); nb = len(sysB)

        binary_matrix = self.binaryMatrix(gA)

        rank = self.gf2_rank(binary_matrix)
        
        return rank - na
    
    def get_circuit(self) -> stim.Circuit:
        """Return the constructed circuit"""
        return self.circuit
