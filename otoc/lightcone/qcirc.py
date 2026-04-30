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
        
    def generate_random_configs(self, max_attempts:int=1000):
        
        all_groups = [list(range(i, i + self.graph_qubits)) for i in range(self.total_qubits - self.graph_qubits + 1)]

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


        self.circuit = self.circuit + current_circuit
    


    def otoc(self, layers: int, defect_pos: int = None):
        
        commutator = np.zeros((layers, self.total_qubits), dtype = bool)
        
        if not defect_pos:
            defect_pos = int(self.total_qubits//2) - 1
        
        pauli_str = ["I"] * self.total_qubits
        pauli_str[defect_pos] = "X"
        pauli_str = "".join(pauli_str)  
        
        Xi = stim.PauliString(pauli_str)
                
        for layer in range(layers):
            self.evolve_circuit()
            evolved_pauli = Xi.after(self.circuit)
            Xt, Zt = evolved_pauli.to_numpy()
            commutator[layer] = np.logical_xor(Xt, Zt)
            
        return commutator
        
    
    def get_circuit(self) -> stim.Circuit:
        """Return the constructed circuit"""
        return self.circuit


def otoc_evolution(graph_state, 
                   total_qubits, 
                   avg_gates_per_layer, 
                   layers, 
                   defect_pos = None):

    random_ame_circuit = RandomGraphCircuit(total_qubits=total_qubits, 
                                            graph_state=graph_state, 
                                            avg_gates_per_layer=avg_gates_per_layer)
    
    e = random_ame_circuit.otoc(layers=layers, defect_pos = defect_pos)

    return np.array(e, dtype = int)

def average_otoc_evolution(graph_state, 
                           total_qubits, 
                           avg_gates_per_layer, 
                           layers, num_shots, 
                           defect_pos=None):

    num_workers = -1
    entropy = Parallel(n_jobs=num_workers)(delayed(otoc_evolution)(graph_state, 
                                                                   total_qubits, 
                                                                   avg_gates_per_layer, 
                                                                   layers, 
                                                                   defect_pos) for i in (range(num_shots)))
    
    mean_val = np.zeros((layers, total_qubits), dtype = float)
    
    for i in range(num_shots):
        mean_val += (1/num_shots)*np.array(entropy[i], dtype = float)

    return mean_val


def butterfly_vel(graph, 
                  total_qubits, 
                  avg_gates_per_layer, 
                  layers, 
                  num_shots
                  ):

    ct = average_otoc_evolution(graph, 
                                total_qubits, 
                                avg_gates_per_layer, 
                                layers, 
                                num_shots)

    center = int(total_qubits//2) - 1 
    threshold = 0.05

    lightcone = np.zeros(layers , dtype = float)

    for i in range(layers):
        for k in range(center, total_qubits, 1):
            if ct[i, k] <= threshold:
                lightcone[i] = k
                break

    xvals = np.arange(layers) + center

    params = np.polyfit(xvals, lightcone - center, deg = 1)

    return params[0]


def count_cross_edges(edges, partA, partB):
    """
    Count how many edges connect nodes from partA to nodes from partB.
    
    Parameters:
        edges (list of tuple): list of (u, v) edges
        partA (list or set): nodes in partition A
        partB (list or set): nodes in partition B
    
    Returns:
        int: number of crossing edges
    """
    setA = set(partA)
    setB = set(partB)
    count = 0
    
    for u, v in edges:
        if (u in setA and v in setB) or (u in setB and v in setA):
            count += 1
    
    return count
