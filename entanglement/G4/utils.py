import numpy as np
import stim
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import networkx as nx
from scipy.linalg import null_space
from typing import List, Tuple, Union
from itertools import cycle
from joblib import Parallel, delayed


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = 'k'
plt.rcParams['axes.labelcolor'] = 'k'
plt.rcParams['text.usetex'] = True


pal = ["003049","d62828","f77f00","fcbf49","eae2b7"]
pal = ['#' + i for i in pal]

marks = [
    '.',  # point
    ',',  # pixel
    'o',  # circle
    'v',  # triangle_down
    '^',  # triangle_up
    '<',  # triangle_left
    '>',  # triangle_right
    '1',  # tri_down
    '2',  # tri_up
    '3',  # tri_left
    '4',  # tri_right
    '8',  # octagon
    's',  # square
    'p',  # pentagon
    'P',  # plus (filled)
    '*',  # star
    'h',  # hexagon1
    'H',  # hexagon2
    '+',  # plus
    'x',  # x
    'X',  # x (filled)
    'D',  # diamond
    'd',  # thin_diamond
    '|',  # vline
    '_',  # hline
]


class GraphState:
    def __init__(self,
                 num_nodes: int, 
                 adjacency_matrix: Union[np.ndarray, None] = None, 
                 edges: Union[List[Tuple[int, int]], None] = None,
                 ):
        """
        Initialize with either adjacency matrix or edge list.
        
        Args:
            adjacency_matrix: Symmetric binary matrix where 1 indicates an edge
            edges: List of tuples representing edges (either is sufficient)
        """
        self.num_qubits = num_nodes
        self.edges = edges
        if self.edges == None:
            self.get_edges()
        if adjacency_matrix is not None:
            assert len(adjacency_matrix.shape) == 2, "Adjacency matrix must be 2D"
            assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1], "Adjacency matrix must be square"
            assert np.all(adjacency_matrix == adjacency_matrix.T), "Adjacency matrix must be symmetric"
            self.adjacency_matrix = adjacency_matrix
            # self.num_qubits = adjacency_matrix.shape[0]
        elif edges is not None:
            self.adjacency_matrix = np.zeros((self.num_qubits, self.num_qubits), dtype=int)
            for i, j in edges:
                self.adjacency_matrix[i, j] = 1
                self.adjacency_matrix[j, i] = 1
        else:
            raise ValueError("Must provide either adjacency_matrix or edges")
            
    def get_edges(self) -> List[Tuple[int, int]]:
        """Returns list of edges (as tuples) in the graph"""
        self.edges = []
        for i in range(self.num_qubits):
            for j in range(i+1, self.num_qubits):  # Avoid duplicates and self-loops
                if self.adjacency_matrix[i, j]:
                    self.edges.append((i, j))
        # return edges
    
    def apply_to_circuit(self, 
                        circuit: stim.Circuit,
                        target_qubits: List[int],
                        random_order: bool = False) -> stim.Circuit:
        """
        Applies graph state preparation to specified qubits in an existing circuit.
        
        Args:
            circuit: Existing stim circuit to modify
            target_qubits: Qubit indices to apply graph state to (must match graph size)
            random_order: If True, applies gates in random order
        Returns:
            Modified stim circuit
        """
        assert len(target_qubits) == self.num_qubits, \
            f"Target qubits list length ({len(target_qubits)}) must match graph size ({self.num_qubits})"
        
        # Apply Hadamards
        hadamard_qubits = [target_qubits[i] for i in range(self.num_qubits)]
        if random_order:
            np.random.shuffle(hadamard_qubits)
            
        for q in hadamard_qubits:
            circuit.append("H", [q])
        
        # Apply CZ gates
        # edges = self.get_edges()
        if random_order:
            np.random.shuffle(self.edges)
            
        for i, j in self.edges:
            circuit.append("CZ", [target_qubits[i], target_qubits[j]])
            
        return circuit

    def draw_graph(self, with_labels: bool = True, node_size: int = 500, font_size: int = 10, title: str = None):
            
            """
            Draws the graph using networkx and matplotlib.
            
            Args:
                with_labels: Whether to show node labels
                node_size: Size of the nodes
                font_size: Font size for labels
            """
            
            G = nx.Graph()
            G.add_edges_from(self.edges)
            blue = '#14213d'
            plt.figure(figsize=(4, 4))
            pos = nx.circular_layout(G)  # positions for all nodes
            
            nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=blue)
            nx.draw_networkx_edges(G, pos)
            
            if with_labels:
                nx.draw_networkx_labels(G, pos, font_size=font_size, font_color="white")
            if title:
                plt.title(rf'{title}',loc = 'center')
            plt.axis("off")
            plt.show()


    def __repr__(self):
        return f"GraphState(n={self.num_qubits}, edges={self.get_edges()})"





E9 = [(0,1),(0,2),(0,3),(0,4),(0,5)]
G9 = GraphState(num_nodes=6, edges=E9)

E10 = [(0,5), (1,5), (2,5), (3,4), (4,5)]
G10 = GraphState(num_nodes=6, edges=E10)

E11 = [(0,5), (1,5), (2,4), (3,4), (5,4)]
G11 = GraphState(num_nodes=6, edges=E11)

E12 = [(0,1), (1,2), (2,3), (3,4), (1,5)]
G12 = GraphState(num_nodes=6, edges=E12)

E13 = [(0,1), (1,2), (2,5), (2, 3), (3,4)]
G13 = GraphState(num_nodes=6, edges=E13)

E14 = [(0,1), (1,2), (2,3), (3,4), (4,5)]
G14 = GraphState(num_nodes=6, edges=E14)

E15 = [(0,5), (5,2), (5,4), (2,3), (3,4), (1,3)]
G15 = GraphState(num_nodes=6, edges=E15)

E16 = [(0,1), (1,2), (2,3), (1,3), (3,4), (2,5)]
G16 = GraphState(num_nodes=6, edges=E16)

E17 = [(0,5), (0,1), (1,2), (2,3), (3,4), (0,4)]
G17 = GraphState(num_nodes=6, edges=E17)

E18 = [(0,1), (1,2), (2,3), (3,4), (4,5), (0,5)]
G18 = GraphState(num_nodes=6, edges=E18)

E19 = [(0,1), (0,2), (1,2), (2,3), (1,4), (3,5), (3,4), (4,5), (0,5)]
G19 = GraphState(num_nodes=6, edges=E19)

Q6_graph = [G9, G10, G11, G12, G13, G14, G15, G16, G17, G18, G19]

# ==============================================

E20 = [(0,1), (0,2), (0,3), (0,4), (0,5), (0,6)]
G20 = GraphState(num_nodes = 7, edges = E20)

E21 = [(0,6), (1,6), (2,6), (3,6), (4,5), (5,6)]
G21 = GraphState(num_nodes = 7, edges = E21)

E22 = [(0,6), (1,6), (2,6), (3,5), (4,5), (5,6)]
G22 = GraphState(num_nodes=7, edges=E22)

E23 = [(0,6), (1,6), (2,6), (3,4), (4,5), (5,6)]
G23 = GraphState(num_nodes=7, edges=E23)

E24 = [(0,6), (1,6), (5,6), (4,5), (2,4), (3,4)]
G24 = GraphState(num_nodes=7, edges=E24)

E25 = [(0,1), (0,6), (2,6), (3,6), (4,5), (5,6)]
G25 = GraphState(num_nodes=7, edges=E25)

E26 = [(0,6), (1,6), (5,6), (2,5), (4,5), (3,4)]
G26 = GraphState(num_nodes=7, edges=E26)

E27 = [(0,1), (1,6), (1,2), (2,3), (3,4), (4,5)]
G27 = GraphState(num_nodes=7, edges=E27)

E28 = [(0,1), (1,2), (2,3), (2,4), (4,5), (5,6)]
G28 = GraphState(num_nodes=7, edges = E28)

E29 = [(0,1), (1,2), (2,5), (5,6), (2,3), (3,4)]
G29 = GraphState(num_nodes=7, edges=E29)

E30 = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6)]
G30 = GraphState(num_nodes=7, edges=E30)

E31 = [(0,2), (1,2), (2,3), (2,5), (3,4), (4,5)]
G31 = GraphState(num_nodes=7, edges = E31)

E32 = [(0,6), (1,6), (5,6), (2,5), (4,6), (4,5), (3,4)]
G32= GraphState(num_nodes=7, edges=E32)

E33 = [(0,2), (1,2), (2,6), (2,3), (3,4), (4,5), (5,6)]
G33 = GraphState(num_nodes=7, edges = E33)

E34 = [(1,2), (0,3), (2,3), (2,5), (3,4), (4,5), (5,6)]
G34 = GraphState(num_nodes=7, edges = E34)

E35 = [(0,5), (1,2), (2,6), (2,3), (3,4), (4,5), (5,6)]
G35 = GraphState(num_nodes=7, edges = E35)

E36 = [(0,1), (1,2), (2,3), (2,4), (3,6), (4,5)]
G36 = GraphState(num_nodes=7, edges = E36)

E37 = [(0,6), (1,2), (2,6), (2,3), (3,4), (4,5), (5,6)]
G37 = GraphState(num_nodes=7, edges = E37)

E38 = [(0,5), (0,1), (1,2), (2,3), (3,4), (4,5), (5,6)]
G38 = GraphState(num_nodes=7, edges = E38)

E39 = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (0,4)]
G39 = GraphState(num_nodes=7, edges = E39)

E40 = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,0)]
G40 = GraphState(num_nodes=7, edges=E40)

E41 = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (0,4), (0,5)]
G41 = GraphState(num_nodes=7, edges = E41)

E42 = [(1,2), (2,3), (3,4), (4,5), (5,6), (0,6), (0,2)]
G42 = GraphState(num_nodes=7, edges = E42)

E43 = [(0,1), (1,2), (2,3), (3,4), (4,5), (0,3), (0,6), (4,6), (2,5)]
G43 = GraphState(num_nodes=7, edges = E43)

E44 = [(1,2), (2,3), (3,4), (4,5), (5,6), (6,0), (0,3), (1,6), (2,4)]
G44 = GraphState(num_nodes=7, edges = E44)

E45 = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (1,6), (1,4), (2,6), (3,5)]
G45 = GraphState(num_nodes=7, edges=E45)


Q7_graph = [G20,G21,G22,G23,G24,G25,G26,G27,G28,G29,G30,G31,G32,G33,G34,G35,G36,G37,G38,G39,G40,G41,G42,G43,G44,G45]


E3 = [(0,1), (0,2), (0,3)]
G3 = GraphState(num_nodes = 4, edges = E3)

E4 = [(0,1), (1,2), (2,3)]
G4 = GraphState(num_nodes=4, edges = E4)

Q4_graph = [G3, G4]

# ==============================================



class RandomAMECircuit:

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
        # gB = [stim.PauliString([s[q] for q in sysB]) for s in stabilizers]

        na = len(sysA); nb = len(sysB)

        # gA = [g_in_A[i] for i in range(self.total_qubits) if (g_in_B[i] == stim.PauliString(nb)) or (g_in_B[i] == - stim.PauliString(nb))]
        
        # if not gA:
            # return na 

        binary_matrix = self.binaryMatrix(gA)

        rank = self.gf2_rank(binary_matrix)
        
        return rank - na
    
    def get_circuit(self) -> stim.Circuit:
        """Return the constructed circuit"""
        return self.circuit
