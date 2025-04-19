from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
import numpy as np

# Define the ASCII for "Hello" -> H = 72, e = 101, l = 108, o = 111
hello_string = "Hello"
ascii_values = [ord(c) for c in hello_string]

# Create a quantum circuit with 8 qubits for each character
circuits = []

for val in ascii_values:
    qc = QuantumCircuit(8, 8)
    bin_val = format(val, '08b')
    
    # Initialize the qubits to represent each bit of the ASCII character
    for i, bit in enumerate(reversed(bin_val)):
        if bit == '1':
            qc.x(i)  # Flip the qubit to 1

    qc.measure(range(8), range(8))
    circuits.append(qc)

# Use Aer's qasm simulator
simulator = Aer.get_backend('qasm_simulator')

# Execute each circuit and print the result
print("Quantum Hello:")
for i, qc in enumerate(circuits):
    # Transpile the circuit for the simulator
    compiled_circuit = transpile(qc, simulator)
    
    # Run the circuit on the simulator
    job = simulator.run(compiled_circuit, shots=1)
    result = job.result()
    
    counts = result.get_counts()
    measured_bin = list(counts.keys())[0]
    char = chr(int(measured_bin, 2))
    print(char, end="")
