from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.primitives import Sampler # Import Sampler from qiskit.primitives
from qiskit_aer import Aer  # Import Aer from qiskit_aer
from qiskit.visualization import plot_circuit_layout

# Create quantum and classical registers
qr = QuantumRegister(1, 'q')
cr = ClassicalRegister(1, 'c')
qc = QuantumCircuit(qr, cr)

# Explanation:
# - Our qubit starts in |0>.
# - Our target state is |+>, which we can create by applying a Hadamard gate (H) to |0>.
# - We simulate an iterative feedback loop where we:
#     a. Rotate the state into the X basis (using H).
#     b. Measure the qubit.
#     c. If the measurement shows the qubit is in the undesired state, apply an X gate to correct it.

n_iter = 50  # Number of feedback iterations

for i in range(n_iter):
    # Step 1: Rotate to X basis for measurement in that basis.
    qc.h(qr[0])
    
    # Step 2: Measure the qubit. In the X basis, outcome 0 indicates |+> and 1 indicates |–>.
    qc.measure(qr[0], cr[0])
    
    # Step 3: Conditional correction:
    # If the measurement result is 1 (i.e., the state was |–>), then apply an X gate to flip the state.
    qc.x(qr[0]).c_if(cr, 1)
    
    # Optionally, you could add a reset instruction on the classical bit here,
    # but for demonstration we reuse the same classical register.
    
# Visualize the circuit
print(qc.draw(output='text'))

# Simulate the circuit on the Qiskit Aer simulator
# Instead of 'execute', use simulator.run directly:
backend = Aer.get_backend('qasm_simulator')
job = backend.run(qc, shots=1024) # Changed to backend.run
result = job.result()
counts = result.get_counts(qc)

print("\nMeasurement outcomes:")
print(counts)
