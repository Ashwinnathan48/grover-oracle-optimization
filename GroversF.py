import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# Property used: H · X · H = Z
# 
# STANDARD ORACLE (Universal - works for any N)
# 
def build_oracle(qc, n_qubits, target):
    """Standard general oracle - phase flips target state"""
    for i in range(n_qubits):
        if not (target >> i & 1): # Checks if target is 0, if it is 0 apply X gate
            qc.x(i)
    qc.h(n_qubits - 1) # Apply Hamamard to last Qubit
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1) # MCX gate needs all to qubits to be |1>
    qc.h(n_qubits - 1) # Apply another Hadamard to last qubit to switch back to original basis
    for i in range(n_qubits): # Unflip or uncommputation
        if not (target >> i & 1):
            qc.x(i)

# 
# OPTIMIZED ORACLES (N=4 and N=8 only)
# 
def build_oracle_optimized_n4(qc, target):
    """Optimized oracle for N=4 (2 qubits) - uses CZ gate""" 
    if not (target >> 0 & 1):   # Manually check each qubit
        qc.x(0)
    if not (target >> 1 & 1):
        qc.x(1)
    qc.cz(0, 1)
    if not (target >> 0 & 1):
        qc.x(0)
    if not (target >> 1 & 1):
        qc.x(1)

def build_oracle_optimized_n8(qc, target):
    """Optimized oracle for N=8 (3 qubits) - uses CCZ gate"""
    if not (target >> 0 & 1):
        qc.x(0)
    if not (target >> 1 & 1):
        qc.x(1)
    if not (target >> 2 & 1):
        qc.x(2)
    qc.ccz(0, 1, 2)
    if not (target >> 0 & 1):
        qc.x(0)
    if not (target >> 1 & 1):
        qc.x(1)
    if not (target >> 2 & 1):
        qc.x(2)

# 
# DIFFUSER (Same for all circuits)
# 
def build_diffuser(qc, n_qubits):
    """Reflect amplitudes over mean"""
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))

# 
# GROVER RUNNER
# 
def run_grover(n_qubits, target, optimized=False, shots=1000, noise_model=None):
    """Build and run Grover's circuit and return success probability and gate count"""
    N = 2 ** n_qubits
    iterations = max(1, int(np.floor((np.pi / 4) * np.sqrt(N))))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    for _ in range(iterations):
        if optimized:
            if n_qubits == 2:
                build_oracle_optimized_n4(qc, target)
            elif n_qubits == 3:
                build_oracle_optimized_n8(qc, target)
        else:
            build_oracle(qc, n_qubits, target)
        qc.barrier()
        build_diffuser(qc, n_qubits)
        qc.barrier()

    # Get gate count before measuring
    gate_count = sum(qc.count_ops().values())

    qc.measure(range(n_qubits), range(n_qubits))

    # Run on simulator with or without noise
    simulator = AerSimulator()
    job = simulator.run(qc, shots=shots, noise_model=noise_model)
    result = job.result()
    counts = result.get_counts()

    # Get success probability
    target_str = format(target, f'0{n_qubits}b')
    success_count = counts.get(target_str, 0)
    success_probability = success_count / shots

    return success_probability, iterations, gate_count, qc

# 
# NOISE MODEL BUILDER
# 
def build_noise_model(error_rate):
    """Build a depolarizing noise model at a given error rate"""
    noise_model = NoiseModel()

    # Single qubit gate error
    single_qubit_error = depolarizing_error(error_rate, 1)
    # Two qubit gate error
    two_qubit_error = depolarizing_error(error_rate * 2, 2)
    # Three qubit gate error
    three_qubit_error = depolarizing_error(error_rate * 3, 3)

    # Apply to single qubit gates
    noise_model.add_all_qubit_quantum_error(single_qubit_error, ['h', 'x', 'u1', 'u2', 'u3'])
    # Apply to two qubit gates
    noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx', 'cz'])
    # Apply to three qubit gates
    noise_model.add_all_qubit_quantum_error(three_qubit_error, ['ccx', 'ccz'])

    return noise_model

# 
# MANUAL CIRCUIT DISPLAY
# 
def display_oracle_circuits_manual():
    """Constructed oracle circuits for N=4 and N=8"""

    # N=4
    qc_std_n4 = QuantumCircuit(2)
    qc_std_n4.h(1)
    qc_std_n4.cx(0, 1)
    qc_std_n4.h(1)

    qc_opt_n4 = QuantumCircuit(2)
    qc_opt_n4.cz(0, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    fig.suptitle('Oracle Circuits for N=4 (2 qubits, target |11⟩)', fontsize=13)
    qc_std_n4.draw(output='mpl', ax=ax1)
    ax1.set_title(f'Standard Oracle (Gates: {sum(qc_std_n4.count_ops().values())})')
    qc_opt_n4.draw(output='mpl', ax=ax2)
    ax2.set_title(f'Optimized Oracle (Gates: {sum(qc_opt_n4.count_ops().values())})')
    plt.tight_layout()
    plt.show()

    # N=8
    qc_std_n8 = QuantumCircuit(3)
    qc_std_n8.h(2)
    qc_std_n8.ccx(0, 1, 2)
    qc_std_n8.h(2)

    qc_opt_n8 = QuantumCircuit(3)
    qc_opt_n8.ccz(0, 1, 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    fig.suptitle('Oracle Circuits for N=8 (3 qubits, target |111⟩)', fontsize=13)
    qc_std_n8.draw(output='mpl', ax=ax1)
    ax1.set_title(f'Standard Oracle (Gates: {sum(qc_std_n8.count_ops().values())})')
    qc_opt_n8.draw(output='mpl', ax=ax2)
    ax2.set_title(f'Optimized Oracle (Gates: {sum(qc_opt_n8.count_ops().values())})')
    plt.tight_layout()
    plt.show()

# 
# MAIN
# 
N_values = [4, 8]
qubit_counts = [2, 3]
NOISE_RATE = 0.01        # 1% error rate per gate
NOISY_SHOTS = 1000      # shots for noise simulation
CLEAN_SHOTS = 1000       # shots for noiseless baseline

grover_iterations = []
std_gate_counts = []
opt_gate_counts = []
std_success_probs = []
opt_success_probs = []
std_noisy_probs = []
opt_noisy_probs = []

# Build noise model
noise_model = build_noise_model(NOISE_RATE)

# Noiseless baseline
print("=" * 74)
print("NOISELESS RESULTS: Grover's vs Classical, Standard vs Optimized Oracle")
print("=" * 74)
print(f"{'N':<6} {'Qubits':<8} {'Iterations':<12} {'Classical N/2':<16} {'Std Gates':<12} {'Opt Gates':<12} {'Std Prob':<12} {'Opt Prob':<12}")
print("-" * 74)

for N, n_qubits in zip(N_values, qubit_counts):
    target = N - 1

    std_prob, iterations, std_gates, _ = run_grover(n_qubits, target, optimized=False, shots=CLEAN_SHOTS)
    opt_prob, _, opt_gates, _ = run_grover(n_qubits, target, optimized=True, shots=CLEAN_SHOTS)

    grover_iterations.append(iterations)
    std_gate_counts.append(std_gates)
    opt_gate_counts.append(opt_gates)
    std_success_probs.append(std_prob)
    opt_success_probs.append(opt_prob)

    print(f"{N:<6} {n_qubits:<8} {iterations:<12} {N/2:<16} {std_gates:<12} {opt_gates:<12} {std_prob:<12.3f} {opt_prob:<12.3f}")

# Noisy simulation 
print()
print("=" * 74)
print(f"NOISY RESULTS: 1% gate error rate, {NOISY_SHOTS} shots")
print("=" * 74)
print(f"{'N':<6} {'Std Prob (noisy)':<20} {'Opt Prob (noisy)':<20} {'Improvement':<12}")
print("-" * 74)

for N, n_qubits in zip(N_values, qubit_counts):
    target = N - 1

    std_prob_noisy, _, _, _ = run_grover(n_qubits, target, optimized=False, shots=NOISY_SHOTS, noise_model=noise_model)
    opt_prob_noisy, _, _, _ = run_grover(n_qubits, target, optimized=True, shots=NOISY_SHOTS, noise_model=noise_model)

    improvement = opt_prob_noisy - std_prob_noisy
    std_noisy_probs.append(std_prob_noisy)
    opt_noisy_probs.append(opt_prob_noisy)

    print(f"{N:<6} {std_prob_noisy:<20.3f} {opt_prob_noisy:<20.3f} {improvement:+.3f}")

# 
# GRAPH 1 - Grover's vs Classical
# 
x = np.arange(len(N_values))
width = 0.35
theoretical_classical = [N / 2 for N in N_values]

plt.figure(figsize=(7, 5))
plt.bar(x - width/2, grover_iterations, width, label="Grover's Iterations (√N)")
plt.bar(x + width/2, theoretical_classical, width, label='Classical Linear Search (N/2)')
plt.xlabel('N (Search Space Size)')
plt.ylabel('Average Checks / Iterations')
plt.title("Grover's Algorithm vs Classical Linear Search")
plt.xticks(x, [f'N={N}' for N in N_values])
plt.legend()
plt.grid(True, axis='y')
plt.show()

# 
# GRAPH 2 - Gate Count
# 
plt.figure(figsize=(7, 5))
plt.bar(x - width/2, std_gate_counts, width, label='Standard Oracle Gates')
plt.bar(x + width/2, opt_gate_counts, width, label='Optimized Oracle Gates')
plt.xlabel('N (Search Space Size)')
plt.ylabel('Total Gate Count')
plt.title("Standard vs Optimized Oracle: Gate Count")
plt.xticks(x, [f'N={N}' for N in N_values])
plt.legend()
plt.grid(True, axis='y')
plt.show()

# 
# GRAPH 3 - Noiseless Success Probability
# 
plt.figure(figsize=(7, 5))
plt.bar(x - width/2, std_success_probs, width, label='Standard Oracle')
plt.bar(x + width/2, opt_success_probs, width, label='Optimized Oracle')
plt.xlabel('N (Search Space Size)')
plt.ylabel('Success Probability')
plt.title("Standard vs Optimized Oracle: Success Probability (Noiseless)")
plt.xticks(x, [f'N={N}' for N in N_values])
plt.legend()
plt.grid(True, axis='y')
plt.ylim(0, 1.1)
plt.show()

# 
# GRAPH 4 - Noisy Success Probability
# 
plt.figure(figsize=(7, 5))
plt.bar(x - width/2, std_noisy_probs, width, label='Standard Oracle')
plt.bar(x + width/2, opt_noisy_probs, width, label='Optimized Oracle')
plt.xlabel('N (Search Space Size)')
plt.ylabel('Success Probability')
plt.title("Standard vs Optimized Oracle: Success Probability (1% Noise)")
plt.xticks(x, [f'N={N}' for N in N_values])
plt.legend()
plt.grid(True, axis='y')
plt.ylim(0, 1.1)
plt.show()

# Displays manual circuit diagrams
display_oracle_circuits_manual()
