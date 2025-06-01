import random
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector

import sympy as sp
from sympy import sqrt, I, Matrix, eye, zeros, symbols, kronecker_product
from sympy.polys.polytools import gcd_list
from IPython.display import display, Math


## Stuff Required for Matrix Multiplications

# 1. Computational basis states
zero  = Matrix([1, 0])
one   = Matrix([0, 1])

# 2. ± states
plus  = 1/sqrt(2) * (zero + one)
minus = 1/sqrt(2) * (zero - one)

# 3. Single-qubit gates
X = Matrix([[0, 1],
            [1, 0]])

Y = Matrix([[0, -I],
            [I,  0]])

Z = Matrix([[1,  0],
            [0, -1]])

H = 1/sqrt(2) * Matrix([[1,  1],
                        [1, -1]])

S = Matrix([[1, 0],
            [0, I]])            # the “phase” (S) gate

# 4. CNOT (control qubit = first, target = second)
CNOT = Matrix([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,0,1],
    [0,0,1,0],
])

# 5. Toffoli (CCNOT) on 3 qubits:
#    flips qubit3 iff qubit1 AND qubit2 are 1.
#    basis ordering is |abc⟩ with a,b,c ∈ {0,1}.
Toffoli = eye(8)               # start with 8×8 identity
# swap the |110⟩ ↔ |111⟩ amplitudes:
Toffoli[6,6] = 0
Toffoli[7,7] = 0
Toffoli[6,7] = 1
Toffoli[7,6] = 1

import sympy as sp
from sympy.polys.polytools import gcd_list
from sympy import kronecker_product
from IPython.display import display, Math

def check_normalized(vec):
    """
    Verify that a state vector is normalized (⟨ψ|ψ⟩ = 1).

    Parameters
    ----------
    vec : sp.Matrix
        Column vector representing a quantum state.

    Raises
    ------
    ValueError
        If the vector is not normalized.
    """
    # Compute ⟨ψ|ψ⟩ = ψ† ψ
    norm2 = (vec.conjugate().T * vec)[0]
    # Check it equals 1
    if sp.simplify(norm2 - 1) != 0:
        raise ValueError(f"State is not normalized: ⟨ψ|ψ⟩ = {sp.simplify(norm2)}")

def check_unitary(mat):
    """
    Verify that a matrix is unitary (M† M = I).

    Parameters
    ----------
    mat : sp.Matrix
        Square matrix to check.

    Raises
    ------
    ValueError
        If the matrix is not square or not unitary.
    """
    rows, cols = mat.shape
    # Must be square
    if rows != cols:
        raise ValueError(f"Matrix is not square: shape = {mat.shape}")
    # Check M†M == I
    if sp.simplify(mat.H * mat - sp.eye(rows)) != sp.zeros(rows):
        raise ValueError("Matrix is not unitary: M†M ≠ I")

def _nice_factor(f):
    """
    Internal helper: if f = sqrt(b)/b, rewrite as 1/sqrt(b), else leave it alone.

    Parameters
    ----------
    f : sympy.Expr
        Candidate scalar factor.

    Returns
    -------
    sympy.Expr
        Simplified factor or original.
    """
    num, den = sp.simplify(f).as_numer_denom()
    # pattern: sqrt(b)/b  →  1/sqrt(b)
    if num.is_Pow and num.exp == sp.Rational(1, 2) and den == num.base:
        return 1/sp.sqrt(num.base)
    return f

def apply_unitaries(initial_vec, unitaries):
    """
    Apply a sequence of unitary matrices to an initial state, displaying
    each intermediate LaTeX‐rendered step in a Jupyter notebook.

    Parameters
    ----------
    initial_vec : sp.Matrix
        Column vector for the initial quantum state.
    unitaries : list of sp.Matrix
        Sequence of unitary matrices to apply in order.

    Raises
    ------
    ValueError
        If the initial vector isn’t a column, isn’t normalized,
        or any unitary fails the shape or unitarity checks.
    """
    # 1) Vector must be a column
    if initial_vec.cols != 1:
        raise ValueError(f"Initial state must be a column vector, got shape {initial_vec.shape}")
    # 2) Must be normalized
    check_normalized(initial_vec)

    # 3) Each U must be square, unitary, and match dimension
    dim = initial_vec.rows
    for U in unitaries:
        check_unitary(U)
        if U.rows != dim:
            raise ValueError(f"Dimension mismatch: got U shape {U.shape} but state dim = {dim}")

    # Prepare LaTeX labels
    psi_lbl   = r"\lvert \psi \rangle"
    vec_ltx   = sp.latex(initial_vec)
    mats_ltx  = [sp.latex(U) for U in unitaries]

    # Display first line: |ψ> = |initial> U₁ U₂ …
    first_rhs = vec_ltx + "".join(r"\," + M for M in mats_ltx)
    display(Math(f"{psi_lbl} = {first_rhs}"))

    # Step through each unitary
    psi = initial_vec
    for i, U in enumerate(unitaries):
        # Apply and simplify
        psi = sp.simplify(U * psi)

        # Factor out any common scalar for prettier output
        entries    = [psi[j,0] for j in range(psi.rows)]
        raw_factor = gcd_list(entries) or 1
        factor     = _nice_factor(raw_factor)

        if factor != 1:
            core    = sp.simplify(psi / factor)
            psi_str = sp.latex(factor) + r"\," + sp.latex(core)
        else:
            psi_str = sp.latex(psi)

        # Remaining unitaries
        tail = "".join(r"\," + M for M in mats_ltx[i+1:])
        display(Math(r"=\, " + psi_str + tail))


def tensor(a, b):
    """
    Compute the tensor (Kronecker) product of two Sympy matrices or vectors.

    Parameters
    ----------
    a : sp.Matrix
        First matrix or column vector.
    b : sp.Matrix
        Second matrix or column vector.

    Returns
    -------
    sp.Matrix
        The Kronecker product a ⊗ b.
    """
    return kronecker_product(a, b)


def show_latex(mat):
    """
    Render a Sympy matrix or vector as LaTeX in a Jupyter notebook.

    Parameters
    ----------
    mat : MatrixBase
        Any Sympy matrix or column vector (mutable or immutable).
    """
    display(Math(sp.latex(mat)))

## End of Stuff Required for Matrix Multiplications


def random_qubit_state():
    """
    Returns (alpha, beta) such that |alpha|^2 + |beta|^2 = 1,
    uniformly at random (ignoring global phase).
    """
    # pick |alpha|^2 = p ~ Uniform(0,1)
    p = np.random.rand()
    # pick relative phase phi ~ Uniform(0,2π)
    phi = np.random.rand() * 2 * np.pi

    alpha = np.sqrt(p)
    beta  = np.sqrt(1 - p) * np.exp(1j * phi)
    return alpha, beta


def _fmt_cplx(z: complex, prec: int = 3) -> str:
    """
    Format a complex number for LaTeX:
     - if imag(z) != 0: return "(a + b i)" with chosen precision
     - else: return "a" (real only)
    """
    a, b = z.real, z.imag
    if abs(b) > 10**(-prec):
        return f"({a:.{prec}f} {b:+.{prec}f}i)"
    else:
        return f"{a:.{prec}f}"


def plot_single_qubit_bloch(qc: QuantumCircuit, prec: int = 3):
    """
    Plot the Bloch sphere for a single-qubit circuit (no 'qubit 0' label),
    and display the statevector as
      |ψ> = α|0> + β|1>
    with α,β formatted as (Re + Im i) or just Re if purely real.
    
    Parameters:
      qc   – 1-qubit QuantumCircuit (no measurements)
      prec – number of decimals for real/imag parts
    """
    if qc.num_qubits != 1:
        raise ValueError("Circuit must have exactly one qubit.")
    
    # get statevector
    state = Statevector.from_instruction(qc)
    
    # plot Bloch sphere
    fig = plot_bloch_multivector(state)
    ax = fig.axes[0]
    ax.set_title("")                 # remove "qubit 0"
    
    # format amplitudes
    α_str = _fmt_cplx(state.data[0], prec)
    β_str = _fmt_cplx(state.data[1], prec)
    
    # build LaTeX string
    latex_str = (
        r"$|\psi\rangle = "
        + α_str + r"\,|0\rangle + "
        + β_str + r"\,|1\rangle$"
    )
    
    # place it below the sphere
    fig.text(0.5, 0.175, latex_str, ha='center', fontsize=12)
    fig.subplots_adjust(bottom=0.2)
    
    plt.show()
    return fig


def simulate_circuit(qc: QuantumCircuit, shots: int) -> dict:
    """
    Simulate a Qiskit QuantumCircuit on the AER noiseless qasm_simulator.
    
    Parameters:
    qc (QuantumCircuit): The quantum circuit to simulate.
    shots (int): Number of measurement shots to run.
    
    Returns:
    dict: A counts dictionary mapping bitstrings to occurrence counts.
    """
    simulator = AerSimulator()
    # Transpile the circuit for the simulator backend
    compiled = transpile(qc, simulator)
    # Run and get results
    result = simulator.run(compiled, shots=shots).result()
    return result.get_counts(compiled)
    

def plot_binary_histogram(counts_dict):
    """
    Plot histogram of counts for binary strings.
    
    Parameters:
    counts_dict (dict): Keys are binary strings, values are positive integer counts.
    """
    # Extract binary strings and their counts
    keys = list(counts_dict.keys())
    values = [counts_dict[k] for k in keys]

    # Create bar chart
    fig, ax = plt.subplots()
    ax.bar(keys, values)
    ax.set_xlabel('Binary strings')
    ax.set_ylabel('Counts')
    ax.set_title('Histogram of Binary String Occurrences')

    # Improve layout and readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def deutsch_oracle(qc, fun_case):
    """
    Append a 2-qubit Deutsch oracle to the given circuit.

    The oracle implements one of the four possible Boolean functions 
    f:{0,1}→{0,1} as follows:

      fun_case = 1 → f(x) = 0          (constant zero)
      fun_case = 2 → f(x) = x          (balanced identity)
      fun_case = 3 → f(x) = ¬x         (balanced negation)
      fun_case = 4 → f(x) = 1          (constant one)

    This assumes qubit 0 is the “input” and qubit 1 is the “output.”

    Args:
        qc (QuantumCircuit): a circuit with at least 2 qubits.
        fun_case (int): which of the four Deutsch functions to apply (1–4).

    Returns:
        QuantumCircuit: the same circuit with the oracle gates appended.

    Raises:
        ValueError: if `fun_case` is not in [1, 2, 3, 4].
    """
    # 1) Validate that we have a correct case number
    if fun_case not in [1, 2, 3, 4]:
        raise ValueError("`fun_case` must be 1, 2, 3, or 4.")
    
    # 2) Barrier for visual separation in the circuit diagram
    qc.barrier()
    
    # 3) For the two balanced functions (cases 2 & 3), flip the output if
    #    the input qubit is |1〉 → implements f(x)=x or f(x)=¬x after the X
    if fun_case in [2, 3]:
        qc.cx(0, 1)
    
    # 4) For functions that output 1 (case 4) or negate the parity (case 3),
    #    apply an X on the target qubit
    if fun_case in [3, 4]:
        qc.x(1)
    
    # 5) Final barrier to end the oracle section cleanly
    qc.barrier()
    
    return qc


def deutsch_jozsa_oracle(qc: QuantumCircuit, is_balanced: bool) -> QuantumCircuit:
    """
    Apply a Deutsch–Jozsa oracle to `qc`.  
    - Input wires: 0 … n-2  
    - Output wire:  n-1

    If is_balanced=False, performs a constant function (always 0 or always 1).  
    If is_balanced=True, performs a balanced parity function on a random non-empty subset
    of the input bits (which has exactly half the inputs map to 0 and half to 1).

    Returns the modified QuantumCircuit.
    """
    n = qc.num_qubits
    inputs = list(range(n - 1))
    output = n - 1

    qc.barrier()
    if not is_balanced:
        # constant function: randomly choose f(x)=0 (do nothing) or f(x)=1 (flip output)
        k = random.randint(1, n)
        subset = random.sample(list(range(n)), k=k)
        qc.x(subset)
        qc.barrier()
        return qc

    # balanced function: pick a random non-empty subset of inputs
    k = random.randint(1, len(inputs))
    subset = random.sample(inputs, k=k)
    # implement f(x)=⊕_{i in subset} x_i
    for i in subset:
        qc.cx(i, output)
    qc.barrier()
    return qc