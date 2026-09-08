import numpy as np, pennylane as qml, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from qkn.simulator import Sim

def test_against_pennylane():
    n = 5; sim = Sim(n); rng = np.random.default_rng(0)
    X = rng.uniform(0, 2*np.pi, (3, n)); th = rng.uniform(0, 2*np.pi, n)
    dev = qml.device('default.qubit', wires=n)
    @qml.qnode(dev)
    def circ(x):
        for q in range(n): qml.Hadamard(wires=q)
        for q in range(n): qml.RZ(x[q], wires=q)
        for q in range(n): qml.RY(th[q], wires=q)
        for q in range(n-1): qml.CNOT(wires=[q, q+1])
        qml.CNOT(wires=[n-1, 0])
        for q in range(n): qml.IsingZZ(x[q]*x[(q+1)%n], wires=[q, (q+1)%n])
        for q in range(n): qml.CZ(wires=[q, (q+1)%n])
        return qml.state()
    psi = sim.zero(3); psi = sim.h_layer(psi); psi = sim.rz_data(psi, X); psi = sim.ry_layer(psi, th)
    psi = sim.cnot_ring(psi)
    pairs = [(q, (q+1)%n) for q in range(n)]
    psi = sim.zz_data(psi, np.stack([X[:, j]*X[:, k] for j, k in pairs], 1), pairs)
    psi = sim.cz_ring(psi)
    for i in range(3):
        ref = np.asarray(circ(X[i]))
        assert np.allclose(psi[i], ref, atol=1e-10), np.abs(psi[i]-ref).max()

if __name__ == '__main__':
    test_against_pennylane(); print('simulator matches PennyLane')
