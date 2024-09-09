import sys
sys.path.append(".")

from qOMM import qOMM, qOMMResult
from qiskit_aer import AerSimulator
import numpy as np
from qiskit_aer import Aer
from qiskit_algorithms import NumPyEigensolver
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.mappers import QubitMapper, qubit_mapper
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.algorithms import ExcitedStatesEigensolver
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit.quantum_info import Statevector
from qiskit_algorithms.utils import algorithm_globals
from qiskit_aer import QasmSimulator
from qiskit.circuit.library import StatePreparation
import matplotlib.pyplot as plt
from time import perf_counter, process_time
from datetime import datetime
import os
import platform
import getpass

algorithm_globals.massive = True

PID = os.getpid()
print(f'PID: {PID}')
date_and_time = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
print(f'Program started at {date_and_time}')

molecule = MoleculeInfo(["H", "H"], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.735)], charge=0, multiplicity=1)

driver = PySCFDriver.from_molecule(molecule, basis="sto3g")

problem = driver.run()

transformer = FreezeCoreTransformer()

# and you also apply transformers explicitly
q_molecule = transformer.transform(problem)

es_problem = ElectronicStructureProblem(q_molecule.hamiltonian)

second_q_op = es_problem.hamiltonian.second_q_op()
qubit_converter = JordanWignerMapper()
qubit_op = qubit_converter.map(second_q_op)

#print(es_problem,qubit_op)
#print(qubit_op)

num_qubits = qubit_op.num_qubits

basis_states = np.identity(2**num_qubits, int)

initial_states = np.array([basis_states[2], # |0000001100000011>
                           basis_states[3] # |0000010100000011>
                           ])

initial_states = [StatePreparation(s) for s in initial_states]

var_forms = []
for it in range(2):
    var_forms.append(UCCSD(qubit_mapper=qubit_converter,
                            num_particles=(1,1),
                            num_spatial_orbitals=es_problem.num_spatial_orbitals,
                            reps=5))
    
simulator = AerSimulator()
optimizer = COBYLA()

qomm_obj = qOMM(qubit_op,
                2,
                var_forms,
                initial_states,
                optimizer = optimizer,
                simulator = simulator,
                shots = 8192
                )

qomm_obj.compute_eigenvalues()