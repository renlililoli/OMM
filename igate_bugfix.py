from qiskit import QuantumCircuit 
from qiskit.circuit.library.standard_gates.i import IGate 

def _define(self):
    self.definition = QuantumCircuit(1, name=self.name)
      
setattr(IGate, "_define", _define)
