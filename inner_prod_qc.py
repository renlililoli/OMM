from qiskit import QuantumRegister, AncillaRegister
from qiskit import ClassicalRegister, QuantumCircuit
import igate_bugfix
from qiskit.quantum_info import Pauli

def inner_prod_qc(
                var_form1,
                var_form2,
                init_state=None,
                init_state1=None,
                init_state2=None,
                re_or_im=True
    ):

    return inner_prod_op_qc(var_form1=var_form1,
                            var_form2=var_form2,
                            op=None,
                            init_state=init_state,
                            init_state1=init_state1, 
                            init_state2=init_state2,
                            re_or_im=re_or_im
            )

def inner_prod_op_qc(
                    var_form1,
                    var_form2,
                    op,
                    init_state=None,
                    init_state1=None,
                    init_state2=None,
                    re_or_im=True
    ):

    if re_or_im:
        return inner_prod_op_real_qc(
                                var_form1=var_form1,
                                var_form2=var_form2,
                                op=op,
                                init_state=init_state,
                                init_state1=init_state1, 
                                init_state2=init_state2,
                                imag_flag=False)
    else:
        return inner_prod_op_real_qc(
                                var_form1=var_form1,
                                var_form2=var_form2,
                                op=op,
                                init_state=init_state,
                                init_state1=init_state1, 
                                init_state2=init_state2,
                                imag_flag=True
                )

#==========================================================================
# Calculate either the real or the imaginary part (imag_flag) of the inner
#  product with respect to the operator.
def inner_prod_op_real_qc(
                        var_form1,
                        var_form2,
                        op: Pauli,
                        init_state=None,
                        init_state1=None,
                        init_state2=None,
                        imag_flag=False
    ):

    num_vf_qubits = var_form1.num_qubits

    qr  = QuantumRegister(num_vf_qubits, 'q')
    anc = AncillaRegister(1, 'a')
    cr  = ClassicalRegister(1, 'c')

    # Circuit construction
    circuit = QuantumCircuit(anc, qr, cr)
    circuit.h(anc)
    if init_state is not None:
        circuit.compose(init_state, qr, inplace=True)
    elif init_state is None and init_state1 is not None:
        circuit.compose(init_state1.control(), inplace=True)
    circuit.compose(var_form1.control(), inplace=True)
    if op is not None:
        op = op.to_instruction()
        circuit.compose(op.control(), inplace=True)
    if imag_flag:
        circuit.s(anc)
    circuit.x(anc)
    if init_state is None and init_state2 is not None:
        circuit.compose(init_state2.control(), inplace=True)
    circuit.compose(var_form2.control(), inplace=True)
    circuit.h(anc)
    circuit.measure(anc, cr)

    return circuit