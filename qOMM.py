# MIT License
#
# Copyright (c) 2024 Joel Bierman, Guorui Zhu, Jianfeng Lu, Yingzhou Li
#
# This file is licensed under the MIT License. See the LICENSE file for more details.

from __future__ import annotations

from collections.abc import Callable, Sequence, Iterable
from typing import Any, cast, Optional
import logging
from time import time

import numpy as np

import copy

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2

from qiskit_algorithms.optimizers import Optimizer, Minimizer, OptimizerResult
from qiskit_algorithms.variational_algorithm import VariationalAlgorithm, VariationalResult
from qiskit_algorithms import Eigensolver, EigensolverResult
from qiskit_algorithms.exceptions import AlgorithmError
from qiskit.circuit import Instruction

from inner_prod_qc import *

# private function as we expect this to be updated in the next release
from qiskit_algorithms.utils.set_batching import _set_default_batchsize

logger = logging.getLogger(__name__)

class qOMM(VariationalAlgorithm, Eigensolver):

    def __init__(
        self,
        operator: BaseOperator,
        kstates: int,
        var_forms: list[QuantumCircuit]|QuantumCircuit,
        init_states: Instruction,#list[QuantumCircuit]|QuantumCircuit,
        init_params_val = None,
        initial_point: Optional[np.ndarray] = None,
        optimizer: Optimizer | Minimizer | Sequence[Optimizer | Minimizer] = None,
        simulator: AerSimulator = None,
        nuclear_repulsion_energy = 0,
        callback: Callable[[int, np.ndarray, float, dict[str, Any], int], None] | None = None,
        shots: int = 8192
    ) -> None:
        
        super().__init__()

        self.simulator = simulator

        self.kstates = kstates
        self.operator = operator
        self.constructed_qc = False
        self._initial_point = initial_point
        self.preferred_init_points = None
        

        if isinstance(var_forms, list):
            self.var_forms = []
            for it,vf in enumerate(var_forms):
                namestr = 'P'+str(it)
                self.var_forms.append(vf.assign_parameters(
                    ParameterVector(namestr, length=vf.num_parameters)))
        elif var_forms is None:
            raise ValueError('var_forms cannot be None')
        else:
            namestr = 'P'+str(0)
            self.var_forms = []
            self.var_forms.append(var_forms.assign_parameters(
                ParameterVector(namestr, length=var_forms.num_parameters)))

        self.num_qubits  = self.var_forms[0].num_qubits

        if isinstance(init_states, list):
            self.init_states = copy.deepcopy(init_states)
        else:
            self.init_states = [init_states]

        self._op_to_list()

        if isinstance(init_params_val, list):
            self.params_val = copy.deepcopy(init_params_val)
        elif init_params_val is None:
            self.params_val = []
            for vf in self.var_forms:
                if hasattr(vf, 'preferred_init_points'):
                    if vf.preferred_init_points is None:
                        low  = [-2*np.pi]*vf.num_parameters
                        high = [2*np.pi]*vf.num_parameters
                        self.params_val.append(
                            np.random.uniform(low, high))
                        #print("hello")
                    else:
                        self.params_val.append(vf.preferred_init_points)
                else:
                    self.params_val.append(None) 
        else:
            self.params_val = [copy.deepcopy(init_params_val)]


        self.nuclear_repulsion_energy = nuclear_repulsion_energy
        self.shots = shots


        self.optimizer = optimizer 

        self.nparams = []
        self._totparams = 0
        for vf in self.var_forms:
            self.nparams.append(vf.num_parameters)
            self._totparams += vf.num_parameters

        self._eval_count = 0
        self._shot_count = 0

        #TODO: finish result
        self._ret = None

        self.callback = callback
        self.constructed_qc = False

        if not self.constructed_qc:
            self._construct_quantum_circuit()


    @property
    def initial_point(self) -> np.ndarray | None:
        """Returns initial point."""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: np.ndarray | None) -> None:
        """Sets initial point."""
        self._initial_point = initial_point


    def _check_operator_ansatz(self):
        """Check that the number of qubits of operator and ansatz match."""
        operator = self.operator
        if operator is not None and self.var_forms is not None:
            if operator.num_qubits != self.num_qubits:
                # try to set the number of qubits on the ansatz, if possible
                try:
                    self.var_forms[0].num_qubits = operator.num_qubits
                except AttributeError as exc:
                    raise AlgorithmError(
                        "The number of qubits of the ansatz does not match the "
                        "operator, and the ansatz does not allow setting the "
                        "number of qubits using `num_qubits`."
                    ) from exc

    
    @classmethod
    def supports_aux_operators(cls) -> bool:
        return False

    
    def _op_to_list(self):
        
        pair_list = self.operator.to_list()
        self.op_list=[]
        for p in pair_list:
            self.op_list.append([Pauli(p[0]),p[1]])
    

    def _construct_quantum_circuit(self):
        self.inner_qc_list = []
        for it in range(self.kstates):
            self.inner_qc_list.append([])
            for jt in range(self.kstates):
                self.inner_qc_list[it].append([])
                if jt < it:
                    continue
                if len(self.var_forms) == 1:
                    vf1 = self.var_forms[0]
                    vf2 = self.var_forms[0]
                else:
                    vf1 = self.var_forms[it]
                    vf2 = self.var_forms[jt]
                if len(self.init_states) == 1:
                    self.inner_qc_list[it][jt].append(
                            transpile(
                            inner_prod_qc(vf1,vf2,
                                        init_state=self.init_states[0],
                                        re_or_im=True),
                                        backend=self.simulator))
                    self.inner_qc_list[it][jt].append(
                            transpile(
                            inner_prod_qc(vf1,vf2,
                                        init_state=self.init_states[0],
                                        re_or_im=False),
                                        backend=self.simulator))
                else:
                    self.inner_qc_list[it][jt].append(
                            transpile(
                            inner_prod_qc(vf1,vf2,
                                        init_state1=self.init_states[it],
                                        init_state2=self.init_states[jt],
                                        re_or_im=True),
                                        backend=self.simulator))
                    self.inner_qc_list[it][jt].append(
                            transpile(
                            inner_prod_qc(vf1,vf2,
                                        init_state1=self.init_states[it],
                                        init_state2=self.init_states[jt],
                                        re_or_im=False),
                                        backend=self.simulator))

        self.inner_op_qc_list = []

        for it in range(self.kstates):

            self.inner_op_qc_list.append([])

            for jt in range(self.kstates):

                self.inner_op_qc_list[it].append([])

                if jt < it:
                    continue

                for ot,_op in enumerate(self.op_list):

                    self.inner_op_qc_list[it][jt].append([])
                    

                    # if all vector in the pauli series are all I, 
                    #then this equals to the innerproduct.
                    op = _op[0]
                    if not any(op.x) and not any(op.z):
                        self.inner_op_qc_list[it][jt][ot].append(
                            self.inner_qc_list[it][jt][0])
                        self.inner_op_qc_list[it][jt][ot].append(
                            self.inner_qc_list[it][jt][1])
                        self.inner_op_qc_list[it][jt][ot].append(_op[1])
                        continue

                    if len(self.var_forms) == 1:
                        vf1 = self.var_forms[0]
                        vf2 = self.var_forms[0]
                    else:
                        vf1 = self.var_forms[it]
                        vf2 = self.var_forms[jt]

                    if len(self.init_states) == 1:
                        self.inner_op_qc_list[it][jt][ot].append(
                                transpile(
                                inner_prod_op_qc(vf1,vf2,op=op,
                                            init_state=self.init_states[0],
                                            re_or_im=True),
                                            backend=self.simulator))
                        self.inner_op_qc_list[it][jt][ot].append(
                                transpile(
                                inner_prod_op_qc(vf1,vf2,op=op,
                                            init_state=self.init_states[0],
                                            re_or_im=False),
                                            backend=self.simulator))
                    else:
                        self.inner_op_qc_list[it][jt][ot].append(
                                transpile(
                                inner_prod_op_qc(vf1,vf2,op=op,
                                        init_state1=self.init_states[it],
                                        init_state2=self.init_states[jt],
                                        re_or_im=True),
                                        backend=self.simulator))
                        self.inner_op_qc_list[it][jt][ot].append(
                                transpile(
                                inner_prod_op_qc(vf1,vf2,op=op,
                                        init_state1=self.init_states[it],
                                        init_state2=self.init_states[jt],
                                        re_or_im=False),
                                        backend=self.simulator))
                        
                    self.inner_op_qc_list[it][jt][ot].append(_op[1])

        self.constructed_qc = True
    

    def _inner_prod(self, i_idx, j_idx, paras1, paras2=None,
                    re_im_part=None):
        if re_im_part is None:
            re_im_vec = [0,1]
        else:
            re_im_vec = []
            if 're' in re_im_part:
                re_im_vec.append(0)
            if 'im' in re_im_part:
                re_im_vec.append(1)

        res = []

        for re_im_idx in re_im_vec:

            if i_idx == j_idx or paras2 is None:
                qc = self.inner_qc_list[i_idx][j_idx][
                    re_im_idx].assign_parameters(paras1)
            else:
                qc = self.inner_qc_list[i_idx][j_idx][
                    re_im_idx].assign_parameters(np.concatenate(
                    (paras1,paras2)))
                
            job = self.simulator.run(qc, shots=self.shots)
            self._shot_count += self.shots
            counts = job.result().get_counts()

            if '0' not in counts:
                counts['0'] = 0

            res.append(2*counts['0']/self.shots-1)

        if re_im_part is None:
            return complex(res[0],res[1])
        if 're' in re_im_part and 'im' in re_im_part:
            return complex(res[0],res[1])
        
        return res[0]

    def _inner_prod_op(self, i_idx, j_idx, paras1, paras2=None,
                        re_im_part=None):
        
        if re_im_part is None:
            re_im_vec = [0,1]
        else:
            re_im_vec = []
            if 're' in re_im_part:
                re_im_vec.append(0)
            if 'im' in re_im_part:
                re_im_vec.append(1)

        res = []

        for re_im_idx in re_im_vec:

            sum = 0

            for op_idx in range(len(self.op_list)):

                if i_idx == j_idx or paras2 is None:
                    qc = self.inner_op_qc_list[i_idx][j_idx][op_idx][
                        re_im_idx].assign_parameters(paras1)
                else:
                    qc = self.inner_op_qc_list[i_idx][j_idx][op_idx][
                        re_im_idx].assign_parameters(np.concatenate(
                        (paras1,paras2)))
                    
                job = self.simulator.run(qc, shots=self.shots)
                self._shot_count += self.shots
                counts = job.result().get_counts()

                if '0' not in counts:
                    counts['0'] = 0
                sum += (2*counts['0']/self.shots-1)*self.inner_op_qc_list[
                    i_idx][j_idx][op_idx][2]
            res.append(sum)

        if re_im_part is None:
            return complex(res[0],res[1])
        
        if 're' in re_im_part and 'im' in re_im_part:
            return complex(res[0],res[1])
        
        return res[0]
    


    def _cost_fn(self, parameter):

        params = []

        if len(self.var_forms) > 1:
            npsum = 0
            for it in range(self.kstates):
                tmp = self.nparams[it]
                params.append(parameter[npsum:npsum+tmp])
                npsum += tmp
        else:
            params.append(parameter)

        start_time = time()

        mat_inner_op = np.zeros((self.kstates,self.kstates),
                                dtype=complex)
        mat_inner    = np.zeros((self.kstates,self.kstates),
                                dtype=complex)

        for it in range(self.kstates):

            for jt in range(self.kstates):

                if jt < it:
                    mat_inner[it,jt] = np.conj(mat_inner[jt,it])
                    mat_inner_op[it,jt] = np.conj(mat_inner_op[jt,it])
                    continue

                if len(self.var_forms) > 1:
                    if jt == it:
                        mat_inner[it,jt] = self._inner_prod(it,jt,
                                                paras1=params[it],
                                                re_im_part='re')
                        mat_inner_op[it,jt] = self._inner_prod_op(it,jt,
                                                paras1=params[it],
                                                re_im_part='re')
                    else:
                        mat_inner[it,jt] = self._inner_prod(it,jt,
                                                paras1=params[it],
                                                paras2=params[jt])
                        mat_inner_op[it,jt] = self._inner_prod_op(it,jt,
                                                paras1=params[it],
                                                paras2=params[jt])
                else:
                    if jt == it:
                        mat_inner[it,jt] = self._inner_prod(it,jt,
                                                paras1=params[0],
                                                re_im_part='re')
                        mat_inner_op[it,jt] = self._inner_prod_op(it,jt,
                                                paras1=params[0],
                                                re_im_part='re')
                    else:
                        mat_inner[it,jt] = self._inner_prod(it,jt,
                                                paras1=params[0])
                        mat_inner_op[it,jt] = self._inner_prod_op(it,jt,
                                                paras1=params[0])
        
        loss = np.real(np.trace(
            (2*np.eye(self.kstates) - mat_inner) @ mat_inner_op))

        self._eval_count += 1

        end_time = time()

        print("Loss: %.6f, Time: %.2f"%(loss, end_time - start_time))

        return loss
    

    def compute_eigenvalues(
        self,
        cost_fn: Callable[[np.ndarray], float | np.ndarray] = None
    ) -> qOMMResult:
        #self.print_information()

        if cost_fn is None:
            cost_fn = self._cost_fn

        self._check_operator_ansatz()

        parameter = np.array([])

        if len(self.var_forms) > 1:
            for it in range(self.kstates):
                parameter = np.concatenate((parameter,self.params_val[it]))
        else:
            parameter = self.params_val[0]

        '''
        parameter, opt_val, num_optimizer_evals = self.optimizer.optimize(
                    self._totparams, cost_fn,
                    initial_point = parameter)

        '''

        start_time = time()

        optimizer_result = self.optimizer.minimize(
            fun = cost_fn,
            x0 = parameter
        )

        eval_time = time() - start_time

        res = self._build_qOMM_result()

        self._update_vqd_result(res, optimizer_result, eval_time)

        self._ret = res

        return res
    

    @staticmethod
    def _build_qOMM_result() -> qOMMResult:
        
        result = qOMMResult()
        
        result.optimal_parameters = []
        result.cost_function_evals = np.array([], dtype=int)
        result.eigenvalues = []  # type: ignore[assignment]
        result.cost_function_evals = np.array([])
        result.optimizer_times = np.array([])
        result.optimizer_results = []
        result.optimal_circuits = []
        result.optimal_points = np.array([])

        return result

    #@staticmethod
    def _update_vqd_result(
        self,
        result: qOMMResult,
        opt_result: OptimizerResult,
        eval_time: float,
        #ansatz: QuantumCircuit|list[QuantumCircuit]
    ) -> qOMMResult:
        
        parameter = opt_result.x
        
        if len(self.var_forms) > 1:
            npsum = 0
            for it in range(self.kstates):

                tmp = self.nparams[it]
                result.optimal_parameters.append(copy.deepcopy(parameter[npsum:npsum+tmp]))
                npsum += tmp
        else:
            result.optimal_parameters.append(copy.deepcopy(parameter))

        result.optimal_values = opt_result.fun
        #result.cost_function_evals = np.concatenate([result.cost_function_evals, [opt_result.nfev]])
        result.optimizer_times = eval_time

        result.optimizer_results.append(copy.deepcopy(opt_result))

        if len(self.var_forms) > 1:
            for it in range(self.kstates):
                tmp = copy.deepcopy(self.var_forms[it])
                result.optimal_circuits.append(copy.deepcopy(tmp.assign_parameters(result.optimal_parameters[it])))
        else:
            tmp = copy.deepcopy(self.var_forms)
            result.optimal_circuits.append(copy.deepcopy(tmp.assign_parameters(result.optimal_parameters[0])))
            
        if len(self.var_forms) > 1:
            for it in range(self.kstates):
                
                estimator = EstimatorV2(mode = self.simulator)
                tmp = estimator.run([(self.var_forms[it], self.operator, result.optimal_parameters[it])]).result()
                result.eigenvalues.append(tmp[0].data.evs)
                result.eigenvalues.sort()
        else:
            estimator = EstimatorV2(mode = self.simulator)
            tmp = estimator.run([(self.var_forms, self.operator, result.optimal_parameters[0])]).result()
            result.eigenvalues.append(tmp[0].data.evs)

        return result
    
    


# TODO: 

class qOMMResult(VariationalResult, EigensolverResult):
    """qOMM Result."""

    def __init__(self) -> None:
        super().__init__()

        self._cost_function_evals: np.ndarray | None = None
        self._optimizer_times: np.ndarray | None = None
        self._optimal_values: np.ndarray | None = None
        self._optimal_points: np.ndarray | None = None
        self._optimal_parameters: list[dict] | None = None
        self._optimizer_results: list[OptimizerResult] | None = None
        self._optimal_circuits: list[QuantumCircuit] | None = None

    @property
    def cost_function_evals(self) -> np.ndarray | None:
        """Returns number of cost optimizer evaluations"""
        return self._cost_function_evals

    @cost_function_evals.setter
    def cost_function_evals(self, value: np.ndarray) -> None:
        """Sets number of cost function evaluations"""
        self._cost_function_evals = value

    @property
    def optimizer_times(self) -> np.ndarray | None:
        """Returns time taken for optimization for each step"""
        return self._optimizer_times

    @optimizer_times.setter
    def optimizer_times(self, value: np.ndarray) -> None:
        """Sets time taken for optimization for each step"""
        self._optimizer_times = value

    @property
    def optimal_values(self) -> np.ndarray | None:
        """Returns optimal value for each step"""
        return self._optimal_values

    @optimal_values.setter
    def optimal_values(self, value: np.ndarray) -> None:
        """Sets optimal values"""
        self._optimal_values = value

    @property
    def optimal_points(self) -> np.ndarray | None:
        """Returns optimal point for each step"""
        return self._optimal_points

    @optimal_points.setter
    def optimal_points(self, value: np.ndarray) -> None:
        """Sets optimal points"""
        self._optimal_points = value

    @property
    def optimal_parameters(self) -> list[dict] | None:
        """Returns the optimal parameters for each step"""
        return self._optimal_parameters

    @optimal_parameters.setter
    def optimal_parameters(self, value: list[dict]) -> None:
        """Sets optimal parameters"""
        self._optimal_parameters = value

    @property
    def optimizer_results(self) -> list[OptimizerResult] | None:
        """Returns the optimizer results for each step"""
        return self._optimizer_results

    @optimizer_results.setter
    def optimizer_results(self, value: list[OptimizerResult]) -> None:
        """Sets optimizer results"""
        self._optimizer_results = value

    @property
    def optimal_circuits(self) -> list[QuantumCircuit] | None:
        """The optimal circuits. Along with the optimal parameters,
        these can be used to retrieve the different eigenstates."""
        return self._optimal_circuits

    @optimal_circuits.setter
    def optimal_circuits(self, optimal_circuits: list[QuantumCircuit]) -> None:
        self._optimal_circuits = optimal_circuits
    