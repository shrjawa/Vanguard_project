import numpy as np
import os
import docplex.mp.model
from docplex.mp.model_reader import ModelReader
from numba import jit
from itertools import combinations
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit.library import efficient_su2, n_local, real_amplitudes, evolved_operator_ansatz, excitation_preserving
from qiskit_aer import AerSimulator, noise
from scipy.optimize import minimize
from qiskit_ibm_runtime import Batch, Session

from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
from qiskit_ibm_runtime import SamplerV2 as RuntimeSamplerV2
from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimatorV2
from logging import exception
def get_service(token: str, instance: str):
  from qiskit_ibm_runtime import QiskitRuntimeService
  token="token"
  service = QiskitRuntimeService.save_account(
      token=token, # Your token is confidential.
      # Do not share your token in public code.
      instance=instance, # Optionally specify the instance to use.
      #plans_preference="['Open']", # Optionally set the types of plans to prioritize.  This is ignored if the instance is specified.
      region="Washington DC (us-east)", # Optionally set the region to prioritize. This is ignored if the instance is specified.
      #name="<account-name>", # Optionally name this set of account credentials.
      set_as_default=True, # Optionally set these as your default credentials.
      overwrite=True
    )
  service = QiskitRuntimeService()
  return service
def get_Q_h_c_A_b_penalty(model: docplex.mp.model.Model):
  num_vars = model.number_of_binary_variables
  num_ctr = model.number_of_constraints
  print("number of variables:", num_vars, num_ctr)
  print("number of linear constraints:", num_ctr)

  # Parsing objective under assumption:
  # - the objective is in the form quadratic + linear + c
  # Output: Q (as a dense matrix) and c such that      objective = x^T Q x + c

  Q = np.zeros((num_vars, num_vars))
  h = np.zeros(num_vars)
  c = model.objective_expr.get_constant()

  # For dense Q
  for i, dvari in enumerate(model.iter_variables()):
    for j, dvarj in enumerate(model.iter_variables()):
      Q[i,j] = model.objective_expr.get_quadratic_coefficient(dvari, dvarj) / 2
      if i == j:
        Q[i,i] = 0
        # diagonal elements of Q are absorbed into linear term h (since for a binary vatiable, (y_i)^2 = y_i where y_i = 0 or 1)
        h[i] = model.objective_expr.get_quadratic_coefficient(dvari, dvari) + model.objective_expr.linear_part.get_coef(dvari)

  # Parsing constraints under the assumption:
  # - they are all linear inequalities *** it could be generalized to equalities!
  # Retrieving A and b such that constraints write as    A x - b ≥ 0.

  A = np.zeros((num_ctr, num_vars))
  b = np.zeros(num_ctr)

  for i, ctr in enumerate(model.iter_constraints()):
    sense = 1 if ctr.sense == docplex.mp.constants.ComparisonType.GE else -1
    for j, dvarj in enumerate(model.iter_variables()):
      A[i,j] = sense * ctr.lhs.get_coef(dvarj)
    b[i] = sense * ctr.rhs.get_constant()

  # Rescale constraints so that the minimum coefficient of a variable in each constraint is 1 (in abs)
  min_A_by_row = np.zeros(num_ctr)

  for i,row in enumerate(A):
    min_A_by_row[i] = np.min(np.abs(row[np.nonzero(row)]))

  A = A / min_A_by_row.reshape(num_ctr, 1)
  b = b / min_A_by_row

  # Translate constraints into obj terms under the following assumptions:
  # - minimization problem
  # - all vars are bin (or integer)
  # - each coef in constraints is ≥ 1 (in abs)
  # Remark: the resulting unconstr_obj_fn is not polynomial (contains maximum), and it's designed to be used in sampling VQE

  max_obj = np.sum(Q, where=Q>0)
  min_obj = np.sum(Q, where=Q<0)
  penalty = (max_obj-min_obj) * 1.1

  return Q, h, c, A, b, penalty

def get_pauli_x_y_z_correlation_encoding_node_lists(Q):
  num_nodes = Q.shape[0] ### Same as number of binary variables
  list_size = num_nodes // 3 ### partition binary variables into 3 list of almost equal length
  node_x_list = [i for i in range(list_size)]
  node_y_list = [i for i in range(list_size, 2 * list_size)]
  node_z_list = [i for i in range(2 * list_size, num_nodes)]
  return node_x_list, node_y_list, node_z_list

def hamiltoninas_for_pauli_x_y_z_correlation_encoding(num_qubits, node_x_list, node_y_list, node_z_list, k=2):
  def build_pauli_correlation_encoding(pauli, node_list, num_qubits, k):
    pauli_correlation_encoding = []
    for idx, c in enumerate(combinations(range(num_qubits), k)):
      if idx >= len(node_list):
        break
      paulis = ["I"] * num_qubits
      paulis[c[0]], paulis[c[1]] = pauli, pauli
      pauli_correlation_encoding.append(("".join(paulis)[::-1], 1))

    hamiltonian = []
    for pauli, weight in pauli_correlation_encoding:
      hamiltonian.append(SparsePauliOp.from_list([(pauli, weight)]))
    return hamiltonian

  pce_x = build_pauli_correlation_encoding("X", node_x_list, num_qubits, k)
  pce_y = build_pauli_correlation_encoding("Y", node_y_list, num_qubits, k)
  pce_z = build_pauli_correlation_encoding("Z", node_z_list, num_qubits, k)
  return pce_x, pce_y, pce_z

def get_isa_ansatz_isa_hamiltoninas_for_pauli_x_y_z_correlation_encoding(ansatz, pce_x, pce_y, pce_z, backend, optimization_label):
  pass_manager = generate_preset_pass_manager(optimization_level=optimization_label, backend=backend)
  isa_ansatz = pass_manager.run(ansatz)
  isa_pce_x = [op.apply_layout(isa_ansatz.layout) for op in pce_x]
  isa_pce_y = [op.apply_layout(isa_ansatz.layout) for op in pce_y]
  isa_pce_z = [op.apply_layout(isa_ansatz.layout) for op in pce_z]
  return isa_ansatz, isa_pce_x, isa_pce_y, isa_pce_z

@jit
def obj_fn_embedding_constraints(y):
  return y @ Q @ y + h @ y + c + penalty * np.sum(np.maximum(b - A @ y, 0)**2)

# Simple
def get_Simple_Noise_Model(prob_1,prob_2,single_gates=['u1', 'u2', 'u3'], double_gates=['cz']):
  #### prob_1 = 1-qubit gate error probabilities
  #### prob_2 = 2-qubit gate error probabilities

  # Depolarizing quantum errors
  error_1 = noise.depolarizing_error(prob_1, 1)
  error_2 = noise.depolarizing_error(prob_2, 2)

  # Add errors to noise model
  noise_model = noise.NoiseModel()
  noise_model.add_all_qubit_quantum_error(error_1, single_gates)
  noise_model.add_all_qubit_quantum_error(error_2, double_gates)

  return noise_model

# run Quadratic Contrained Optimization Problem (QCBO) with linear constraints

def run_QCBO(init_params, alpha, ansatz, pce_x, pce_y, pce_z, node_x_list, node_y_list, node_z_list, Q, h, c, A, b, penalty, simulation_type, shots = 1024, method="COBYLA", tol=1e-4, optimization_label=3, noise_model=None, token=None, instance=None):
  if simulation_type == 'exact' and noise_model == None:
    backend = AerSimulator(method='statevector')
  if simulation_type == 'shot_based' and shots != None and noise_model == None:
    backend = AerSimulator()
  if simulation_type == "noisy_shot_based" and shots != None and noise_model != None:
    backend = AerSimulator(noise_model=noise_model)
  if simulation_type=='Noisy_backend_real': #simulation based on noise from real device
    if not token or not instance:
      raise ValueError("Token and instance must be provided for 'Noisy_backend_real' simulation.")
        
    service = get_service(token, instance)
    backend = service.backend("ibm_brisbane")

  isa_ansatz, isa_pce_x, isa_pce_y, isa_pce_z = get_isa_ansatz_isa_hamiltoninas_for_pauli_x_y_z_correlation_encoding(ansatz, pce_x, pce_y, pce_z, backend, optimization_label)
  

  cost_list = []
  params_list = []

  bitstrings= np.zeros(Q.shape[0])
  def get_cost(params, alpha, isa_ansatz, isa_pce_x, isa_pce_y, isa_pce_z, Q, h, c, A, b, penalty, estimator):
    
    lenx = len(isa_pce_x)
    leny = len(isa_pce_y)
    lenz = len(isa_pce_z)

    pubs_x = [(isa_ansatz, isa_pce_x[i], params) for i in range(lenx)]
    pubs_y = [(isa_ansatz, isa_pce_y[i], params) for i in range(leny)]
    pubs_z = [(isa_ansatz, isa_pce_z[i], params) for i in range(lenz)]

    all_pubs = pubs_x + pubs_y + pubs_z


    job = estimator.run(all_pubs)
    results = job.result()
    
    expectation_x = np.array([results[i].data.evs for i in range(lenx)])
    expectation_y = np.array([results[i + lenx].data.evs for i in range(leny)])
    expectation_z = np.array([results[i + lenx + leny].data.evs for i in range(lenz)])

    y = np.zeros(Q.shape[0])
    y[node_x_list] = (1 - np.tanh(alpha* expectation_x))/2
    y[node_y_list] = (1 - np.tanh(alpha* expectation_y))/2
    y[node_z_list] = (1 - np.tanh(alpha* expectation_z))/2

    bitstrings[node_x_list] = (1 - np.sign( expectation_x))/2
    bitstrings[node_y_list] = (1 - np.sign( expectation_y))/2
    bitstrings[node_z_list] = (1 - np.sign( expectation_z))/2

  
    cost = y @ Q @ y + h @ y + c + penalty * np.sum(np.maximum(b - A @ y, 0)**2)
    print(cost)
    cost_list.append(cost)
    params_list.append(params)
    return cost

  if simulation_type == 'shot_based' or simulation_type == 'noisy_shot_based':
    with Session(backend=backend) as session:
      estimator = RuntimeEstimatorV2(mode=session)
      estimator.options.default_shots = shots
      result = minimize(get_cost,
                        init_params, args=(alpha, isa_ansatz, isa_pce_x, isa_pce_y, isa_pce_z, Q, h, c, A, b, penalty, estimator), method=method, tol=tol,options={'maxiter':10000})

  if simulation_type == 'exact':
    estimator=AerEstimatorV2()
    result = minimize(get_cost,
                        init_params, args=(alpha, isa_ansatz, isa_pce_x, isa_pce_y, isa_pce_z, Q, h, c, A, b, penalty, estimator), method=method, tol=tol, options={'maxiter':120})
    
  if simulation_type == 'Noisy_backend_real': 

    from qiskit_aer.noise import NoiseModel
    noise_model = NoiseModel.from_backend(backend)
    
    estimator=AerEstimatorV2()
    estimator_options = {
        "backend_options": {
            "noise_model": noise_model
        },
        "run_options": {
            "shots": shots
        }
    }
    noisy_estimator = AerEstimatorV2(options=estimator_options)

    try:
      result = minimize(get_cost,init_params,args=(alpha, isa_ansatz, isa_pce_x, isa_pce_y, isa_pce_z, Q, h, c, A, b, penalty, noisy_estimator), 
        method=method,tol=tol,options={'maxiter':1200})
    except KeyboardInterrupt:
      print("\n--- Optimization stopped manually ---")
      print("Returning the best results found so far.")
    '''
    import spsa
    from scipy.optimize import OptimizeResult
    cost_function_for_spsa = lambda p: get_cost(p, alpha, isa_ansatz, isa_pce_x, isa_pce_y, isa_pce_z, Q, h, c, A, b, penalty, estimator)
    optimized_params = spsa.minimize(cost_function_for_spsa, init_params, iterations=1200)
    final_cost = cost_function_for_spsa(optimized_params)
    result = OptimizeResult(x=optimized_params, fun=final_cost, success=True, nit=1200)''' 


  return cost_list, params_list, bitstrings




def polish_on_real_device(
    initial_params, # The best parameters from the local simulation become the starting point
    ansatz,
    pce_x, pce_y, pce_z,
    node_x_list, node_y_list, node_z_list,
    Q, h, c, A, b, penalty,
    alpha,
    token, # Your IBM Quantum token
    instance, # Your hub/group/project string e.g., "ibm-q/open/main"
    method="COBYQA", # The classical optimizer to use
    maxiter=20, # Keep the number of hardware iterations low
    shots=8192
):
    """
    Takes a near-optimal set of parameters and performs a final, short
    optimization on a real quantum device with error mitigation.
    """
    print("\n--- Starting Phase 2: Polishing parameters on real hardware ---")
    service = QiskitRuntimeService(token=token, instance=instance, channel="ibm_quantum")
    backend = service.backend("ibm_brisbane")
    print(f"Using real backend: {backend.name} for final optimization.")

    # These lists will store the history of the hardware optimization
    cost_list = []
    params_list = []
    bitstrings = np.zeros(Q.shape[0]) # Use a mutable object to store the final bitstring

    # Use a Session for better performance by managing jobs sent to the cloud
    with Session(backend=backend) as session:
        # Create the RUNTIME estimator, which works with Sessions
        runtime_estimator = RuntimeEstimatorV2(session=session)

        # Set the runtime error mitigation and other options
        runtime_estimator.options.resilience_level = 1
        runtime_estimator.options.execution.shots = shots
        runtime_estimator.options.dynamical_decoupling.enable = True

        # This nested function is what scipy.minimize will call at each step
        def get_cost_on_hardware(params):
            nonlocal bitstrings # Modify the bitstrings variable from the outer scope
            
            pubs_x = [(ansatz, obs, params) for obs in pce_x]
            pubs_y = [(ansatz, obs, params) for obs in pce_y]
            pubs_z = [(ansatz, obs, params) for obs in pce_z]
            all_pubs = pubs_x + pubs_y + pubs_z

            # Submit the job to the real hardware
            job = runtime_estimator.run(all_pubs)
            print(f"Job {job.job_id()} submitted for cost evaluation. Waiting for result...")
            results = job.result()
            print("Result received.")

            # --- Calculate cost from hardware results ---
            lenx, leny, lenz = len(pce_x), len(pce_y), len(pce_z)
            expectation_x = np.array([results[i].data.evs for i in range(lenx)])
            expectation_y = np.array([results[i + lenx].data.evs for i in range(leny)])
            expectation_z = np.array([results[i + lenx + leny].data.evs for i in range(lenz)])

            y = np.zeros(Q.shape[0])
            y[node_x_list] = (1 - np.tanh(alpha * expectation_x)) / 2
            y[node_y_list] = (1 - np.tanh(alpha * expectation_y)) / 2
            y[node_z_list] = (1 - np.tanh(alpha * expectation_z)) / 2
            
            bitstrings = (1 - np.sign(np.concatenate([expectation_x, expectation_y, expectation_z]))) / 2

            cost = y @ Q @ y + h @ y + c + penalty * np.sum(np.maximum(b - A @ y, 0)**2)
            
            print(f"Hardware Cost: {cost}")
            cost_list.append(cost)
            params_list.append(params)
            return cost

       
        result = minimize(
            get_cost_on_hardware,
            initial_params, 
            method=method,
            options={'maxiter': maxiter}, # Crucially, very few iterations
            tol=1e-5 # A tighter tolerance for the final polish
        )

    return cost_list, params_list, bitstrings
