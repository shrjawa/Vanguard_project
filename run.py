from src.pce import (
    get_Q_h_c_A_b_penalty,
    get_pauli_x_y_z_correlation_encoding_node_lists,
    hamiltoninas_for_pauli_x_y_z_correlation_encoding,
    run_QCBO
)
from src.step_1 import model_to_obj

my_token = "YOUR_CONFIDENTIAL_TOKEN"
my_instance = "YOUR_INSTANCE_CRN"
LP_FILE_PATH = os.path.join('data', '31bonds', 'docplex-bin-avgonly-nocplexvars.lp')
model = ModelReader.read(LP_FILE_PATH)
Q,h,c,A,b,penalty = get_Q_h_c_A_b_penalty(model)

node_x_list, node_y_list, node_z_list = get_pauli_x_y_z_correlation_encoding_node_lists(Q)
print("List_x:", node_x_list,"List_y:", node_y_list,"List_z:", node_z_list)
num_qubits = 6
pce_x, pce_y, pce_z = hamiltoninas_for_pauli_x_y_z_correlation_encoding(num_qubits, node_x_list, node_y_list, node_z_list)

shots = 20000
method_list = ['COBYLA','L-BFGS-B','Nelder-Mead','Powell','SLSQP','TNC','CG']

efficient_su2_ansatz = efficient_su2(num_qubits, ["ry", "rz"], reps=2)
n_local_ansatz = n_local(num_qubits,rotation_blocks='ry',entanglement_blocks='cz' ,reps=2)
ansatz_list = [efficient_su2_ansatz,n_local_ansatz]

method = method_list[1]
ansatz = ansatz_list[1]

init_params = np.zeros(ansatz.num_parameters)

np.random.seed(42)
init_params = np.random.rand(ansatz.num_parameters)


alpha = 15
cost, params, bitstring = run_QCBO(init_params,alpha,ansatz,pce_x,pce_y,pce_z,node_x_list,node_y_list,node_z_list,  Q,h,c,A,
    b,penalty,'Noisy_backend_real',method=method,tol=1e-1,token=my_token,instance=my_instance)

model_file = "data/1/31bonds/docplex-bin-avgonly-nocplexvars.lp"
model: docplex.mp.model.Model = ModelReader.read(model_file)
obj_fn = model_to_obj(model)
print("benchmark cost",obj_fn(ideal_bitstring))
