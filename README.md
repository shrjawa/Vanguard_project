# Vanguard_project
WISER Vanguard project
Quantum-Classical Hybrid Optimization for QCBO
<img width="5068" height="3916" alt="Shreeram-Jawadekar-af44" src="https://github.com/user-attachments/assets/6d1d537d-4450-462c-ae2b-675fbf24e1a8" />

A Python project that uses a Variational Quantum Eigensolver (VQE) with Pauli correlation encoding to solve Quadratic Constrained Binary Optimization (QCBO) problems.
Pauli Correlation Encoding scheme is particularly useful for problems with a large number of binary variables, as it allows the problem to be executed on current quantum computers, which have a limited number of available qubits.

Features
QCBO Problem Mapping: Reads a mathematical optimization problem from a .lp file and maps it to a quantum Hamiltonian.

Pauli Correlation Encoding (PCE): Implements a specific encoding scheme to represent binary variables using Pauli operators.

Hybrid VQE Algorithm: Utilizes a hybrid quantum-classical approach to find an approximate solution to the optimization problem.

Flexible Simulation: Supports running the algorithm on an exact simulator, a shot-based simulator, or a noisy simulator that models a real backend.

Multiple Ansatz Circuits: Includes support for different quantum circuits (EfficientSU2, NLocal) to serve as the ansatz for the VQE.

Classical Optimization: Employs classical optimizers like L-BFGS-B to find the optimal parameters for the ansatz circuit.


1.  **Clone the repository:**
    ```bash
    git clone [[https://github.com/your-username/your-repo-name.git](https://github.com/shrjawa/Vanguard_project.git)]([https://github.com/your-username/your-repo-name.git](https://github.com/shrjawa/Vanguard_project.git))
    cd your-repo-name
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    cd Vanguard_project
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
    if conda:
    ```
    
    conda create --name project_name
    conda activate project_name
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    For conda packages can be installed one by one, some may not be available on conda channel
    
    






Usage
Set up IBM Quantum Credentials:
The script can run on a noisy simulator based on a real IBM Quantum device. To do this, you need to provide your IBM Quantum API token and CRN (Cloud Resource Name).

Find your credentials: Log in to your IBM Quantum account. Go to your dashboard to find your API token and the CRN for your preferred instance 

Update run.py:
Open run.py and replace the placeholder values for my_token and my_instance with your actual credentials.
Warning: Do not commit your personal tokens to a public repository. If you are not using a real backend simulation, you can leave these as placeholders.
Choose a Simulation Type:
The run_QCBO function uses the simulation_type argument to determine the execution environment. You can modify this in run.py to select the desired simulation.

'exact': Runs on an AerSimulator using the statevector method. This provides the most precise results but is limited to a small number of qubits due to memory constraints.

'shot_based': Runs on an AerSimulator that uses a finite number of measurement shots. This simulates a real quantum computer's measurement process, introducing statistical noise.

'noisy_shot_based': Runs on an AerSimulator that incorporates a custom noise model. This allows for a more realistic simulation of a quantum device's behavior. The get_Simple_Noise_Model function is used for this purpose.

'Noisy_backend_real': This option is designed for advanced use cases where a real IBM Quantum backend is used to generate the noise model for a local simulation. It requires a token and instance to access the backend's calibration data. The code will connect to the ibm_brisbane backend to retrieve its noise characteristics.

Execute the run.py file from the root directory of the repository.
The script will begin the VQE optimization, printing the cost at each iteration. It will then output the final benchmark cost of the solution found.

