from import_summary import *

def add_X_fields(operator):
    nqubits = operator.get_qubit_count()
    for k in range(nqubits):
        operator.add_operator(1.0,"X {0}".format(k)) 
    return operator


def add_Z_fields(operator):
    nqubits = operator.get_qubit_count()
    for k in range(nqubits):
        operator.add_operator(1.0,"Z {0}".format(k)) 
    return operator

def add_Y_fields(operator):
    nqubits = operator.get_qubit_count()
    for k in range(nqubits):
        operator.add_operator(1.0,"Y {0}".format(k)) 
    return operator

def add_ZZ_interactions(operator,ListOfInt):
    nqubits = operator.get_qubit_count()
    
    for k in range(len(ListOfInt)):
        operator.add_operator(1.0,"Z {0}".format(ListOfInt[k][0])+"Z {0}".format(ListOfInt[k][1]))
    return operator

def add_parametric_gates_from_observable(parametric_circuit,layer_observable):

    nqubits = layer_observable.get_qubit_count()

    for j in range(layer_observable.get_term_count()):
        pauli = layer_observable.get_term(j)

        # Get the subscript of each pauli symbol
        index_list = pauli.get_index_list()

        # Get pauli symbols (I,X,Y,Z -> 0,1,2,3)
        pauli_id_list = pauli.get_pauli_id_list()

        parametric_circuit.add_parametric_multi_Pauli_rotation_gate(index_list, pauli_id_list, 0.0)
    return parametric_circuit


def show_distribution(state):
    nqubits = state.get_qubit_count()
    plt.figure(figsize=(2*nqubits,5))
    mpl.rcParams['axes.xmargin'] = 0
    mpl.rcParams['axes.ymargin'] = 0.1
    
    plt.grid(axis='y', color='r', linestyle='dotted', linewidth=1)
    plt.bar([i for i in range(pow(2,nqubits))],pow(abs(state.get_vector()),2), tick_label=[format(i,'#0'+str(nqubits+2)+'b') for i in range(pow(2,nqubits))])
    plt.xticks(rotation=90)
    plt.show()

def qasm_to_qulacs_fromfile(input_filepath,qulacs_circuit):

    with open(input_filepath, "r") as ifile:
        lines = ifile.readlines()
        

        for line in lines:
            s = re.search(r"qreg|cx|u3|u1", line)

            if s is None:
                continue

            elif s.group() == 'qreg':
                match = re.search(r"\d\d*", line)
                print(match)
                continue

            elif s.group() == 'cx':
                match = re.findall(r"\[\d\d*\]", line)  # int??????
                c_qbit = int(match[0].strip('[]'))
                t_qbit = int(match[1].strip('[]'))
                qulacs_circuit.add_gate(CNOT(c_qbit,t_qbit))   

                continue

            elif s.group() == 'u3':
                m_r = re.findall(r"[-]?\d\.\d\d*", line)  # real??????, ???????????????
                m_i = re.findall(r"\[\d\d*\]", line)  # int??????

                # target_bit = m_i[0]
                # u3parameters = m_r
                qulacs_circuit.add_gate(U3(int(m_i[0].strip('[]')),float(m_r[0]),float(m_r[1]),float(m_r[2])))

                continue

            elif s.group() == 'u1':
                m_r = re.findall(r"[-]?\d\.\d\d*", line)  # real??????
                m_i = re.findall(r"\[\d\d*\]", line)  # int??????

                qulacs_circuit.add_gate(U1(int(m_i[0].strip('[]')),float(m_r[0])))

                continue


def show_observable(hamiltonian):
    for j in range(hamiltonian.get_term_count()):
        pauli=hamiltonian.get_term(j)

        # Get the subscript of each pauli symbol
        index_list = pauli.get_index_list()

        # Get pauli symbols (I,X,Y,Z -> 0,1,2,3)
        pauli_id_list = pauli.get_pauli_id_list()

        # Get pauli coefficient
        coef = pauli.get_coef()

        # Create a copy of pauli operator
        another_pauli = pauli.copy()

        s = ["I","X","Y","Z"]
        pauli_str = [s[i] for i in pauli_id_list]
        terms_str = [item[0]+str(item[1]) for item in zip(pauli_str,index_list)]
        full_str = str(coef) + " " + " ".join(terms_str)
        print(full_str)

#################
# Mitarai
################

from scipy.sparse import csr_matrix, kron
sigmaz = csr_matrix([[1, 0], [0, -1]])
sigmay = csr_matrix([[0, -1j], [1j, 0]])
sigmax = csr_matrix([[0, 1], [1, 0]])
sigmai = csr_matrix([[1, 0], [0, 1]])
sigma_list = [sigmai, sigmax, sigmay, sigmaz]
import numpy as np

def _kron_n(*ops):
    """
    takes tensor product of given scipy matrix
    Args:
        ops (:class:`list`) 
    """
    if len(ops) == 2:
        return kron(ops[0], ops[1])
    else:
        return kron(_kron_n(*ops[:-1]), ops[-1])


def get_matrix(obs):
    """
    returns matrix of an observable
    Args:
        obs (qulacs_core.Observable)
    Return:
        scipy.sparse.csr_matrix
    """
    n_terms = obs.get_term_count()
    n_qubits = obs.get_qubit_count()
    result = csr_matrix((2**n_qubits, 2**n_qubits), dtype=np.complex128)
    for i in range(n_terms):
        pauli = obs.get_term(i)
        pauli_id_list = pauli.get_pauli_id_list()
        pauli_target_list = pauli.get_index_list()
        pauli_string = [sigmai for q in range(n_qubits)]
        for j, target in enumerate(pauli_target_list):
            pauli_string[target] = sigma_list[pauli_id_list[j]]
        result += pauli.get_coef()*_kron_n(*(pauli_string[::-1]))
    return result
