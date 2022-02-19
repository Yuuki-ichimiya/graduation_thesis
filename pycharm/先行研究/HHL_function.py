from import_summary import *

def input_state_gate(start_bit, end_bit, vec):
    """ 
    Making a quantum gate which transform |0> to \sum_i x[i]|i>m where x[i] is input vector.
    !!! this uses 2**n times 2**n matrix, so it is quite memory-cosuming.
    !!! this gate is not unitary (we assume that the input state is |0>)
    Args:
      int start_bit: first index of qubit which the gate applies 
      int end_bit:   last index of qubit which the gate applies
      np.ndarray vec:  input vector.
    Returns:
      qulacs.QuantumGate 
    """
    nbit = end_bit - start_bit + 1
    assert vec.size == 2**nbit
    mat_0tox = np.eye(vec.size, dtype=complex)
    mat_0tox[:,0] = vec
    return DenseMatrix(np.arange(start_bit, end_bit+1), mat_0tox)


def CPhaseGate(target, control, angle):
    """ 
    Create controlled phase gate diag(1,e^{i*angle}) with controll. (Qulacs.gate is requried)

    Args:
      int target:  index of target qubit.
      int control:  index of control qubit.
      float64 angle: angle of phase gate.
    Returns:
      QuantumGateBase.DenseMatrix: diag(1, exp(i*angle)).
    """
    CPhaseGate = DenseMatrix(target, np.array( [[1,0], [0,np.cos(angle)+1.j*np.sin(angle)]])  )
    CPhaseGate.add_control_qubit(control, 1)
    return CPhaseGate

def QFT_gate(start_bit, end_bit, Inverse = False):
    """ 
    Making a gate which performs quantum Fourier transfromation between start_bit to end_bit.
    (Definition below is the case when start_bit = 0 and end_bit=n-1)
    We associate an integer  j = j_{n-1}...j_0 to quantum state |j_{n-1}...j_0>.
    We define QFT as
    |k> = |k_{n-1}...k_0> = 1/sqrt(2^n) sum_{j=0}^{2^n-1} exp(2pi*i*(k/2^n)*j) |j>.
    then, |k_m > = 1/sqrt(2)*(|0> + exp(i*2pi*0.j_{n-1-m}...j_0)|1> )
    When Inverse=True,  the gate represents Inverse QFT,
    |k> = |k_{n-1}...k_0> = 1/sqrt(2^n) sum_{j=0}^{2^n-1} exp(-2pi*i*(k/2^n)*j) |j>.

    Args:
      int start_bit:  first index of qubits where we apply QFT.
      int end_bit:    last  index of qubits where we apply QFT.
      bool Inverse: When True, the gate perform inverse-QFT ( = QFT^{\dagger}).
    Returns:
      qulacs.QuantumGate: QFT gate which acts on a region between start_bit and end_bit.
    """

    gate = Identity(start_bit) ## make empty gate
    n = end_bit - start_bit + 1  ## size of QFT

    ## loop from j_{n-1} 
    for target in range(end_bit, start_bit-1, -1):
        gate = merge(gate, H(target)) ## 1/sqrt(2)(|0> + exp(i*2pi*0.j_{target})|1>)
        for control in range(start_bit, target):
            gate = merge( gate, CPhaseGate(target, control, (-1)**Inverse * 2.*np.pi/2**(target-control+1)) )
    ## perform SWAP between (start_bit + s)-th bit and (end_bit - s)-th bit
    for s in range(n//2):  ## s runs 0 to n//2-1
        gate = merge(gate, SWAP(start_bit + s, end_bit - s))
    ## return final circuit
    return gate

def HHL_algorithm(W, b_vec, matrix_size, reg_nbit, scale_fac):
    nbit = int(np.ceil(np.log2(matrix_size))) ## 状態に使うビット数
    N = 2**nbit

    W_enl = np.zeros((N, N)) ## enl は enlarged の略、N*Nの0成分で埋めた行列
    W_enl[:W.shape[0], :W.shape[1]] = W.copy() # 元々のWからコピー、余った部分は0のまま
    b_enl = np.zeros(N) # 0成分で埋めたN次元ベクトル
    b_enl[:len(b_vec)] = b_vec.copy() # 元々の右辺ベクトルからコピー、余った部分は0のまま

    ## W_enl をスケール(定数倍)する
    W_enl_scaled = scale_fac * W_enl

    ## W_enl_scaledの固有値として想定する最小の値
    ## 今回は射影が100%成功するので, レジスタで表せる最小値の定数倍でとっておく
    C = 0.5*(2 * np.pi * (1. / 2**(reg_nbit) ))
    
    ## 対角化. AP = PD <-> A = P*D*P^dag 
    D, P = np.linalg.eigh(W_enl_scaled)

    #####################################
    ### HHL量子回路を作る. 0番目のビットから順に、Aの作用する空間のbit達 (0番目 ~ nbit-1番目), 
    ### register bit達 (nbit番目 ~ nbit+reg_nbit-1番目), conditional回転用のbit (nbit+reg_nbit番目)
    ### とする.
    #####################################
    total_qubits = nbit + reg_nbit + 1
    total_circuit = QuantumCircuit(total_qubits)

    ## ------ 0番目~(nbit-1)番目のbitに入力するベクトルbの準備 ------
    ## 本来はqRAMのアルゴリズムを用いるべきだが, ここでは自作の入力ゲートを用いている. 
    ## qulacsではstate.load(b_enl)でも実装可能.
    state = QuantumState(total_qubits)
    state.set_zero_state() 
    b_gate = input_state_gate(0, nbit-1, b_enl)
    total_circuit.add_gate(b_gate)

    ## ------- レジスターbit に Hadamard gate をかける -------
    for register in range(nbit, nbit+reg_nbit): ## from nbit to nbit+reg_nbit-1
        total_circuit.add_H_gate(register)

    ## ------- 位相推定を実装 -------
    ## U := e^{i*A*t), その固有値をdiag( {e^{i*2pi*phi_k}}_{k=0, ..., N-1) )とおく.
    ## Implement \sum_j |j><j| exp(i*A*t*j) to register bits
    for register in range(nbit, nbit+reg_nbit):
        ## U^{2^{register-nbit}} を実装.
        ## 対角化した結果を使ってしまう
        U_mat = reduce(np.dot,  [P, np.diag(np.exp( 1.j * D * (2**(register-nbit)) )), P.T.conj()]  )
        U_gate = DenseMatrix(np.arange(nbit), U_mat)
        U_gate.add_control_qubit(register, 1) ## control bitの追加
        total_circuit.add_gate(U_gate)

    ## ------- Perform inverse QFT to register bits -------
    total_circuit.add_gate(QFT_gate(nbit, nbit+reg_nbit-1, Inverse=True))

    ## ------- conditional rotation を掛ける -------
    ## レジスター |phi> に対応するA*tの固有値は l = 2pi * 0.phi = 2pi * (phi / 2**reg_nbit).
    ## conditional rotationの定義は (本文と逆)
    ## |phi>|0> -> C/(lambda)|phi>|0> + sqrt(1 - C^2/(lambda)^2)|phi>|1>.
    ## 古典シミュレーションなのでゲートをあらわに作ってしまう.
    condrot_mat = np.zeros( (2**(reg_nbit+1), (2**(reg_nbit+1))), dtype=complex)
    for index in range(2**reg_nbit):
        lam = 2 * np.pi * (float(index) / 2**(reg_nbit) )
        index_0 = index ## integer which represents |index>|0>
        index_1 = index + 2**reg_nbit ## integer which represents |index>|1>
        if lam >= C:
            if lam >= np.pi: ## あらかじめ[-pi, pi]内に固有値をスケールしているので、[pi, 2pi] は 負の固有値に対応
                lam = lam - 2*np.pi
            condrot_mat[index_0, index_0] = C / lam
            condrot_mat[index_1, index_0] =   np.sqrt( 1 - C**2/lam**2 )
            condrot_mat[index_0, index_1] = - np.sqrt( 1 - C**2/lam**2 )
            condrot_mat[index_1, index_1] = C / lam

        else:
            condrot_mat[index_0, index_0] = 1.
            condrot_mat[index_1, index_1] = 1.
    ## DenseGateに変換して実装
    condrot_gate = DenseMatrix(np.arange(nbit, nbit+reg_nbit+1), condrot_mat) 
    total_circuit.add_gate(condrot_gate)

    ## ------- Perform QFT to register bits -------
    total_circuit.add_gate(QFT_gate(nbit, nbit+reg_nbit-1, Inverse=False))

    ## ------- 位相推定の逆を実装(U^\dagger = e^{-iAt}) -------
    for register in range(nbit, nbit+reg_nbit): ## from nbit to nbit+reg_nbit-1
        ## {U^{\dagger}}^{2^{register-nbit}} を実装.
        ## 対角化した結果を使ってしまう
        U_mat = reduce(np.dot,  [P, np.diag(np.exp( -1.j* D * (2**(register-nbit)) )), P.T.conj()]  )
        U_gate = DenseMatrix(np.arange(nbit), U_mat)
        U_gate.add_control_qubit(register, 1) ## control bitの追加
        total_circuit.add_gate(U_gate)

    ## ------- レジスターbit に Hadamard gate をかける -------
    for register in range(nbit, nbit+reg_nbit): 
        total_circuit.add_H_gate(register)

    ## ------- 補助ビットを0に射影する. qulacsでは非ユニタリゲートとして実装されている -------
    total_circuit.add_P0_gate(nbit+reg_nbit)

    #####################################
    ### HHL量子回路を実行し, 結果を取り出す
    #####################################
    total_circuit.update_quantum_state(state)

    ## 0番目から(nbit-1)番目の bit が計算結果 |x>に対応
    result = state.get_vector()[:2**nbit].real
    x_HHL = result/C * scale_fac

    ################################################################################################################
    
    return x_HHL


############################################ 使い方　############################################
# 要約
# W = np.array([[0.         ,0.         ,0.30021458 ,0.4111915  ,0.43015563 ,0.46456748],
#               [0.         ,0.         ,1.         ,1.         ,1.         ,1.        ],
#               [0.30021458 ,1.         ,0.02369003 ,0.01330327 ,0.01838175 ,0.0216144 ],
#               [0.4111915  ,1.         ,0.01330327 ,0.03111914 ,0.01629129 ,0.01887665],
#               [0.43015563 ,1.         ,0.01838175 ,0.01629129 ,0.02885482 ,0.02333747],
#               [0.46456748 ,1.         ,0.0216144  ,0.01887665 ,0.02333747 ,0.04412049]]
#             )
# b = np.array([0.1, 1. , 0., 0., 0., 0.])

# np.set_printoptions(linewidth=200)
# print(W)
# ## Wの固有値を確認 -> [-pi, pi] に収まっている
# print(np.linalg.eigh(W)[0])

# nbit = 3 ## 状態に使うビット数
# N = 2**nbit

# W_enl = np.zeros((N, N)) ## enl は enlarged の略、N*Nの0成分で埋めた行列
# W_enl[:W.shape[0], :W.shape[1]] = W.copy() # 元々のWからコピー、余った部分は0のまま
# b_enl = np.zeros(N) # 0成分で埋めたN次元ベクトル
# b_enl[:len(b)] = b.copy() # 元々の右辺ベクトルからコピー、余った部分は0のまま

# # 位相推定に使うレジスタの数
# reg_nbit = 10

# ## W_enl をスケール(定数倍)する係数
# scale_fac = 1.
# W_enl_scaled = scale_fac * W_enl

# ## W_enl_scaledの固有値として想定する最小の値
# ## 今回は射影が100%成功するので, レジスタで表せる最小値の定数倍でとっておく
# C = 0.5*(2 * np.pi * (1. / 2**(reg_nbit) ))

# start = time.time()
# x_HHL = HHL_algorithm(W_enl_scaled, b_enl, nbit, reg_nbit, C, scale_fac)
# t = time.time() - start

# ## 厳密解
# x_exact = np.linalg.lstsq(W_enl, b_enl, rcond=0)[0]

# print("経過時間[s]:", t)
# print("HHL:  ", x_HHL)
# print("exact:", x_exact)
# rel_error = np.linalg.norm(x_HHL- x_exact) / np.linalg.norm(x_exact)
# print("rel_error", rel_error)