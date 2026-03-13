import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import diags, eye
from warnings import filterwarnings
import pickle

# 忽略稀疏矩阵运算中的部分效率警告
filterwarnings('ignore')

class Simulation():
    """
    供应链仿真引擎 (支持随机需求与自动微分)。
    
    核心功能: 
    1. 前向计算 (Forward): 模拟供应链运作，计算总成本 (持有成本 + 缺货惩罚)。
    2. 反向传播 (Backward): 计算总成本对初始库存(S)的梯度，用于SGD优化。
    """
    def __init__(self, data_type, data_path, network_name, distribution_filename,
                 penalty_filename, simulation_duration=365):
        self.__data_type = data_type
        self.__duration = simulation_duration 
        print(f'数据类型: {data_type}, 模拟时长: {simulation_duration}')
        
        # 初始化并加载所有数据
        self.prepare_data(data_path, network_name, distribution_filename, penalty_filename, data_type)

    def prepare_data(self, data_path, network_name, distribution_filename, penalty_filename, data_type):
        """
        数据准备函数：加载网络结构、销量分布、成本参数及惩罚系数。
        将图结构转换为稀疏矩阵形式以便进行向量化计算。
        """

        def count_layer(B):
            """辅助函数：计算网络/BOM的层级深度，用于确定前向计算的顺序"""
            B = B.tocsr().astype(self.__data_type)
            temp = eye(B.shape[0], dtype=self.__data_type).tocsr()
            maxlayer = 0
            for i in range(B.shape[0]):
                temp = B @ temp; temp.eliminate_zeros()
                if temp.nnz == 0:
                    maxlayer = i; break
            return maxlayer + 1

        # --- 1. 加载网络图 ---
        path = os.path.join(data_path, network_name)
        with open(path, 'rb') as f:
            G = pickle.load(f)
        # 确保节点ID是连续整数 (0..N-1)
        G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')

        # 构建邻接矩阵 B (Target x Source)
        # B[i, j] > 0 表示节点 j 是节点 i 的上游（或者 i 消耗 j）
        self.__B = nx.adjacency_matrix(G, weight='weight').astype(data_type)
        self.__nodes_num = self.__B.shape[0]
        print('节点数：', self.__nodes_num)
        
        # --- 2. 加载销量分布 (Mean, Std) ---
        distribution_path = os.path.join(data_path, distribution_filename)
        df_dist = pd.read_csv(distribution_path)

        self.__demand_mean = np.zeros(self.__nodes_num, dtype=data_type)
        self.__demand_std = np.zeros(self.__nodes_num, dtype=data_type)
        
        node_name_to_id_map = {data['name']: node_id for node_id, data in G.nodes(data=True)}
        for row in df_dist.itertuples(index=False):
            node_name = f"{row.LocationCode}_{row.SKUCode}"
            if node_name in node_name_to_id_map:
                node_id = node_name_to_id_map[node_name]
                self.__demand_mean[node_id] = row.SaleQtyMean
                self.__demand_std[node_id] = row.SaleQtyStd
        
        # 计算BOM/网络深度
        self.__stage_num = count_layer(self.__B)
        
        # --- 3. 获取持有成本和提前期 ---
        self.__hold_coef = np.array(list(nx.get_node_attributes(G, 'holdcost').values()), dtype=data_type)
        self.__hold_coef = np.expand_dims(self.__hold_coef, axis=0) # shape: (1, N)
        self.__lead_time = np.array(list(nx.get_node_attributes(G, 'leadtime').values()))

        # --- 4. 加载并应用惩罚系数 (Penalty Cost) ---
        print(f"--- 正在加载惩罚系数 ---")
        penalty_file_path = os.path.join(data_path, penalty_filename)
        
        penalty_map = {}
        if os.path.exists(penalty_file_path):
            df_penalty = pd.read_csv(penalty_file_path)
            penalty_map = df_penalty.set_index(['LocationCode', 'SKUCode'])['PenaltyFactor'].to_dict()
        else:
            print("警告：未找到惩罚系数文件，将完全使用默认策略。")

        # 基于持有成本初始化
        self.penalty_coef = np.zeros_like(self.__hold_coef)
        
        # 设置每个节点的缺货惩罚系数
        for node_id, node_data in G.nodes(data=True):
            loc_code = node_data['location']
            sku_code = node_data['sku']
            key = (loc_code, sku_code)
            holding_cost_val = self.__hold_coef[0, node_id]
            
            if key in penalty_map:
                # 使用文件中的系数
                factor = penalty_map[key]
                self.penalty_coef[0, node_id] = factor * holding_cost_val
            else:
                # 默认缺省值逻辑
                if node_data.get('is_customer_facing', False):
                     self.penalty_coef[0, node_id] = 500.0 * holding_cost_val
                else:
                     self.penalty_coef[0, node_id] = 200.0 * holding_cost_val

        # --- 5. 补货参数 ---
        self.__replenishment_cycles = np.array([G.nodes[i].get('replenishment_cycle', 1) for i in range(self.__nodes_num)], dtype=int)
        self.__replenishment_cycles[self.__replenishment_cycles == 0] = 1

        # 最小起订量 (Min Lot Size)
        self.__min_lot_sizes = {}
        for u, v, data in G.edges(data=True):
            if data.get('type') == 'replenishment' and data.get('min_lot_size', 0) > 0:
                self.__min_lot_sizes[v] = data['min_lot_size']
        
        # --- 6. 矩阵掩码 (用于区分成品和原材料) ---
        is_customer_facing = np.array(list(nx.get_node_attributes(G, 'is_customer_facing').values()), dtype=bool)
        self.__customer_facing_mask = np.expand_dims(is_customer_facing.astype(data_type), axis=0)
        
        # --- 7. 稀疏矩阵辅助变量 (用于加速反向传播) ---
        self.__B_T = self.__B.T.tocsr() # 转置矩阵
        self.__E = eye(self.__nodes_num, dtype=data_type).tocsr() # 单位矩阵
        self.__E_B_T = (self.__E - self.__B).T.tocsr() # (I - B)^T
        self.__E_B_T.eliminate_zeros()
        
        # 定义常量
        self.__zero = data_type(0.0)
        self.__one_minus = data_type(-1.0)
        self.__one = data_type(1.0)
        self.__equal_tole = data_type(1e-5) if data_type == np.float32 else data_type(1e-11)

        # 识别原材料节点 (出度为0的通常是原材料/采购件)
        out_degree_values = np.expand_dims([v for _, v in G.out_degree()], axis=0)
        self.__raw_material_node = np.where(out_degree_values == 0, self.__one, self.__zero)
        # 制造件/中间件 (非原材料)
        mau_item = self.__one - self.__raw_material_node
        self.__mau_item_diag = diags(mau_item[0])

        # 缓存生产节点的BOM索引，加速稀疏矩阵行切片
        idx_mau = np.nonzero(mau_item)[1]
        self.__B_indices_list = {i: self.__B.getrow(i).indices for i in idx_mau if self.__B.getrow(i).nnz > 0}

    def evaluate_cost_gradient(self, I_S, rep_num=1, mean_flag=True):
        """
        评估入口函数：执行多次蒙特卡洛模拟，并计算平均成本和平均梯度。
        """
        total_cost = 0.0
        total_grad = np.zeros_like(I_S)
        total_holding = 0.0
        
        for _ in range(rep_num):
            # 1. 生成随机需求矩阵 (Duration x Nodes)
            D_random = np.random.normal(loc=self.__demand_mean, scale=self.__demand_std, 
                                        size=(self.__duration, self.__nodes_num))
            D_random = np.round(D_random) # 需求取整
            D_random = np.maximum(D_random, 0).astype(self.__data_type) # 保证非负
            
            # 2. 打包所有静态参数传入核心计算函数
            args = (I_S, self.__duration, self.__nodes_num, self.__zero, self.__one, self.__one_minus, self.__stage_num,
                    self.__lead_time, self.__data_type, self.__B_indices_list, self.__equal_tole,
                    self.__hold_coef, self.penalty_coef, self.__mau_item_diag, self.__raw_material_node, 
                    self.__B, self.__B_T, self.__E_B_T, D_random,
                    self.__replenishment_cycles, self.__min_lot_sizes, self.__customer_facing_mask)
            
            # 3. 运行前向仿真 + 反向传播
            cost, grad, holding_cost = simulate_and_bp(args)
            
            total_cost += cost
            total_grad += grad
            total_holding += holding_cost
            
        if mean_flag:
            return total_cost / rep_num, total_grad / rep_num, total_holding / rep_num
        else:
            return total_cost, total_grad, total_holding

def simulate_and_bp(args):
    """
    静态核心函数：包含前向仿真 (Forward) 和 反向传播 (Backward)。
    这里实现了核心的库存逻辑方程和自动微分逻辑。
    """
    # 解包参数
    (I_S, duration, nodes_num, zero, one, one_minus, stage_num, lead_time, data_type, 
    B_indices_list, equal_tolerance, holding_cost, penalty_cost, mau_item_diag, raw_material_node,
    B_full, B_T, E_B_T, current_demand, replenishment_cycles, min_lot_sizes, customer_facing_mask) = args
    
    # 构建需求矩阵 (补一行0方便计算)
    D = current_demand
    D_order = np.vstack([D, np.zeros((1, nodes_num), dtype=data_type)]) 
    
    # 本地化常用numpy函数以提速
    minimum, maximum, where = np.minimum, np.maximum, np.where
    np_sum, np_multiply, np_array, np_ceil = np.sum, np.multiply, np.array, np.ceil
    zeros_like, nonzero, np_abs = np.zeros_like, np.nonzero, np.abs
    
    # --- 初始化状态变量 ---
    M_backlog = zeros_like(I_S)       # 生产积压 (因原材料短缺导致的未生产量)
    D_queue = zeros_like(I_S)         # 缺货积压 (Backlog/Shortage)
    P = np.zeros((duration + 1, nodes_num), dtype=data_type) # 在途库存/在制品队列 (Pipeline)
    I_t = I_S.copy()                  # 现有库存 (On-hand Inventory)
    I_position = I_S.copy()           # 库存位置 (Inventory Position = OnHand + OnOrder - Backlog)
    
    cost = data_type(0.0)
    total_holding_cost = data_type(0.0)

    # --- 初始化反向传播所需的中间变量存储容器 ---
    # 这些列表用于在前向过程中记录“开关”状态(Mask)，供反向传播使用
    d_It_d_Yt, d_Dq_d_Yt = [], []
    d_O_d_Ipformer = [[] for _ in range(duration)]
    d_M_d_man_o = [zeros_like(I_S) for _ in range(duration)]
    d_M_d_r_r, d_r_r_d_I, d_r_r_d_r_n = [{} for _ in range(duration)], [], []
    P_orders = [{} for _ in range(duration)] # 记录订单发出时间和到达时间

    # =========================================================
    # Phase 1: Forward Simulation (前向仿真：计算成本)
    # =========================================================
    for t in range(duration):
        # --- 1. Lost Sales 清零逻辑 ---
        # 判断今天是否是补货周期的第一天
        is_cycle_start = (np.mod(t, replenishment_cycles) == 0).astype(data_type)
        is_cycle_start = np.expand_dims(is_cycle_start, axis=0) # shape (1, N)
        
        # 计算清零掩码: 周期起点且为成品时，掩码为0(清零)，否则为1
        queue_reset_mask = one - (is_cycle_start * customer_facing_mask)
        D_queue = D_queue * queue_reset_mask
        
        # --- 2. 订货逻辑 (Base Stock Policy) ---
        # 更新库存位置 IP = IP - 今日需求
        I_position = I_position - D_order[t, :]
        
        # 计算理想补货量：Order = Target(S) - IP
        O_t_ideal = -minimum(zero, (I_position - I_S))
        flag = where((I_position - I_S) < 0, one_minus, zero)
        d_O_d_Ipformer[t].insert(0, diags(flag[0])) # 记录梯度开关
        
        # 处理多级 BOM 的库存位置联动 (Ripple Effect)
        for _ in range(stage_num - 1):
            temp_I_position = I_position - (O_t_ideal @ B_full)
            O_t_ideal = -minimum(zero, (temp_I_position - I_S))
            flag = where((temp_I_position - I_S) < 0, one_minus, zero)
            d_O_d_Ipformer[t].insert(0, diags(flag[0]))
        
        # 实际订单：只有在补货日 (is_cycle_start) 才发出
        O_t = O_t_ideal * is_cycle_start
        
        # 应用最小起订量 (Min Lot Size) 约束
        if min_lot_sizes:
            for node_idx, lot_size in min_lot_sizes.items():
                if O_t[0, node_idx] > 0:
                    O_t[0, node_idx] = np_ceil(O_t[0, node_idx] / lot_size) * lot_size

        # 更新库存位置 IP = IP + Order - 下游消耗的Order
        I_position = I_position + O_t - (O_t @ B_full)
        
        # --- 3. 库存平衡与缺货计算 ---
        # 核心方程：临时库存 = 期初 - 旧缺货 - 今日需求 + 今日入库
        temp_I_t = I_t - D_queue - D[t] + P[t]
        
        # 分离正库存 (Inventory) 和 缺货 (Backlog)
        I_t = maximum(zero, temp_I_t)
        flag = where(temp_I_t > 0, one, zero)
        d_It_d_Yt.append(diags(flag[0])) # 记录梯度流向 I_t 的开关
        
        D_queue = -minimum(zero, temp_I_t)
        flag = where(temp_I_t <= 0, one_minus, zero)
        d_Dq_d_Yt.append(diags(flag[0])) # 记录梯度流向 D_queue 的开关

        # --- 4. 成本计算 ---
        # 持有成本
        daily_holding = np_sum(np_multiply(I_t, holding_cost))
        total_holding_cost += daily_holding
        
        # 缺货成本 (Daily Stockout Penalty)
        daily_shortage = np_sum(np_multiply(D_queue, penalty_cost))
        
        # 总成本累加
        cost += daily_holding + daily_shortage
        
        # --- 5. 生产/采购执行 (考虑 BOM 约束) ---
        # 区分外购件订单和自制件订单
        purchase_order = O_t * raw_material_node 
        mau_order = O_t - purchase_order + M_backlog # 自制件订单包含之前的积压
        
        idx_purch, idx_mau = nonzero(purchase_order)[1], nonzero(mau_order)[1]
        
        # 计算原材料可用率 (齐套率检查)
        resource_needed = mau_order @ B_full
        temp_resource_rate = I_t / resource_needed
        temp_resource_rate[resource_needed == 0] = one # 避免除零
        
        # 记录梯度辅助变量
        temp1, temp2 = one / resource_needed, -np_multiply(temp_resource_rate, one / resource_needed)
        temp1[resource_needed == 0], temp2[resource_needed == 0] = one, one
        resource_rate = minimum(one, temp_resource_rate)
        
        flag2 = where(temp_resource_rate < 1, one, zero)
        d_r_r_d_I.append(diags(np_multiply(flag2, temp1)[0]))
        d_r_r_d_r_n.append(diags(np_multiply(flag2, temp2)[0]))
        
        # 处理外购件入库 (Pipeline)
        for index in idx_purch:
            temp_lead = max(1, int(lead_time[index]))
            if t + temp_lead < duration + 1:
                P[t + temp_lead, index] += purchase_order[0, index]
                P_orders[t][index] = int(lead_time[index])
        
        # 处理自制件入库 (受限于原材料短缺)
        M_actual = zeros_like(M_backlog)
        mau_indices_in_list = [i for i in idx_mau if i in B_indices_list]
        
        if mau_indices_in_list:
            # 找到每个成品的最短板原材料比例 (Min Rate)
            min_rate = np_array([resource_rate[0, B_indices_list[i]].min() for i in mau_indices_in_list])
            M_actual[0, mau_indices_in_list] = min_rate * mau_order[0, mau_indices_in_list]
            
            # 记录哪个原材料是瓶颈，用于反向传播梯度
            for i, index in enumerate(mau_indices_in_list):
                col = B_indices_list[index]
                col2 = col[np_abs(resource_rate[0, col] - min_rate[i]) < equal_tolerance]
                if len(col2) > 0:
                    d_M_d_r_r[t][index] = (data_type(1.0 / len(col2)) * mau_order[0, index], col2)
            d_M_d_man_o[t][0, mau_indices_in_list] = min_rate
        
        # 无BOM约束的节点直接满足
        mau_indices_not_in_list = [i for i in idx_mau if i not in B_indices_list]
        if mau_indices_not_in_list:
            M_actual[0, mau_indices_not_in_list] = mau_order[0, mau_indices_not_in_list]
            
        # 将实际生产量放入 Pipeline
        for index in idx_mau:
            temp_lead = int(lead_time[index])
            if t + temp_lead < duration + 1:
                P[t + temp_lead, index] += M_actual[0, index]
                P_orders[t][index] = temp_lead
        
        # 更新生产积压和原材料消耗
        M_backlog = mau_order - M_actual
        I_t = I_t - (M_actual @ B_full)
    
    # =========================================================
    # Phase 2: Backward Propagation (反向传播：计算梯度)
    # =========================================================
    d_S = zeros_like(I_S)          # 对初始策略 S 的总梯度
    d_Ipt = zeros_like(I_S)        # 对库存位置 IP 的梯度
    d_Mt_backlog = zeros_like(I_S) # 对生产积压的梯度
    
    # 初始化最后一天的梯度源 (Cost 对 It 和 Dback 的导数)
    d_It = holding_cost.copy() 
    d_Dback = penalty_cost.copy()
    
    d_O = np.zeros((duration, 1, nodes_num), dtype=data_type) # 对订单 O 的梯度
    d_P_d_Mq = np.zeros((duration + 1, 1, nodes_num), dtype=data_type)
    
    # 时间倒流：从 T-1 到 0
    for t in range(duration - 1, -1, -1):
        # 1. 重构当天的状态掩码
        is_cycle_start = (np.mod(t, replenishment_cycles) == 0).astype(data_type)
        is_cycle_start = np.expand_dims(is_cycle_start, axis=0)
        
        # 重构清零掩码
        queue_reset_mask = one - (is_cycle_start * customer_facing_mask)
        
        # 2. 生产逻辑的反向传播 (通过BOM传导梯度)
        d_Mact = -(d_It @ B_T) # 原材料消耗导致的库存梯度
        d_Mq = d_Mact - d_Mt_backlog + d_P_d_Mq[t]
        d_mau_o = d_Mt_backlog + np_multiply(d_Mq, d_M_d_man_o[t]) 
        
        # 处理组装约束的梯度分配 (将成品的梯度分给瓶颈原材料)
        d_res_r = zeros_like(I_S)
        for index, (k, col2_list) in d_M_d_r_r[t].items():
            temp_k = k * d_Mq[0, index]
            d_res_r[0, col2_list] += temp_k
            
        d_It = d_It + (d_res_r @ d_r_r_d_I[t])
        d_res_n = d_res_r @ d_r_r_d_r_n[t]
        d_mau_o = d_mau_o + (d_res_n @ B_T)
        
        # 订单梯度更新
        d_O[t] = d_O[t] + (d_mau_o @ mau_item_diag) * is_cycle_start
        
        # 3. 库存平衡方程的反向传播 (d_Yt 是对 temp_I_t 的梯度)
        d_Yt = (d_It @ d_It_d_Yt[t]) + (d_Dback @ d_Dq_d_Yt[t])
        
        # 4. 订货策略的反向传播 (Base Stock)
        d_O[t] = d_O[t] + (d_Ipt @ E_B_T) * is_cycle_start
        d_temp_O = d_O[t].copy()
        for i in range(stage_num):
            d_temp_Ipt = d_temp_O @ d_O_d_Ipformer[t][i]
            d_S = d_S - d_temp_Ipt # 累积到 S 的梯度
            d_Ipt = d_Ipt + d_temp_Ipt
            if i < stage_num - 1:
                d_temp_O = -(d_temp_Ipt @ B_T)
        
        # 5. 时间回溯 (计算 t-1 时刻的梯度源)
        if t > 0:
            d_Mt_backlog = d_mau_o.copy()
            
            # 准备回传给上一天 D_queue 的梯度
            d_It = d_Yt + holding_cost  
            d_Dback = (-d_Yt * queue_reset_mask) + penalty_cost 
            
            # 处理 Pipeline Delay (将库存梯度回传给 L 天前发出订单的那一刻)
            for i in range(nodes_num):
                L = int(lead_time[i])
                origin_t = t - L
                
                if origin_t >= 0:
                    if raw_material_node[0, i] == 1:
                         d_O[origin_t, 0, i] += d_Yt[0, i]
                    else:
                         d_P_d_Mq[origin_t, 0, i] += d_Yt[0, i]
        else:
            # t=0 时，梯度汇入初始状态
            d_S = d_S + d_Yt + d_Ipt
            
    return cost, d_S, total_holding_cost