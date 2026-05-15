import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import diags, eye
import scipy.sparse as sp
from warnings import filterwarnings
import pickle
filterwarnings('ignore')

class Simulation():
    """
    供应链仿真引擎 (支持随机需求与自动微分)。
    动态配合外部优化器，支持松弛约束期(Relaxation)。
    """
    def __init__(self, data_type, data_path, network_name, distribution_filename, penalty_filename, simulation_duration=365):
        self.__data_type = data_type
        self.__duration = simulation_duration
        print(f'数据类型: {data_type}, 模拟时长: {simulation_duration}')

        self.prepare_data(data_path, network_name, distribution_filename, penalty_filename, data_type)
        
    def prepare_data(self, data_path, network_name, distribution_filename, penalty_filename, data_type):
        def count_layer(B):
            B = B.tocsr().astype(self.__data_type)
            temp = eye(B.shape[0], dtype=self.__data_type).tocsr()
            maxlayer = 0
            for i in range(B.shape[0]):
                temp = B @ temp; temp.eliminate_zeros()
                if temp.nnz == 0:
                    maxlayer = i; break
            return maxlayer + 1

        # 加载网络图
        path = os.path.join(data_path, network_name)
        with open(path, 'rb') as f:
            G = pickle.load(f)
        G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')

        self.__B = sp.csr_matrix(nx.adjacency_matrix(G, weight='weight').astype(data_type))
        self.__nodes_num = self.__B.shape[0]
        
        # 加载销量分布
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
        
        self.__stage_num = count_layer(self.__B)
        
        self.__hold_coef = np.array(list(nx.get_node_attributes(G, 'holdcost').values()), dtype=data_type)
        self.__hold_coef = np.expand_dims(self.__hold_coef, axis=0) 
        self.__lead_time = np.array(list(nx.get_node_attributes(G, 'leadtime').values()))

        self.target_sl = np.array([G.nodes[i].get('service_level', 0.95) for i in range(self.__nodes_num)], dtype=data_type)
        self.target_sl = np.expand_dims(self.target_sl, axis=0)
        
        alpha = np.clip(self.target_sl, 0.01, 0.999) 

        default_penalty = 100 * alpha / (1.0 - alpha)
        penalty_file_path = os.path.join(data_path, penalty_filename)
        
        if os.path.exists(penalty_file_path):
            try:
                df_penalty = pd.read_csv(penalty_file_path)
                self.__penalty_coef = default_penalty.copy()

                for row in df_penalty.itertuples(index=False):
                    node_name = f"{row.LocationCode}_{row.SKUCode}"
                    if node_name in node_name_to_id_map:
                        node_id = node_name_to_id_map[node_name]
                        self.__penalty_coef[0, node_id] = row.PenaltyFactor
                        
                print(f"成功从 penalty.csv 加载自定义缺货惩罚系数。")
            except Exception as e:
                print(f"读取 penalty.csv 失败 ({e})，将使用默认公式计算惩罚系数。")
                self.__penalty_coef = default_penalty
        else:
            self.__penalty_coef = default_penalty

        # 补货参数
        self.__replenishment_cycles = np.array([G.nodes[i].get('replenishment_cycle', 1) for i in range(self.__nodes_num)], dtype=int)
        self.__replenishment_cycles[self.__replenishment_cycles == 0] = 1

        self.__min_lot_sizes = {}
        for u, v, data in G.edges(data=True):
            if data.get('type') == 'replenishment' and data.get('min_lot_size', 0) > 0:
                self.__min_lot_sizes[v] = data['min_lot_size']
        
        # 矩阵掩码
        is_customer_facing = np.array(list(nx.get_node_attributes(G, 'is_customer_facing').values()), dtype=bool)
        self.__customer_facing_mask = np.expand_dims(is_customer_facing.astype(data_type), axis=0)
        
        # 稀疏矩阵辅助变量
        self.__B_T = self.__B.T.tocsr()
        self.__E = eye(self.__nodes_num, dtype=data_type).tocsr()
        self.__E_B_T = (self.__E - self.__B).T.tocsr()
        self.__E_B_T.eliminate_zeros()
        
        self.__zero = data_type(0.0)
        self.__one_minus = data_type(-1.0)
        self.__one = data_type(1.0)
        self.__equal_tole = data_type(1e-5) if data_type == np.float32 else data_type(1e-11)

        out_degree_values = np.expand_dims([v for _, v in G.out_degree()], axis=0)
        self.__raw_material_node = np.where(out_degree_values == 0, self.__one, self.__zero)
        mau_item = self.__one - self.__raw_material_node
        self.__mau_item_diag = diags(mau_item[0])

        idx_mau = np.nonzero(mau_item)[1]
        self.__B_indices_list = {i: self.__B[i].indices for i in idx_mau if self.__B[i].nnz > 0}

    def evaluate_cost_gradient(self, I_S, rep_num=1, mean_flag=True, epoch=0, max_epochs=50, current_penalty=None):
        total_cost = 0.0
        total_grad = np.zeros_like(I_S)
        total_holding = 0.0
        total_csl = np.zeros_like(I_S) 
        
        active_penalty = self.__penalty_coef if current_penalty is None else current_penalty
        
        for _ in range(rep_num):
            D_random = np.random.normal(loc=self.__demand_mean, scale=self.__demand_std, 
                                        size=(self.__duration, self.__nodes_num))
            D_random = np.round(D_random)
            D_random = np.maximum(D_random, 0).astype(self.__data_type) 
            
            args = (I_S, self.__duration, self.__nodes_num, self.__zero, self.__one, self.__one_minus, self.__stage_num,
                    self.__lead_time, self.__data_type, self.__B_indices_list, self.__equal_tole,
                    self.__hold_coef, active_penalty, self.__mau_item_diag, self.__raw_material_node, 
                    self.__B, self.__B_T, self.__E_B_T, D_random,
                    self.__replenishment_cycles, self.__min_lot_sizes, self.__customer_facing_mask, epoch, max_epochs)
            
            cost, grad, holding_cost, csl = simulate_and_bp(args)
            
            total_cost += cost
            total_grad += grad
            total_holding += holding_cost
            total_csl += csl
            
        if mean_flag:
            return total_cost / rep_num, total_grad / rep_num, total_holding / rep_num, total_csl / rep_num
        else:
            return total_cost, total_grad, total_holding, total_csl

def simulate_and_bp(args):
    (I_S, duration, nodes_num, zero, one, one_minus, stage_num, lead_time, data_type, 
    B_indices_list, equal_tolerance, holding_cost, penalty_cost, mau_item_diag, raw_material_node,
    B_full, B_T, E_B_T, current_demand, replenishment_cycles, min_lot_sizes, customer_facing_mask, epoch, max_epochs) = args
    
    D = current_demand
    D_order = np.vstack([D, np.zeros((1, nodes_num), dtype=data_type)]) 
    
    minimum, maximum, where = np.minimum, np.maximum, np.where
    np_sum, np_multiply, np_array, np_ceil = np.sum, np.multiply, np.array, np.ceil
    zeros_like, nonzero, np_abs = np.zeros_like, np.nonzero, np.abs
    
    M_backlog = zeros_like(I_S)       
    D_queue = zeros_like(I_S)         

    max_lead = int(np.max(lead_time)) if lead_time.size > 0 else 0
    P = np.zeros((duration + max_lead + 1, nodes_num), dtype=data_type) 
    
    I_t = I_S.copy()                  
    I_position = I_S.copy()           
    
    cost = data_type(0.0)
    total_holding_cost = data_type(0.0)

    cycle_stockout_flag = zeros_like(I_S)
    total_stockout_cycles = zeros_like(I_S)
    total_cycles_count = zeros_like(I_S)

    d_It_d_Yt, d_Dq_d_Yt = [], []
    d_O_d_Ipformer = [[] for _ in range(duration)]
    d_M_d_man_o = [zeros_like(I_S) for _ in range(duration)]
    d_M_d_r_r, d_r_r_d_I, d_r_r_d_r_n = [{} for _ in range(duration)], [], []
    P_orders = [{} for _ in range(duration)] 

    relaxation_threshold = int(max_epochs * 0.8)

    for t in range(duration):
        is_cycle_start = (np.mod(t, replenishment_cycles) == 0).astype(data_type)
        is_cycle_start = np.expand_dims(is_cycle_start, axis=0) 

        if t > 0:
            total_stockout_cycles += cycle_stockout_flag * is_cycle_start
            cycle_stockout_flag = cycle_stockout_flag * (one - is_cycle_start)
            
        total_cycles_count += is_cycle_start
        
        queue_reset_mask = one - (is_cycle_start * customer_facing_mask)
        D_queue = D_queue * queue_reset_mask

        I_position = I_position - D_order[t, :]
        temp_I_position = I_position.copy()
        O_t_actual = zeros_like(I_S)
        
        for i in range(stage_num):
            O_t_ideal = -minimum(zero, (temp_I_position - I_S))
            flag = where((temp_I_position - I_S) < 0, one_minus, zero)
            O_t_actual = O_t_ideal * is_cycle_start

            if min_lot_sizes and epoch >= relaxation_threshold:
                for node_idx, lot_size in min_lot_sizes.items():
                    if O_t_actual[0, node_idx] > 0:
                        O_t_actual[0, node_idx] = np_ceil(O_t_actual[0, node_idx] / lot_size) * lot_size

            d_O_d_Ipformer[t].insert(0, diags((flag * is_cycle_start)[0])) 
            if i < stage_num - 1:
                temp_I_position = I_position - (O_t_actual @ B_full)     
        O_t = O_t_actual

        I_position = I_position + O_t - (O_t @ B_full)
        
        temp_I_t = I_t - D_queue - D[t] + P[t]

        ext_stockout = where(temp_I_t < -1e-5, one, zero)
        
        I_t = maximum(zero, temp_I_t)
        flag = where(temp_I_t > 0, one, zero)
        d_It_d_Yt.append(diags(flag[0])) 
        
        D_queue = -minimum(zero, temp_I_t)
        flag = where(temp_I_t <= 0, one_minus, zero)
        d_Dq_d_Yt.append(diags(flag[0])) 

        daily_shortage = np_sum(np_multiply(D_queue, penalty_cost))
        
        purchase_order = O_t * raw_material_node 
        mau_order = O_t - purchase_order + M_backlog 
        idx_purch, idx_mau = nonzero(purchase_order)[1], nonzero(mau_order)[1]
        
        resource_needed = mau_order @ B_full
        
        valid_mask = (resource_needed > 1e-11)
        temp_resource_rate = zeros_like(resource_needed)
        temp1 = zeros_like(resource_needed)
        temp2 = zeros_like(resource_needed)

        temp_resource_rate[valid_mask] = I_t[valid_mask] / resource_needed[valid_mask]
        temp1[valid_mask] = one / resource_needed[valid_mask]
        temp2[valid_mask] = -np_multiply(temp_resource_rate[valid_mask], one / resource_needed[valid_mask])

        temp_resource_rate[~valid_mask] = one 
        resource_rate = minimum(one, temp_resource_rate)

        int_stockout = where(temp_resource_rate < 1 - 1e-5, one, zero)
        day_stockout = maximum(ext_stockout, int_stockout)

        cycle_stockout_flag = maximum(cycle_stockout_flag, day_stockout)
        
        flag2 = where(temp_resource_rate < 1, one, zero)
        d_r_r_d_I.append(diags(np_multiply(flag2, temp1)[0]))
        d_r_r_d_r_n.append(diags(np_multiply(flag2, temp2)[0]))
        
        # 发货填充在途矩阵（移除边界截断，已通过外扩 P 容纳所有订单）
        for index in idx_purch:
            temp_lead = max(1, int(lead_time[index]))
            P[t + temp_lead, index] += purchase_order[0, index]
            P_orders[t][index] = int(lead_time[index])
        
        M_actual = zeros_like(M_backlog)
        mau_indices_in_list = [i for i in idx_mau if i in B_indices_list]
        
        if mau_indices_in_list:
            min_rate = np_array([resource_rate[0, B_indices_list[i]].min() for i in mau_indices_in_list])
            M_actual[0, mau_indices_in_list] = min_rate * mau_order[0, mau_indices_in_list]

            for i, index in enumerate(mau_indices_in_list):
                col = B_indices_list[index]
                col2 = col[np_abs(resource_rate[0, col] - min_rate[i]) < equal_tolerance]
                if len(col2) > 0:
                    d_M_d_r_r[t][index] = (data_type(1.0 / len(col2)) * mau_order[0, index], col2)
            d_M_d_man_o[t][0, mau_indices_in_list] = min_rate
        
        mau_indices_not_in_list = [i for i in idx_mau if i not in B_indices_list]
        if mau_indices_not_in_list:
            M_actual[0, mau_indices_not_in_list] = mau_order[0, mau_indices_not_in_list]
            
        for index in idx_mau:
            temp_lead = int(lead_time[index])
            P[t + temp_lead, index] += M_actual[0, index]
            P_orders[t][index] = temp_lead
        
        M_backlog = mau_order - M_actual
        I_t = I_t - (M_actual @ B_full)

        daily_holding = np_sum(np_multiply(I_t, holding_cost))
        total_holding_cost += daily_holding
        cost += daily_holding + daily_shortage

    total_stockout_cycles += cycle_stockout_flag
    
    terminal_days = max(10.0, float(max_lead))
    
    term_hold = np_sum(np_multiply(I_t, holding_cost)) * terminal_days
    term_pipe = np_sum(np_multiply(np_sum(P[duration:, :], axis=0), holding_cost)) * terminal_days
    term_short_ext = np_sum(np_multiply(D_queue, penalty_cost)) * terminal_days
    term_short_int = np_sum(np_multiply(M_backlog, penalty_cost)) * terminal_days

    cost += term_hold + term_pipe + term_short_ext + term_short_int
    total_holding_cost += term_hold + term_pipe
    
    # === 反向传播 ===
    d_S = zeros_like(I_S); d_Ipt = zeros_like(I_S) 

    d_It = holding_cost * (1.0 + terminal_days)
    d_Dback = penalty_cost * (1.0 + terminal_days)
    d_Mt_backlog = penalty_cost * terminal_days 

    d_O = np.zeros((duration, 1, nodes_num), dtype=data_type) 
    d_P_d_Mq = np.zeros((duration + max_lead + 1, 1, nodes_num), dtype=data_type)
    
    for arr_t in range(duration, duration + max_lead + 1):
        for i in range(nodes_num):
            L = int(lead_time[i])
            origin_t = arr_t - L
            if 0 <= origin_t < duration:
                if raw_material_node[0, i] == 1:
                    d_O[origin_t, 0, i] += holding_cost[0, i] * terminal_days
                else:
                    d_P_d_Mq[origin_t, 0, i] += holding_cost[0, i] * terminal_days

    for t in range(duration - 1, -1, -1):
        is_cycle_start = (np.mod(t, replenishment_cycles) == 0).astype(data_type)
        is_cycle_start = np.expand_dims(is_cycle_start, axis=0)
        queue_reset_mask = one - (is_cycle_start * customer_facing_mask)
        d_Mact = -(d_It @ B_T) 
        d_Mq = d_Mact - d_Mt_backlog + d_P_d_Mq[t]
        d_mau_o = d_Mt_backlog + np_multiply(d_Mq, d_M_d_man_o[t]) 
        d_res_r = zeros_like(I_S)
        for index, (k, col2_list) in d_M_d_r_r[t].items():
            d_res_r[0, col2_list] += k * d_Mq[0, index]
        d_It = d_It + (d_res_r @ d_r_r_d_I[t])
        d_res_n = d_res_r @ d_r_r_d_r_n[t]
        d_mau_o = d_mau_o + (d_res_n @ B_T)
        d_O[t] = d_O[t] + (d_mau_o @ mau_item_diag) * is_cycle_start
        d_Yt = (d_It @ d_It_d_Yt[t]) + (d_Dback @ d_Dq_d_Yt[t])
        d_O[t] = d_O[t] + (d_Ipt @ E_B_T) 
        d_temp_O = d_O[t].copy()
        
        for i in range(stage_num):
            d_temp_Ipt = d_temp_O @ d_O_d_Ipformer[t][i]
            d_S = d_S - d_temp_Ipt 
            d_Ipt = d_Ipt + d_temp_Ipt
            if i < stage_num - 1:
                d_temp_O = -(d_temp_Ipt @ B_T)
        
        if t > 0:
            d_Mt_backlog = d_mau_o.copy()
            d_It = d_Yt + holding_cost  
            d_Dback = (-d_Yt * queue_reset_mask) + penalty_cost 
            for i in range(nodes_num):
                L = int(lead_time[i])
                origin_t = t - L
                if origin_t >= 0:
                    if raw_material_node[0, i] == 1:
                         d_O[origin_t, 0, i] += d_Yt[0, i]
                    else:
                         d_P_d_Mq[origin_t, 0, i] += d_Yt[0, i]
        else:
            d_S = d_S + d_Yt + d_Ipt

    csl = one - (total_stockout_cycles / maximum(total_cycles_count, one))
            
    return cost, d_S, total_holding_cost, csl