import os
import numpy as np
from time import time
from datetime import datetime
from rnnisa.utils.tool_function import my_dump

class SimOpt():
    """
    仿真优化器：执行 SGD 梯度下降。
    """
    def __init__(self, data_path, rep_num, step_size, positive_flag,
                 grad_f, max_epochs, convergence_tolerance, patience, target_cost=0.0):
        self.__data_path = data_path
        self.__rep_num = rep_num
        self.__step_size = step_size
        self.__positive_flag = positive_flag
        self.__grad_f = grad_f
        self.__max_epochs = max_epochs
        self.__tol = convergence_tolerance
        self.__patience = patience
        self.__target_cost = target_cost

    def SGD(self, I_S_0):
        """
        执行随机梯度下降 (Stochastic Gradient Descent).
        """
        print(f"SGD启动: LR={self.__step_size}, MaxEpochs={self.__max_epochs}, TargetCost={self.__target_cost:.2e}")
        print("格式: (总成本, 持有成本, Epoch, 非零节点数, L2距离, 耐心计数)")

        I_S = I_S_0.copy()
        history = []
        patience_counter = 0
        l2_dist = 0.0
        
        t_start = time()

        for epoch in range(self.__max_epochs):
            # 1. 计算梯度和成本
            cost, grad, holding = self.__grad_f(I_S, self.__rep_num)
            
            # 2. 记录与打印
            nz_nodes = np.count_nonzero(np.floor(I_S))
            print(f"({cost:.3e}, {holding:.3e}, {epoch}, {nz_nodes}, {l2_dist:.3e}, {patience_counter})")
            history.append((cost, holding, epoch, nz_nodes))
            
            if self.__target_cost > 0 and cost < self.__target_cost:
                print(f"早停: 当前总成本 {cost:.3e} 已低于目标值 {self.__target_cost:.3e}。")
                break

            # 3. 更新参数
            I_S_prev = I_S.copy()
            I_S = I_S - self.__step_size * grad
            
            if self.__positive_flag:
                I_S = np.maximum(I_S, 0)

            # 4. 收敛性检查
            l2_dist = np.linalg.norm(I_S - I_S_prev)
            
            if l2_dist < self.__tol:
                patience_counter += 1
            else:
                patience_counter = 0

            if patience_counter >= self.__patience:
                print(f"早停: 参数更新已连续 {self.__patience} 次小于容差，且未达到目标成本。")
                break

        print(f"优化完成。耗时: {time() - t_start:.2f}s")
        
        # 保存历史
        timestamp = datetime.now().strftime('%Y-%m-%d %H-%M')
        dump_path = os.path.join(self.__data_path, f'history_SGD_{timestamp}.pkl')
        my_dump(history, dump_path)
        
        return I_S, history