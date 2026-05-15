import os
import numpy as np
from time import time

class SimOpt():
    """
    仿真优化器：执行 Adam 梯度下降。
    """
    def __init__(self, data_path, rep_num, step_size, positive_flag, grad_f, 
                 max_epochs):
        self.__data_path = data_path
        self.__rep_num = rep_num
        self.__step_size = step_size
        self.__positive_flag = positive_flag
        self.__grad_f = grad_f
        self.__max_epochs = max_epochs

    def Adam(self, I_S_0, beta1=0.9, beta2=0.999, epsilon=1e-8):
        is_vector_lr = isinstance(self.__step_size, np.ndarray)
        lr_info = "Vector (Node-Specific)" if is_vector_lr else self.__step_size
        print(f"Adam优化器启动: LR={lr_info}, MaxEpochs={self.__max_epochs}")
        
        I_S = I_S_0.copy()
        history = []
        
        m = np.zeros_like(I_S)
        v = np.zeros_like(I_S)
        t_step = 0
        t_start = time()

        checkpoint_dir = os.path.join(self.__data_path, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        for epoch in range(self.__max_epochs):
            t_step += 1
            
            # 1. 计算梯度和成本
            cost, grad, holding, actual_sl = self.__grad_f(
                I_S, self.__rep_num, mean_flag=True, epoch=epoch
            )
            
            nz_nodes = np.count_nonzero(np.floor(I_S))
            avg_sl = np.mean(actual_sl) if actual_sl.size > 0 else 0.0
            print(f"(总: {cost:.3e}, 持有: {holding:.3e}, Epoch: {epoch}, 非零: {nz_nodes}, 均SL: {avg_sl:.2%})")

            history.append((cost, holding, epoch, nz_nodes))

            # 2. Adam 参数更新
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t_step)
            v_hat = v / (1 - beta2 ** t_step)
            
            I_S = I_S - self.__step_size * m_hat / (np.sqrt(v_hat) + epsilon)
            
            if self.__positive_flag:
                I_S = np.maximum(I_S, 0)

            if (epoch + 1) % 20 == 0:
                ckpt_filename = f'I_S_epoch_{epoch + 1}.npy'
                ckpt_path = os.path.join(checkpoint_dir, ckpt_filename)
                np.save(ckpt_path, I_S)
                print(f"  [Checkpoint] 中间策略已保存至: {ckpt_path}")

        print(f"当前阶段优化完成。耗时: {time() - t_start:.2f}s")
        return I_S, history