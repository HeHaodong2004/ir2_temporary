# test_std.py

import os
import torch
import numpy as np
import pandas as pd
from model import PolicyNet
from multi_agent_worker import MultiAgentWorker
from parameter import (
    model_path,
    NODE_INPUT_DIM, EMBEDDING_DIM,
    USE_GPU,
    TEST_EPISODES  # 确保 parameter.py 里有: TEST_EPISODES = 50
)

def load_policy_model(checkpoint_path, device):
    """
    加载保存的 policy 模型。
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
    net.load_state_dict(checkpoint['policy_model'])
    net.to(device)
    net.eval()
    return net

def main():
    device = torch.device('cuda') if USE_GPU else torch.device('cpu')

    ckpt_file = os.path.join(model_path, 'checkpoint.pth')
    if not os.path.isfile(ckpt_file):
        raise FileNotFoundError(f"找不到模型文件: {ckpt_file}")

    # 1) 加载模型
    policy_net = load_policy_model(ckpt_file, device)

    # 2) 准备存放指标
    metrics = {
        'travel_dist':      [],  # 团队总距离
        'explored_rate':    [],
        'success_rate':     [],
        'map_area_std':     [],  # 各 agent 已知栅格数的标准差
        'travel_dist_std':  []   # 各 agent 行驶距离的标准差
    }

    # 3) 多次测试
    for ep in range(TEST_EPISODES):
        worker = MultiAgentWorker(
            meta_agent_id=0,
            policy_net=policy_net,
            global_step=ep,
            device=device,
            save_image=False
        )
        worker.run_episode()

        # 3.1) 团队指标
        perf = worker.perf_metrics
        metrics['travel_dist'].append(perf['travel_dist'])
        metrics['explored_rate'].append(perf['explored_rate'])
        metrics['success_rate'].append(perf['success_rate'])

        # 3.2) map_area_std：每个 agent 已知栅格数的 std
        known_counts = [
            np.sum(worker.env.get_agent_map(i).map != -1)
            for i in range(worker.n_agent)
        ]
        metrics['map_area_std'].append(np.std(known_counts))

        # 3.3) travel_dist_std：每个 agent 的 travel_dist 属性的 std
        travels = [agent.travel_dist for agent in worker.robot_list]
        metrics['travel_dist_std'].append(np.std(travels))

    # 4) 汇总并打印
    print(f"\n测试完成，共 {TEST_EPISODES} 个 episode，平均性能：")
    for k, v in metrics.items():
        arr = np.array(v)
        print(f"  {k:13s}: mean={arr.mean():.4f}, std={arr.std():.4f}")

    # 5) 保存到 CSV（可选）
    df = pd.DataFrame(metrics)
    df.index.name = 'episode'
    csv_path = os.path.join(model_path, 'test_metrics.csv')
    df.to_csv(csv_path)
    print(f"\n已保存每个 episode 的指标到: {csv_path}")

if __name__ == '__main__':
    main()
