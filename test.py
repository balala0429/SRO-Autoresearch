from main import get_probing_points, NUM_POINTS
from tool.Node import Node
import torch.nn.functional as F
import torch
from tool.OP_TO_ID import OP_TO_ID
from tool.SymbolicAutoencoder import SymbolicAutoencoder


def run_equivalence_test(model, device, op_to_id):
    model.eval()
    test_cases = [
        ("等价性 (x+x vs 2*x)",
         Node('+', left=Node('x'), right=Node('x')),
         Node('*', left=Node('const', value=2.0), right=Node('x'))),

        ("微小差异 (sin(x) vs sin(x)+0.01)",
         Node('sin', left=Node('x')),
         Node('+', left=Node('sin', left=Node('x')), right=Node('const', value=0.01))),

        ("完全不同 (exp(x) vs sin(x))",
         Node('exp', left=Node('x')),
         Node('sin', left=Node('x')))
    ]

    print("\n" + "=" * 50)
    print("      Encoder 语义等价性测试 (验证结果)      ")
    print("=" * 50)

    with torch.no_grad():
        for description, tree_a, tree_b in test_cases:
            # 这里的 .encoder(tree_a) 内部会递归处理
            v_a, _ = model.encoder(tree_a)
            v_b, _ = model.encoder(tree_b)

            similarity = F.cosine_similarity(v_a, v_b).item()
            dist = torch.dist(v_a, v_b).item()

            print(f"项目: {description}")
            print(f" -> 相似度: {similarity:.6f} (越接近1越好)")
            print(f" -> 距离:   {dist:.6f}")
            print("-" * 30)


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_test = get_probing_points(NUM_POINTS)
    actual_points = len(x_test)

    # 2. 实例化模型并加载权重
    model = SymbolicAutoencoder(
        vocab_size=len(OP_TO_ID),
        embed_dim=128,
        num_points=actual_points,
        device=device
    )

    # 确保权重文件路径正确，并移动到对应设备
    try:
        model.load_state_dict(torch.load("D:\yjs\SRO_PRO\weight1\encoder_stage1_final.pth", map_location=device))
        model.to(device)
        print("Successfully loaded model weights from epoch 60.")
    except FileNotFoundError:
        print("Error: Weight file not found. Check the path.")
        exit()

    # 3. 执行等价性测试
    # 注意：函数定义要在调用之前（或者放在文件上方）
    run_equivalence_test(model, device, OP_TO_ID)