import pickle


def build_library(model, device, x_test, size=50000):
    model.eval()
    library = []
    seen_hashes = set()  # 用于去重

    print(f"正在构建包含 {size} 个组件的库...")

    while len(library) < size:
        # 生成稍深一点的树（深度2-3），增加库的多样性
        tree, y_val = generate_safe_tree(depth=np.random.randint(1, 4), x_samples=x_test)

        if tree is None: continue

        # 简单的去重逻辑（基于公式字符串）
        tree_str = str(tree)  # 假设你 Node 类有 __str__
        if tree_str in seen_hashes: continue
        seen_hashes.add(tree_str)

        with torch.no_grad():
            v_h, _ = model.encoder(tree)
            # 存储：公式结构、对应的向量、以及数值响应（备用）
            library.append({
                'structure': tree,
                'vector': v_h.cpu().numpy(),
                'y_response': y_val
            })

        if len(library) % 5000 == 0:
            print(f"已收集 {len(library)} 个组件...")

    with open("library.pkl", "wb") as f:
        pickle.dump(library, f)
    print("库已保存至 library.pkl")

# 在 main.py 训练结束后调用
# build_library(model, device, x_test)