import torch
import clip
import seaborn as sns
import matplotlib.pyplot as plt

def check_prompt_similarity():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    
    # 以 "dog" 为例
    class_name = "dog"
    templates = [
        f"elementary visual features, edges, and textures of a {class_name}", # Stage 1
        f"visual patterns, parts, and components of a {class_name}",        # Stage 2
        f"the visual shape, geometry, and structure of a {class_name}",      # Stage 3
        f"a photo of a {class_name}"                                         # Stage 4
    ]
    
    print("Prompts being tested:")
    for t in templates:
        print(f"- {t}")

    text_inputs = clip.tokenize(templates).to(device)
    
    with torch.no_grad():
        # 提取特征并归一化
        feats = model.encode_text(text_inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        
        # 计算相似度矩阵 (4x4)
        similarity_matrix = feats @ feats.T
        
    print("\nSimilarity Matrix:")
    print(similarity_matrix)
    
    # 简单的判断逻辑
    sim_1_4 = similarity_matrix[0, 3].item()
    print(f"\nSimilarity between Stage 1 (Texture) and Stage 4 (Semantic): {sim_1_4:.4f}")
    
    if sim_1_4 > 0.95:
        print("⚠️ 警告：CLIP 认为这两个 Prompt 几乎一样！分层对齐可能失效。")
        print("建议：需要大幅修改 Prompt，或者放弃分层。")
    elif sim_1_4 < 0.85:
        print("✅ 通过：CLIP 能够区分底层特征和高层语义。Prompt 设计有效。")
    else:
        print("😐 中规中矩：有一定的区分度，但重叠较高。")

check_prompt_similarity()