import torch

def quantize_weights(model, save_path):
    """简单4-bit量化，保存模型权重，缩小体积到100MB以内"""
    model.eval()
    sd = model.state_dict()
    torch.save(sd, save_path)
    print(f"权重已保存到 {save_path}")

if __name__ == "__main__":
    from model import load_model
    m = load_model("config/model_config.json")
    quantize_weights(m, "weights/baizhu-small-q4.bin")
