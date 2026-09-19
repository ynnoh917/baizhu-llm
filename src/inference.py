import torch
from model import load_model
from tokenizer import BaizhuTokenizer

def generate(model, tokenizer, prompt, max_new_tokens=128, temperature=0.7):
    model.eval()
    tokens = [tokenizer.bos_id] + tokenizer.encode(prompt)
    tokens = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        cache = None
        for _ in range(max_new_tokens):
            logits, cache = model(tokens, cache)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            if next_id.item() == tokenizer.eos_id():
                break
            tokens = torch.cat([tokens, next_id], dim=-1)
    out_ids = tokens[0].tolist()
    return tokenizer.decode(out_ids)

def main():
    device = torch.device("cpu")
    print("Baizhu-LLM 白猪大模型 (CPU模式)")
    print("正在加载模型...")
    model = load_model("config/model_config.json", "weights/baizhu-small-q4.bin")
    model = model.to(device)
    tokenizer = BaizhuTokenizer("weights/tokenizer.model")
    print("加载完成！输入 quit 退出")
    while True:
        user_input = input("\n你：")
        if user_input.strip().lower() == "quit":
            break
        res = generate(model, tokenizer, user_input)
        print(f"白猪：{res}")

if __name__ == "__main__":
    main()
