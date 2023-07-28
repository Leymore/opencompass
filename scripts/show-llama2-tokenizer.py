from llama import Tokenizer

tokenizer = Tokenizer("/mnt/petrelfs/share_data/basemodel/checkpoints/llm/llama2/llama/tokenizer.model")
print(f"#words: {tokenizer.n_words} - BOS ID: {tokenizer.bos_id} - EOS ID: {tokenizer.eos_id}")
bos_str = tokenizer.decode([tokenizer.bos_id])
print(bos_str, tokenizer.encode(bos_str, bos=False, eos=False))
eos_str = tokenizer.decode([tokenizer.eos_id])
print(eos_str, tokenizer.encode(eos_str, bos=False, eos=False))
