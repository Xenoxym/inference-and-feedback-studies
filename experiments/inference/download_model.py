import pathlib,urllib.request,hashlib,json
ROOT=pathlib.Path(__file__).resolve().parents[2]
REV='93efa2f097d58c2a74874c7e644dbc9b0cee75a2'
FILES=['README.md','config.json','generation_config.json','merges.txt','model.safetensors','special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.json']
d=ROOT/'sources/SmolLM2-135M';d.mkdir(parents=True,exist_ok=True)
for f in FILES:
 p=d/f
 if not p.exists():urllib.request.urlretrieve(f'https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/{REV}/{f}',p)
 print(f,p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest())
assert hashlib.sha256((d/'model.safetensors').read_bytes()).hexdigest()=='80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1'
(d/'revision.txt').write_text(REV)
