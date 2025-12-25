# SLAI_NLP
NLP_Final_Project: Machine Translation between Chinese and English
## Data Preprocess
### Data Cleaning
```bash
python ./script/data_preprocess.py \
--train_in /path/to/your/project/SLAI_NLP/data/raw/train_100k.jsonl \
--valid_in /path/to/your/project/SLAI_NLP/data/raw/valid.jsonl \
--test_in /path/to/your/project/SLAI_NLP/data/raw/test.jsonl \
--out_dir /path/to/your/project/SLAI_NLP/data/processed_data
```
### Tokenization & Vocabulary Construction

```bash
python ./script/tokenization_jieba_nltk.py \
  --clean_dir /path/to/your/project/SLAI_NLP/data/processed_data/clean \
  --out_dir /path/to/your/project/SLAI_NLP/data/processed_data \
  --min_freq_zh 2 --min_freq_en 2 \
  --max_vocab_zh 30000 --max_vocab_en 30000 \
  --max_tokens 256 \
  --len_mode filter \
  --ratio_min 0.2 --ratio_max 5.0
```
### Word Embedding Initialization
```bash
python ./script/init_embeddings.py \
  --vocab_json /data1/yinghao/slai/SLAI_NLP/data/processed_data/word_tok/vocab_en.json \
  --vec_path  pretrained/zh.vec \
  --out_dir   /path/to/your/project/SLAI_NLP/data/processed_data/word_tok \
  --out_name  emb_zh \
  --normalize
```
## RNN
train RNN
```bash
python RNN_train.py \
  --train_script script/train_rnn.py \
  --data_dir /mnt/afs/250010023/lyh_code/slai/SLAI_NLP/data/processed_data/word_tok \
  --vocab_zh /mnt/afs/250010023/lyh_code/slai/SLAI_NLP/data/processed_data/word_tok/vocab_zh.json \
  --vocab_en /mnt/afs/250010023/lyh_code/slai/SLAI_NLP/data/processed_data/word_tok/vocab_en.json \
  --save_root runs/rnn_sweep \
  --epochs 25 --batch_size 512 --lr 3e-4 \
  --skip_existing
```
Greedy BLEU
```bash
python RNN_inference.py \
  --save_root runs/rnn_sweep \
  --data_dir /mnt/afs/250010023/lyh_code/slai/SLAI_NLP/data/processed_data/word_tok \
  --vocab_zh /mnt/afs/250010023/lyh_code/slai/SLAI_NLP/data/processed_data/word_tok/vocab_zh.json \
  --vocab_en /mnt/afs/250010023/lyh_code/slai/SLAI_NLP/data/processed_data/word_tok/vocab_en.json \
  --clean_test_jsonl /mnt/afs/250010023/lyh_code/slai/SLAI_NLP/data/processed_data/clean/test.jsonl \
  --decode greedy \
  --device cuda \
  --skip_existing
```
beam BLEU
```bash
python RNN_inference.py \
  --save_root runs/rnn_sweep \
  --data_dir /mnt/afs/250010023/lyh_code/slai/SLAI_NLP/data/processed_data/word_tok \
  --vocab_zh /mnt/afs/250010023/lyh_code/slai/SLAI_NLP/data/processed_data/word_tok/vocab_zh.json \
  --vocab_en /mnt/afs/250010023/lyh_code/slai/SLAI_NLP/data/processed_data/word_tok/vocab_en.json \
  --clean_test_jsonl /mnt/afs/250010023/lyh_code/slai/SLAI_NLP/data/processed_data/clean/test.jsonl \
  --decode beam --beam_size 5 \
  --device cuda \
  --skip_existing
```
## Transformer
norm exp
```bash
python Transformer_train.py \
  --train_script scripts/train_transformer.py \
  --data_dir /mnt/afs/250010023/lyh_code/slai/SLAI_NLP/data/processed_data/word_tok \
  --vocab_zh /mnt/afs/250010023/lyh_code/slai/SLAI_NLP/data/processed_data/word_tok/vocab_zh.json \
  --vocab_en /mnt/afs/250010023/lyh_code/slai/SLAI_NLP/data/processed_data/word_tok/vocab_en.json \
  --save_root runs/transformer_sweep/norm \
  --pos_embs absolute relative \
  --norms layernorm rmsnorm \
  --scales 256,4,4,1024 \
  --batch_sizes 64 \
  --lrs 3e-4 \
  --epochs 10 \
  --device cuda \
  --skip_existing
```
batch / lr
```bash
python Transformer_train.py \
  --train_script scripts/train_transformer.py \
  --data_dir /mnt/afs/250010023/lyh_code/slai/SLAI_NLP/data/processed_data/word_tok \
  --vocab_zh /mnt/afs/250010023/lyh_code/slai/SLAI_NLP/data/processed_data/word_tok/vocab_zh.json \
  --vocab_en /mnt/afs/250010023/lyh_code/slai/SLAI_NLP/data/processed_data/word_tok/vocab_en.json \
  --save_root runs/transformer_sweep/hparam \
  --pos_embs absolute \
  --norms layernorm \
  --scales 256,4,4,1024 \
  --batch_sizes 32 64 128 \
  --lrs 1e-4 3e-4 1e-3 \
  --epochs 10 \
  --device cuda \
  --skip_existing
```
small/base/large-ish
```bash
python Transformer_train.py \
  --train_script scripts/train_transformer.py \
  --data_dir /mnt/afs/250010023/lyh_code/slai/SLAI_NLP/data/processed_data/word_tok \
  --vocab_zh /mnt/afs/250010023/lyh_code/slai/SLAI_NLP/data/processed_data/word_tok/vocab_zh.json \
  --vocab_en /mnt/afs/250010023/lyh_code/slai/SLAI_NLP/data/processed_data/word_tok/vocab_en.json \
  --save_root runs/transformer_sweep/scale \
  --pos_embs absolute \
  --norms layernorm \
  --scales 256,4,4,1024 512,8,6,2048 768,12,8,3072 \
  --batch_sizes 64 \
  --lrs 3e-4 \
  --epochs 10 \
  --device cuda \
  --skip_existing
```
BLEU：greedy
```bash
python Transformer_inference.py \
  --save_root runs/transformer_sweep/scale \
  --data_dir /mnt/afs/250010023/lyh_code/slai/SLAI_NLP/data/processed_data/word_tok \
  --vocab_zh /mnt/afs/250010023/lyh_code/slai/SLAI_NLP/data/processed_data/word_tok/vocab_zh.json \
  --vocab_en /mnt/afs/250010023/lyh_code/slai/SLAI_NLP/data/processed_data/word_tok/vocab_en.json \
  --clean_test_jsonl /mnt/afs/250010023/lyh_code/slai/SLAI_NLP/data/processed_data/clean/test.jsonl \
  --decode greedy \
  --device cuda \
  --skip_existing
```
## Transformer Pretrained T5
t5-small/base 
```bash
python T5_train.py \
  --train_script scripts/finetune_t5.py \
  --clean_dir /mnt/afs/250010023/lyh_code/slai/SLAI_NLP/data/processed_data/clean \
  --save_root runs/t5_sweep \
  --models t5-small t5-base \
  --batch_sizes 8 16 \
  --lrs 1e-4 3e-4 \
  --epochs 5 \
  --max_src_lens 128 \
  --max_tgt_lens 128 \
  --fp16 \
  --skip_existing
```
BLEU:greedy
```bash
python T5_inference.py \
  --save_root runs/t5_sweep \
  --clean_test_jsonl /mnt/afs/250010023/lyh_code/slai/SLAI_NLP/data/processed_data/clean/test.jsonl \
  --decode greedy \
  --device cuda \
  --skip_existing
```