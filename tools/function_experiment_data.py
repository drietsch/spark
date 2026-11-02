"""Pinned public WikiText splits and reproducible article-separated mask windows."""
import hashlib
import json
from pathlib import Path
import re


def hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def resolve_dataset(protocol):
    from huggingface_hub import HfApi
    info = HfApi(token=False).dataset_info(protocol['dataset'])
    if not re.fullmatch('[0-9a-f]{40}', info.sha): raise ValueError('dataset revision is not immutable')
    return info.sha


def load_articles(protocol, revision, split):
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as parquet
    prefix = protocol['dataset_config'] + '/' + split + '-'
    files = sorted(f for f in HfApi(token=False).list_repo_files(protocol['dataset'], repo_type='dataset', revision=revision)
                   if f.startswith(prefix) and f.endswith('.parquet'))
    if not files: raise ValueError('dataset has no parquet files for ' + split)
    from spark.function_ir import digest
    provenance = []; articles = []; title = None; content = []
    def flush():
        if title is not None and content:
            normalized = ' '.join(title.casefold().split())
            articles.append({'id': hashlib.sha256(normalized.encode()).hexdigest(), 'title': title, 'text': ''.join(content)})
    for filename in files:
        path = hf_hub_download(protocol['dataset'], filename, repo_type='dataset', revision=revision, token=False)
        provenance.append({'file': filename, 'sha256': digest(path)})
        for line in parquet.read_table(path, columns=['text'])['text'].to_pylist():
            heading = re.fullmatch(r'\s*=\s*([^=\n]+?)\s*=\s*', line or '')
            if heading:
                flush(); title = heading[1]; content = []
            elif line:
                if title is None: title = split + ':preamble'
                content.append(line)
    flush()
    if not articles: raise ValueError('no top-level articles found')
    return articles, provenance


def prepare_split(protocol, revision, split, tokenizer_directory, *, excluded_articles=(), excluded_windows=()):
    import numpy as np
    from tokenizers import Tokenizer
    directory = Path(tokenizer_directory)
    tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
    spec = json.loads((directory / 'spark_tokenizer.json').read_text())
    articles, source_files = load_articles(protocol, revision, split)
    all_ids = sorted({article['id'] for article in articles}); excluded_articles = set(excluded_articles)
    seen = set(excluded_windows); count = protocol['split_windows'][split]
    body_length = protocol['context_tokens'] - 2; candidates = []
    for article in articles:
        if article['id'] in excluded_articles: continue
        tokens = tokenizer.encode(article['text'], add_special_tokens=False).ids
        starts = list(range(0, len(tokens) - body_length + 1, body_length))
        rng = np.random.default_rng(int(hash_json([protocol['seed'], split, article['id']])[:16], 16))
        rng.shuffle(starts)
        windows = []
        for start in starts[:protocol['max_windows_per_article']]:
            ids = [spec['cls_id'], *tokens[start:start + body_length], spec['sep_id']]
            identity = hash_json(ids)
            if identity in seen: continue
            mask_rng = np.random.default_rng(int(hash_json([protocol['seed'], identity, 'masks'])[:16], 16))
            queries = sorted(int(i) for i in mask_rng.choice(np.arange(1, len(ids)-1), size=protocol['masks_per_window'], replace=False))
            labels = [ids[i] for i in queries]; masked = list(ids)
            for i in queries: masked[i] = spec['mask_id']
            windows.append({'id': identity, 'article': article['id'], 'title': article['title'], 'labels': labels,
                            'inputs': {'tokens': masked, 'mask': [1] * len(ids), 'positions': list(range(len(ids))), 'queries': queries}})
        if windows: candidates.append(windows)
    order_rng = np.random.default_rng(protocol['seed'] + {'train':0, 'validation':1, 'test':2}[split])
    order_rng.shuffle(candidates); result = []
    # Round-robin balances article contributions before taking extra windows.
    for index in range(protocol['max_windows_per_article']):
        for windows in candidates:
            if index >= len(windows): continue
            window = windows[index]
            if window['id'] in seen: continue
            result.append(window); seen.add(window['id'])
            if len(result) == count:
                return result, {'revision': revision, 'split': split, 'source_files': source_files,
                                'all_article_ids': all_ids, 'selected_articles': len({r['article'] for r in result}),
                                'windows_sha256': hash_json(result), 'windows': count}
    raise ValueError(f'not enough disjoint {split} windows: {len(result)} < {count}')
