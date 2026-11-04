"""Evaluate the original SBERT-NLI-base model on BIOSSES (100 biomedical pairs)."""
import argparse
import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='sentence-transformers/bert-base-nli-mean-tokens')
    parser.add_argument('--output', default='biosses_sbert_results.csv')
    args = parser.parse_args()

    print('Loading BIOSSES from tabilab/biosses ...')
    ds = load_dataset('tabilab/biosses', split='train')
    df = ds.to_pandas()[['sentence1', 'sentence2', 'score']].copy()
    df['score'] = pd.to_numeric(df['score'], errors='raise')
    if df[['sentence1','sentence2','score']].isna().any().any():
        raise ValueError('BIOSSES contains missing values')
    if not df['score'].between(0, 4).all():
        raise ValueError('Unexpected BIOSSES score range')
    print(f'Pairs: {len(df)}; score range: {df.score.min():g}–{df.score.max():g}')
    if len(df) != 100:
        print('WARNING: expected 100 pairs; check dataset version')

    print(f'Loading model: {args.model}')
    model = SentenceTransformer(args.model)
    print('Encoding BIOSSES sentence pairs ...')
    left = model.encode(df['sentence1'].tolist(), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    right = model.encode(df['sentence2'].tolist(), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    df['sbert_cosine'] = np.sum(left * right, axis=1)
    result = spearmanr(df['score'].to_numpy(), df['sbert_cosine'].to_numpy())
    print('\n=== BIOSSES RESULT ===')
    print(f'Number of pairs: {len(df)}')
    print(f'Spearman correlation: {result.statistic:.6f}')
    print(f'Spearman x 100: {result.statistic * 100:.3f}')
    print(f'p-value: {result.pvalue:.6g}')
    df.insert(0, 'id', range(1, len(df) + 1))
    df.to_csv(args.output, index=False)
    print(f'Saved pair-level results: {args.output}')


if __name__ == '__main__':
    main()
