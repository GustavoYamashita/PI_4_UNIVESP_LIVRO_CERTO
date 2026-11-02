import numpy as np
import pandas as pd
from recomendador.services import recomendacao as rec
from sklearn.metrics.pairwise import cosine_similarity

def avaliar_modelo(fn_recomendar, k=5, n_usuarios=50):
    s = rec._load_state()
    mat = s['mat']
    if mat.empty:
        return {'precision@k': 0, 'recall@k': 0, 'hit_rate': 0, 'ndcg@k': 0, 'n_aval': 0}

    alunos = [a for a in mat.index if mat.loc[a].sum() >= 2]
    np.random.seed(42)
    if len(alunos) > n_usuarios:
        alunos = list(np.random.choice(alunos, n_usuarios, replace=False))

    hits, precs, recs, ndcgs = [], [], [], []

    for aluno in alunos:
        linha = mat.loc[aluno]
        lidos = linha[linha > 0].index.tolist()
        if len(lidos) < 2:
            continue
        alvo = lidos[-1]

        mat_tmp = mat.copy()
        mat_tmp.loc[aluno, alvo] = 0

        # Substituição temporária do estado para simular a remoção do livro
        orig = rec._load_state
        state = orig()
        new_state = {**state, 'mat': mat_tmp}

        def fake_load():
            return new_state
        rec._load_state = fake_load

        try:
            df = fn_recomendar(aluno, k)
        finally:
            rec._load_state = orig

        if df is None or df.empty:
            continue

        top = df['Livro'].tolist()[:k]
        hit = 1 if alvo in top else 0
        hits.append(hit)
        precs.append(hit / k)
        recs.append(float(hit))
        if hit:
            pos = top.index(alvo) + 1
            ndcgs.append(1 / np.log2(pos + 1))
        else:
            ndcgs.append(0.0)

    if not hits:
        return {'precision@k': 0, 'recall@k': 0, 'hit_rate': 0, 'ndcg@k': 0, 'n_aval': 0}

    return {
        'precision@k': float(np.mean(precs)),
        'recall@k': float(np.mean(recs)),
        'hit_rate': float(np.mean(hits)),
        'ndcg@k': float(np.mean(ndcgs)),
        'n_aval': len(hits),
    }

def avaliar_todos(k=5, n_usuarios=50):
    def rec_pop(aluno, kk):
        lidos = set(rec.historico_aluno(aluno))
        return rec.recomendar_popularidade(n=kk, excluir=lidos)

    def rec_knn(aluno, kk):
        return rec.recomendar_por_historico(aluno, n=kk)

    def rec_km(aluno, kk):
        return rec.recomendar_por_cluster(aluno, n=kk)

    def rec_arv(aluno, kk):
        return rec.recomendar_por_arvore(aluno, n=kk)

    resultados = {
        'Popularidade': avaliar_modelo(rec_pop, k, n_usuarios),
        'KNN (conteúdo)': avaliar_modelo(rec_knn, k, n_usuarios),
        'KMeans (cluster)': avaliar_modelo(rec_km, k, n_usuarios),
        'Árvore de decisão': avaliar_modelo(rec_arv, k, n_usuarios),
    }
    return pd.DataFrame(resultados).T