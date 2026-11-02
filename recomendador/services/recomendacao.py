from functools import lru_cache
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, silhouette_score
)
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from recomendador.models import Livro, Aluno, Emprestimo


# =========================================================
# CARREGAMENTO EM MEMÓRIA (cacheado)
# =========================================================
@lru_cache(maxsize=1)
def _load_state():
    livros = list(Livro.objects.all())
    emprestimos = list(Emprestimo.objects.select_related('aluno', 'livro').all())

    livros_df = pd.DataFrame([{
        'id': l.id,
        'Livro': l.titulo,
        'Autor': l.autor,
        'Gênero': l.genero,
        'Resumo': l.resumo,
        'emprestimos': l.emprestimos,
    } for l in livros])

    if not emprestimos:
        emp_df = pd.DataFrame(columns=['Aluno', 'Livro'])
    else:
        emp_df = pd.DataFrame([{
            'Aluno': e.aluno.nome,
            'Livro': e.livro.titulo,
        } for e in emprestimos])

    # matriz aluno x livro
    if emp_df.empty:
        mat = pd.DataFrame()
    else:
        mat = emp_df.pivot_table(
            index='Aluno', columns='Livro', aggfunc='size', fill_value=0
        )
        for t in livros_df['Livro']:
            if t not in mat.columns:
                mat[t] = 0
        mat = mat[livros_df['Livro'].tolist()]

    # TF-IDF
    corpus = (
        livros_df['Gênero'].str.lower() + ' ' +
        livros_df['Autor'].str.lower() + ' ' +
        livros_df['Resumo'].str.lower()
    )
    vec = TfidfVectorizer(ngram_range=(1, 2))
    X = vec.fit_transform(corpus)

    # KMeans
    k = min(5, len(livros_df) - 1) if len(livros_df) > 1 else 1
    if k >= 2:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = km.fit_predict(X)
        sil = silhouette_score(X, clusters)
    else:
        clusters = np.zeros(len(livros_df), dtype=int)
        sil = 0.0
    livros_df['cluster'] = clusters

    return {
        'livros_df': livros_df,
        'mat': mat,
        'X': X,
        'clusters': clusters,
        'silhouette': float(sil),
        'k': int(k),
    }


def invalidar_cache():
    _load_state.cache_clear()


# =========================================================
# MODELO 1 — POPULARIDADE
# =========================================================
def recomendar_popularidade(n=5, excluir=None):
    excluir = excluir or set()
    df = _load_state()['livros_df']
    df = df[~df['Livro'].isin(excluir)]
    return df.sort_values('emprestimos', ascending=False).head(n)


# =========================================================
# MODELO 2 — KNN CONTEÚDO
# =========================================================
def recomendar_por_livro(titulo_alvo, n=5):
    s = _load_state()
    livros_df, X = s['livros_df'], s['X']
    if titulo_alvo not in livros_df['Livro'].values:
        return pd.DataFrame()
    idx = livros_df.index[livros_df['Livro'] == titulo_alvo][0]
    sims = cosine_similarity(X[idx], X).ravel()
    ordem = sims.argsort()[::-1]
    ordem = [i for i in ordem if i != idx][:n]
    rec = livros_df.iloc[ordem].copy()
    rec['similaridade'] = sims[ordem]
    return rec


def recomendar_por_historico(aluno, n=5):
    s = _load_state()
    livros_df, X, mat = s['livros_df'], s['X'], s['mat']
    if mat.empty or aluno not in mat.index:
        return pd.DataFrame()

    vetor = mat.loc[aluno].values
    if vetor.sum() == 0:
        return pd.DataFrame()

    # CORREÇÃO: Multiplicação correta (1x50) @ (50x670) = (1x670)
    perfil = vetor.reshape(1, -1).dot(X)
    
    # Agora ambos têm 670 colunas e a comparação funciona
    sims = cosine_similarity(perfil, X).ravel()
    
    lidos = mat.columns[vetor > 0].tolist()
    idx_lidos = {livros_df.index[livros_df['Livro'] == l][0] for l in lidos}

    ordem = sims.argsort()[::-1]
    ordem = [i for i in ordem if i not in idx_lidos][:n]
    rec = livros_df.iloc[ordem].copy()
    rec['similaridade'] = sims[ordem]
    return rec


# =========================================================
# MODELO 3 — KMEANS
# =========================================================
def recomendar_por_cluster(aluno, n=5):
    s = _load_state()
    livros_df, mat = s['livros_df'], s['mat']
    if mat.empty or aluno not in mat.index:
        return pd.DataFrame()

    lidos = mat.columns[mat.loc[aluno].values > 0].tolist()
    if not lidos:
        return pd.DataFrame()

    clusters_lidos = livros_df[livros_df['Livro'].isin(lidos)]['cluster'].tolist()
    if not clusters_lidos:
        return pd.DataFrame()

    cluster_fav = pd.Series(clusters_lidos).mode()[0]
    cand = livros_df[
        (livros_df['cluster'] == cluster_fav) &
        (~livros_df['Livro'].isin(lidos))
    ].sort_values('emprestimos', ascending=False)
    return cand.head(n)


# =========================================================
# MODELO 4 — ÁRVORE DE DECISÃO
# =========================================================
def _construir_features_aluno():
    s = _load_state()
    livros_df, mat = s['livros_df'], s['mat']
    if mat.empty:
        return pd.DataFrame()

    generos = livros_df.set_index('Livro')['Gênero'].to_dict()
    autores = livros_df.set_index('Livro')['Autor'].to_dict()

    rows = []
    for aluno in mat.index:
        lidos = mat.columns[mat.loc[aluno].values > 0].tolist()
        if not lidos:
            continue
        gs = [generos.get(l, 'Não informado') for l in lidos]
        aus = [autores.get(l, 'Desconhecido') for l in lidos]
        rows.append({
            'Aluno': aluno,
            'n_emprestimos': len(lidos),
            'n_generos_distintos': len(set(gs)),
            'n_autores_distintos': len(set(aus)),
            'genero_mais_frequente': pd.Series(gs).mode()[0],
        })
    return pd.DataFrame(rows)


def treinar_arvore():
    df = _construir_features_aluno()
    if df.empty or df['genero_mais_frequente'].nunique() < 2:
        return None, None, None

    le = LabelEncoder()
    y = le.fit_transform(df['genero_mais_frequente'])
    Xf = df[['n_emprestimos', 'n_generos_distintos', 'n_autores_distintos']].values

    try:
        Xtr, Xte, ytr, yte = train_test_split(
            Xf, y, test_size=0.3, random_state=42, stratify=y
        )
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(
            Xf, y, test_size=0.3, random_state=42
        )

    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)

    metricas = {
        'accuracy': float(accuracy_score(yte, ypred)),
        'precision_macro': float(precision_score(yte, ypred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(yte, ypred, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(yte, ypred, average='macro', zero_division=0)),
        'report': classification_report(yte, ypred, zero_division=0, output_dict=True),
    }
    return clf, le, metricas


def recomendar_por_arvore(aluno, n=5):
    s = _load_state()
    livros_df, mat = s['livros_df'], s['mat']
    clf, le, _ = treinar_arvore()
    if clf is None or mat.empty or aluno not in mat.index:
        return pd.DataFrame()

    lidos = mat.columns[mat.loc[aluno].values > 0].tolist()
    gs = livros_df.set_index('Livro').loc[lidos, 'Gênero'].tolist() if lidos else []
    aus = livros_df.set_index('Livro').loc[lidos, 'Autor'].tolist() if lidos else []

    feat = np.array([[len(lidos), len(set(gs)), len(set(aus))]])
    pred = le.inverse_transform(clf.predict(feat))[0]

    return livros_df[
        (livros_df['Gênero'] == pred) & (~livros_df['Livro'].isin(lidos))
    ].sort_values('emprestimos', ascending=False).head(n)


# =========================================================
# ALUNO NOVO (questionário)
# =========================================================
def recomendar_aluno_novo(generos, qtd_habito, pref_ext, tema, n=5):
    livros_df = _load_state()['livros_df'].copy()
    cand = livros_df.copy()

    if generos:
        mask = cand['Gênero'].apply(
            lambda g: any(gf.lower() in str(g).lower() for gf in generos)
        )
        cand = cand[mask]

    if tema and tema != '(nenhum)':
        cand = cand[cand['Gênero'].str.lower().str.contains(tema.lower(), na=False)]

    if cand.empty:
        cand = livros_df.copy()

    cand = cand.sort_values('emprestimos', ascending=(qtd_habito > 5))
    return cand.head(n)


def listar_alunos():
    return sorted(Aluno.objects.values_list('nome', flat=True))


def listar_generos():
    s = _load_state()
    return sorted(s['livros_df']['Gênero'].dropna().unique().tolist())


def historico_aluno(aluno):
    s = _load_state()
    if s['mat'].empty or aluno not in s['mat'].index:
        return []
    row = s['mat'].loc[aluno]
    return row[row > 0].index.tolist()


def diversidade(rec_df):
    if rec_df is None or rec_df.empty:
        return 0.0
    return rec_df['Gênero'].nunique() / len(rec_df)