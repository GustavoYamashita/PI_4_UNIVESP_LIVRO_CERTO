import pandas as pd
from django.shortcuts import render
from recomendador.forms import AlunoExistenteForm, AlunoNovoForm
from recomendador.services import recomendacao as rec
from recomendador.services import avaliacao
from recomendador.models import Livro, Aluno, Emprestimo


def home(request):
    ctx = {
        'n_livros': Livro.objects.count(),
        'n_alunos': Aluno.objects.count(),
        'n_emprestimos': Emprestimo.objects.count(),
        'top10': Livro.objects.order_by('-emprestimos')[:10],
    }
    return render(request, 'recomendador/home.html', ctx)


def aluno_existente(request):
    form = AlunoExistenteForm(request.POST or None)
    historico, resultado, diversidade, modelo_usado = [], None, None, None

    if request.method == 'POST' and form.is_valid():
        aluno = form.cleaned_data['aluno']
        modelo = form.cleaned_data['modelo']
        k = form.cleaned_data['k']

        historico = rec.historico_aluno(aluno)

        if modelo == 'knn':
            df = rec.recomendar_por_historico(aluno, n=k)
        elif modelo == 'cluster':
            df = rec.recomendar_por_cluster(aluno, n=k)
        elif modelo == 'arvore':
            df = rec.recomendar_por_arvore(aluno, n=k)
        else:
            df = rec.recomendar_popularidade(n=k, excluir=set(historico))

        if df is not None and not df.empty:
            resultado = df[[
                c for c in ['Livro', 'Autor', 'Gênero', 'emprestimos', 'similaridade']
                if c in df.columns
            ]].to_dict('records')
            diversidade = rec.diversidade(df)
        modelo_usado = dict(form.fields['modelo'].choices)[modelo]

    return render(request, 'recomendador/aluno_existente.html', {
        'form': form,
        'historico': historico,
        'resultado': resultado,
        'diversidade': diversidade,
        'modelo_usado': modelo_usado,
    })


def aluno_novo(request):
    form = AlunoNovoForm(request.POST or None)
    resultado = None
    populares = None

    if request.method == 'POST' and form.is_valid():
        cd = form.cleaned_data
        df = rec.recomendar_aluno_novo(
            generos=cd['generos'],
            qtd_habito=cd['qtd_habito'],
            pref_ext=cd['pref_ext'],
            tema=cd['tema'],
            n=cd['k'],
        )
        if df is not None and not df.empty:
            resultado = df[['Livro', 'Autor', 'Gênero', 'emprestimos', 'Resumo']].to_dict('records')

        pop = rec.recomendar_popularidade(n=cd['k'])
        populares = pop[['Livro', 'Autor', 'Gênero', 'emprestimos']].to_dict('records')

    return render(request, 'recomendador/aluno_novo.html', {
        'form': form,
        'resultado': resultado,
        'populares': populares,
    })


def metricas(request):
    k = int(request.GET.get('k', 5))
    n_us = int(request.GET.get('n', 50))
    tabela, arvore_metricas, kmeans_info = None, None, None

    if request.GET.get('rodar') == '1':
        df = avaliacao.avaliar_todos(k=k, n_usuarios=n_us)
        tabela = []
        for modelo, row in df.iterrows():
            tabela.append({
                'modelo': modelo,
                'precision': round(row['precision@k'], 3),
                'recall': round(row['recall@k'], 3),
                'hit_rate': round(row['hit_rate'], 3),
                'ndcg': round(row['ndcg@k'], 3),
                'n': int(row['n_aval']),
            })

        _, _, arvore_metricas = rec.treinar_arvore()
        s = rec._load_state()
        kmeans_info = {'k': s['k'], 'silhouette': round(s['silhouette'], 3),
                       'n_livros': len(s['livros_df'])}

    return render(request, 'recomendador/metricas.html', {
        'k': k, 'n': n_us,
        'tabela': tabela,
        'arvore': arvore_metricas,
        'kmeans': kmeans_info,
    })


def dados(request):
    s = rec._load_state()
    livros_df = s['livros_df'].sort_values(['cluster', 'emprestimos'],
                                           ascending=[True, False])
    cluster_list = livros_df[['Livro', 'Autor', 'Gênero', 'emprestimos', 'cluster']].to_dict('records')

    top10 = livros_df.sort_values('emprestimos', ascending=False).head(10)
    top10_list = top10[['Livro', 'Autor', 'Gênero', 'emprestimos']].to_dict('records')

    # empréstimos por gênero
    emp = Emprestimo.objects.select_related('livro').all()
    cont = {}
    for e in emp:
        cont[e.livro.genero] = cont.get(e.livro.genero, 0) + 1
    generos = sorted(cont.items(), key=lambda x: -x[1])

    return render(request, 'recomendador/dados.html', {
        'clusters': cluster_list,
        'top10': top10_list,
        'generos': generos,
        'silhouette': round(s['silhouette'], 3),
        'k': s['k'],
    })


def recarregar(request):
    rec.invalidar_cache()
    from django.shortcuts import redirect
    return redirect('recomendador:home')