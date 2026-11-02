import pandas as pd
from pathlib import Path
from django.conf import settings
from recomendador.models import Livro, Aluno, Emprestimo


def importar_excel(caminho: Path | None = None) -> dict:
    """Importa as 3 abas do Excel para o banco. Retorna contagens."""
    caminho = Path(caminho or settings.EXCEL_PATH)
    if not caminho.exists():
        raise FileNotFoundError(f'Arquivo não encontrado: {caminho}')

    livros_df = pd.read_excel(caminho, sheet_name='Livros')
    alunos_df = pd.read_excel(caminho, sheet_name='Alunos')
    emp_df = pd.read_excel(caminho, sheet_name='Empréstimo')

    # normalização
    livros_df['Livro'] = livros_df['Livro'].astype(str).str.strip()
    livros_df['Autor'] = livros_df['Autor'].fillna('Desconhecido').astype(str).str.strip()
    livros_df['Gênero'] = livros_df['Gênero'].fillna('Não informado').astype(str).str.strip()
    livros_df['Resumo'] = livros_df['Resumo'].fillna('').astype(str).str.strip()

    emp_df['Aluno'] = emp_df['Aluno'].astype(str).str.strip()
    emp_df['Livro'] = emp_df['Livro'].astype(str).str.strip()

    # ---------- LIVROS ----------
    # remove duplicatas no Excel mantendo o primeiro
    livros_df = livros_df.drop_duplicates(subset=['Livro'], keep='first')

    # popularidade
    pop = emp_df.groupby('Livro').size().rename('emprestimos').reset_index()
    livros_df = livros_df.merge(pop, on='Livro', how='left').fillna({'emprestimos': 0})
    livros_df['emprestimos'] = livros_df['emprestimos'].astype(int)

    Livro.objects.all().delete()
    Livro.objects.bulk_create([
        Livro(
            titulo=row['Livro'],
            autor=row['Autor'],
            genero=row['Gênero'],
            resumo=row['Resumo'],
            emprestimos=int(row['emprestimos']),
        )
        for _, row in livros_df.iterrows()
    ])

    # ---------- ALUNOS ----------
    Aluno.objects.all().delete()
    Aluno.objects.bulk_create([
        Aluno(nome=str(n).strip()) for n in alunos_df['Aluno'].dropna().unique()
    ])

    # ---------- EMPRÉSTIMOS ----------
    Emprestimo.objects.all().delete()
    mapa_aluno = {a.nome: a for a in Aluno.objects.all()}
    mapa_livro = {l.titulo: l for l in Livro.objects.all()}

    emprestimos = []
    for _, row in emp_df.iterrows():
        aluno = mapa_aluno.get(row['Aluno'])
        livro = mapa_livro.get(row['Livro'])
        if aluno and livro:
            emprestimos.append(Emprestimo(
                aluno=aluno,
                livro=livro,
                data_emprestimo=pd.to_datetime(row['Data Empréstimo']).date(),
                data_devolucao=pd.to_datetime(row['Data Devolução']).date(),
            ))
    Emprestimo.objects.bulk_create(emprestimos)

    return {
        'livros': Livro.objects.count(),
        'alunos': Aluno.objects.count(),
        'emprestimos': Emprestimo.objects.count(),
    }