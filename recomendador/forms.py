from django import forms
from recomendador.services.recomendacao import listar_generos, listar_alunos


class AlunoExistenteForm(forms.Form):
    aluno = forms.ChoiceField(label='Aluno')
    modelo = forms.ChoiceField(
        label='Modelo',
        choices=[
            ('knn', 'KNN (conteúdo)'),
            ('cluster', 'KMeans (cluster)'),
            ('arvore', 'Árvore de decisão'),
            ('popularidade', 'Popularidade'),
        ],
        initial='knn',
    )
    k = forms.IntegerField(label='Quantidade (K)', min_value=3, max_value=10, initial=5)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['aluno'].choices = [(a, a) for a in listar_alunos()]


class AlunoNovoForm(forms.Form):
    generos = forms.MultipleChoiceField(
        label='Quais gêneros você mais gosta?',
        widget=forms.CheckboxSelectMultiple,
        required=False,
    )
    qtd_habito = forms.IntegerField(
        label='Quantos livros lê por mês?', min_value=0, max_value=10, initial=2
    )
    pref_ext = forms.ChoiceField(
        label='Preferência',
        choices=[
            ('curto', 'Livros curtos e rápidos'),
            ('medio', 'Livros médios'),
            ('longo', 'Livros longos e densos'),
        ],
        initial='medio',
    )
    tema = forms.CharField(label='Tema de interesse (opcional)', required=False, max_length=100)
    k = forms.IntegerField(label='Quantidade (K)', min_value=3, max_value=10, initial=5)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['generos'].choices = [(g, g) for g in listar_generos()]