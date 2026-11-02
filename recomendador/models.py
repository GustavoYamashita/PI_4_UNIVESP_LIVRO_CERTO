from django.db import models


class Livro(models.Model):
    titulo = models.CharField(max_length=255, unique=True)
    autor = models.CharField(max_length=255, blank=True, default='Desconhecido')
    genero = models.CharField(max_length=255, blank=True, default='Não informado')
    resumo = models.TextField(blank=True, default='')
    emprestimos = models.IntegerField(default=0)
    cluster = models.IntegerField(null=True, blank=True)

    class Meta:
        ordering = ['-emprestimos', 'titulo']
        verbose_name = 'Livro'
        verbose_name_plural = 'Livros'

    def __str__(self):
        return self.titulo


class Aluno(models.Model):
    nome = models.CharField(max_length=100, unique=True)

    class Meta:
        ordering = ['nome']

    def __str__(self):
        return self.nome


class Emprestimo(models.Model):
    aluno = models.ForeignKey(Aluno, on_delete=models.CASCADE, related_name='emprestimos')
    livro = models.ForeignKey(Livro, on_delete=models.CASCADE, related_name='historico_emprestimos')
    data_emprestimo = models.DateField()
    data_devolucao = models.DateField()

    class Meta:
        ordering = ['-data_emprestimo']

    def __str__(self):
        return f'{self.aluno} → {self.livro}'