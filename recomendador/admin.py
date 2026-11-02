from django.contrib import admin
from .models import Livro, Aluno, Emprestimo


@admin.register(Livro)
class LivroAdmin(admin.ModelAdmin):
    list_display = ('titulo', 'autor', 'genero', 'emprestimos', 'cluster')
    search_fields = ('titulo', 'autor', 'genero')
    list_filter = ('genero',)


@admin.register(Aluno)
class AlunoAdmin(admin.ModelAdmin):
    list_display = ('nome',)
    search_fields = ('nome',)


@admin.register(Emprestimo)
class EmprestimoAdmin(admin.ModelAdmin):
    list_display = ('aluno', 'livro', 'data_emprestimo', 'data_devolucao')
    list_filter = ('aluno', 'livro')
    search_fields = ('aluno__nome', 'livro__titulo')