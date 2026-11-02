from django.core.management.base import BaseCommand
from recomendador.services.loader import importar_excel
from recomendador.services.recomendacao import invalidar_cache


class Command(BaseCommand):
    help = 'Importa dados do Excel para o banco'

    def add_arguments(self, parser):
        parser.add_argument('--caminho', type=str, default=None)

    def handle(self, *args, **options):
        res = importar_excel(options['caminho'])
        invalidar_cache()
        self.stdout.write(self.style.SUCCESS(
            f"Importado: {res['livros']} livros, "
            f"{res['alunos']} alunos, {res['emprestimos']} empréstimos."
        ))