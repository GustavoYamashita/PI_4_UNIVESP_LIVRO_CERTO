from django.urls import path
from . import views

app_name = 'recomendador'

urlpatterns = [
    path('', views.home, name='home'),
    path('aluno-existente/', views.aluno_existente, name='aluno_existente'),
    path('aluno-novo/', views.aluno_novo, name='aluno_novo'),
    path('metricas/', views.metricas, name='metricas'),
    path('dados/', views.dados, name='dados'),
    path('recarregar/', views.recarregar, name='recarregar'),
]