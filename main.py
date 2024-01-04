import numpy as np
from rbm import RBM

rbm = RBM(num_visible=6, num_hidden=2)

base = np.array([[1,1,0,0,0,0],
                [0,0,0,1,0,1],
                [1,0,1,0,0,0],
                [0,0,1,0,0,0],
                [0,0,0,0,1,1],
                [0,0,0,1,0,1]])

filmes = ['A Bruxa', 'Invocação do Mal', 'O Chamado', 'Se Beber não Case', 'Gente Grande', 'American Pie']

rbm.train(base, max_epochs=5000)

usuario = np.array([[0,1,1,0,0,0]])

camada_oculta = rbm.run_visible(usuario)
recomendacao = rbm.run_hidden(camada_oculta)

print(camada_oculta)
print(recomendacao)

for i in range(len(usuario[0])):
    if usuario[0,i] == 0 and recomendacao[0,i] == 1:
        print(filmes[i])

