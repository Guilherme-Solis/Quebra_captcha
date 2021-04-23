import cv2
import os
import numpy as np
import pickle
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from scrips.helpers import resize_to_fit


dados = []  # Onde vai ficar os dados -> As imagens
rotulos = []  # Onde vai ficar os Rótulos, as respostas
pasta_base_dados = 'BD_LN'

imagens = paths.list_images(pasta_base_dados)

for arquivo in imagens:
    # Vai separar os rótulos em " \ "(os.path.sep), [-2] pra pegar somente a Letra/Número q ta na pasta
    rotulo = arquivo.split(os.path.sep)[-2]
    imagem = cv2.imread(arquivo)  # Abrir a imagem
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)  # Transformar em tons de cinza

    # Padronizar imagem por 20x20
    imagem = resize_to_fit(imagem, 20, 20)

    # Adicionar uma Dimenção para que o Keras consiga ler a imagem
    imagem = np.expand_dims(imagem, axis=2)

    # Adicionar os Rótulos e dados da imagem
    rotulos.append(rotulo)
    dados.append(imagem)

dados = np.array(dados, dtype="float") / 255  # Transformar os dados em Array do Numpy
rotulos = np.array(rotulos)  # Transformar os rotulos em Array do Numpy

# Separação em dados de Treino (75%) e dados de teste (25%)
# X são as Imagens, São os dados, e o Y são as Respostas

(X_train, X_test, Y_train, Y_test) = train_test_split(dados, rotulos, test_size=0.25, random_state=0)

# Converte com o oneHot_encoding, para transformar as letras em números
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# Salvar o LabelBunarizer = A Inteligência Artificial já treinada
# Irá salvar a variável " lb ", em formato .dat para usar depois para descodificar

with open('Rotulos_modelo.dat', 'wb') as arquivo_pickle:
    pickle.dump(lb, arquivo_pickle)

# ________ Criar e treinar a IA _________

modelo = Sequential()

# Crair as camadas da rede Neural

modelo.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Criar a 2ª camada da Rede Neural

modelo.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Criar a 3ª camada da Rede Neural

modelo.add(Flatten())
modelo.add(Dense(500, activation="relu"))

# Camada Final
# O Dense é a quantidade de "Bolinhas"/os neuronios que vão passar os dados

# ______ O Dense de 34 equivale as 26 Letras mais os números da Base de dados _______
# Ou 26 que serão as 26 Letras do alfabeto

modelo.add(Dense(26, activation="softmax"))

# Compilar todas as camadas
modelo.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Treinar a IA
# epochs = quantidade de treinos, verbose=0(Não mostra nada), 1(Mostra o progresso)

# ___________ O batch_size é de acordo com a camada Final, as respostas __________

modelo.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=33, epochs=25, verbose=1)

# Salvar o modelo em um arquivo

modelo.save("modelo_treinado.hdf5")


