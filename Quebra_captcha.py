from keras.models import load_model
from scrips.helpers import resize_to_fit
from imutils import paths
import numpy as np
import cv2
import pickle
from Tratamento_de_Captcha import Tratamento_Imagens


def quebrar_captcha():
    lista_dos_capthas = []
    # Importar o modelo de IA treinado e Importar o tradutor

    # ____ Abrirá o arquivo de tradutor com o pickle ____
    # _____ Alterar onde está pegando a base de DADOS onde pega o tradutor de rotulos

    with open("IA_BD_LN/rotulos_modelo.dat", "rb") as arquivo_tradutor:
        lb = pickle.load(arquivo_tradutor)

    # _____ ALterar a base de DADOS onde pega o modelo treinado
    modelo = load_model("IA_BD_LN/modelo_treinado.hdf5")

    # Usar o modelo para resolver os Captchas
    Tratamento_Imagens("Captchas_para_resolver", pasta_de_destino="Captchas_para_resolver")

    # _____ Tratar a imagem _____
    arquivos = list(paths.list_images("Captchas_para_resolver"))

    # Listar todos os arquivos dentro da pasta designada
    for arquivo in arquivos:
        imagem = cv2.imread(arquivo)
        imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)

        # em preto e branco
        _, imagem_preta = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY_INV)

        # encontrar os contornos de cada letra
        contornos, _ = cv2.findContours(imagem_preta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regiao_letras = []

        # filtrar os contornos que são realmente de letras
        for contorno in contornos:
            (x, y, largura, altura) = cv2.boundingRect(contorno)
            area = cv2.contourArea(contorno)
            if area > 115:
                regiao_letras.append((x, y, largura, altura))

        # Ordenar as letras--Não está organizando de forma correta -- O x do lambda é uma lista, não o eixo da imagem

        regiao_letras = sorted(regiao_letras, key=lambda x: x[0])

        # desenhar os contornos e separar as letras em arquivos individuais

        imagem_final = cv2.merge([imagem] * 3)

        palpites_do_modelo = []

        for retangulo in regiao_letras:
            x, y, largura, altura = retangulo
            imagem_letra = imagem[y - 2:y + altura + 2, x - 2:x + largura + 2]

            # _____ Pegar as letras e dar para a IA resolver _____
            # ___ Primero tratar a imagem para 20 pixels
            imagem_letra = resize_to_fit(imagem_letra, 20, 20)

            # Tratamento para o keras ler a imagem, para a IA
            # Irá adicionar duas dimenções na imagem = EX: (1, 0 - 255, 0 - 255, 1)

            imagem_letra = np.expand_dims(imagem_letra, axis=2)
            imagem_letra = np.expand_dims(imagem_letra, axis=0)

            # A resposta da IA

            letra_prevista = modelo.predict(imagem_letra)

            # Traduzindo a resposta

            letra_prevista = lb.inverse_transform(letra_prevista)[0]
            palpites_do_modelo.append(letra_prevista)

        # _____ Juntar as letras em 1 texto _____
        texto_captcha = "".join(palpites_do_modelo)
        lista_dos_capthas.append(texto_captcha)

    print('... Os Captchas foram resolvidos ...')
    i = 0
    for captcha in lista_dos_capthas:
        i += 1
        print(f'O {i}ª Captcha é : {captcha}')
    print('Obrigado por usar Quebra_captchas')
    print('\033[34m<3 Thank you <3 \033[m')


if __name__ == "__main__":
    quebrar_captcha()
