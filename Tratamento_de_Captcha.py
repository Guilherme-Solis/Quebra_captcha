import cv2
import os
import glob
from PIL import Image


def Tratamento_Imagens(pasta_de_origem, pasta_de_destino="Captchas_para_tratar"):
    # Primero tratamento: Retirar os ricos e defeitos

    arquivos = glob.glob(f"{pasta_de_origem}/*")  # Pegará todos os arquivos da pasta de origem

    for arquivo in arquivos:  # Fará uma varedura na pasta de origem
        imagem = cv2.imread(arquivo)  # Abrirá a imagem
        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
        # Colocará em tons de cinza, o "_" é um valor inutilizavel da tupla retornavel

        _, imagem_tratada = cv2.threshold(imagem_cinza, 127, 125,
                                          cv2.THRESH_TRUNC or cv2.THRESH_OTSU)  # Aplicará o método
        nome_arquivo = os.path.basename(arquivo)  # O nome do arquivo que foi pego será o nome do arquivo final
        cv2.imwrite(f'{pasta_de_destino}/{nome_arquivo}', imagem_tratada)  # Salvara a imagem no pasta Semifinal

    # Segundo Tratamento: Reprodução da imagem em Preto e branco

    arquivos = glob.glob(f"{pasta_de_destino}/*")

    for arquivo in arquivos:
        imagem = Image.open(arquivo)
        imagem = imagem.convert("P")  # Transforma em tons de cinza
        imagem_copy = Image.new("P", imagem.size, 255)

        # Imagens RGB, quanto mais perto do 0:Mais escura, quanto mais perto do 255: Mais clara

        for x in range(imagem.size[1]):  # Pra cada pixel na largura/ comprimento da imagem
            for y in range(imagem.size[0]):  # Pra cada pixel na altura da imagem
                cor_pixel = imagem.getpixel((y, x))  # Localização do pixel da imagem
                if cor_pixel < 115:  # Se a cor do pixel for menor/mais escura
                    imagem_copy.putpixel((y, x), 0)  # Transforma em preto o pixel

        nome_arquivo = os.path.basename(arquivo)
        imagem_copy.save(f"{pasta_de_destino}/{nome_arquivo}")  # Salvando a imagem


# Só será reproduzido a função se estiver executando o arquivo, não será executado em importações


# Tratamento_Imagens('bdcaptcha')

if __name__ == '__main__':
    Tratamento_Imagens('Captchas_para_tratar')
