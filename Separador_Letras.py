import os
from glob import glob
import cv2

arquivos = glob("Captchas_para_resolver/*")

for arquivo in arquivos:

    imagem = cv2.imread(arquivo)
    imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    _, imagem_preta = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY_INV)  # Transforma a imagem em Preto e Braco

    # Encontra os contornos das letras
    contornos, _ = cv2.findContours(imagem_preta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Uma lista com as letras contornadas // Tomar cuidado por que ele pode conciderar outro espaço como uma letra
    lista_de_letras = []

    # Filtra as letras dos riscos/defeitos da imagem, e pega as informações do contorno

    for letra_contornada in contornos:

        # Irá pegar as informações do contorno feito na letra
        (x, y, largura, altura) = cv2.boundingRect(letra_contornada)

        # Calculará a área do contorno
        area = cv2.contourArea(letra_contornada)
        if area > 115:
            lista_de_letras.append((x, y, largura, altura))

    if len(lista_de_letras) != 5:
        continue

    # Não realiza o resto do FOR, e pula para a próxima imagem # ____ MUDAR O Nº DE CARACTERES ____

    # if len(lista_de_letras) != 4:
    #    continue

    # Irá realizar o contorno e separar os arquivos em imagens de Letras

    # Transformar a imagem em RGB

    # Não é tão importante, mas salva a imagem com os contornos
    imagem_final = cv2.merge([imagem] * 3)

    indice = 0
    for retangulo in lista_de_letras:
        x, y, largura, altura = retangulo
        # Coloca-se -5 para não cortar a imagem da letra muito colada a letra, dar uma espaço
        # Borda da imagem
        # EX: y-5:y+altura+5, y-5 é a posição inicial de Y, e y+altura+5 é a posição final de Y das cordenadas, para fazer o contorno

        imagem_da_letra = imagem[y - 5:y + altura + 5,
                          x - 5:x + largura + 5]  # As imagens no python funcionam de jeito que primero mostra o Y depois o X

        indice += 1
        nome_arquivo = os.path.basename(arquivo).replace(".png", f"numero{indice}.png")
        # Salvar a imagem
        cv2.imwrite(f"Letras/{nome_arquivo}", imagem_da_letra)

        # Irá criar uma imagem com os retângulos das letras
        # Parâmentros: A imagem // Posição Inicial das cordenadas da letra // Posição Final // A cor // A espessura

        cv2.rectangle(imagem_final, (x - 5, y - 5), (x + largura + 5, y + altura + 5), (0, 255, 0), 1)

    # Cria as imagens com contorno das letras

    nome_arquivo = os.path.basename(arquivo)
    cv2.imwrite(f"Imagem_contornada/{nome_arquivo}", imagem_final)
