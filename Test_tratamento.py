import cv2
from PIL import Image


metodos = [
    cv2.THRESH_BINARY,
    cv2.THRESH_BINARY_INV,
    cv2.THRESH_TRUNC, # Vamos usar esse método junto ao OTSU
    cv2.THRESH_TOZERO,
    cv2.THRESH_TOZERO_INV

]

imagem = cv2.imread("com_numeros/numeros_0.png") # Abre a imagem no cv2

# Transformar a imagem em escala de cinza

imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY) # Transforma em tons de cinza

i = 0
for metodo in metodos: # Vai tratar a imagem original com o métodos propostos na lista de Metodos
    i += 1
    _, imagem_tratada = cv2.threshold(imagem_cinza, 127, 125, metodo or cv2.THRESH_OTSU)
    cv2.imwrite(f'Teste_metodos/Imagem_tratada_{i}.png', imagem_tratada)

# Usando O Pillow para modificar a imagem

imagem = Image.open("Teste_metodos/Imagem_tratada_3.png")
imagem = imagem.convert("P") # Transforma em tons de cinza
imagem_copy = Image.new("P", imagem.size, 255)

for x in range(imagem.size[1]): # Pra cada pixel na largura/ comprimento da imagem
    for y in range(imagem.size[0]): # Pra cada pixel na altura da imagem
        cor_pixel = imagem.getpixel((y, x)) # Localização do pixel da imagem
        if cor_pixel < 115: # Se a cor do pixel for menor/mais escura
            imagem_copy.putpixel((y, x), 0) # Transforma em preto o pixel

imagem_copy.save("Teste_metodos/imagem_tratada.png") # Salvando a imagem
