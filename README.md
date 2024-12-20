# dl-ImageNet-ResNet50-12-24
Este código demonstra como utilizar um modelo de deep learning pré-treinado para classificar imagens com alta precisão. O ResNet50 é conhecido por seu excelente desempenho em tarefas de classificação de imagens, e este exemplo mostra sua capacidade de identificar corretamente diferentes espécies de elefantes com um alto grau de confiança.
---
# Classificação de Imagens com ResNet50 e ImageNet

## Importações
```python
import keras
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
```

## Carregamento do Modelo
```python
model = ResNet50(weights='imagenet')
```

## Processamento da Imagem
```python
img_path = 'elephant.jpg'
img = keras.utils.load_img(img_path, target_size=(224, 224))
x = keras.utils.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
```

## Visualização da Imagem Original
```python
img = mpimg.imread(img_path)
imgplot = plt.imshow(img)
plt.show()
```

## Realizando a Predição
```python
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])
```

## Explicação Detalhada

### 1. Importações
O código começa importando as bibliotecas necessárias:
- **Keras**: Framework principal para deep learning
- **ResNet50**: Modelo de rede neural pré-treinado
- **NumPy**: Para manipulação eficiente de arrays
- **Matplotlib**: Para visualização da imagem

### 2. Modelo
Carrega o ResNet50 pré-treinado com os pesos do ImageNet, que é um grande banco de dados com mais de um milhão de imagens classificadas em 1000 categorias diferentes.

### 3. Processamento de Imagem
A imagem passa por várias etapas de processamento:
- Redimensionamento para 224x224 pixels
- Conversão para array NumPy
- Adição de dimensão extra para batch
- Pré-processamento específico do modelo

### 4. Visualização
Exibe a imagem original usando matplotlib, permitindo uma verificação visual do que está sendo classificado.

### 5. Predição
O modelo analisa a imagem e retorna as três principais previsões. No exemplo do elefante, o resultado foi:
- Elefante Africano: 86.5% de probabilidade
- Tusker: 9.9% de probabilidade
- Elefante Indiano: 3.3% de probabilidade
