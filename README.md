# Detector de Gatos e Cachorros em Imagens

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Hub-yellow)](https://huggingface.co/)
[![Weights & Biases](https://img.shields.io/badge/W&B-Tracking-ff69b4)](https://wandb.ai/)

Um classificador de imagens baseado em Rede Neural Convolucional (CNN) para detectar e diferenciar gatos e cachorros em fotos. O modelo é treinado com um dataset público do Hugging Face, monitorado via Weights & Biases (W&B) e salvo no Hugging Face Hub para fácil reutilização.

Este projeto demonstra uma pipeline completa de visão computacional: carregamento de dados, treinamento de CNN, avaliação de métricas e deploy na nuvem.

## 📸 Demo

![Evolução do Treinamento](training_plots.png)

*Gráficos de Loss e Accuracy ao longo das épocas (gerados via W&B ou Matplotlib). À esquerda: Perda (Loss) diminuindo consistentemente. À direita: Acurácia (Accuracy) subindo para ~99% no treino e ~98% na validação.*

## 🚀 Visão Geral

- **Arquitetura**: CNN personalizada (camadas convolucionais, pooling e fully connected).
- **Tarefa**: Classificação binária de imagens (gato vs. cachorro).
- **Dataset**: [Cats vs Dogs no Hugging Face](https://huggingface.co/datasets/cats_vs_dogs) ou similar (ex: Oxford Pets).
- **Tecnologias**:
  - Framework: TensorFlow/Keras (ou PyTorch, adaptável).
  - Dataset: Hugging Face Datasets.
  - Tracking: Weights & Biases para logs de métricas, gráficos e artefatos.
  - Deploy: Modelo salvo no Hugging Face Hub.
- **Métricas Principais**:
  - Accuracy
  - F1-Score
  - AUC-ROC
  - Top-1 Accuracy
  - Top-5 Accuracy (útil para extensões multi-classe)

O modelo atinge **acurácia de validação de até 98.72%** após 5 épocas, com baixa perda e sem sinais claros de overfitting.

## 📊 Resultados do Treinamento

Treinamento realizado por 5 épocas com batch size implícito (~32-64) e otimizador Adam.

| Época | Train Loss | Train Acc | Train F1 | Train AUC | Val Loss | Val Acc | Val F1 | Val AUC |
|-------|------------|-----------|----------|-----------|----------|---------|--------|---------|
| 1     | 0.0773    | 0.9689   | 0.9689  | 0.9962   | 0.0464  | 0.9812 | 0.9812 | 0.9984 |
| 2     | 0.0391    | 0.9865   | 0.9865  | 0.9989   | 0.0488  | 0.9791 | 0.9791 | 0.9988 |
| 3     | 0.0268    | 0.9905   | 0.9905  | 0.9995   | 0.0348  | 0.9872 | 0.9872 | 0.9993 |
| 4     | 0.0246    | 0.9910   | 0.9910  | 0.9996   | 0.0539  | 0.9795 | 0.9795 | 0.9985 |
| 5     | 0.0229    | 0.9919   | 0.9919  | 0.9996   | 0.0419  | 0.9825 | 0.9825 | 0.9991 |

- **Melhor Val Accuracy**: 98.72% (Época 3)
- **Top-1/Top-5 Accuracy**: Consistentemente >98% / 100% em validação.
- **Observações**: A perda de validação varia ligeiramente, indicando estabilidade. O modelo generaliza bem, com AUC próximo de 1.0.

Gráficos de evolução:
- **Loss**: Diminui rapidamente no treino; validação oscila mas permanece baixa.
- **Accuracy**: Aumenta monotonicamente no treino; validação atinge pico cedo.

## 🛠️ Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/detector-gatos-cachorros.git
   cd detector-gatos-cachorros
   ```

2. Crie um ambiente virtual (recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

   `requirements.txt` exemplo:
   ```
   tensorflow>=2.10
   # ou torch>=1.12 torchvision
   datasets[huggingface]
   wandb
   matplotlib
   scikit-learn
   ```

4. Configure credenciais:
   - Hugging Face: `huggingface-cli login`
   - Weights & Biases: `wandb login`

## 📂 Estrutura do Projeto

```
.
├── train.py              # Script principal de treinamento
├── model.py              # Definição da CNN
├── inference.py          # Script para predições
├── requirements.txt      # Dependências
├── training_plots.png    # Gráficos de métricas
├── wandb/                # Logs do W&B (gerados automaticamente)
├── models/               # Modelo salvo localmente
└── README.md             # Este arquivo
```

## ⚙️ Uso

### Treinamento
```bash
python train.py --epochs 5 --batch-size 32 --lr 0.001
```

- Integração com W&B: Métricas são logadas automaticamente.
- Salvamento: Modelo pushado para `seu-usuario/cat-dog-classifier` no Hugging Face Hub.

### Inferência
```bash
python inference.py --image path/to/image.jpg --model hf://seu-usuario/cat-dog-classifier
```

Exemplo de saída:
```
Predição: Gato (Confiança: 99.2%)
```

### Carregando o Modelo do Hub
```python
from transformers import pipeline

classifier = pipeline("image-classification", model="seu-usuario/cat-dog-classifier")
result = classifier("path/to/image.jpg")
print(result)
```

## 🔍 Explicação da Arquitetura

A CNN inclui:
- Camadas Convolucionais (Conv2D) com ReLU.
- MaxPooling para redução dimensional.
- Dropout para regularização.
- Camadas Dense finais com Softmax (binário: Sigmoid).

Exemplo em Keras:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binário
])
```

## 📈 Métricas Detalhadas

- **Accuracy**: Proporção de predições corretas.
- **F1-Score**: Média harmônica de Precision e Recall (ideal para classes balanceadas).
- **AUC**: Área sob a curva ROC, medindo separabilidade.
- **Top-k**: Extensível para multi-classe futura.

Para detecção de objetos (extensão futura): IoU e mAP.

## 🤝 Contribuições

Contribuições são bem-vindas! Abra uma issue ou pull request.

1. Fork o repositório.
2. Crie uma branch: `git checkout -b feature/nova-funcionalidade`.
3. Commit: `git commit -m 'Adiciona nova funcionalidade'`.
4. Push e abra PR.

## 📄 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para detalhes.

## 👨‍💻 Autor

- GitHub: [@cl4y70n](https://github.com/cl4y70n)

---

⭐ Se gostou, dê uma estrela no repositório! Para mais projetos de visão computacional, siga-me. 

*Projeto criado em novembro de 2025.*
