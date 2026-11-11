# ♻️ Identificador de Recicláveis

App web em Python (Flask) que usa a câmera do celular para identificar o
tipo de material reciclável em uma foto (papelão, vidro, metal, papel,
plástico ou rejeito), usando um modelo de visão computacional treinado
por transfer learning (MobileNetV2).

## Estrutura

```
recycling-app/
├── app.py              # Servidor Flask (página + endpoint /predict)
├── train_model.py      # Script de treino do modelo
├── requirements.txt
├── templates/
│   └── index.html      # Página com acesso à câmera do celular
└── model/               # Onde o modelo treinado é salvo
```

## Passo 1 — Instalar dependências

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Passo 2 — Conseguir um dataset de recicláveis

Baixe um dataset de imagens de lixo reciclável, por exemplo:

- **Garbage Classification** (Kaggle, baseado no TrashNet):
  https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification

Extraia e organize as imagens em pastas por classe dentro de uma pasta
`dataset/` na raiz do projeto:

```
dataset/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

## Passo 3 — Treinar o modelo

```bash
python train_model.py
```

Isso vai gerar:
- `model/recycle_model.h5` — o modelo treinado
- `model/class_indices.json` — mapeamento de índice → nome da classe

O treino é feito em duas fases (cabeça nova + fine-tuning das últimas
camadas da MobileNetV2) para melhor acurácia com pouco dado.

## Passo 4 — Rodar o servidor

```bash
python app.py
```

Isso sobe o servidor em `http://0.0.0.0:5000`.

## Passo 5 — Acessar pela câmera do celular

Navegadores só liberam a câmera em `localhost` ou `https`. Para testar
no celular pela rede Wi-Fi, escolha uma opção:

**Opção A — ngrok (mais simples)**
```bash
ngrok http 5000
```
Acesse a URL `https://...ngrok...` gerada, pelo navegador do celular.

**Opção B — HTTPS local autoassinado**

No final de `app.py`, troque:
```python
app.run(host="0.0.0.0", port=5000, debug=True)
```
por:
```python
app.run(host="0.0.0.0", port=5000, ssl_context="adhoc")
```
Acesse `https://<IP-DO-COMPUTADOR>:5000` no celular (mesma rede Wi-Fi) e
aceite o aviso de certificado não confiável.

## Como funciona

1. A página abre a câmera do celular (traseira por padrão, com botão
   para trocar).
2. Ao tocar em "Capturar e identificar", a foto é convertida em base64
   e enviada para o endpoint `/predict`.
3. O Flask decodifica a imagem, redimensiona para 224x224, normaliza
   e passa pelo modelo treinado.
4. O resultado (classe prevista + confiança + dica de descarte) volta
   como JSON e é exibido na tela.

## Próximos passos possíveis

- Adicionar mais classes (ex: eletrônicos, orgânico separado de rejeito)
- Guardar histórico de itens identificados
- Empacotar como app Android real com Kivy/Buildozer, se quiser sair do
  navegador
