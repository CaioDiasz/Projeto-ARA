"""
train_model.py
----------------

Ajuste DATASET_DIR abaixo se salvar em outro lugar.

COMO RODAR
-----------
    pip install -r requirements.txt
    python train_model.py

Ao final, o modelo treinado é salvo em model/recycle_model.h5
e o mapeamento de classes em model/class_indices.json
"""

import json
import os

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ----------------------------
# Configurações
# ----------------------------
DATASET_DIR = "dataset"          # pasta com subpastas por classe
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 10                 # treino só da "cabeça" nova
EPOCHS_FINE_TUNE = 5             # fine-tuning das últimas camadas da base
MODEL_OUT = "model/recycle_model.h5"
CLASS_INDEX_OUT = "model/class_indices.json"


def build_datasets():
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
    )

    train_gen = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_gen = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    return train_gen, val_gen


def build_model(num_classes):
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # congela a base inicialmente

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


def main():
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(
            f"Pasta '{DATASET_DIR}' não encontrada. Baixe um dataset de "
            "recicláveis (ex: TrashNet / Garbage Classification no Kaggle) "
            "e organize em subpastas por classe, conforme o cabeçalho deste "
            "arquivo."
        )

    os.makedirs("model", exist_ok=True)

    train_gen, val_gen = build_datasets()
    num_classes = len(train_gen.class_indices)
    print(f"Classes encontradas: {train_gen.class_indices}")

    model, base_model = build_model(num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\n=== Fase 1: treinando apenas a cabeça (base congelada) ===")
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_HEAD)

    print("\n=== Fase 2: fine-tuning das últimas camadas da base ===")
    base_model.trainable = True
    # Congela tudo, exceto as últimas ~30 camadas
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_FINE_TUNE)

    # Avaliação final
    loss, acc = model.evaluate(val_gen)
    print(f"\nAcurácia de validação final: {acc*100:.2f}%")

    # Salva modelo e mapeamento de classes
    model.save(MODEL_OUT)
    with open(CLASS_INDEX_OUT, "w", encoding="utf-8") as f:
        # inverte para {indice: nome_classe}
        idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
        json.dump(idx_to_class, f, ensure_ascii=False, indent=2)

    print(f"\nModelo salvo em: {MODEL_OUT}")
    print(f"Classes salvas em: {CLASS_INDEX_OUT}")


if __name__ == "__main__":
    main()
