import tensorflow as tf
import time

# Configura para ver qué dispositivo está usando
gpus = tf.config.list_physical_devices('GPU')
print("GPUs disponibles:", gpus)

# Carga datos MNIST
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_train = x_train[..., tf.newaxis]  # Añade canal

# Modelo simple
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento y benchmark
EPOCHS = 3

print("Entrenando en GPU (si está disponible)...")
start = time.time()
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=128, verbose=2)
print(f"Tiempo de entrenamiento: {time.time()-start:.2f} segundos")