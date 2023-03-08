
# Importamos las bibliotecas necesarias
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Definimos la secuencia de entrada
raw_seq = [2, 3, 4, 5, 6, 7, 8, 9, 0, 2, 3, 4, 5, 6, 7, 6, 5]

# Definimos la longitud de las secuencias a usar para entrenamiento y pruebas
seq_length = 10

# Preparamos los datos para el entrenamiento
train_X, train_y = [], []
for i in range(seq_length, len(raw_seq)):
    train_X.append(raw_seq[i - seq_length:i])
    train_y.append(raw_seq[i])
train_X, train_y = np.array(train_X), np.array(train_y)

# Damos forma a los datos de entrada para que estén en el formato [muestras, pasos de tiempo, características]
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))

# Dividimos los datos en conjuntos de entrenamiento y validación
train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2)

# Definimos el modelo LSTM
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

user_input = input("Entrenamiento normal o DQN 'normal' o 'DQN': ")

if user_input == 'normal':

    # Ajustamos el modelo a los datos de entrenamiento y validación
    history = model.fit(train_X, train_y, epochs=200, batch_size=32, verbose=2, validation_data=(val_X, val_y))

    # Graficamos la pérdida de entrenamiento y la pérdida de validación
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Pérdida de entrenamiento y validación')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend(['Entrenamiento', 'Validación'])
    plt.show()

    # Usamos el modelo entrenado para generar predicciones para n valores siguientes
    n = 10
    last_X = np.array(raw_seq[-seq_length:]).reshape((1, seq_length, 1))
    predictions = []
    for i in range(n):
        pred = model.predict(last_X, verbose=0)
        predictions.append(pred[0][0])
        last_X = np.append(last_X[:, 1:, :], [[pred[0]]], axis=1)

    # Calculamos el error cuadrático medio entre los valores de prediccion y los valores reales
    mse = np.mean((np.array(predictions) - np.array(raw_seq[-n:])) ** 2)

    # Imprimimos los valores estimados y el MSE
    print('Valores estimados:', predictions)
    print('Error cuadrático medio:', mse)

    # Graficamos los valores de entrada y estimados
    plt.plot(raw_seq, label='Datos de entrada')
    plt.plot(np.arange(len(raw_seq), len(raw_seq) + n), predictions, label='Valores estimados')
    plt.legend()
    plt.show()

elif user_input == 'DQN':

    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 32
    memory = deque(maxlen=2000)
    alpha = 0.5
    target_update_frequency = 10

    predictions = np.zeros(len(raw_seq))

    n_states = 10 ** seq_length
    n_actions = len(raw_seq)
    Q = np.zeros((n_states, n_actions))

    epochs = 100
    model_target = Sequential()
    model_target.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
    model_target.add(Dense(1))
    model_target.compile(optimizer='adam', loss='mse')

    cumulative_reward = 0

    # Bucle for que itera a través de cada epoch (época)
    for epoch in range(epochs):
        # Reshape de los datos de entrenamiento a la forma correcta
        train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
        # Ajustar el modelo de keras para un epoch utilizando los datos de entrenamiento
        model.fit(train_X, train_y, epochs=1, batch_size=batch_size, verbose=0)
        # Generar predicciones para cada punto de datos en el conjunto de datos de entrenamiento
        for i in range(len(raw_seq) - seq_length):
            X = np.array(raw_seq[i:i + seq_length]).reshape((1, seq_length, 1))
            predictions[i + seq_length] = model.predict(X, verbose=0)
        # reducir el valor de epsilon si es mayor que el valor mínimo
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        # Actualizar el modelo objetivo para igualar el modelo actual si se ha alcanzado la frecuencia de actualización objetivo
        if epoch % target_update_frequency == 0:
            model_target.set_weights(model.get_weights())
        # Escoger una acción aleatoria con probabilidad epsilon, de lo contrario escoger la acción con el mayor valor Q
        if np.random.rand() <= epsilon:
            action = np.random.randint(0, len(raw_seq))
        else:
            state = np.random.randint(0, n_states)
            action = np.argmax(Q[state])
        # Calcular el nuevo estado a partir del estado actual y la acción escogida
        new_state = (state * 10 + action) % n_states
        # Calcular la recompensa por la acción escogida
        reward = np.abs(raw_seq[action] - predictions[action])
        # Añadir la transición a la memoria y actualizar la recompensa acumulativa
        memory.append((state, action, reward, new_state))
        cumulative_reward += reward
        # Escoger una muestra aleatoria de transiciones de memoria y actualizar la matriz Q usando Q-learning
        batch = random.choices(list(memory), weights=None, k=batch_size)
        for state, action, reward, new_state in batch:
            Q[state, action] += alpha * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
        # Imprimir el número de epoch, la recompensa actual y la recompensa acumulativa
        print(f"Epoch {epoch + 1}, Reward: {reward}, Cumulative Reward: {cumulative_reward}")

    # Usamos el modelo entrenado para generar predicciones para n valores siguientes
    n = 10
    last_X = np.array(raw_seq[-seq_length:]).reshape((1, seq_length, 1))
    predictions = []
    for i in range(n):
        pred = model.predict(last_X, verbose=0)
        predictions.append(pred[0][0])
        last_X = np.append(last_X[:, 1:, :], [[pred[0]]], axis=1)

    # Calculamos el error cuadrático medio entre los valores de prediccion y los valores reales
    mse = np.mean((np.array(predictions) - np.array(raw_seq[-n:])) ** 2)

    # Imprimimos los valores estimados y el MSE
    print('Valores estimados:', predictions)
    print('Error cuadrático medio:', mse)

    # Graficamos los valores de entrada y estimados
    plt.plot(raw_seq, label='Datos de entrada')
    plt.plot(np.arange(len(raw_seq), len(raw_seq) + n), predictions, label='Valores estimados')
    plt.legend()
    plt.show()
else:
    print("Entrada inválida, por favor ingrese 'normal' o 'DQN'")
