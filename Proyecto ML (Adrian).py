#!/usr/bin/env python
# coding: utf-8

# # Renata Vargas - A01025281

# In[1]:


# Importar bibliotecas

from sklearn import datasets, model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt


# ## Primera Parte: Perceptrón

# In[21]:


#Datos
dataset = datasets.load_breast_cancer()
X = dataset['data']
y = dataset['target']

# División del conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=44)

# Define la función de activación: una función escalón para el perceptrón
def step_function(z):
    return np.where(z >= 0, 1, 0)

# Inicialización de pesos y sesgo para el perceptrón
np.random.seed(0)
perceptron_weights = np.random.rand(X_train.shape[1])
perceptron_bias = np.random.rand()

# Tasa de aprendizaje para el perceptrón
perceptron_learning_rate = 0.01

# Número de épocas de entrenamiento
epochs = 100

# Listas para almacenar la precisión durante el entrenamiento
perceptron_accuracy_history = []
nn_accuracy_history = []

# Entrenamiento del perceptrón
for epoch in range(epochs):
    perceptron_correct_predictions = 0
    for i in range(X_train.shape[0]):
        
        # Calcula la salida del perceptrón
        z = np.dot(X_train[i], perceptron_weights) + perceptron_bias
        y_pred = step_function(z)

        # Actualiza los pesos y el sesgo utilizando la regla de aprendizaje del perceptrón
        perceptron_weights += perceptron_learning_rate * (y_train[i] - y_pred) * X_train[i]
        perceptron_bias += perceptron_learning_rate * (y_train[i] - y_pred)

        # Calcula la precisión durante el entrenamiento del perceptrón
        perceptron_correct_predictions += (y_pred == y_train[i])

    perceptron_accuracy = perceptron_correct_predictions / X_train.shape[0]
    perceptron_accuracy_history.append(perceptron_accuracy)


# In[22]:


# Predicciones del perceptrón 
# Define la función de activación (en este caso, una función escalón)
def step_function(z):
    return np.where(z >= 0, 1, 0)

# Realiza predicciones con el modelo del perceptrón en el conjunto de prueba (X_test)
perceptron_predictions = [step_function(np.dot(x, perceptron_weights) + perceptron_bias) for x in X_test]

# Imprime las predicciones del perceptrón
print("Predicciones del Perceptrón:")
print(perceptron_predictions)


# In[23]:


# Evalúa el perceptrón en el conjunto de entrenamiento
train_correct_predictions = 0
for i in range(X_train.shape[0]):
    z = np.dot(X_train[i], perceptron_weights) + perceptron_bias
    y_pred = step_function(z)
    train_correct_predictions += (y_pred == y_train[i])

train_accuracy = train_correct_predictions / X_train.shape[0]

# Evalúa el perceptrón en el conjunto de prueba
test_correct_predictions = 0
for i in range(X_test.shape[0]):
    z = np.dot(X_test[i], perceptron_weights) + perceptron_bias
    y_pred = step_function(z)
    test_correct_predictions += (y_pred == y_test[i])

test_accuracy = test_correct_predictions / X_test.shape[0]

# Diagnóstico de Sesgo (Bias)
if train_accuracy > test_accuracy:
    bias_diagnosis = "Sesgo alto (overfitting hacia datos de entrenamiento)"
elif train_accuracy < test_accuracy:
    bias_diagnosis = "Sesgo alto (overfitting hacia datos de prueba)"
else:
    bias_diagnosis = "Sesgo bajo (bien equilibrado)"

# Diagnóstico de Varianza
# Calcula la diferencia entre las precisión de entrenamiento y prueba
accuracy_difference = train_accuracy - test_accuracy

if accuracy_difference > 0.1:
    variance_diagnosis = "Varianza alta (sobreajuste pronunciado)"
elif accuracy_difference < 0.1:
    variance_diagnosis = "Varianza baja (bien generalizado)"
else:
    variance_diagnosis = "Varianza media (generalización moderada)"


# In[24]:


print("Diagnóstico de Sesgo:", bias_diagnosis)
print("Diagnóstico de Varianza: ", variance_diagnosis)


# In[ ]:





# ## Segunda parte: Red Neuronal

# In[25]:


# Cargar la base de datos desde scikit-learn

dataset = datasets.load_breast_cancer()
X = dataset['data']
y = dataset['target']

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=44)

def RedNeuronal_CancerDeMama():
    # Definir el modelo
    model = Sequential(name='Breast_Cancer')
    model.add(Dense(units=64, input_shape=(X_train.shape[1],), activation='relu',
                    kernel_initializer=tf.keras.initializers.HeUniform(seed=0),
                    bias_initializer='ones', name='hiddenlayer1'))
    model.add(Dense(units=128, activation='sigmoid', name='hiddenlayer2'))
    model.add(Dense(units=128, activation='tanh', name='hiddenlayer3'))
    model.add(Dense(units=128, activation='elu', name='hiddenlayer4'))
    model.add(Dense(units=128, activation='swish', name='hiddenlayer5'))
    model.add(Dense(units=1, activation='sigmoid', name='outputlayer'))  # Sigmoide para clasificación binaria
    model.summary()
    return model

model = RedNeuronal_CancerDeMama()

adam = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy']) 


# In[26]:


# Entrenamiento de modelo 
training_history_1 = model.fit(X_train, y_train, epochs=100, validation_split=0.15, batch_size=40)
nn_accuracy_history = []
nn_accuracy_history.extend(training_history_1.history['accuracy'])


# In[27]:


# Evaluación del modelo
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# In[28]:


# Predicciones del modelo
# Realiza predicciones con la red neuronal en el conjunto de prueba (X_test)
nn_predictions = model.predict(X_test)

# Convierte las predicciones de la red neuronal a etiquetas binarias (0 o 1)
nn_predictions_binary = [1 if p >= 0.5 else 0 for p in nn_predictions]

# Imprime las predicciones de la red neuronal
print("Predicciones de la Red Neuronal:")
print(nn_predictions_binary)


# In[29]:


# Diagnóstico de Sesgo (Bias)
# Calcula el accuracy en el conjunto de entrenamiento y prueba
train_acc = model.evaluate(X_train, y_train, verbose=0)[1]
test_acc = model.evaluate(X_test, y_test, verbose=0)[1]

if train_acc > test_acc:
    bias_diagnosis = "Sesgo alto (overfitting hacia datos de entrenamiento)"
elif train_acc < test_acc:
    bias_diagnosis = "Sesgo alto (overfitting hacia datos de prueba)"
else:
    bias_diagnosis = "Sesgo bajo (bien equilibrado)"

# Diagnóstico de Varianza
# Calcula la diferencia entre las pérdidas de entrenamiento y prueba
train_loss = model.evaluate(X_train, y_train, verbose=0)[0]
test_loss = model.evaluate(X_test, y_test, verbose=0)[0]
loss_difference = train_loss - test_loss

if loss_difference > 0.1:
    variance_diagnosis = "Varianza alta (sobreajuste pronunciado)"
elif loss_difference < 0.1:
    variance_diagnosis = "Varianza baja (bien generalizado)"
else:
    variance_diagnosis = "Varianza media (generalización moderada)"


# In[30]:


# Diagnóstico de Ajuste del Modelo
if train_acc < 0.8:
    fitting_diagnosis = "Underfitting (modelo demasiado simple)"
elif train_acc >= 0.8 and train_acc < 0.95:
    fitting_diagnosis = "Ajuste adecuado (modelo bien equilibrado)"
else:
    fitting_diagnosis = "Overfitting (modelo complejo, posible sobreajuste)"


# In[31]:


print("Diagnóstico de Sesgo:", bias_diagnosis)
print("Diagnóstico de Ajuste del Modelo:", fitting_diagnosis)
print("Diagnóstico de Varianza: ", variance_diagnosis)


# ## Tercera Parte: Comparación

# In[32]:


# Comparación gráfica de los dos modelos
plt.figure(figsize=(12, 6))
plt.plot(range(1, epochs + 1), perceptron_accuracy_history, label='Perceptron', marker='o',color='blue')
plt.plot(range(1, epochs + 1), nn_accuracy_history, label='Red Neuronal', marker='o',color='red')
plt.xlabel('Épocas')
plt.ylabel('Precisión de Entrenamiento')
plt.title('Entrenamiento del Perceptrón vs. Red Neuronal')
plt.legend()
plt.grid(True)
plt.show()


# In[33]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# Calcular la precisión del perceptrón
perceptron_accuracy = accuracy_score(y_test, perceptron_predictions)
print(f'Precisión del Perceptrón: {perceptron_accuracy * 100:.2f}%')

# Calcular la matriz de confusión del perceptrón
perceptron_confusion_matrix = confusion_matrix(y_test, perceptron_predictions)
print('Matriz de Confusión del Perceptrón:')
print(perceptron_confusion_matrix)

# Calcular la precisión, exhaustividad y puntuación F1 del perceptrón
perceptron_precision = precision_score(y_test, perceptron_predictions)
perceptron_recall = recall_score(y_test, perceptron_predictions)
perceptron_f1_score = f1_score(y_test, perceptron_predictions)
print(f'Precisión del Perceptrón: {perceptron_precision * 100:.2f}%')
print(f'Exhaustividad del Perceptrón: {perceptron_recall * 100:.2f}%')
print(f'Puntuación F1 del Perceptrón: {perceptron_f1_score * 100:.2f}%')


# In[34]:


# Calcular la precisión de la red neuronal
nn_accuracy = accuracy_score(y_test, nn_predictions_binary)
print(f'Precisión de la Red Neuronal: {nn_accuracy * 100:.2f}%')

# Calcular la matriz de confusión de la red neuronal
nn_confusion_matrix = confusion_matrix(y_test, nn_predictions_binary)
print('Matriz de Confusión de la Red Neuronal:')
print(nn_confusion_matrix)

# Calcular la precisión, exhaustividad y puntuación F1 de la red neuronal
nn_precision = precision_score(y_test, nn_predictions_binary)
nn_recall = recall_score(y_test, nn_predictions_binary)
nn_f1_score = f1_score(y_test, nn_predictions_binary)
print(f'Precisión de la Red Neuronal: {nn_precision * 100:.2f}%')
print(f'Exhaustividad de la Red Neuronal: {nn_recall * 100:.2f}%')
print(f'Puntuación F1 de la Red Neuronal: {nn_f1_score * 100:.2f}%') 

