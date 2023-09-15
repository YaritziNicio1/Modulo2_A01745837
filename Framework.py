#Yaritzi Itzayana Nicio Nicolás A01745837
#Modelo implementado: KNN

#Importamos librerías 
from sklearn import datasets 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, f1_score, confusion_matrix

#Se carga el set de datos desde sklearn
df = datasets.load_iris()
# Obtenemos características (X) y etiquetas (y) del conjunto de datos
X, y = df.data, df.target


# Se dividen los datos en train, test y validation 
#Se genera la semilla con el stándar de 42 para que el modelo sea reproducible 
#Primero se divide en un conjunto temporal y en prueba
X_temp, X_valid, y_temp, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
#Se divide de nuevo, ahora en un conjunto de entrenamiento y validación con el conjunto temporal anterior 
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Vvisualización de dispersión de los datos, conlas características de petal width y petal lenght
plt.figure()
plt.scatter(X[:,3] , X[:,2] , c = y , s=20)
plt.title("Dispersión de los datos")
plt.show()


#Visualización de los datos de entrenamiento, validación y prueba
#Train
plt.subplot(131)
plt.scatter(X_train_c[:, 3], X_train_c[:, 2], c=y_train_c, s=20)
plt.title('Conjunto de Entrenamiento')
plt.xlabel('Longitud del pétalo')
plt.ylabel('Ancho del pétalo')

#Test
plt.subplot(132)
plt.scatter(X_test_c[:, 3], X_test_c[:, 2], c=y_test_c, s=20)
plt.title('Conjunto de prueba')
plt.xlabel('Longitud del pétalo')
plt.ylabel('Ancho del pétalo')

#Validation
plt.subplot(133)
plt.scatter(X_valid[:, 3], X_valid[:, 2], c=y_valid, s=20)
plt.title('Conjunto de vaidación')
plt.xlabel('Longitud del pétalo')
plt.ylabel('Ancho del pétalo')

plt.show()

#Definimos el número de k vecinos cercanos a implementar en el modelo 
k = [3,6,9]

print("*Train* Usando un valor de k=3")
# Impolementación del modelo KNN con los valores de K correspondientes
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train_c, y_train_c)
predic_t3 = clf.predict(X_train_c)
print("Se obtienen las siguientes predicciones: ",predic_t3)

#Se calcula el accuracy del modelo
    #Compara las etiquetas predichas con las verdaderas y suma cuántas son correctas contando las coincidencias 
correct_predictions = np.sum(predic_t3 == y_train_c)
    #Se obtiene el número total de predicciones hechas 
total_predictions = len(y_train_c)
    #Calculo de las predicciones correctas entre el total de predicciones, asi se obtiene el accuracy 
accuracy = correct_predictions / total_predictions
print(accuracy)

# Calculamos el F1 Score
f1_t = f1_score(y_train_c, predic_t3, average='weighted')
print("Puntaje F1:", f1_t)

#Se calcula el MSE
mse_t = mean_squared_error(y_train_c, predic_t3)
print("MSE:", mse_t)

# Se hace una matriz de confusión
confusion_train = confusion_matrix(y_train_c, predic_t3)
print("Matriz de Confusión: ")
print(confusion_train)

#--------------------------------------------------------------------------------------------------------
#Ahora, se utiliza un valor de k=6
print("*Train* Usando un valor de k=6")
# Impolementación del modelo KNN con los valores de K correspondientes
clf = KNeighborsClassifier(n_neighbors=6)
clf.fit(X_train_c, y_train_c)
predic_t6 = clf.predict(X_train_c)
print("Se obtienen las siguientes predicciones: ",predic_t6)

#Se calcula el accuracy del modelo
    #Compara las etiquetas predichas con las verdaderas y suma cuántas son correctas contando las coincidencias 
correct_predictions6 = np.sum(predic_t6 == y_train_c)
    #Se obtiene el número total de predicciones hechas 
total_predictions6 = len(y_train_c)
    #Calculo de las predicciones correctas entre el total de predicciones, asi se obtiene el accuracy 
accuracy6 = correct_predictions6 / total_predictions6
print(accuracy6)

# Calculamos el F1 Score
f1_t6 = f1_score(y_train_c, predic_t6, average='weighted')
print("Puntaje F1:", f1_t6)

#Se calcula el MSE
mse_t6 = mean_squared_error(y_train_c, predic_t6)
print("MSE:", mse_t6)

# Se hace una matriz de confusión
confusion_t6 = confusion_matrix(y_train_c, predic_t6)
print("Matriz de Confusión: ")
print(confusion_t6)


#------------------------------------------------------------------------------------------------------------------
#Por último, se utiliza un valor de k=9
print("*Train* Usando un valor de k=9")
# Impolementación del modelo KNN con los valores de K correspondientes
clf = KNeighborsClassifier(n_neighbors=9)
clf.fit(X_train_c, y_train_c)
predic_t9 = clf.predict(X_train_c)
print("Se obtienen las siguientes predicciones: ",predic_t9)

#Se calcula el accuracy del modelo
    #Compara las etiquetas predichas con las verdaderas y suma cuántas son correctas contando las coincidencias 
correct_predictions9 = np.sum(predic_t9 == y_train_c)
    #Se obtiene el número total de predicciones hechas 
total_predictions9 = len(y_train_c)
    #Calculo de las predicciones correctas entre el total de predicciones, asi se obtiene el accuracy 
accuracy9 = correct_predictions9 / total_predictions9
print(accuracy9)

# Calculamos el F1 Score
f1_t9 = f1_score(y_train_c, predic_t9, average='weighted')
print("Puntaje F1:", f1_t9)

#Se calcula el MSE
mse_t9 = mean_squared_error(y_train_c, predic_t9)
print("MSE:", mse_t9)

# Se hace una matriz de confusión
confusion_t9 = confusion_matrix(y_train_c, predic_t9)
print("Matriz de Confusión: ")
print(confusion_t9)

#Se hace el mismo procedimiento para los datos de validation y test 
    #Se involucra un ciclo for 
for i in range(len(k)):
    #****Para datos de validación********* 
    print("*Validación* Usando un valor de k={0}".format(k[i]))
    # Implementación del modelo KNN con los valores de K correspondientes
    clf = KNeighborsClassifier(n_neighbors=k[i])
    clf.fit(X_valid, y_valid)
    predict_v = clf.predict(X_valid)
    print("Se obtienen las siguientes predicciones: ",predict_v)
    #Se calcula el accuracy del modelo
    #Compara las etiquetas predichas con las verdaderas y suma cuántas son correctas contando las coincidencias 
    correct_pred_v = np.sum(predict_v == y_valid)
    #Se obtiene el número total de predicciones hechas 
    total_pred_v = len(y_valid)
    #Calculo de las predicciones correctas entre el total de predicciones, asi se obtiene el accuracy 
    accuracy_v = correct_pred_v / total_pred_v
    print("Accuracy: ",accuracy_v)

    # Calculamos el F1 Score
    f1_v = f1_score(y_valid, predict_v, average='weighted')
    print("Puntaje F1:", f1_v)

    #Se calcula el MSE
    mse_v = mean_squared_error(y_valid, predict_v)
    print("MSE:", mse_v)

    # Se hace una matriz de confusión
    confusion_v = confusion_matrix(y_valid, predict_v)
    print("Matriz de Confusión: ")
    print(confusion_v)


    #****Para datos de test********* 
    print("*Test* Usando un valor de k={0}".format(k[i]))
    # Implementación del modelo KNN con los valores de K correspondientes
    clf = KNeighborsClassifier(n_neighbors=k[i])
    clf.fit(X_test_c, y_test_c)
    predict_t = clf.predict(X_test_c)
    print("Se obtienen las siguientes predicciones: ",predict_t)
    #Se calcula el accuracy del modelo
    #Compara las etiquetas predichas con las verdaderas y suma cuántas son correctas contando las coincidencias 
    co_pred_test = np.sum(predict_t == y_test_c)
    #Se obtiene el número total de predicciones hechas 
    total_pred_test = len(y_test_c)
    #Calculo de las predicciones correctas entre el total de predicciones, asi se obtiene el accuracy 
    accuracyte = co_pred_test / total_pred_test
    print("Accuracy: ",accuracyte)

    # Calculamos el F1 Score
    f1_te = f1_score(y_test_c, predict_t, average='weighted')
    print("Puntaje F1:", f1_te)

    #Se calcula el MSE
    mse_te = mean_squared_error(y_test_c, predict_t)
    print("MSE:", mse_te)

    # Se hace una matriz de confusión
    confus_te = confusion_matrix(y_test_c, predict_t)
    print("Matriz de Confusión: ")
    print(confus_te)

    print("\n\n")


#***

'''
for i in range(len(k)):
    # Impolementación del modelo KNN con los valores de K correspondientes
    clf = KNeighborsClassifier(n_neighbors=k[i])

    # Ajuste del modelo k-NN a los datos de entrenamiento
    clf.fit(X_train_c, y_train_c)

    #Imprimimos una frase que muestre el valor de k que estamos usando
    print("Usando valor de k={0}\n".format(k_values[i]))

    # Realización de predicciones en los datos de entrenamiento
    predictions_train = clf.predict(X_train)

    # Impresión de las predicciones
    print("Predicciones con datos de entrenamiento:")
    print(predictions_train)


# Escala tus datos (esto es importante para KNN)
scaler_reg = StandardScaler()
X_train_reg = scaler_reg.fit_transform(X_train_reg)
X_test_reg = scaler_reg.transform(X_test_reg)

# Crea una instancia del regresor KNN y ajústalo a tus datos de entrenamiento
knn_regressor = KNeighborsRegressor(n_neighbors=3)  # Puedes ajustar el número de vecinos según tus necesidades
knn_regressor.fit(X_train_reg, y_train_reg)

# Realiza predicciones en los datos de prueba y evalúa el rendimiento
y_pred_reg = knn_regressor.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
print(f"Error cuadrático medio (MSE) del regresor KNN: {mse}")

#Se definen todas las métricas a obtener }

'''

