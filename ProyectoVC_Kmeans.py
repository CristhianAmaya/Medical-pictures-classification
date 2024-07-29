import os
import re
import cv2
import numpy as np
from scipy.signal import convolve2d
from skimage.measure import shannon_entropy
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from sklearn.cluster import KMeans

# Cambia el directorio de trabajo
os.chdir('C:\\Camilo\\Clases UMNG\\Visión por computador\\Proyecto_VC\\Imágenes proyecto')

# Lista de imágenes en el directorio
imagenes = [file for file in os.listdir() if file.endswith('.jpg')]

def extract_number(text):
    match = re.search(r'\d+', text)
    return int(match.group()) if match else -1

# Ordenar las imágenes por nombre teniendo en cuenta los números
imagenes_con_enfermedad = sorted([img for img in imagenes if img.startswith('Anormal')], key=extract_number)
imagenes_sin_enfermedad = sorted([img for img in imagenes if img.startswith('Normal')], key=extract_number)
imagenes_prueba = sorted([img for img in imagenes if img.startswith('Prueba')], key=extract_number)

# Verificar la lista ordenada de imágenes
print("Imágenes con enfermedades:", imagenes_con_enfermedad)
print("Imágenes sin enfermedades:", imagenes_sin_enfermedad)
print("Imágenes de prueba:", imagenes_prueba)

imagenes = imagenes_con_enfermedad + imagenes_sin_enfermedad + imagenes_prueba

# Verificar el orden de las imágenes después de la concatenación
print("Todas las imágenes ordenadas:", imagenes)

# Umbral para detección de bordes
Umbral = 50

# Kernel para detección de bordes
h = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
v = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

# Listas para guardar características de las imágenes
imagenes_guardadas = []

# Procesamiento de las imágenes
for nombre in imagenes:
    # Lectura y conversión a escala de grises
    I = cv2.imread(nombre)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

    # Convolución para detección de bordes
    Ih = convolve2d(I, v, mode='same')
    Iv = convolve2d(I, h, mode='same')
    mag = np.sqrt(Iv ** 2 + Ih ** 2)

    # Características a calcular
    densidad_bordes = np.sum(mag >= Umbral) / (mag.shape[0] * mag.shape[1])
    entropia = shannon_entropy(mag)
    varianza = np.var(mag)
    media = np.mean(mag)
    PSNR = (media ** 2) / varianza

    # Guardar características
    imagenes_guardadas.append([densidad_bordes, entropia, varianza, PSNR])

    # Imprimir el nombre de la imagen y sus características calculadas
    print(f"Procesando imagen: {nombre}")
    print(f"Características - Densidad de bordes: {densidad_bordes}, Entropía: {entropia}, Varianza: {varianza}, PSNR: {PSNR}")

# Convertir lista de características a array numpy
imagenes_guardadas = np.array(imagenes_guardadas)

# Aplicar K-means
kmeans = KMeans(n_clusters=2, random_state=0).fit(imagenes_guardadas[:100])
clusters = kmeans.predict(imagenes_guardadas)

# Características de las imágenes de prueba (En total son 10)
valores_pruebaMedias = np.array(imagenes_guardadas[100:])

# Se crea una matriz de 5x3 para asignar los valores obtenidos
agregado = np.zeros((10, 3))
agregado[:, 0] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Clasificar las imágenes de prueba usando K-means
for i in range(len(valores_pruebaMedias)):
    cluster = kmeans.predict([valores_pruebaMedias[i]])
    agregado[i, 2] = cluster

# Crear la interfaz gráfica
def mostrar_imagenes(indices):
    ventana_imagenes = tk.Toplevel(root)
    ventana_imagenes.title("Imágenes")

    # Crear un marco para el canvas y el scrollbar
    frame = ttk.Frame(ventana_imagenes)
    frame.grid(row=0, column=0, sticky="nsew")

    # Crear un canvas más grande dentro del frame
    canvas = tk.Canvas(frame, width=1000, height=600)
    canvas.grid(row=0, column=0, sticky="nsew")

    # Añadir un scrollbar al canvas
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Crear otro frame dentro del canvas para contener las imágenes
    image_frame = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=image_frame, anchor="nw")

    # Añadir las imágenes al frame
    for i, idx in enumerate(indices):
        img = Image.open(imagenes[idx])
        img = img.resize((200, 200), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = ttk.Label(image_frame, image=img)
        panel.image = img
        panel.grid(row=i // 5, column=i % 5)

    # Configurar el tamaño del canvas para que coincida con el contenido
    image_frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

def mostrar_anormales():
    mostrar_imagenes(range(50))

def mostrar_normales():
    mostrar_imagenes(range(50, 100))

def mostrar_prueba():
    mostrar_imagenes(range(100, 110))

def mostrar_agregado():
    ventana_agregado = tk.Toplevel(root)
    ventana_agregado.title("Matriz resultado")
    tree = ttk.Treeview(ventana_agregado, columns=("# de imagen de prueba", "Clasificación numérica", "Declaración"))
    tree.heading("#0", text="# de imagen de prueba")
    tree.heading("#1", text="Clasificación numérica")
    tree.heading("#2", text="Declaración")
    tree.column("#0", width=150)
    tree.column("#1", width=150)
    tree.column("#2", width=150)

    for i, (indice, clase) in enumerate(zip(agregado[:, 0], agregado[:, 2])):
        if clase == 0:
            declaracion = "Sin enfermedad"
        else:
            declaracion = "Con enfermedad"
        tree.insert("", "end", text=int(indice), values=(int(clase), declaracion))
    tree.pack()

root = tk.Tk()
root.title("Clasificación de Imágenes")

btn_anormales = ttk.Button(root, text="Mostrar base de datos 1", command=mostrar_anormales)
btn_anormales.pack(pady=10)

btn_normales = ttk.Button(root, text="Mostrar base de datos 2", command=mostrar_normales)
btn_normales.pack(pady=10)

btn_prueba = ttk.Button(root, text="Mostrar imágenes de prueba", command=mostrar_prueba)
btn_prueba.pack(pady=10)

btn_agregado = ttk.Button(root, text="Mostrar resultados", command=mostrar_agregado)
btn_agregado.pack(pady=10)

root.mainloop()
