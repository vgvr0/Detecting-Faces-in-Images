import cv2

# Carga el archivo XML de cascada frontal de Haar para detección de rostros
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Carga la imagen
image = cv2.imread('imagen.jpg')

# Convierte la imagen a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detecta rostros en la imagen
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Dibuja un rectángulo alrededor de cada rostro detectado
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)

# Muestra la imagen con los rostros detectados
cv2.imshow('Facial Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
