import cv2 as cv
import matplotlib.pyplot as plt

class Imaginator:

    def __init__(self, path) -> None:
        self.image = cv.imread(path)

    def convert2rgb(self):
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)
    
    def convert2gray(self):
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
    
    def resize(self, x, y):
        self.image = cv.resize(self.image, (x, y), cv.INTER_LINEAR)
    
    def crop(self, x, y, w, h):
        self.image = self.image[y:y+h, x:x+w]
    
    def add_threshold_filter(self, threshold_value):
        ret, self.image = cv.threshold(self.image, threshold_value, 255, cv.THRESH_BINARY)
    
    def add_blur_filter(self, blur_value):
        self.image = cv.blur(self.image, blur_value)
    
    def add_canny_filter(self, min_val, max_val):
        self.image = cv.Canny(self.image, min_val, max_val)
        
    def show(self, title="", axis="on", cmap="brg"):
        plt.imshow(self.image, cmap=cmap)
        plt.axis(axis) # On enlève les axes des absysses et des ordonnées (facultatif)
        plt.title(title) # On définit un titre (facultatif)
        plt.show()
    
    def save(self, path):
        cv.imwrite(path, self.image)


if __name__ == "__main__":
    # 1. Afficher l'image originale
    img = Imaginator('img/mante.jpeg')
    img.convert2rgb()
    img.show(title='Mante religieuse styley', axis='off')

    # 2. Afficher l'image en noir et blanc
    img = Imaginator('img/mante.jpeg')
    img.convert2gray()
    img.show(title='Mante religieuse styley en noir et blanc', axis='off', cmap='gray')
    img.save('img/mante_gray.png') # Vous pouvez enregistrer/convertir l'image (facultatif)

    # 3. Redimensionner l'image
    img = Imaginator('img/mante.jpeg')
    img.convert2rgb()
    img.resize(100, 300)
    img.show(title='Mante religieuse styley resized', axis='off')

    # 4. Rogner l'image
    img = Imaginator('img/mante.jpeg')
    img.convert2rgb()
    img.crop(210, 50, 400, 400)
    img.show(title='Mante religieuse styley rognée')

    # 5. Appliquer un filtre de seuillage
    img = Imaginator('img/mante.jpeg')
    img.convert2gray()
    img.add_threshold_filter(200)
    img.show(title='Mante religieuse styley dark', axis='off', cmap='gray')

    # 6. Appliquer un filtre de flou
    img = Imaginator('img/mante.jpeg')
    img.convert2gray()
    img.add_blur_filter((10, 10))
    img.show(title='Mante religieuse styley flou', axis='off', cmap='gray')

    # 7. Appliquer un filtre Canny
    img = Imaginator('img/mante.jpeg')
    img.convert2gray()
    img.add_canny_filter(100, 200)
    img.show(title='Mante religieuse styley Canny', axis='off', cmap='gray')

