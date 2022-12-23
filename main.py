import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog as fido
from descriptor import ColorDescriptor, TextureDescriptor
from searcher import Searcher
import cv2


class App(tk.Tk):

    def __init__(self):
        super().__init__()

        self.imgs = []

        # app info
        self.title("Moteur de recherche d'image")
        self.geometry("1050x720")
        self.resizable(False, False)

        # creating a container
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)

        # creating the left and right sides of the GUI
        left_side = tk.Frame(container, width=300, height=720, bg="grey")
        left_side.pack(fill="both", expand=1, side="left")
        self.right_side = tk.Frame(container, width=700, height=720)
        self.right_side.pack(fill="both", expand=1, side="right")

        # dividing the left side into two parts: top and bottom
        left_top = tk.Frame(left_side, bg='grey')
        left_top.grid(row=0)
        left_bot = tk.Frame(left_side, bg='grey')
        left_bot.grid(row=1, pady=50)

        # adding image choice to top section
        # adding label to display chosen image
        self.selected_image = ImageTk.PhotoImage(Image.open("test.jpg").resize((320, 300)))
        self.selected_image_name = "./test.jpg"
        self.image_label = tk.Label(left_top, image=self.selected_image)
        self.image_label.pack()

        # add the select image button
        select_img_btn = tk.Button(left_top, text="Selectionner une image", command=self.select_image, font=("Arial", 15))
        select_img_btn.pack(pady=20)

        # adding configuration to bottom section
        # adding inputs to select % of texture, % of color and how many samples to show
        tk.Label(left_bot, text="% de contribution couleur:", bg="grey", fg="white", font=("Arial", 15)).grid(row=0)
        self.percent_color = tk.Entry(left_bot, font=("Arial", 13))
        self.percent_color.insert(0, "0.5")
        self.percent_color.grid(row=1, pady=12)

        tk.Label(left_bot, text="% de contribution texture:", bg="grey", fg="white", font=("Arial", 15)).grid(row=2)
        self.percent_texture = tk.Entry(left_bot, font=("Arial", 13))
        self.percent_texture.insert(0, "0.5")
        self.percent_texture.grid(row=3, pady=12)

        tk.Label(left_bot, text="Nombre d'exemples a afficher:", bg="grey", fg="white", font=("Arial", 15)).grid(row=4)
        self.n_samples = tk.Entry(left_bot, font=("Arial", 13))
        self.n_samples.grid(row=5, pady=12)
        self.n_samples.insert(0, "10")

        # adding button to start search
        search_img_btn = tk.Button(left_bot, text="Rechercher", command=self.search, font=("Arial", 15))
        search_img_btn.grid(row=6, pady=12)
    
    # action to execute when select image button is pressed
    def select_image(self):
        self.selected_image_name = fido.askopenfilename(title = "Choisir une image")
        if self.selected_image_name:
            self.selected_image = ImageTk.PhotoImage(Image.open(self.selected_image_name).resize((308, 300)))
            self.image_label.config(image=self.selected_image)

    # search for selected image
    def search(self):

        dataset = "data.csv"

        # clear the existing images
        for widget in self.right_side.winfo_children():
            widget.destroy()
        
        # initialize the image descriptor
        cd = ColorDescriptor((8, 12, 3))
        # initialize the texture descriptor
        td = TextureDescriptor()

        # load the query image and describe it
        query = cv2.imread(self.selected_image_name)
        query = cv2.resize(query, (220, 220))
        features = [cd.describe(query), td.extract_textures(query)]

        # perform the search
        searcher = Searcher(dataset)
        results = searcher.search(features, textures = float(self.percent_texture.get()), colors = float(self.percent_color.get()), limit = int(self.n_samples.get()))

        # loop over the results and display images
        col = 0
        row = 0
        for _, resultID in results:
            print(resultID, row, col)
            self.imgs.append(ImageTk.PhotoImage(Image.open(resultID).resize((100, 100))))
            tk.Label(self.right_side, image=self.imgs[-1]).grid(row=row, column=col)

            col+=1
            if col >= 7:
                col = 0
                row += 1
            
if __name__ == "__main__":

    App().mainloop()