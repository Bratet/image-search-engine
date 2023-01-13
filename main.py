import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog as fido
from searcher import Searcher
import cv2


class App(tk.Tk):

    def __init__(self):
        super().__init__()

        self.imgs = []

        # app info
        self.title("Moteur de recherche d'image")
        self.geometry("1050x950")
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
        self.selected_image = ImageTk.PhotoImage(
            Image.open("test.jpg").resize((240, 200)))
        self.selected_image_name = "./test.jpg"
        self.image_label = tk.Label(left_top, image=self.selected_image)
        self.image_label.pack()

        # add the select image button
        select_img_btn = tk.Button(
            left_top, text="Selectionner une image", command=self.select_image, font=("Arial", 15))
        select_img_btn.pack(pady=13)

        # adding configuration to bottom section
        # adding inputs to select % of texture, % of color, % of shapes and how many samples to show
        tk.Label(left_bot, text="% de contribution couleur:",
                 bg="grey", fg="white", font=("Arial", 15)).grid(row=0)
        self.percent_color = tk.Entry(left_bot, font=("Arial", 13))
        self.percent_color.insert(0, "0.33")
        self.percent_color.grid(row=1, pady=9)

        tk.Label(left_bot, text="% de contribution texture:",
                 bg="grey", fg="white", font=("Arial", 15)).grid(row=2)
        self.percent_texture = tk.Entry(left_bot, font=("Arial", 13))
        self.percent_texture.insert(0, "0.33")
        self.percent_texture.grid(row=3, pady=9)

        tk.Label(left_bot, text="% de contribution des formes:",
                 bg="grey", fg="white", font=("Arial", 15)).grid(row=4)
        self.percent_formes = tk.Entry(left_bot, font=("Arial", 13))
        self.percent_formes.grid(row=5, pady=9)
        self.percent_formes.insert(0, "0.33")

        tk.Label(left_bot, text="Nombre d'exemples a afficher:",
                 bg="grey", fg="white", font=("Arial", 15)).grid(row=6)
        self.n_samples = tk.Entry(left_bot, font=("Arial", 13))
        self.n_samples.grid(row=7, pady=9)
        self.n_samples.insert(0, "10")


        # adding drop down lists to choose methods
        # drop down to choose colors method
        tk.Label(left_bot, text="Methode a utiliser pour filtrer en couleur:",
                 bg="grey", fg="white", font=("Arial", 15)).grid(row=8)
        color_options = ['hsv', 'rgb', 'mean']
        self.color_variable = tk.StringVar()
        self.color_variable.set(color_options[0])
        self.color_method = tk.OptionMenu(left_bot, self.color_variable, *color_options)
        self.color_method.grid(row=9, pady=9)

        # drop down to choose texture method
        tk.Label(left_bot, text="Methode a utiliser pour filtrer en couleur:",
                 bg="grey", fg="white", font=("Arial", 15)).grid(row=10)
        texture_options = ['lbp', 'glcm', 'haralick']
        self.texture_variable = tk.StringVar()
        self.texture_variable.set(texture_options[0])
        self.texture_method = tk.OptionMenu(left_bot, self.texture_variable, *texture_options)
        self.texture_method.grid(row=11, pady=9)

        # drop down to choose shape method
        tk.Label(left_bot, text="Methode a utiliser pour filtrer en forme:",
                 bg="grey", fg="white", font=("Arial", 15)).grid(row=12)
        formes_options = ['zernike', 'hu']
        self.formes_variable = tk.StringVar()
        self.formes_variable.set(formes_options[0])
        self.formes_method = tk.OptionMenu(left_bot, self.formes_variable, *formes_options)
        self.formes_method.grid(row=13, pady=9)

        # adding button to start search
        search_img_btn = tk.Button(
            left_bot, text="Rechercher", command=self.search, font=("Arial", 15))
        search_img_btn.grid(row=14, pady=9)

    # action to execute when select image button is pressed
    def select_image(self):
        self.selected_image_name = fido.askopenfilename(
            title="Choisir une image")
        if self.selected_image_name:
            self.selected_image = ImageTk.PhotoImage(
                Image.open(self.selected_image_name).resize((240, 200)))
            self.image_label.config(image=self.selected_image)

    # search for selected image
    def search(self):

        data = "data.json"

        # get chosen methods from droplists
        list_of_methods = list_of_methods = {
            "colors": self.color_variable.get(),
            "textures": self.texture_variable.get(),
            "shapes": self.formes_variable.get()
        }

        # clear the existing images
        for widget in self.right_side.winfo_children():
            widget.destroy()


        # load the query image and describe it
        query = cv2.imread(self.selected_image_name)
        query = cv2.resize(query, (220, 220))

        # perform the search
        searcher = Searcher(data)
        results = searcher.hybrid_search(query, list_of_methods, textures=float(self.percent_texture.get(
        )), colors=float(self.percent_color.get()), shapes=float(self.percent_formes.get()), limit=int(self.n_samples.get()))

        # loop over the results and display images
        col = 0
        row = 0
        for _, resultID in results:

            if resultID < 10:
                result = 'dataset\cat_000' + str(resultID) + '.jpg'
            elif resultID < 100:
                result = 'dataset\cat_00' + str(resultID) + '.jpg'
            else:
                result = 'dataset\cat_0' + str(resultID) + '.jpg'

            self.imgs.append(ImageTk.PhotoImage(
                Image.open(result).resize((100, 100))))
            tk.Label(self.right_side,
                     image=self.imgs[-1]).grid(row=row, column=col)

            col += 1
            if col >= 7:
                col = 0
                row += 1

if __name__ == "__main__":

    App().mainloop()
