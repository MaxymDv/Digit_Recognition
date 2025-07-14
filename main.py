import tkinter as tk
from tkinter import ttk, messagebox

import visualkeras
from PIL import Image, ImageDraw, ImageTk
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy import ndimage
import cv2
from keras.utils import plot_model



class ImprovedMNISTApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Система розпізнавання цифр MNIST")
        self.root.geometry("800x600")

        # Створення головного фрейму
        main_frame = ttk.Frame(root)
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Фрейм для малювання
        draw_frame = ttk.LabelFrame(main_frame, text="Область малювання")
        draw_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Canvas для малювання (збільшений розмір)
        self.drawing_area = tk.Canvas(draw_frame, width=320, height=320, bg="white", cursor="pencil")
        self.drawing_area.pack(padx=10, pady=10)
        self.drawing_area.bind("<B1-Motion>", self.draw)
        self.drawing_area.bind("<Button-1>", self.start_draw)
        self.drawing_area.bind("<ButtonRelease-1>", self.finalize_drawing)

        # Кнопки управління
        button_frame = ttk.Frame(draw_frame)
        button_frame.pack(pady=10)

        self.clear_button = ttk.Button(button_frame, text="Очистити", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.predict_button = ttk.Button(button_frame, text="Розпізнати", command=self.predict_digit)
        self.predict_button.pack(side=tk.LEFT, padx=5)

        # Фрейм для результатів
        results_frame = ttk.LabelFrame(main_frame, text="Результати розпізнавання")
        results_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Відображення найбільш вірогідної цифри
        self.result_label = ttk.Label(results_frame, text="Результат: -", font=("Arial", 16, "bold"))
        self.result_label.pack(pady=10)

        # Фрейм для ймовірностей
        prob_canvas_frame = ttk.Frame(results_frame)
        prob_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Canvas для відображення гістограми ймовірностей
        self.prob_canvas = tk.Canvas(prob_canvas_frame, width=250, height=250, bg="white")
        self.prob_canvas.pack()

        # Фрейм для обробленого зображення
        processed_frame = ttk.LabelFrame(main_frame, text="Оброблене зображення")
        processed_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        self.processed_image_label = ttk.Label(processed_frame)
        self.processed_image_label.pack(pady=10)

        # Ініціалізація змінних для малювання
        self.image = Image.new("L", (320, 320), "white")
        self.draw_obj = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None
        self.line_width = 15

        # Завантаження або навчання моделі
        self.model = self.load_or_train_model()

        # Ініціалізація відображення
        self.clear_canvas()

    def start_draw(self, event):
        """Початок малювання"""
        self.last_x, self.last_y = event.x, event.y

    def draw(self, event):
        """Малювання на canvas"""
        if self.last_x is not None and self.last_y is not None:
            # Малювання на canvas
            self.drawing_area.create_line(self.last_x, self.last_y, event.x, event.y,
                                          width=self.line_width, fill="black",
                                          capstyle=tk.ROUND, smooth=tk.TRUE)

            # Малювання на PIL зображенні
            self.draw_obj.line((self.last_x, self.last_y, event.x, event.y),
                               fill="black", width=self.line_width)

        self.last_x, self.last_y = event.x, event.y

    def finalize_drawing(self, event):
        """Завершення малювання"""
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        """Очищення canvas та скидання результатів"""
        self.drawing_area.delete("all")
        self.image = Image.new("L", (320, 320), "white")
        self.draw_obj = ImageDraw.Draw(self.image)

        # Скидання результатів
        self.result_label.config(text="Результат: -")
        self.prob_canvas.delete("all")
        self.processed_image_label.config(image="")

    def center_image(self, img_array):
        """Центрування зображення для кращого розпізнавання"""
        # Знаходження центру мас
        cy, cx = ndimage.center_of_mass(img_array)

        # Обчислення зсуву для центрування
        rows, cols = img_array.shape
        shiftx = np.round(cols / 2.0 - cx).astype(int)
        shifty = np.round(rows / 2.0 - cy).astype(int)

        # Застосування зсуву
        return ndimage.shift(img_array, [shifty, shiftx])

    def preprocess_image(self):
        """Покращена обробка зображення"""
        # Конвертація у numpy array
        img_array = np.array(self.image)

        # Інвертування кольорів (чорне тло, білі цифри)
        img_array = 255 - img_array

        # Знаходження меж об'єкта для обрізання
        coords = np.column_stack(np.where(img_array > 50))
        if len(coords) == 0:
            # Якщо нічого не намальовано, повертаємо порожнє зображення
            return np.zeros((1, 28, 28, 1))

        # Обрізання зображення навколо цифри з відступом
        top, left = coords.min(axis=0)
        bottom, right = coords.max(axis=0)

        # Додавання відступу
        padding = 20
        top = max(0, top - padding)
        left = max(0, left - padding)
        bottom = min(img_array.shape[0], bottom + padding)
        right = min(img_array.shape[1], right + padding)

        # Обрізання
        cropped = img_array[top:bottom, left:right]

        # Створення квадратного зображення
        size = max(cropped.shape)
        square_img = np.zeros((size, size))
        y_offset = (size - cropped.shape[0]) // 2
        x_offset = (size - cropped.shape[1]) // 2
        square_img[y_offset:y_offset + cropped.shape[0], x_offset:x_offset + cropped.shape[1]] = cropped

        # Зміна розміру до 20x20
        resized = cv2.resize(square_img, (20, 20), interpolation=cv2.INTER_AREA)

        # Розміщення в центрі 28x28 зображення
        final_img = np.zeros((28, 28))
        final_img[4:24, 4:24] = resized

        # Центрування за центром мас
        final_img = self.center_image(final_img)

        # Нормалізація
        final_img = final_img / 255.0

        # Перетворення для моделі
        final_img = final_img.reshape(1, 28, 28, 1)

        return final_img, square_img

    def predict_digit(self):
        """Розпізнавання цифри"""
        if self.model is None:
            messagebox.showerror("Помилка", "Модель не завантажена.")
            return

        try:
            # Обробка зображення
            processed_img, display_img = self.preprocess_image()

            # Відображення обробленого зображення
            self.show_processed_image(processed_img[0, :, :, 0])

            # Передбачення
            predictions = self.model.predict(processed_img, verbose=0)[0]

            # Знаходження найбільш вірогідної цифри
            predicted_digit = np.argmax(predictions)
            confidence = predictions[predicted_digit] * 100

            # Оновлення результату
            self.result_label.config(text=f"Результат: {predicted_digit} ({confidence:.1f}%)")

            # Відображення гістограми ймовірностей
            self.show_probability_chart(predictions)

        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка при розпізнаванні: {str(e)}")

    def show_processed_image(self, img_array):
        """Відображення обробленого зображення"""
        # Перетворення для відображення
        display_img = (img_array * 255).astype(np.uint8)
        display_img = Image.fromarray(display_img, mode='L')
        display_img = display_img.resize((84, 84))  # Збільшення для кращого відображення

        # Конвертація для Tkinter
        photo = ImageTk.PhotoImage(display_img)
        self.processed_image_label.config(image=photo)
        self.processed_image_label.image = photo  # Збереження посилання

    def show_probability_chart(self, probabilities):
        """Відображення гістограми ймовірностей"""
        self.prob_canvas.delete("all")

        # Параметри гістограми
        width = 240
        height = 180
        margin = 10
        bar_width = (width - 2 * margin) / 10
        max_height = height - 4 * margin

        # Максимальна ймовірність для масштабування
        max_prob = max(probabilities) if max(probabilities) > 0 else 1

        # Малювання стовпців
        for i, prob in enumerate(probabilities):
            x1 = margin + i * bar_width
            x2 = x1 + bar_width - 2

            bar_height = (prob / max_prob) * max_height
            y1 = height - margin
            y2 = y1 - bar_height

            # Колір стовпця (червоний для найбільшої ймовірності)
            color = "red" if i == np.argmax(probabilities) else "lightblue"

            # Малювання стовпця
            self.prob_canvas.create_rectangle(x1, y2, x2, y1, fill=color, outline="black")

            # Підпис цифри
            self.prob_canvas.create_text(x1 + bar_width / 2 - 1, y1 + 15, text=str(i), font=("Arial", 8))

            # Значення ймовірності
            if prob > 0.01:  # Відображаємо лише значущі ймовірності
                self.prob_canvas.create_text(x1 + bar_width / 2 - 1, y2 - 10,
                                             text=f"{prob * 100:.0f}%", font=("Arial", 7))

    def load_or_train_model(self):
        """Завантаження або навчання моделі"""
        try:
            from tensorflow import keras
            model = keras.models.load_model("improved_mnist_cnn_model.h5")
            visualkeras.layered_view(model, to_file='model_visual.png', legend=True)
            print("Завантажено попередньо навчену покращену модель.")
            return model
        except FileNotFoundError:
            print("Попередньо навчена модель не знайдена. Розпочинаємо навчання моделі...")
            return self.train_model()

    def train_model(self):
        """Навчання моделі CNN"""
        try:
            print("Завантаження даних MNIST...")
            mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
            X, y = mnist.data, mnist.target.astype(int)

            # Підготовка даних
            X = X.reshape(-1, 28, 28, 1) / 255.0
            y = to_categorical(y, num_classes=10)

            # Розділення на навчальну та тестову вибірки
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            print("Створення архітектури CNN...")

            model = Sequential([
                # Перший блок згортки
                Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Dropout(0.25),

                # Другий блок згортки
                Conv2D(64, (3, 3), activation='relu'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Dropout(0.25),

                # Третій блок згортки
                Conv2D(128, (3, 3), activation='relu'),
                BatchNormalization(),
                Dropout(0.25),

                # Повнозв'язні шари
                Flatten(),
                Dense(512, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(256, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(10, activation='softmax')
            ])

            # Компіляція моделі з покращеними параметрами
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            print("Архітектура моделі:")
            model.summary()

            # Колбеки для покращення навчання
            early_stopping = EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            )

            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.00001
            )

            print("Розпочинаємо навчання моделі...")
            # Навчання моделі
            history = model.fit(
                X_train, y_train,
                batch_size=128,
                epochs=20,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping, lr_scheduler],
                verbose=1
            )

            # Оцінка моделі
            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
            print(f"Точність на тестовій вибірці: {test_accuracy:.4f}")

            # Збереження моделі
            model.save("improved_mnist_cnn_model.h5")
            plot_model(model, to_file="model_architecture.png", show_shapes=True, show_layer_names=True)
            print("Навчена модель збережена як 'mnist_cnn_model.h5'.")

            return model

        except Exception as e:
            print(f"Помилка при навчанні моделі: {str(e)}")
            messagebox.showerror("Помилка", f"Не вдалося навчити модель: {str(e)}")
            return None


if __name__ == "__main__":

    root = tk.Tk()
    app = ImprovedMNISTApp(root)
    root.mainloop()