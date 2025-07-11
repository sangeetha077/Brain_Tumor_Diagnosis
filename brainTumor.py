
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance, ImageOps
import numpy as np
import cv2
import os
import math
import random
import time
from skimage.feature import graycomatrix, graycoprops
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
# -------------------- GLCM Feature Extraction --------------------
class GLCMFeatures:
    def __init__(self):
        self.properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        
    def extract_features(self, image):
     if isinstance(image, Image.Image):
        image = np.array(image)
    
     if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    # Compute GLCM
     glcm = graycomatrix(
        image, 
        distances=[1], 
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True, 
        normed=True
    )
    
    # Print GLCM for angle 0 (for debugging)
     print("GLCM Matrix (Angle=0):")
     print(glcm[:, :, 0, 0])  # First distance, first angle
    
     features = []
     for prop in self.properties:
        features.append(graycoprops(glcm, prop).mean())
        
     return features

# -------------------- BAT Algorithm for Feature Selection --------------------
class BatAlgorithm:
    def __init__(self, dim, population_size=20, generations=100):
        self.dim = dim  # Dimension of problem (number of features)
        self.pop_size = population_size
        self.generations = generations
        self.f_min = 0  # Minimum frequency
        self.f_max = 2  # Maximum frequency
        self.A = 0.5  # Loudness
        self.r = 0.5  # Pulse rate
        self.alpha = 0.9  # Loudness decrease factor
        self.gamma = 0.9  # Pulse rate increase factor
        
    def initialize_bats(self):
        self.bats = np.random.rand(self.pop_size, self.dim)
        self.velocities = np.zeros((self.pop_size, self.dim))
        self.frequencies = np.zeros(self.pop_size)
        self.loudness = np.ones(self.pop_size) * self.A
        self.pulse_rates = np.zeros(self.pop_size)
        self.fitness = np.zeros(self.pop_size)
        
    def evaluate_fitness(self, X, y):
        """Evaluate fitness using SVM classifier"""
        for i in range(self.pop_size):
            # Get selected features (binary mask)
            selected = self.bats[i] > 0.5
            
            if np.sum(selected) == 0:
                self.fitness[i] = 0
                continue
                
            # Train SVM on selected features
            clf = svm.SVC(kernel='rbf')
            X_selected = X[:, selected]
            
            # Split data for validation
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=0.2, random_state=42)
                
            clf.fit(X_train, y_train)
            score = accuracy_score(y_test, clf.predict(X_test))
            
            # Fitness is accuracy minus a penalty for too many features
            self.fitness[i] = score - 0.01 * np.sum(selected)
            
    def run(self, X, y):
        self.initialize_bats()
        self.evaluate_fitness(X, y)
        
        best_bat = np.argmax(self.fitness)
        best_solution = self.bats[best_bat].copy()
        best_fitness = self.fitness[best_bat]
        
        for t in range(self.generations):
            for i in range(self.pop_size):
                # Update frequency
                self.frequencies[i] = self.f_min + (self.f_max - self.f_min) * np.random.rand()
                
                # Update velocity
                self.velocities[i] += (self.bats[i] - best_solution) * self.frequencies[i]
                
                # Update position
                new_solution = self.bats[i] + self.velocities[i]
                
                # Apply random walk if condition met
                if np.random.rand() > self.pulse_rates[i]:
                    new_solution = best_solution + 0.001 * np.random.randn(self.dim)
                
                # Apply bounds
                new_solution = np.clip(new_solution, 0, 1)
                
                # Evaluate new solution
                selected = new_solution > 0.5
                if np.sum(selected) > 0:
                    clf = svm.SVC(kernel='rbf')
                    X_selected = X[:, selected]
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_selected, y, test_size=0.2, random_state=42)
                    clf.fit(X_train, y_train)
                    new_fitness = accuracy_score(y_test, clf.predict(X_test)) - 0.01 * np.sum(selected)
                else:
                    new_fitness = 0
                
                # Update if solution improves and meets loudness condition
                if (new_fitness > self.fitness[i]) and (np.random.rand() < self.loudness[i]):
                    self.bats[i] = new_solution
                    self.fitness[i] = new_fitness
                    self.loudness[i] *= self.alpha
                    self.pulse_rates[i] = self.r * (1 - np.exp(-self.gamma * t))
                    
                    # Update global best if needed
                    if new_fitness > best_fitness:
                        best_solution = new_solution.copy()
                        best_fitness = new_fitness
            
            print(f"Generation {t+1}, Best Fitness: {best_fitness:.4f}")
        
        return best_solution > 0.5  # Return binary feature mask

# -------------------- SVM Classifier --------------------
class SVMTumorClassifier:
    def __init__(self):
        self.clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
        self.scaler = StandardScaler()
        self.glcmextractor = GLCMFeatures()
        self.feature_selector = None
        self.selected_features = None
        self.trained = False

    def extract_features(self, image):
        """Extract features including GLCM"""
        # Basic intensity features
        gray = np.array(image.convert("L"))
        intensity_features = [
            gray.mean(), gray.std(), gray.min(), gray.max()
        ]
        
        # GLCM texture features
        glcm_features = self.glcmextractor.extract_features(gray)
        
        # Combine all features
        return np.concatenate([intensity_features, glcm_features])

    def train(self, image_paths, labels):
        """Train with feature selection using BAT algorithm"""
        # Extract features from all images
        features = []
        for path in image_paths:
            img = Image.open(path)
            features.append(self.extract_features(img))
        
        X = np.array(features)
        y = np.array(labels)

        #
        if X.size == 0:
            print("❌ Error: No features found. Check if 'norm' folder has valid images and feature extraction is working.")
            return
        #
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Feature selection using BAT algorithm
        print("Running BAT algorithm for feature selection...")
        self.feature_selector = BatAlgorithm(dim=X.shape[1])
        self.selected_features = self.feature_selector.run(X, y)
        print(f"Selected {np.sum(self.selected_features)} features")
        
        # Train SVM on selected features
        X_selected = X[:, self.selected_features]
        self.clf.fit(X_selected, y)
        self.trained = True
        
        # # Evaluate
        # preds = self.clf.predict(X_selected)
        # acc = accuracy_score(y, preds)
        # cm = confusion_matrix(y, preds)
        
        # print(f"Training Accuracy: {acc:.2f}")
        # print("Confusion Matrix:")
        # print(cm)
        
        return None

    def predict(self, image):
        """Predict class for a single image"""
        if not self.trained:
            raise ValueError("Model not trained yet")
            
        features = self.extract_features(image)
        features = self.scaler.transform([features])
        if self.selected_features is not None:
            features = features[:, self.selected_features]
        return self.clf.predict(features)[0]

# -------------------- MainFrame (Primary GUI) --------------------
class Details:
    def __init__(self):
       
        self.trPath = r"C:\Users\Admin\Documents\final year project\final year project\TrainImage"
        self.grPath = r"C:\Users\Admin\Documents\final year project\final year project\GroundTruth"
        self.biasPath = os.path.join(os.getcwd(), "Bias")
        self.normPath = os.path.join(os.getcwd(), "Norm")
        
        # Create directories if they don't exist
        os.makedirs(self.biasPath, exist_ok=True)
        os.makedirs(self.normPath, exist_ok=True)
class MainFrame(tk.Tk):
    def __init__(self):
        super().__init__()
        self.details = Details()
        self.svm_classifier = SVMTumorClassifier()
        self.title("Brain Tumor Classification - SVM with BAT Feature Selection")
        self.configure(bg="white")
        self.create_widgets()
        self.deleteImages("Bias")
        self.deleteImages("Norm")

    def create_widgets(self):
        btn1 = tk.Button(self, text="Load Training Images", font=("Andalus", 17), command=self.loadTrainingImages)
        btn2 = tk.Button(self, text="Preprocess Train Set", font=("Andalus", 17), command=self.preprocessTrainSet)
        btn3 = tk.Button(self, text="Train SVM with BAT", font=("Andalus", 17), command=self.trainSVM, state="disabled")
        btn4 = tk.Button(self, text="Test Image", font=("Andalus", 17), command=self.testImage, state="disabled")
        btn1.pack(pady=10)
        btn2.pack(pady=10)
        btn3.pack(pady=10)
        btn4.pack(pady=10)
        self.btn3 = btn3
        self.btn4 = btn4

    # ... [Keep all other methods exactly the same as before]
    def deleteImages(self, folder):
        if os.path.exists(folder):
            for f in os.listdir(folder):
                os.remove(os.path.join(folder, f))

    def loadTrainingImages(self):
        #TrainImgFrame().mainloop()
        #Population().mainloop()
        TrainImgFrame()
        Population()

    def preprocessTrainSet(self):
      input_folder = self.details.trPath
      output_folder = os.path.join(os.getcwd(), "Preprocessed")
    
      try:
        os.makedirs(output_folder, exist_ok=True)
        files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(input_folder, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                # Basic preprocessing - adjust as needed
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                img = cv2.medianBlur(img, 3)
                
                output_path = os.path.join(output_folder, file)
                cv2.imwrite(output_path, img)
                
        messagebox.showinfo("Success", f"Preprocessed {len(files)} images")  # ✅ Shows correct count
        self.btn3.config(state="normal")
      except Exception as e:
        messagebox.showerror("Error", f"Preprocessing failed: {str(e)}")
    def batTraining(self):
        unitsPerPixel = 1
        # folder = "Norm"
        folder = r"C:\Users\Admin\Documents\final year project\final year project\Norm"
        files = os.listdir(folder)
        imgset = Imageset()
        for fname in files:
            path = os.path.join(folder, fname)
            image = Image.open(path)
            inst = ImageInst(image)
            imgset.add(inst)
        if imgset.getSize() > 0:
            sampleImage = imgset.getImages()[0]
            inVecSize = sampleImage.getWidth() * sampleImage.getHeight() * unitsPerPixel + 1
            trainFeatVec = []
            for inst in imgset.getImages():
                gray = inst.getGrayImage() / 255.0
                flat = gray.flatten().tolist()
                flat.append(0) # dummy label
                trainFeatVec.append(flat)
            countActivationMapsConv = 10
            kernelSizeConv = 5
            countClasses = 2
            padding = 0
            stride = 1
            hparam = (padding << 28) | (countClasses << 24) | (stride << 16) | (kernelSizeConv << 8) | countActivationMapsConv
            CNN = ConvNet(trainFeatVec, hparam, False)
            errorCount = CNN.trainCNN(trainFeatVec)
            messagebox.showinfo("Info", "Bat Training Completed....")
            self.btn4.config(state="normal")

    def trainSVM(self):
        folder = r"C:\Users\Admin\Documents\final year project\final year project\Norm"
        files = os.listdir(folder)
        
        # Create dummy labels (0 for normal, 1 for tumor)
        # Replace with your actual labels
        labels = [random.randint(0, 1) for _ in files]  
        
        # Get full paths
        image_paths = [os.path.join(folder, f) for f in files]
        
        # Train SVM with BAT feature selection
        accuracy = self.svm_classifier.train(image_paths, labels)
        messagebox.showinfo("Info SVM Training Completed")
        self.btn4.config(state="normal")

    def testImage(self):
        TestFrame(self.svm_classifier)

# ... [Keep all other classes exactly the same as before]
class TrainImgFrame(tk.Toplevel):
    def __init__(self):
        super().__init__()
        self.title("Training Images")
        self.geometry("800x600")
        self.details = Details()
        self.labels = []  # Initialize labels list
        self.photos = []  # To keep references to PhotoImage objects
        self.displayImages()

    def displayImages(self):
        # Clear any existing widgets
        for widget in self.winfo_children():
            widget.destroy()
        
        self.labels = []  # Reset labels list
        self.photos = []  # Reset photo references
        
        folder = self.details.trPath
        if not folder or not os.path.exists(folder):
            messagebox.showerror("Error", "Training images path not set or invalid")
            self.destroy()
            return

        # Get all image files
        image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            messagebox.showinfo("Info", "No images found in the selected folder")
            self.destroy()
            return

        # Create a canvas with scrollbar
        canvas = tk.Canvas(self)
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Display images in a grid
        rows = math.ceil(len(image_files) / 4)
        for i, file in enumerate(image_files):
            try:
                img_path = os.path.join(folder, file)
                img = Image.open(img_path)
                img.thumbnail((150, 150))  # Resize for display
                photo = ImageTk.PhotoImage(img)
                self.photos.append(photo)  # Keep reference

                # Create label for each image
                label = tk.Label(scrollable_frame, image=photo)
                label.grid(row=i//4, column=i%4, padx=5, pady=5)
                self.labels.append(label)  # Add to labels list

                # Add filename below image
                tk.Label(scrollable_frame, text=file).grid(row=i//4, column=i%4, sticky="n", pady=(0,5))

            except Exception as e:
                print(f"Error loading {file}: {str(e)}")

        # Add a close button
        close_btn = tk.Button(self, text="Close", command=self.destroy)
        close_btn.pack(side="bottom", pady=10)

# -------------------- Population (Ground Truth GUI) --------------------
class Population(tk.Toplevel):
    def __init__(self):
        super().__init__()
        self.title("Ground Truth Images")
        self.configure(bg="white")
        self.create_widgets()
        self.displayImages()

    def create_widgets(self):
        lbl = tk.Label(self, text="Sample Brain Tumor Ground Truth", font=("Andalus", 30), bg="white")
        lbl.pack(pady=10)
        self.frame = tk.Frame(self, bg="white")
        self.frame.pack(pady=10)
        self.labels = []
        for _ in range(5):
            l = tk.Label(self.frame, bg="white")
            l.pack(side=tk.LEFT, padx=5)
            self.labels.append(l)

    def resize_image(self, image, width, height):
       return image.resize((width, height), Image.LANCZOS)


    def displayImages(self):
        dt = Details()
        folder = dt.grPath
        files = os.listdir(folder)
        for i in range(min(5, len(files))):
            path = os.path.join(folder, files[i])
            image = Image.open(path)
            image = self.resize_image(image, 128, 128)
            photo = ImageTk.PhotoImage(image)
            self.labels[i].config(image=photo)
            self.labels[i].image = photo

# -------------------- PreprocessFrame (Preprocessed Image GUI) --------------------
class PreprocessFrame(tk.Toplevel):
    def __init__(self):
        super().__init__()
        self.title("Preprocessed Image")
        self.configure(bg="white")
        self.create_widgets()
        self.display1()
        self.display2()

    def create_widgets(self):
        lbl = tk.Label(self, text="Sample Preprocessed Image", font=("Andalus", 30), bg="white")
        lbl.pack(pady=10)
        self.frame1 = tk.Frame(self, bg="white")
        self.frame1.pack(pady=5)
        self.biasLabels = []
        for _ in range(5):
            l = tk.Label(self.frame1, bg="white")
            l.pack(side=tk.LEFT, padx=5)
            self.biasLabels.append(l)
        self.frame2 = tk.Frame(self, bg="white")
        self.frame2.pack(pady=5)
        self.normLabels = []
        for _ in range(5):
            l = tk.Label(self.frame2, bg="white")
            l.pack(side=tk.LEFT, padx=5)
            self.normLabels.append(l)

    def resize_image(self, image, width, height):
        return image.resize((width, height), Image.LANCZOS)


    def display1(self):
        folder =  r"C:\Users\Admin\Documents\final year project\final year project\Bias"
        files = os.listdir(folder)
        for i in range(min(5, len(files))):
            path = os.path.join(folder, files[i])
            image = Image.open(path)
            image = self.resize_image(image, 128, 128)
            photo = ImageTk.PhotoImage(image)
            self.biasLabels[i].config(image=photo)
            self.biasLabels[i].image = photo

    def display2(self):
        folder =  r"C:\Users\Admin\Documents\final year project\final year project\Norm"
        files = os.listdir(folder)
        for i in range(min(5, len(files))):
            path = os.path.join(folder, files[i])
            image = Image.open(path)
            image = self.resize_image(image, 128, 128)
            photo = ImageTk.PhotoImage(image)
            self.normLabels[i].config(image=photo)
            self.normLabels[i].image = photo

# -------------------- TestFrame (Test Image GUI) --------------------
class Bias1:
    def __init__(self):
        self.kernel_size = 5
        self.sigma = 1.0
        
    def getBias(self, image):
        """Apply bias field correction to an image"""
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        # Apply Gaussian blur to estimate bias field
        bias_field = cv2.GaussianBlur(image.astype(np.float32), 
                                    (self.kernel_size, self.kernel_size), 
                                    self.sigma)
        
        # Correct the image by dividing by bias field (avoid division by zero)
        bias_field[bias_field == 0] = 1e-6
        corrected = image.astype(np.float32) / bias_field
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        
        return Image.fromarray(corrected)
        
    def normalize1(self, image):
        """Normalize image intensities"""
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Normalize to 0-255 range
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return Image.fromarray(normalized.astype(np.uint8))
    
class TestFrame(tk.Toplevel):
    def __init__(self, svm_classifier=None):  # Add optional classifier parameter
        super().__init__()
        self.svm_classifier = svm_classifier  # Store the classifier
        self.title("Test Image")
        self.configure(bg="white")
        self.create_widgets()
        self.current_image = None  # To store the loaded image

    def create_widgets(self):
        lbl = tk.Label(self, text="Brain Tumor Segmentation", font=("Andalus", 30), bg="white")
        lbl.pack(pady=10)
        
        frm = tk.Frame(self, bg="white")
        frm.pack(pady=10)
        
        lbl2 = tk.Label(frm, text="Select Image", font=("Andalus", 17), bg="white")
        lbl2.grid(row=0, column=0, padx=5)
        
        self.entry = tk.Entry(frm, width=50, font=("Andalus", 17))
        self.entry.grid(row=0, column=1, padx=5)
        
        btnBrowse = tk.Button(frm, text="Browse", font=("Andalus", 17), command=self.browse)
        btnBrowse.grid(row=0, column=2, padx=5)
        
        self.preview = tk.Label(self, bg="white")
        self.preview.pack(pady=10)
        
        btnPreprocess = tk.Button(self, text="Preprocess", font=("Andalus", 17), command=self.preprocess)
        btnPreprocess.pack(pady=10)
        
        # Add Classify button if classifier is available
        if self.svm_classifier:
            btnClassify = tk.Button(self, text="Classify Tumor", font=("Andalus", 17), 
                                 command=self.classify_image)
            btnClassify.pack(pady=10)
            
        self.result_label = tk.Label(self, text="", font=("Andalus", 17), bg="white")
        self.result_label.pack(pady=10)

    def resize_image(self, image, width, height):
        return image.resize((width, height), Image.LANCZOS)

    def browse(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if path:
            self.entry.delete(0, tk.END)
            self.entry.insert(0, path)
            try:
                self.current_image = Image.open(path)
                image = self.resize_image(self.current_image, 256, 256)
                photo = ImageTk.PhotoImage(image)
                self.preview.config(image=photo)
                self.preview.image = photo
                self.result_label.config(text="")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def preprocess(self):
        path = self.entry.get().strip()
        if not path:
            messagebox.showwarning("Warning", "Select Test Image")
            return
        
        try:
            bs = Bias1()
            image = Image.open(path)
            
            # Process and save images
            bi2 = bs.getBias(image)
            bi2.save(os.path.join(os.getcwd(), "bias1.jpg"))
            
            bi3 = bs.normalize1(bi2)
            bi3.save(os.path.join(os.getcwd(), "norm1.jpg"))
            
            # Show preprocessing results
            tf = TestPreprocessFrame(path)
            lblBias = tk.Label(tf, image=ImageTk.PhotoImage(bi2))
            lblNorm = tk.Label(tf, image=ImageTk.PhotoImage(bi3))
            lblBias.pack()
            lblNorm.pack()
            
        except Exception as e:
            messagebox.showerror("Error", f"Preprocessing failed: {str(e)}")

    def classify_image(self):
        if not self.current_image:
            messagebox.showwarning("Warning", "No image loaded")
            return
            
        if not self.svm_classifier:
            messagebox.showerror("Error", "Classifier not available")
            return
            
        try:
            # Preprocess the image first
            bs = Bias1()
            processed_img = bs.normalize1(bs.getBias(self.current_image))
            
            # Classify the image
            prediction = self.svm_classifier.predict(processed_img)
            result = "Tumor Detected" if prediction == 1 else "Normal Brain"
            color = "red" if prediction == 1 else "green"
            
            self.result_label.config(text=f"Result: {result}", fg=color)
            
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {str(e)}")

# -------------------- TestPreprocessFrame (Test Preprocessing GUI) --------------------
class TestPreprocessFrame(tk.Toplevel):
    def __init__(self, path):
        super().__init__()
        self.title("Test Preprocessing")
        self.configure(bg="white")
        self.path = path
        self.create_widgets()

    def create_widgets(self):
        lbl = tk.Label(self, text="Preprocessing", font=("Andalus", 30), bg="white")
        lbl.pack(pady=10)
        frm = tk.Frame(self, bg="white")
        frm.pack(pady=10)
        self.jLabel2 = tk.Label(frm, bg="white")
        self.jLabel2.grid(row=0, column=0, padx=10)
        self.jLabel3 = tk.Label(frm, bg="white")
        self.jLabel3.grid(row=0, column=1, padx=10)
        btnSegment = tk.Button(self, text="Segment", font=("Andalus", 17), command=self.segment)
        btnSegment.pack(pady=10)

    def segment(self):
        try:
            # For demonstration, use RegionGrowing segmentation
            image = Image.open(self.path)
            # Convert PIL image to cv2 image
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            rg = RegionGrowing(cv_image, preprocess=True)
            rg.run()
            seg = rg.getOutput()
            #seg.save(r"C:\Users\Admin\Documents\final year project\final year project\seg.jpg")
            import os
            import uuid
# Folder where segmented images will be saved
            norm_path = r"C:\Users\Admin\Documents\final year project\final year project\norm"
# Create the folder if it doesn't exist
            if not os.path.exists(norm_path):
                os.makedirs(norm_path)
                filename = f"{uuid.uuid4().hex}.jpg"
                seg.save(os.path.join(norm_path, filename))
                print("✅ Image saved:", os.path.join(norm_path, filename))
                ResultFrame()
            except Exception as e:
                print(e)

# -------------------- ResultFrame (Segmented Result GUI) --------------------
class ResultFrame(tk.Toplevel):
    def __init__(self):
        super().__init__()
        self.title("Segmented Result")
        self.configure(bg="white")
        lbl = tk.Label(self, text="Segmented Result", font=("Andalus", 30), bg="white")
        lbl.pack(pady=10)
        self.imgLabel = tk.Label(self, bg="white")
        self.imgLabel.pack(pady=10)
        self.display()

    def display(self):
        with Image.open(r"C:\Users\Admin\Documents\final year project\final year project\seg.jpg") as image:
            resized = image.resize((331, 299), resample=Image.LANCZOS)
        photo = ImageTk.PhotoImage(resized)
        self.imgLabel.config(image=photo)
        self.imgLabel.image = photo
# -------------------- ConvNet (Overall Neural Network) --------------------
class ConvNet:
    def __init__(self, inputFeatureVectors, hyperparameters, debugSwitch):
        self.conv1 = Bat(inputFeatureVectors, hyperparameters, False)
        self.maxPool1 = Fitness(self.conv1, False)
        
        # For conv2, create empty feature maps with correct dimensions
        conv2_input_size = self.maxPool1.outputVolume()
        dummy_feature_vectors = [[0]*(conv2_input_size**2 + 1)]  # +1 for label
        self.conv2 = Bat(dummy_feature_vectors, hyperparameters, False)
        
        self.maxPool2 = Fitness(self.conv2, False)
        self.flat = FrequencyLayer(self.maxPool2, False)
        self.out = PulseEmission(self.flat, hyperparameters, False)

    # In your ConvNet class:
def trainCNN(self, trainFeatVec):
    # First conv-pool
    self.conv1.train(trainFeatVec)
    self.maxPool1.train(self.conv1)
    
    # Create proper input for conv2
    conv2_input = []
    for img_features in trainFeatVec:
        # Process through first conv-pool
        self.conv1.train([img_features])
        self.maxPool1.train(self.conv1)
        
        # Extract and format features
        pooled_features = []
        for pm in self.maxPool1.get_P_maps():
            pooled_features.extend(pm.getOutput().flatten().tolist())
        pooled_features.append(img_features[-1])  # Preserve label
        conv2_input.append(pooled_features)
    
    # Second conv-pool
    self.conv2.train(conv2_input)  # Now passing proper input format
    self.maxPool2.train(self.conv2)
    
    # Continue with rest of network
    self.flat.train(self.maxPool2)
    self.out.train(self.flat)
    self.out.backpropagate()
    self.out.printPrediction()
    errorCount += self.out.reportPredictionError()
    
    return errorCount

    def testCNN(self, testFeatureVectors):
        self.out.resetCountCorrect()
        self.out.zeroConfusionMatrix()
        errorCount = 0
        for vec in testFeatureVectors:
            self.conv1.train(vec)
            self.maxPool1.train(self.conv1)
            self.conv2.train(self.maxPool1)
            self.maxPool2.train(self.conv2)
            self.flat.train(self.maxPool2)
            self.out.train(self.flat)
            self.out.printPrediction()
            errorCount += self.out.reportPredictionError()
        self.out.printConfusion()
        return errorCount

# -------------------- RegionGrowing (Fuzzy Region Growing Segmentation) --------------------
class RegionGrowing:
    def __init__(self, image, preprocess):
        # image is a cv2 image
        if preprocess:
            self.input = self.preprocess(image)
        else:
            self.input = image
        self.height = self.input.shape[0]
        self.width = self.input.shape[1]
        self.labels = -np.ones((self.width, self.height), dtype=int)
        self.pixels = np.zeros((self.width, self.height), dtype=np.uint8)  
        for h in range(self.height):
            for w in range(self.width):
                self.pixels[w, h] = int(self.input[h, w, 0])
        self.position = 0
        self.count = {}
        self.numberOfRegions = 0

    def preprocess(self, input):
        gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def run(self):
        stack = []
        for h in range(self.height):
            for w in range(self.width):
                self.position += 1
                if self.labels[w, h] < 0:
                    self.numberOfRegions += 1
                    stack.append((w, h))
                    self.labels[w, h] = self.numberOfRegions
                    self.count[self.numberOfRegions] = 1
                    while stack:
                        x, y = stack.pop(0)
                        for th in range(-1, 2):
                            for tw in range(-1, 2):
                                rx = x + tw
                                ry = y + th
                                if rx < 0 or ry < 0 or ry >= self.height or rx >= self.width:
                                    continue
                                if self.labels[rx, ry] < 0 and self.pixels[rx, ry] == self.pixels[x, y]:
                                    stack.append((rx, ry))
                                    self.labels[rx, ry] = self.numberOfRegions
                                    self.count[self.numberOfRegions] += 1
        self.position = self.width * self.height

    def getNumberOfRegions(self):
        return self.numberOfRegions

    def getPixelCount(self, region):
        return self.count.get(region, -1)

    def getSize(self):
        return self.width * self.height

    def getPosition(self):
        return self.position

    def isFinished(self):
        return self.position == self.width * self.height

    def getOutput(self):
        arr = self.labels.copy()
        arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-6) * 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")

    def getInternalImage(self):
        return Image.fromarray(cv2.cvtColor(self.input, cv2.COLOR_BGR2RGB))


# -------------------- Main --------------------
class Main:
    @staticmethod
    def main():
        mf = MainFrame()
        mf.mainloop()

if __name__ == "__main__":
    Main.main()
