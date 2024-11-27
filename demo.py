import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


# Hàm tiền xử lý dữ liệu
def preprocess_data():
    global data, preprocessed_data
    if data is None:
        messagebox.showerror("Lỗi", "Hãy tải dữ liệu trước!")
        return
    
    try:
        df = data.copy()

        # Xử lý giá trị thiếu
        if missing_values_option.get() == "Mean/Mode":
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        elif missing_values_option.get() == "Drop Rows":
            df.dropna(inplace=True)

        # Chuyển đổi categorical thành numerical
        if categorical_option.get() == "Label Encoding":
            label_encoders = {}
            for col in df.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le

        # Chuẩn hóa dữ liệu
        if normalization_option.get() == "Min-Max Scaling":
            scaler = MinMaxScaler()
            df[df.columns] = scaler.fit_transform(df[df.columns])

        preprocessed_data = df
        display_data_in_treeview(preprocessed_data, treeview_after)
        messagebox.showinfo("Thành công", "Dữ liệu đã được tiền xử lý!")
    except Exception as e:
        messagebox.showerror("Lỗi", f"Đã xảy ra lỗi trong quá trình tiền xử lý: {e}")

# Hàm train model
def train_model():
    if data is None:
        messagebox.showerror("Lỗi", "Hãy tải dữ liệu trước!")
        return
    
    try:
        # Lấy dữ liệu đầu vào và nhãn
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Chọn thuật toán
        algorithm = combobox_algorithm.get()
        if algorithm == "Random Forest":
            model = RandomForestClassifier()
        elif algorithm == "Logistic Regression":
            model = LogisticRegression()
        else:
            messagebox.showerror("Lỗi", "Chưa chọn thuật toán!")
            return
        
        # Huấn luyện và tính độ chính xác
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Hiển thị kết quả
        label_result.configure(text=f"Độ chính xác: {accuracy * 100:.2f}%")
    except Exception as e:
        messagebox.showerror("Lỗi", f"Đã xảy ra lỗi: {e}")

# Cấu hình ứng dụng
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

app = ctk.CTk()
app.title("Ứng dụng Preprocessing và Train Model")
app.geometry("900x700")

# Khung chính
main_frame = ctk.CTkFrame(app, width=900, height=700)
main_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Canvas và Scrollbar
canvas = ctk.CTkCanvas(main_frame, bg="black", highlightthickness=0)
canvas.pack(side="left", fill="both", expand=True)

scrollbar = ctk.CTkScrollbar(main_frame, command=canvas.yview)
scrollbar.pack(side="right", fill="y")
canvas.configure(yscrollcommand=scrollbar.set)

# Frame bên trong Canvas
content_frame = ctk.CTkFrame(canvas, width=900)
canvas.create_window((0, 0), window=content_frame, anchor="nw")

# Xử lý sự kiện cuộn
def configure_canvas(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

content_frame.bind("<Configure>", configure_canvas)


# Header
header_frame = ctk.CTkFrame(main_frame)
header_frame.pack(pady=20, padx=10, fill="x")

label_title = ctk.CTkLabel(header_frame, text="ỨNG DỤNG PREPROCESSING VÀ TRAIN MODEL", font=("Arial", 20, "bold"))
label_title.pack(pady=10)

# Body
body_frame = ctk.CTkFrame(main_frame, corner_radius=10)
body_frame.pack(pady=20, padx=20, fill="both", expand=True)

# Hàm tải file
def load_file():
    global data
    file_path = filedialog.askopenfilename(filetypes=[("CSV/Excel files", "*.csv *.xlsx")])
    if file_path:
        try:
            # Đọc file CSV hoặc Excel
            if file_path.endswith(".csv"):
                data = pd.read_csv(file_path)
            elif file_path.endswith(".xlsx"):
                data = pd.read_excel(file_path)
            else:
                raise ValueError("Định dạng file không được hỗ trợ!")

            label_file.configure(text=f"File: {file_path.split('/')[-1]}")
            display_data_in_treeview(data, treeview_before)
            messagebox.showinfo("Thành công", "Dữ liệu đã được tải!")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải file: {e}")
file_button = ctk.CTkButton(content_frame, text="Tải file CSV/Excel", command=upload_file)
file_button.pack(pady=10)

file_label = ctk.CTkLabel(content_frame, text="Chưa có file được tải.", font=("Arial", 12))
file_label.pack(pady=10)

# Hiển thị dữ liệu trong Treeview
def show_data():
    for widget in data_frame.winfo_children():
        widget.destroy()
    tree = ttk.Treeview(data_frame)
    tree.pack(fill="both", expand=True)

    tree["columns"] = list(data.columns)
    tree["show"] = "headings"
    for column in data.columns:
        tree.heading(column, text=column)
        tree.column(column, width=100)
    
    for _, row in data.iterrows():
        tree.insert("", "end", values=list(row))

data_frame = ctk.CTkFrame(content_frame, width=850, height=200)
data_frame.pack(fill="both", expand=True, pady=10)

# Phần tùy chọn xử lý dữ liệu
preprocessing_frame = ctk.CTkFrame(content_frame)
preprocessing_frame.pack(pady=10, fill="both", expand=True)

preprocessing_label = ctk.CTkLabel(preprocessing_frame, text="Tùy chọn tiền xử lý", font=("Arial", 14, "bold"))
preprocessing_label.grid(row=0, column=0, columnspan=2, pady=10)

fillna_options = ctk.CTkOptionMenu(preprocessing_frame, values=["Mean/Mode", "Median", "Constant"])
fillna_options.grid(row=1, column=0, padx=5, pady=5)

encoding_options = ctk.CTkOptionMenu(preprocessing_frame, values=["Label Encoding", "One-Hot Encoding"])
encoding_options.grid(row=1, column=1, padx=5, pady=5)

scaling_options = ctk.CTkOptionMenu(preprocessing_frame, values=["Min-Max Scaling", "Standard Scaling"])
scaling_options.grid(row=2, column=0, padx=5, pady=5)

process_button = ctk.CTkButton(preprocessing_frame, text="Tiền xử lý dữ liệu")
process_button.grid(row=2, column=1, padx=5, pady=5)

# Nút thực hiện tiền xử lý
btn_preprocess = ctk.CTkButton(body_frame, text="Tiền xử lý dữ liệu", command=preprocess_data)
btn_preprocess.pack(pady=20)

# Treeview hiển thị dữ liệu sau xử lý
processed_data_frame = ctk.CTkFrame(content_frame, width=850, height=200)
processed_data_frame.pack(fill="both", expand=True, pady=10)

processed_data_label = ctk.CTkLabel(processed_data_frame, text="Dữ liệu sau xử lý sẽ hiển thị ở đây.")
processed_data_label.pack()

# Chọn thuật toán
label_algorithm = ctk.CTkLabel(body_frame, text="Chọn thuật toán để train:", font=("Arial", 14))
label_algorithm.pack(pady=5)

combobox_algorithm = ctk.CTkComboBox(body_frame, values=["Random Forest", "Logistic Regression"])
combobox_algorithm.pack(pady=10)

# Train model
btn_train = ctk.CTkButton(body_frame, text="Train Model", command=train_model)
btn_train.pack(pady=20)

# Kết quả
label_result = ctk.CTkLabel(body_frame, text="Kết quả sẽ hiển thị ở đây.", font=("Arial", 14), text_color="green")
label_result.pack(pady=10)

# Chạy ứng dụng
data = None
preprocessed_data = None
app.mainloop()