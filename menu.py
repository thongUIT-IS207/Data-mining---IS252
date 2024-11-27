import os
import pandas as pd
from tkinter import filedialog, messagebox, ttk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
import customtkinter as ctk

# Ứng dụng chính
app = ctk.CTk()
app.geometry("1200x800")
app.title("Ứng dụng Tiền Xử Lý và Train Model")

# Biến toàn cục
data = None
processed_data = None
trained_model = None

# --------------------- Các hàm tiện ích ---------------------
def show_message(title, message):
    messagebox.showinfo(title, message)

def clear_frame():
    for widget in app.winfo_children():
        widget.destroy()
def auto_process_data(data, encoding_option="Label Encoding", scale_data=True):
    try:
        # Điền NaN bằng mode
        data = data.fillna(data.mode().iloc[0])

        # Phân loại cột
        numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
        categorical_columns = data.select_dtypes(include=['object']).columns
        datetime_columns = data.select_dtypes(include=['datetime64']).columns

        # Xử lý cột chuỗi
        if encoding_option == "Label Encoding":
            for col in categorical_columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
        elif encoding_option == "One-Hot Encoding":
            data = pd.get_dummies(data, columns=categorical_columns)

        # Chuyển datetime thành UNIX timestamp
        for col in datetime_columns:
            data[col] = data[col].astype('int64') // 10**9

        # Chuẩn hóa dữ liệu số
        if scale_data and len(numeric_columns) > 0:
            scaler = MinMaxScaler()
            data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

        return data
    except Exception as e:
        messagebox.showerror("Lỗi", f"Lỗi khi xử lý dữ liệu: {e}")
        return None


# Hàm đọc file và tự động nhận dạng
def read_file(file_path):
    try:
        ext = os.path.splitext(file_path)[1]
        if ext == '.csv':
            return pd.read_csv(file_path)
        elif ext in ['.xls', '.xlsx']:
            return pd.read_excel(file_path)
        else:
            messagebox.showerror("Lỗi", "Định dạng file không được hỗ trợ!")
            return None
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể đọc file: {e}")
        return None
def show_data(frame, df):
        """Hiển thị dữ liệu trong Treeview."""
        for widget in frame.winfo_children():
            widget.destroy()
        tree = ttk.Treeview(frame)
        tree.pack(fill="both", expand=True)

        tree["columns"] = list(df.columns)
        tree["show"] = "headings"
        for column in df.columns:
            tree.heading(column, text=column)
            tree.column(column, width=100)

        for _, row in df.iterrows():
            tree.insert("", "end", values=list(row))
def save_processed_data(self):
        if self.processed_data is None:
            messagebox.showwarning("Cảnh báo", "Không có dữ liệu để lưu!")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")])
        if save_path:
            try:
                ext = os.path.splitext(save_path)[1]
                if ext == ".csv":
                    self.processed_data.to_csv(save_path, index=False)
                elif ext in [".xls", ".xlsx"]:
                    self.processed_data.to_excel(save_path, index=False)
                messagebox.showinfo("Thông báo", "Lưu dữ liệu thành công!")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể lưu dữ liệu: {e}")
def process_data(self):
        if not self.file_path:
            messagebox.showwarning("Cảnh báo", "Bạn chưa tải file!")
            return
        data = read_file(self.file_path)
        if data is not None:
            encoding_option = self.encoding_combo.get()
            self.processed_data = auto_process_data(data, encoding_option=encoding_option)
            if self.processed_data is not None:
                messagebox.showinfo("Thông báo", "Tiền xử lý dữ liệu thành công!")
                self.display_data(self.processed_data)
# --------------------- Menu chính ---------------------
def show_main_menu():
    clear_frame()
    title_label = ctk.CTkLabel(app, text="CHỌN TÁC VỤ", font=("Arial", 20, "bold"))
    title_label.pack(pady=20)

    preprocess_button = ctk.CTkButton(app, text="Tiền xử lý dữ liệu", command=show_preprocessing_page, width=300)
    preprocess_button.pack(pady=10)

    train_model_button = ctk.CTkButton(app, text="Train Model Machine Learning", command=show_train_model_page, width=300)
    train_model_button.pack(pady=10)

# --------------------- Trang tiền xử lý dữ liệu ---------------------
def show_preprocessing_page():
    clear_frame()

    title_label = ctk.CTkLabel(app, text="TIỀN XỬ LÝ DỮ LIỆU", font=("Arial", 20, "bold"))
    title_label.pack(pady=10)

    back_button = ctk.CTkButton(app, text="Quay lại menu", command=show_main_menu)
    back_button.pack(pady=10)


    file_button = ctk.CTkButton(app, text="Tải file CSV/Excel", command=read_file)
    file_button.pack(pady=10)

    file_label = ctk.CTkLabel(app, text="Chưa có file được tải.", font=("Arial", 12))
    file_label.pack(pady=10)

    # Hiển thị dữ liệu
    data_frame = ctk.CTkFrame(app, width=850, height=200)
    data_frame.pack(fill="both", expand=True, pady=10)

    # Các tùy chọn tiền xử lý
    options_frame = ctk.CTkFrame(app)
    options_frame.pack(fill="both", expand=True, pady=20)

    ctk.CTkLabel(options_frame, text="Tùy chọn tiền xử lý", font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=10)

    fillna_options = ctk.CTkOptionMenu(options_frame, values=["Mean/Mode", "Median", "Constant"], width=150)
    fillna_options.grid(row=1, column=0, padx=5, pady=5)
    fillna_options.set("Mean/Mode")

    encoding_options = ctk.CTkOptionMenu(options_frame, values=["Label Encoding", "One-Hot Encoding"], width=150)
    encoding_options.grid(row=1, column=1, padx=5, pady=5)
    encoding_options.set("Label Encoding")

    scaling_options = ctk.CTkOptionMenu(options_frame, values=["Min-Max Scaling", "Standard Scaling"], width=150)
    scaling_options.grid(row=2, column=0, padx=5, pady=5)
    scaling_options.set("Min-Max Scaling")

    def process_data():
        global processed_data
        if data is None:
            show_message("Lỗi", "Chưa có dữ liệu để xử lý.")
            return

        processed_data = data.copy()

        # Xử lý NaN
        if fillna_options.get() == "Mean/Mode":
            processed_data = processed_data.fillna(processed_data.mean())
        elif fillna_options.get() == "Median":
            processed_data = processed_data.fillna(processed_data.median())
        elif fillna_options.get() == "Constant":
            processed_data = processed_data.fillna(0)

        # Mã hóa dữ liệu
        if encoding_options.get() == "Label Encoding":
            for col in processed_data.select_dtypes(include=["object"]).columns:
                le = LabelEncoder()
                processed_data[col] = le.fit_transform(processed_data[col])
        elif encoding_options.get() == "One-Hot Encoding":
            processed_data = pd.get_dummies(processed_data)

        # Chuẩn hóa dữ liệu
        if scaling_options.get() == "Min-Max Scaling":
            scaler = MinMaxScaler()
            processed_data = pd.DataFrame(scaler.fit_transform(processed_data), columns=processed_data.columns)
        elif scaling_options.get() == "Standard Scaling":
            scaler = StandardScaler()
            processed_data = pd.DataFrame(scaler.fit_transform(processed_data), columns=processed_data.columns)

        show_message("Thành công", "Tiền xử lý dữ liệu hoàn tất!")
        show_data(data_frame, processed_data)

    process_button = ctk.CTkButton(options_frame, text="Tiền xử lý dữ liệu", command=process_data)
    process_button.grid(row=2, column=1, padx=5, pady=5)

# --------------------- Trang train model ---------------------
def show_train_model_page():
    clear_frame()

    title_label = ctk.CTkLabel(app, text="TRAIN MODEL MACHINE LEARNING", font=("Arial", 20, "bold"))
    title_label.pack(pady=10)

    back_button = ctk.CTkButton(app, text="Quay lại menu", command=show_main_menu)
    back_button.pack(pady=10)

    # Upload file và chọn thuật toán
    def upload_file_for_train():
        global data
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
        if file_path:
            try:
                if file_path.endswith(".csv"):
                    data = pd.read_csv(file_path)
                elif file_path.endswith(".xlsx"):
                    data = pd.read_excel(file_path)
                file_label.configure(text=f"Đã tải file: {file_path.split('/')[-1]}")
                show_data(data_frame, data)
            except Exception as e:
                show_message("Lỗi", f"Lỗi khi tải file: {str(e)}")

    file_button = ctk.CTkButton(app, text="Tải file CSV/Excel", command=upload_file_for_train)
    file_button.pack(pady=10)

    file_label = ctk.CTkLabel(app, text="Chưa có file được tải.", font=("Arial", 12))
    file_label.pack(pady=10)

    algo_frame = ctk.CTkFrame(app)
    algo_frame.pack(pady=10, fill="both", expand=True)

    ctk.CTkLabel(algo_frame, text="Chọn thuật toán Machine Learning", font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=10)

    algo_options = ctk.CTkOptionMenu(algo_frame, values=[
        "Linear Regression", "Logistic Regression", "K-Nearest Neighbors", 
        "Gradient Boosting", "K-Means Clustering"
    ])
    algo_options.grid(row=1, column=0, padx=5, pady=5)

    def train_model():
        global processed_data, trained_model
        if processed_data is None:
            show_message("Lỗi", "Chưa có dữ liệu được tiền xử lý.")
            return

        try:
            X = processed_data.iloc[:, :-1]
            y = processed_data.iloc[:, -1]
            if algo_options.get() == "K-Means Clustering":
                model = KMeans(n_clusters=3, random_state=42)
                model.fit(X)
                show_message("Kết quả", "K-Means Training hoàn tất!")
            else:
                # Chia dữ liệu
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                if algo_options.get() == "Linear Regression":
                    model = LinearRegression()
                elif algo_options.get() == "Logistic Regression":
                    model = LogisticRegression()
                elif algo_options.get() == "K-Nearest Neighbors":
                    model = KNeighborsClassifier()
                elif algo_options.get() == "Gradient Boosting":
                    model = GradientBoostingClassifier()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred) if algo_options.get() != "Linear Regression" else mean_squared_error(y_test, y_pred)
                metric = "Accuracy" if algo_options.get() != "Linear Regression" else "MSE"
                result = f"{metric}: {acc}"
                show_message("Kết quả", f"Training hoàn tất!\n{result}")
        except Exception as e:
            show_message("Lỗi", f"Lỗi trong quá trình train model: {str(e)}")

    train_button = ctk.CTkButton(algo_frame, text="Train Model", command=train_model)
    train_button.grid(row=1, column=1, padx=5, pady=5)

# --------------------- Khởi chạy ứng dụng ---------------------
show_main_menu()
app.mainloop()
