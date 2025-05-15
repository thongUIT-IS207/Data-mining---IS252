# 0.data preprocessing
def preprocessing_page():
    st.title("🛠️ Tiền xử lý dữ liệu")
    st.header("1️⃣ Tải Tập Dữ Liệu")
    uploaded_file = st.file_uploader("Chọn file CSV để tải dữ liệu", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Hiển thị dữ liệu
        st.write("### 🔍 Dữ liệu ban đầu:")
        st.write(df.head())
        st.write(f"**Số lượng mẫu:** {df.shape[0]}, **Số lượng đặc trưng:** {df.shape[1]}")

        # ==========================
        # 2. Tiền xử lý dữ liệu
        # ==========================
        st.sidebar.header("🛠️ Tiền xử lý dữ liệu")

        remove_missing = st.sidebar.checkbox("Loại bỏ giá trị thiếu")
        standardize = st.sidebar.checkbox("Chuẩn hóa dữ liệu (StandardScaler)")
        remove_outliers = st.sidebar.checkbox("Xử lý giá trị ngoại lai (IQR)")
        filter_noise = st.sidebar.checkbox("Xử lý nhiễu (Lọc trung bình)")
        encode_labels = st.sidebar.checkbox("Mã hóa nhãn (Label Encoding)")

        # ==========================
        # 3. Nút Tiền Xử Lý
        # ==========================
        if st.sidebar.button("🚀 Tiến hành Tiền Xử Lý"):
            # Tiền xử lý dữ liệu
            if remove_missing:
                # 1. Xử lý giá trị thiếu
                df["Age"].fillna(df["Age"].median(), inplace=True)  # Điền median cho Age
                df["Discount"].fillna(0, inplace=True)  # Điền 0 cho Discount
                df["OrderDate"] = pd.to_datetime(df["OrderDate"]).fillna(pd.Timestamp("2021-01-01"))  # Điền giá trị mặc định cho OrderDate
                df["City"].fillna("Unknown", inplace=True)  # Điền "Unknown" cho City
                df["Rating"].fillna(df["Rating"].mean(), inplace=True)  # Điền giá trị trung bình cho Rating
                st.success("✅ Đã loại bỏ giá trị thiếu.")

            if standardize:
                scaler = StandardScaler()
                X = pd.DataFrame(scaler.fit_transform(X), columns=data.feature_names)
                st.success("✅ Đã chuẩn hóa dữ liệu.")

            if remove_outliers:
                # Phát hiện và loại bỏ ngoại lai sử dụng IQR
                Q1 = df.quantile(0.25)
                Q3 = df.quantile(0.75)
                IQR = Q3 - Q1
                X = X[~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)]
                st.success("✅ Đã xử lý giá trị ngoại lai (IQR).")

            if filter_noise:
                # Áp dụng bộ lọc trung bình để xử lý nhiễu
                X = X.rolling(window=3, min_periods=1).mean()
                st.success("✅ Đã xử lý nhiễu (lọc trung bình).")

            if encode_labels:
                encoder = LabelEncoder()
                y = encoder.fit_transform(y)
                st.success("✅ Đã mã hóa nhãn.")

            st.write("### 🛠️ Dữ liệu sau tiền xử lý:")
            st.write(X.head())