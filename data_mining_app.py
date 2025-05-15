import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_carousel import carousel
from sklearn.datasets import load_iris, make_blobs
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from mlxtend.frequent_patterns import apriori, association_rules
import time
from minisom import MiniSom
from itertools import combinations

st.markdown(
    """
    <style>
        body {
            background-color: #f4f4f4;
            color: #333333;
        }
        .stButton > button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            font-size: 1.2em;
            border-radius: 12px;
            padding: 10px;
            transition: background-color 0.3s, transform 0.2s;
        }
        .stButton > button:hover {
            background-color: #45a049;
            transform: translateY(-3px);
        }
        .stHeader {
            background-color: #4caf50;
            color: white;
            padding: 10px;
            border-radius: 8px;
        }
        .stCard {
            background-color: #f4f4f4;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
            margin: 10px 0;
            text-align: center;
            transition: all 0.3s ease;
        }
        .stCard:hover {
            background-color: #e6f7ff;
            box-shadow: 0px 6px 20px rgba(0, 0, 0, 0.15);
            transform: translateY(-5px);
        }
        .title {
            font-size: 2.5em;
            text-align: center;
            color: #4CAF50;
            margin-bottom: 1em;
        }
        .stCard h3 {
            color: #007ACC;
            font-size: 1.5em;
            margin-bottom: 10px;
        }
        .stCard p {
            color: #555;
            font-size: 1em;
        }
        .button-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .carousel-container {
            margin-top: 20px;
            margin-bottom: 40px;
        }
    </style>
    """,
    unsafe_allow_html=True
)
# Header
st.markdown("<h1 class='stHeader'>🌈 Ứng Dụng Data Mining</h1>", unsafe_allow_html=True)
# Hàm hiển thị giao diện chính
def main_page():
    st.markdown("<h2>🏠 Trang Chủ</h2>", unsafe_allow_html=True)
    st.write("Chào mừng bạn đến với ứng dụng Data Mining!")

    st.markdown("<div class='carousel-container'>", unsafe_allow_html=True)

    carousel_items = [
    {"title": "🛠️ Data Preprocessing", "text": "Tiền xử lý dữ liệu để làm sạch và chuẩn hóa trước khi phân tích.", "img": "https://via.placeholder.com/600x300?text=Data+Preprocessing"},
    {"title": "🌳 Decision Tree", "text": "Xây dựng và trực quan hóa cây quyết định để phân loại dữ liệu.", "img": "https://via.placeholder.com/600x300?text=Decision+Tree"},
    {"title": "📊 Rough Set Analysis", "text": "Phân tích tập thô để tìm luật và rút gọn dữ liệu.", "img": "https://via.placeholder.com/600x300?text=Rough+Set"},
    {"title": "📉 K-Means Clustering", "text": "Phân cụm dữ liệu thành các nhóm bằng thuật toán K-Means.", "img": "https://via.placeholder.com/600x300?text=K-Means+Clustering"},
    {"title": "🤖 Naive Bayes", "text": "Phân loại dữ liệu bằng thuật toán Naive Bayes.", "img": "https://via.placeholder.com/600x300?text=Naive+Bayes"}
    ]

    carousel(items=carousel_items, width=600)
    st.markdown("</div>", unsafe_allow_html=True)
    # Hiển thị dữ liệu mẫu trong card
    st.markdown("<div class='stCard'><h3>📊 Dữ liệu Mẫu</h3></div>", unsafe_allow_html=True)
    st.write(df)
    st.write("Chọn một thuật toán để khám phá:")
    
    algorithms = {
        "🛠️ Data Preprocessing": preprocessing_page,
        "🌳 Decision Tree": decision_tree_page,
        "📊 Rough Set Analysis": rough_set_page,
        "📉 K-Means Clustering": kmeans_page,
        "🔗 Apriori Association Rule": apriori_page,
        "🧠 Kohonen Self-Organizing Map": kohonen_page,
        "🤖 Naive Bayes Classification": naive_bayes_page,
        "🌲 Random Forest": random_forest_page
    }
    num_cols = 2
    st.markdown("<div class='button-container'>", unsafe_allow_html=True)
    algo_names = list(algorithms.keys())
    for i in range(0, len(algo_names), num_cols):
        cols = st.columns(num_cols)
        for col, algo_name in zip(cols, algo_names[i:i + num_cols]):
            if col.button(algo_name):
                st.session_state["current_page"] = algorithms[algo_name]
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
# Dữ liệu fix cứng
@st.cache_data
def load_data():
    data = {
        'Age': [25, 32, 47, 51, 62, 28, 35, 43, 52, 45],
        'Income': [50000, 60000, 80000, 120000, 150000, 52000, 58000, 79000, 115000, 87000],
        'Education': ['Bachelors', 'Masters', 'PhD', 'PhD', 'Masters', 'Bachelors', 'Bachelors', 'Masters', 'PhD', 'Masters'],
        'Purchase': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes']
    }
    return pd.DataFrame(data)

df = load_data()
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

            # 3. Chuẩn hóa dữ liệu
            if standardize:
                scaler = StandardScaler()
                numeric_columns = ['Age', 'Price', 'Discount', 'Rating']
                df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
                st.success("✅ Đã chuẩn hóa dữ liệu.")

            if remove_outliers:
                numeric_columns = ['Age', 'Price', 'Discount', 'Rating']
                for col in numeric_columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
                st.success("✅ Đã xử lý giá trị ngoại lai (IQR).")


            # 4. Xử lý nhiễu
            if filter_noise:
                numeric_columns = ['Age', 'Price', 'Discount', 'Rating']
                df[numeric_columns] = df[numeric_columns].rolling(window=3, min_periods=1).mean()
                st.success("✅ Đã xử lý nhiễu (lọc trung bình).")

            # 5. Mã hóa nhãn
            if encode_labels:
                categorical_columns = ['CustomerName', 'Product', 'Category', 'City', 'State', 'Country']
                encoder = LabelEncoder()
                for col in categorical_columns:
                    df[col] = encoder.fit_transform(df[col])
                st.success("✅ Đã mã hóa nhãn.")

            st.write("### 🛠️ Dữ liệu sau tiền xử lý:")
            st.write(df.head())
    else:
        st.warning("⚠️ Vui lòng tải file CSV để bắt đầu.")

    if st.button("⬅️ Back to Home"):
        st.session_state["current_page"] = main_page
        st.rerun()

# 1. Decision Tree
def decision_tree_page():
    st.markdown("<h2>🧠 Decision Tree</h2>", unsafe_allow_html=True)
    st.header("1️⃣ Tải Tập Dữ Liệu")
    uploaded_file = st.file_uploader("Chọn file CSV để tải dữ liệu", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)

        target = st.selectbox("Chọn cột mục tiêu (label):", df.columns, index=3)
        features = st.multiselect("Chọn cột đặc trưng (features):", df.columns, default=['Outlook', 'Temperature'])

        if len(features) == 0:
            st.warning("❗ Vui lòng chọn ít nhất một cột đặc trưng.")
            return
        
        # Chọn tiêu chí phân tách (Gini hoặc Entropy)
        criterion = st.selectbox("Chọn tiêu chí phân tách:", ["gini", "entropy"], index=0)

        if st.button("🚀 Tạo Cây Quyết Định"):
            with st.spinner("🔄 Đang tạo cây quyết định..."):
                time.sleep(2)  # Giả lập thời gian xử lý
                for column in features:
                    if df[column].dtype in ['object', 'string']:
                        df, notes = encode_column_with_notes(df, column)
                # Tạo và huấn luyện mô hình với tiêu chí đã chọn
                X = df[features]
                y = df[target]
                # Chuẩn hóa dữ liệu
                model = DecisionTreeClassifier(criterion=criterion, random_state=42)
                model.fit(X, y)

                st.success(f"✅ Cây quyết định đã được tạo thành công với tiêu chí '{criterion}'!")
                plt.figure(figsize=(12, 8))
                plot_tree(model, feature_names=features, class_names=model.classes_, filled=True)
                st.pyplot(plt)

                fig, ax = plt.subplots(figsize=(12, 8))  # Kích thước biểu đồ
                plot_tree(
                    model, 
                    feature_names=features, 
                    class_names=[str(c) for c in model.classes_], 
                    filled=True, 
                    rounded=True,  # Làm tròn ô nút
                    impurity=False,  # Ẩn chỉ số impurity
                    ax=ax
                )
                st.pyplot(fig)
    else:
        st.warning("⚠️ Vui lòng tải file CSV để bắt đầu.")

    if st.button("⬅️ Back to Home"):
        st.session_state["current_page"] = main_page
        st.rerun()

# thuat toan doi kieu chu thanh so
def encode_column_with_notes(df, column):
    """
    Hàm chuyển đổi cột dạng chuỗi thành số và ghi lại chú thích.
    
    Args:
        df (pd.DataFrame): DataFrame cần chuyển đổi.
        column (str): Tên cột cần chuyển đổi.

    Returns:
        df (pd.DataFrame): DataFrame sau khi chuyển đổi.
        notes (dict): Chú thích ghi lại các giá trị cũ và giá trị số tương ứng.
    """
    le = LabelEncoder()
    original_values = df[column].unique()
    df[column] = le.fit_transform(df[column])
    encoded_values = df[column].unique()
    notes = {original: encoded for original, encoded in zip(original_values, encoded_values)}
    return df, notes

# 2. K-Means Clustering
def kmeans_page():
    st.markdown("<h2>📊 K-Means Clustering</h2>", unsafe_allow_html=True)
    st.header("1️⃣ Tải Tập Dữ Liệu")
    uploaded_file = st.file_uploader("Chọn file CSV để tải dữ liệu", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)
        # Người dùng chọn các cột để phân cụm
        selected_features = st.multiselect("Chọn các cột:", df.columns, default=["AnnualIncome", "SpendingScore", "LoyaltyScore"])
        num_clusters = st.slider("Chọn số cụm (k):", 2, 5, 3)

        if st.button("🚀 Thực Hiện Phân Cụm"):
            if len(selected_features) < 2:
                st.warning("⚠️ Vui lòng chọn ít nhất 2 đặc trưng để phân cụm.")
            else:
                df1 = df.copy()# Sao chép dataframe gốc để giữ nguyên dữ liệu ban đầu
                column_notes = {}  # Biến lưu chú thích cho các cột được chuyển đổi

                X = df[selected_features]

                # Chuẩn hóa dữ liệu
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Kiểm tra và chuyển đổi các thuộc tính dạng chuỗi thành dạng số
                for column in selected_features:
                    if df1[column].dtype in ['object', 'string']:
                        df1, notes = encode_column_with_notes(df1, column)
                        column_notes[column] = notes  # Lưu chú thích vào dictionary

                with st.spinner("🔄 Đang thực hiện phân cụm..."):
                    time.sleep(2)

                    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                    df1['Cluster'] = kmeans.fit_predict(df1[selected_features])

                    st.success("✅ Phân cụm hoàn thành!")
                    st.write("### Dữ Liệu Sau Phân Cụm:")
                    st.write(df1)

                    # Vẽ biểu đồ
                    plt.figure(figsize=(10, 6))
                    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df1['Cluster'], cmap='viridis', s=50)
                    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200, label='Centroids')
                    plt.xlabel(selected_features[0])
                    plt.ylabel(selected_features[1])
                    plt.title("K-Means Clustering")
                    plt.legend()
                    st.pyplot(plt)

                    # Hiển thị biểu đồ scatterplot với 2 thuộc tính đầu tiên
                    if len(selected_features) >= 2:
                        plt.title("Scatterplot")
                        plt.legend()
                        plt.figure(figsize=(10, 6))
                        sns.scatterplot(
                            x=selected_features[0], 
                            y=selected_features[1], 
                            hue='Cluster', 
                            data=df1, 
                            palette='viridis', 
                            s=100
                        )
                        plt.xlabel(selected_features[0])
                        plt.ylabel(selected_features[1])
                        st.pyplot(plt)

                    # Hiển thị chú thích của các cột đã chuyển đổi
                    if column_notes:
                        st.write("\n### Chú thích cho các cột chuyển đổi:")
                        for col, notes in column_notes.items():
                            st.write(f"- **{col}**: {notes}")
    else:
        st.warning("⚠️ Vui lòng tải file CSV để bắt đầu.")

    if st.button("⬅️ Back to Home"):
        st.session_state["current_page"] = main_page
        st.rerun()

# 3. Apriori Association Rule
def apriori_page():
    st.markdown("<h2>🔗 Khai Phá Luật Kết Hợp với Apriori</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Chọn file CSV để tải dữ liệu", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)
        # Chuyển đổi dữ liệu thành dạng phù hợp cho Apriori
        st.write("### Tiền Xử Lý Dữ Liệu:")
        transactions = df['Items'].apply(lambda x: x.split(', '))
        unique_items = sorted(set(item for sublist in transactions for item in sublist))

        # Tạo DataFrame dạng one-hot encoding
        encoded_data = pd.DataFrame([[1 if item in transaction else 0 for item in unique_items] for transaction in transactions], columns=unique_items)
        st.write(encoded_data.head())
        
        # Chọn ngưỡng hỗ trợ tối thiểu
        min_support = st.slider("Chọn min support:", 0.01, 0.5, 0.1, step=0.01)

        # Chọn ngưỡng độ tin cậy tối thiểu
        min_confidence = st.slider("Chọn min confidence:", 0.1, 1.0, 0.6, step=0.05)

        if st.button("🚀 Thực Hiện Apriori"):
            with st.spinner("🔄 Đang khai phá tập phổ biến và luật kết hợp..."):
                time.sleep(2)
                st.success("✅ Khai phá luật kết hợp hoàn thành!")
                # Khai thác tập phổ biến
                frequent_itemsets = apriori(encoded_data, min_support=min_support, use_colnames=True)
                st.write("### Tập Phổ Biến:")
                st.write(frequent_itemsets)

                # Khai thác luật kết hợp
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
                st.write("### Luật Kết Hợp:")
                st.write(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

                # Hiển thị các luật kết hợp dưới dạng bảng
                st.write("### Luật Kết Hợp Chi Tiết:")
                for i, row in rules.iterrows():
                    st.write(f"**Luật {i+1}:** Nếu mua {set(row['antecedents'])} thì sẽ mua {set(row['consequents'])}")
                    st.write(f"- **Support:** {row['support']:.2f}")
                    st.write(f"- **Confidence:** {row['confidence']:.2f}")
                    st.write(f"- **Lift:** {row['lift']:.2f}")
                    st.write("---")
    else:
        st.warning("⚠️ Vui lòng tải file CSV để bắt đầu.")

    if st.button("⬅️ Back to Home"):
        st.session_state["current_page"] = main_page
        st.rerun()
#4 thuật toán Kohonen
def kohonen_page():
    st.header("🧠 Kohonen Self-Organizing Map (SOM)")

    # Tải dữ liệu từ file CSV
    uploaded_file = st.file_uploader("Chọn file CSV để tải dữ liệu", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### 🔍 Dữ liệu Gốc:")
        st.write(df.head())

        # Chọn các cột để huấn luyện SOM
        st.write("### 🔧 Chọn Các Cột để Huấn Luyện SOM:")
        selected_columns = st.multiselect(
            "Chọn các cột để huấn luyện:", 
            df.columns, 
            default=["Age", "AnnualIncome", "SpendingScore", "PurchaseFrequency", "AveragePurchaseValue", "Tenure", "OnlinePurchases"]
        )

        if len(selected_columns) < 2:
            st.warning("⚠️ Vui lòng chọn ít nhất 2 cột để huấn luyện SOM.")
            return

        # Xử lý dữ liệu phân loại bằng Label Encoding
        categorical_columns = df.select_dtypes(include=['object']).columns
        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        # Chọn dữ liệu để huấn luyện
        X = df[selected_columns]

        # Chuẩn hóa dữ liệu
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Hiển thị dữ liệu sau khi chuẩn hóa
        st.write("### 📊 Dữ Liệu Sau Khi Chuẩn Hóa:")
        st.write(pd.DataFrame(X_scaled, columns=selected_columns).head())

        # Cấu hình tham số SOM
        st.header("⚙️ Cấu Hình SOM")
        map_size = st.slider("Kích thước bản đồ SOM:", min_value=3, max_value=15, value=7)
        iterations = st.slider("Số lần lặp:", min_value=100, max_value=10000, value=1000, step=100)

        # Tạo và huấn luyện SOM
        som = MiniSom(map_size, map_size, X_scaled.shape[1], sigma=1.0, learning_rate=0.5)
        som.random_weights_init(X_scaled)

        if st.button("🚀 Huấn Luyện SOM"):
            with st.spinner("Đang huấn luyện SOM..."):
                time.sleep(2)
                som.train_random(X_scaled, iterations)
            st.success("✅ Huấn luyện hoàn tất!")

            # Vẽ bản đồ SOM
            st.write("### 🌐 Bản Đồ Kohonen (SOM)")

            plt.figure(figsize=(10, 10))
            for i, x in enumerate(X_scaled):
                w = som.winner(x)
                plt.text(w[0], w[1], str(df['CustomerID'].iloc[i]), 
                         color=plt.cm.rainbow(df['SpendingScore'].iloc[i] / 100),
                         fontdict={'weight': 'bold', 'size': 9})
            plt.xlim(0, map_size)
            plt.ylim(0, map_size)
            plt.title("Kohonen Self-Organizing Map")
            st.pyplot(plt)

            st.write("Màu sắc thể hiện giá trị `SpendingScore` của từng khách hàng.")
    else:
        st.warning("⚠️ Vui lòng tải file CSV để bắt đầu.")
    # Nút quay lại trang chính
    if st.button("⬅️ Back to Home"):
        st.session_state["current_page"] = main_page
        st.rerun()
# 5. Naive Bayes Classification
def naive_bayes_page():
    st.title("🤖 Naive Bayes Classification")
    st.header("1️⃣ Tải Tập Dữ Liệu")
    uploaded_file = st.file_uploader("Chọn file CSV để tải dữ liệu", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### 🔍 Dữ Liệu Đã Tải")
        st.dataframe(df)

        # Chọn cột quyết định
        decision_column = st.selectbox("Chọn cột quyết định:", df.columns)

        # Tách cột quyết định và cột điều kiện
        condition_columns = [col for col in df.columns if col != decision_column]

        st.write("### 📊 Cột Điều Kiện:")
        st.write(condition_columns)
        st.write("### 🏷️ Cột Quyết Định:")
        st.write(decision_column)

        # ==========================
        # 2. Huấn luyện mô hình Naive Bayes
        # ==========================
        st.header("2️⃣ Huấn Luyện Mô Hình Naive Bayes")

        # Chọn 2 đặc trưng để vẽ ranh giới quyết định
        selected_features = st.multiselect("Chọn 2 đặc trưng để vẽ Decision Boundary:", condition_columns, default=condition_columns[:2])

        # Chọn kiểu Naive Bayes
        nb_type = st.selectbox(
            "Chọn loại Naive Bayes:",
            ["GaussianNB", "MultinomialNB", "BernoulliNB"]
        )

        # Chọn tùy chọn làm trơn Laplace
        laplace_smoothing = st.selectbox(
            "Làm trơn Laplace:",
            ["Có", "Không"],
            index=0
        )

        # Chọn tỷ lệ train-test split
        test_size = st.slider(
            "Chọn tỷ lệ kiểm tra (test set):",
            0.1, 0.5, 0.3, step=0.1
        )

        if st.button("🤖 Thực hiện Naive Bayes Classification"):
            if len(selected_features) != 2:
                st.warning("⚠️ Vui lòng chọn đúng 2 đặc trưng để vẽ ranh giới quyết định.")
            else:
                with st.spinner("🔄 Đang thực hiện phân lớp Bayes..."):
                    time.sleep(2)

                    # Tự động encode categorical data thành số
                    encoders = {}
                    for column in df.columns:
                        if df[column].dtype == 'object':  # Nếu cột là dạng categorical
                            encoders[column] = LabelEncoder()
                            df[column] = encoders[column].fit_transform(df[column])

                    X = df[selected_features]
                    y = df[decision_column]

                    
                    # Chuẩn hóa dữ liệu
                    scaler = StandardScaler()
                    X = pd.DataFrame(scaler.fit_transform(X), columns=selected_features)

                    # Train-test split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

                    # Thiết lập giá trị var_smoothing
                    var_smoothing = 1e-9 if laplace_smoothing == "Có" else 1e-12
                    multi_alpha = 1.0 if laplace_smoothing == "Có" else 0.0
                    ber_alpha = 1.0 if laplace_smoothing == "Có" else 0.0

                    # Tạo mô hình dựa trên lựa chọn
                    if nb_type == "GaussianNB":
                        clf = GaussianNB(var_smoothing=var_smoothing)
                    elif nb_type == "MultinomialNB":
                        clf = MultinomialNB(alpha=multi_alpha)
                    elif nb_type == "BernoulliNB":
                        clf = BernoulliNB(alpha=ber_alpha)

                    # Huấn luyện mô hình
                    clf.fit(X_train, y_train)

                    # Dự đoán và đánh giá
                    y_pred = clf.predict(X_test)
                    acc = clf.score(X_test, y_test)

                    st.success("✅ Phân lớp Bayes hoàn thành!")
                    # Hiển thị độ chính xác
                    st.write(f"### 🎯 Độ Chính Xác: {acc:.2f}")

                    # Hiển thị báo cáo chi tiết
                    st.write("### 📋 Báo Cáo Chi Tiết:")
                    st.text(classification_report(y_test, y_pred))

                    # Trực quan hóa confusion matrix
                    st.write("### 🔎 Confusion Matrix:")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots()
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
                    disp.plot(ax=ax, cmap='viridis')
                    st.pyplot(fig)

                    # Trực quan hóa dữ liệu bằng scatter plot
                    st.write("### Biểu đồ Scatter Plot của Dữ Liệu Kiểm Tra:")
                    fig, ax = plt.subplots()
                    scatter = ax.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_pred, cmap='viridis', edgecolor='k', s=100)
                    ax.set_xlabel(selected_features[0])
                    ax.set_ylabel(selected_features[1])
                    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
                    ax.add_artist(legend1)
                    st.pyplot(fig)

                    # ==========================
                    # 3. Vẽ Decision Boundary
                    # ==========================
                    st.write("### 🌐 Decision Boundary")

                    # Tạo lưới điểm để vẽ ranh giới
                    x_min, x_max = X_train[selected_features[0]].min() - 1, X_train[selected_features[0]].max() + 1
                    y_min, y_max = X_train[selected_features[1]].min() - 1, X_train[selected_features[1]].max() + 1
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                                        np.arange(y_min, y_max, 0.01))

                    # Dự đoán cho từng điểm trong lưới
                    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)

                    # Vẽ ranh giới quyết định
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
                    scatter = ax.scatter(X_test[selected_features[0]], X_test[selected_features[1]], c=y_test, edgecolor='k', cmap='viridis', s=100)
                    ax.set_xlabel(selected_features[0])
                    ax.set_ylabel(selected_features[1])
                    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
                    ax.add_artist(legend1)
                    st.pyplot(fig)

    else:
        st.warning("⚠️ Vui lòng tải file CSV để bắt đầu.")

    # Nút quay lại trang chính
    if st.button("⬅️ Back to Home"):
        st.session_state["current_page"] = main_page
        st.rerun()
# 6. Random Forest
def random_forest_page():
    st.header("🌲 Random Forest")

    # Chọn bộ dữ liệu
    dataset_name = st.selectbox(
        "🔹 Chọn bộ dữ liệu:",
        ["Iris", "Wine", "Breast Cancer"]
    )

    # Load dữ liệu tương ứng
    if dataset_name == "Iris":
        data = load_iris()
    elif dataset_name == "Wine":
        data = load_wine()
    else:
        data = load_breast_cancer()

    # Hiển thị thông tin về dữ liệu
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names

    st.write("### 🔍 Dữ liệu:")
    df = pd.DataFrame(X, columns=feature_names)
    df['Target'] = y
    st.write(df.head())

    # Chia dữ liệu thành tập train và test
    test_size = st.slider("🔹 Tỷ lệ dữ liệu test:", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
    random_state = st.number_input("🔹 Random State:", min_value=0, max_value=1000, value=0, step=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Tham số cho Random Forest
    st.write("### ⚙️ Cấu hình Random Forest")
    n_estimators = st.slider("🔸 Số lượng cây (n_estimators):", min_value=10, max_value=500, value=100, step=10)
    max_depth = st.slider("🔸 Độ sâu tối đa (max_depth):", min_value=1, max_value=20, value=5, step=1)
    min_samples_split = st.slider("🔸 Số mẫu tối thiểu để chia (min_samples_split):", min_value=2, max_value=10, value=2, step=1)

    # Huấn luyện mô hình Random Forest
    if st.button("🚀 Huấn Luyện Mô Hình"):
        with st.spinner("🔄 Đang huấn luyện..."):
            time.sleep(2)
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state
            )
            clf.fit(X_train, y_train)

            # Độ chính xác của mô hình
            acc = clf.score(X_test, y_test)
            st.success(f"✅ Độ chính xác: {acc:.2f}")

            # Vẽ ma trận nhầm lẫn
            st.write("### 📊 Ma Trận Nhầm Lẫn")
            y_pred = clf.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)

            fig, ax = plt.subplots(figsize=(8, 6))
            disp.plot(ax=ax, cmap='viridis')
            st.pyplot(fig)

            # Hiển thị biểu đồ Feature Importance
            st.write("### 🌟 Tầm Quan Trọng Của Các Đặc Trưng (Feature Importance)")
            importances = clf.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)

            # Vẽ biểu đồ cột
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                x='Importance',
                y='Feature',
                data=feature_importance_df,
                palette='viridis'
            )
            ax.set_title("Feature Importance")
            st.pyplot(fig)

    if st.button("⬅️ Back to Home"):
        st.session_state["current_page"] = main_page
        st.rerun()
# trang tập thô
def rough_set_page():
    st.title("📊 Rough Set Analysis Application")
    st.header("1️⃣ Tải Tập Dữ Liệu")
    uploaded_file = st.file_uploader("Chọn file excel để tải dữ liệu", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### 🔍 Dữ Liệu Đã Tải")
        st.dataframe(df)
        # Chọn cột quyết định
        decision_column = st.selectbox("Chọn cột quyết định:", df.columns)

        # Tách cột quyết định và cột điều kiện
        condition_columns = [col for col in df.columns if col != decision_column]

        st.write("### 📊 Cột Điều Kiện:")
        st.write(condition_columns)
        st.write("### 🏷️ Cột Quyết Định:")
        st.write(decision_column)

        # ==========================
        # 2. Tính quan hệ bất khả phân biệt
        # ==========================
        st.header("2️⃣ Quan Hệ Bất Khả Phân Biệt")

        def indiscernibility_relation(data, columns):
            indiscernibility_classes = {}
            for i, row in data.iterrows():
                key = tuple(row[columns])
                if key not in indiscernibility_classes:
                    indiscernibility_classes[key] = []
                indiscernibility_classes[key].append(i)
            return indiscernibility_classes

        if st.button("Tính Quan Hệ Bất Khả Phân Biệt"):
            ind_classes = indiscernibility_relation(df, condition_columns)
            st.write("### 🔗 Các Lớp Quan Hệ Bất Khả Phân Biệt:")
            for key, indices in ind_classes.items():
                st.write(f"{key}: {indices}")

        # ==========================
        # 3. Xấp xỉ tập hợp
        # ==========================
        st.header("3️⃣ Xấp Xỉ Tập Hợp")

        target_value = st.text_input("Nhập giá trị của cột quyết định để tính xấp xỉ:", "")

        def lower_upper_approximation(data, condition_columns, decision_column, target_value):
            target_indices = set(data[data[decision_column] == target_value].index)
            ind_classes = indiscernibility_relation(data, condition_columns)

            lower_approx = set()
            upper_approx = set()

            for indices in ind_classes.values():
                indices_set = set(indices)
                if indices_set.issubset(target_indices):
                    lower_approx.update(indices_set)
                if indices_set.intersection(target_indices):
                    upper_approx.update(indices_set)

            return lower_approx, upper_approx

        if st.button("Tính Xấp Xỉ"):
            lower_approx, upper_approx = lower_upper_approximation(df, condition_columns, decision_column, target_value)
            boundary_region = upper_approx - lower_approx

            st.write(f"### ⬇️ Xấp Xỉ Dưới: {lower_approx}")
            st.write(f"### ⬆️ Xấp Xỉ Trên: {upper_approx}")
            st.write(f"### 🔹 Vùng Biên: {boundary_region}")

        # ==========================
        # 4. Tìm reducts
        # ==========================
        st.header("4️⃣ Tìm Reducts")

        def find_reducts(data, condition_columns, decision_column):
            full_ind = indiscernibility_relation(data, condition_columns)
            reducts = []

            for i in range(1, len(condition_columns) + 1):
                for subset in combinations(condition_columns, i):
                    subset_ind = indiscernibility_relation(data, list(subset))
                    if subset_ind == full_ind:
                        reducts.append(subset)

            return reducts

        if st.button("Tìm Reducts"):
            reducts = find_reducts(df, condition_columns, decision_column)
            st.write("### 🔍 Các Reduct Tìm Được:")
            for reduct in reducts:
                st.write(reduct)

        # ==========================
        # 5. Sinh luật quyết định
        # ==========================
        st.header("5️⃣ Sinh Luật Quyết Định")

        def generate_decision_rules(data, condition_columns, decision_column):
            rules = []
            for i, row in data.iterrows():
                condition = " AND ".join([f"{col}={row[col]}" for col in condition_columns])
                decision = f"{decision_column}={row[decision_column]}"
                rule = f"IF {condition} THEN {decision}"
                rules.append(rule)
            return rules

        if st.button("Sinh Luật Quyết Định"):
            rules = generate_decision_rules(df, condition_columns, decision_column)
            st.write("### 📜 Luật Quyết Định:")
            for rule in rules:
                st.write(rule)

    else:
        st.warning("⚠️ Vui lòng tải file CSV để bắt đầu.")
    if st.button("⬅️ Back to Home"):
        st.session_state["current_page"] = main_page
        st.rerun()
# Điều hướng giữa các trang
if "current_page" not in st.session_state:
    st.session_state["current_page"] = main_page

st.session_state["current_page"]()
