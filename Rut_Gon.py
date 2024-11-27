import pandas as pd
from itertools import combinations

# Đọc dữ liệu từ file Excel
df = pd.read_excel('D:/school lecture/New folder/data.xlsx', engine='openpyxl')

# Hàm kiểm tra khả năng phân biệt theo tập thuộc tính cho trước
def distinguishable(df, attributes):
    grouped = df.groupby(attributes)['Kết quả'].nunique()
    return all(grouped == 1)

# Hàm tìm tập rút gọn thuộc tính
def find_reduct(df):
    columns = list(df.columns[:-1])  # Loại bỏ cột 'Kết quả'
    minimal_reducts = []

    # Tìm tất cả các tổ hợp thuộc tính
    for r in range(1, len(columns) + 1):
        for combo in combinations(columns, r):
            if distinguishable(df, combo):
                minimal_reducts.append(combo)

    # Lọc ra tập rút gọn tối thiểu (có kích thước nhỏ nhất)
    min_length = min(len(reduct) for reduct in minimal_reducts)
    return [reduct for reduct in minimal_reducts if len(reduct) == min_length]

# Tìm tập rút gọn thuộc tính
reducts = find_reduct(df)

# In kết quả
print("Tập rút gọn tối thiểu:")
for reduct in reducts:
    print(reduct)
