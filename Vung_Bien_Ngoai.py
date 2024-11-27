import pandas as pd
df = pd.read_excel('C:/New File Downloads/data.xlsx', engine='openpyxl')

def find_outer_boundary(df):
    unique_values = set(df['Ketqua'])
        outer_boundary = []
        for value in unique_values:
            subset = df[df['Ketqua'] == value]
        if subset.shape[0] == 1:  # Nếu chỉ có một đối tượng cho giá trị này
            outer_boundary.append(value)
    return outer_boundary

outer_boundary = find_outer_boundary(df)

print("Vùng biên ngoài:", outer_boundary)