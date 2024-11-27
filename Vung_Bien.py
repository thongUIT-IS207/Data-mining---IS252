import pandas as pd

df = pd.read_excel('C:/New File Downloads/data.xlsx', engine='openpyxl')
X = {'O1', 'O3', 'O4'}
B = ['Troi', 'Gio']

def lower_approximation(df, X, B):
    """Tính xấp xỉ dưới của tập X."""
    lower = set()
    for _, row in df.iterrows():
        matching_objects = df[(df[B] == row[B].values).all(axis=1)]['Ketqua']
        if set(matching_objects).issubset(X):
            lower.add(row['Ketqua'])
    return lower

def upper_approximation(df, X, B):
    """Tính xấp xỉ trên của tập X."""
    upper = set()
    for _, row in df.iterrows():
        matching_objects = df[(df[B] == row[B].values).all(axis=1)]['Ketqua']
        if set(matching_objects).intersection(X):
            upper.add(row['Ketqua'])
    return upper

lower = lower_approximation(df, X, B)
upper = upper_approximation(df, X, B)

boundary_region = upper - lower

print("Xấp xỉ dưới:", lower)
print("Xấp xỉ trên:", upper)
print("Vùng biên:", boundary_region)
