import pandas as pd
import numpy as np
from copy import deepcopy

sales_data = pd.read_csv(r'D:\2024_Study in SMU\5560_Biz Intelligence and data visulization\Final Project\Final Project Materials\Analyze\Datatset2_Sales Trends Dataset\daily_sales.csv')
array_sales_data = sales_data.to_numpy()

# Get the data from Shop 31 (得到第31号商店的数据)

rows = array_sales_data[:, 2] == 31
array = array_sales_data[rows, :]
#timeStamp = np.unique(array[:, 1]) # get 33 month data in total (共计33个月的数据)
timeStamp = np.unique(array[array[:, 1] > 17, 1])
item_index = np.unique(array[:, 3]) # get 1w+ items in total (共计1w+个不同的item)

# Calculate the serial numbers of the n items with the largest sales (计算销量最大的n个item序号)
n = 10
sales = np.zeros(item_index.shape)
for i in range(item_index.shape[0]):
    sales[i] = np.sum(array[array[:, 3] == item_index[i], 4])
sorted_indices = np.argsort(sales)
top_10_indices = sorted_indices[-1 * n:]
item_index = deepcopy(item_index[top_10_indices])


# calculate the sales for each item based on the data of each month (计算每个商品在每个月份的数据--基于销量)
dict = {i: np.zeros(34) for i in item_index}# create a dictionary for "item_index" (创建一个具有item_index个的字典，每一个键值对里的数值都是34个0)
full_array = np.array([[i, j ,0] for i in item_index for j in timeStamp ])
for i in item_index: 
    for j in timeStamp: 
        value = array[(array[:, 3] == i) & (array[:, 1] == j), 4]
        if value.shape != 0:
            full_array[(full_array[:, 0] == i) & (full_array[:, 1] == j), 2] = np.sum(value)


# Calculate the correlation coefficient matrix and select the m pairs of products with the largest correlation (计算相关系数矩阵，取相关性最大的m对商品)
m=10
abc = [full_array[full_array[:, 0] == i, 2] for i in item_index]
r_mat = np.abs(np.corrcoef(abc))
r_mat = r_mat - np.triu(r_mat)
indices = np.argpartition(r_mat.flatten(), -1*m)[-1*m:]
top_10_values = r_mat.flatten()[indices]
row_indices, col_indices = np.unravel_index(indices, r_mat.shape)
results = []
for value, row, col in zip(top_10_values, row_indices, col_indices):
    results.append((value, item_index[row], item_index[col]))
results.sort(reverse=True)
for result in results:
    print("value：{}, item combination：({}, {})".format(result[0], result[1], result[2]))