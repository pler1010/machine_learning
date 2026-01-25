# machine_learning
The final project for machine learning.

复现流程：

- 执行数据处理

  ```
  python driver.py -mode pretreat
  ```

  从`./data/SC-1_dense.h5ad`中读取数据并清洗数据，将清洗后的数据存放在`./data/clean_data.txt`。

- 执行基于统计学的特征选择

  ```
  python driver.py -mode stat_select -d ${k}
  ```

  该命令会执行基于统计学的特征选择，其中 $k$ 是为每个类别选取的特征数量。从`./data/clean_data.txt`中读取数据，特征选择后数据存放在`./data/selected_data.txt`。

- 执行结果评估

  ```
  python driver.py -mode evaluate
  ```

  该命令会评估特征选择的结果，从`./data/selected_data.txt`中读取数据。

