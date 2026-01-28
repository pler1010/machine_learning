# machine_learning
The final project for machine learning.

复现流程：

- 执行数据处理

  ```
  python driver.py --mode pretreat
  ```

  从`./data/SC-1_dense.h5ad`中读取数据并清洗数据，将清洗后的数据存放在`./data/clean_data.txt`。

- 执行基于统计学的特征选择

  ```
  python driver.py --mode stat_select -d ${k}
  ```

  该命令会执行基于统计学的特征选择，其中 $k$ 是为每个类别选取的特征数量。从`./data/clean_data.txt`中读取数据，特征选择后数据存放在`./data/selected_data.txt`。


- 执行基于机器学习的特征选择

  ```
  python driver.py --mode ml_select -d 100 -a 0.5
  ```
  
  该命令会训练 RandomForest 与 XGBoost 多分类模型，并根据特征重要性进行加权融合筛选 Top100 基因。其中 a 表示加权融合中 RandomForest 的权重。

  输出：
    ./data/selected_data.txt（供 evaluate 使用）  
    ./data/ml_importance_scores.csv（各模型重要性+融合分数）

- 执行基于深度学习的特征筛选

  ```
  python driver.py --mode dl_select -d 100 -g abs -e 30
  ```  
  
  其中 -g 表示梯度重要性计算方式，可选 abs 或 square；
  
  -e 表示 MLP 训练轮数
  
  输出：
    ./data/selected_data.txt（供 evaluate 使用）
    ./data/dl_importance_scores_abs.csv（梯度重要性分数）
    ./data/dl_vs_ml_overlap_top100.txt（与任务3重叠率分析）
    ./fig/dl_grad_top20_abs.png 等可视化结果

- 执行结果评估

  ```
  python driver.py --mode evaluate
  ```

  该命令会评估特征选择的结果，从`./data/selected_data.txt`中读取数据。

