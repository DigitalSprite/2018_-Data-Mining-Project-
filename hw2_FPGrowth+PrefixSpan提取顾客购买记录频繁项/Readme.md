## Readme

本次项目采用  `FP-growth` 算法和 `PrefixSpan` 算法对 `trade.csv` 和 `trade_new.csv` 中用户的购买记录做了频繁集的挖掘。



#### 文件结构说明

- **a** ：`fp-growth` 算法对应的文件夹。
  - **ai**：对`uid`进行分组后训练的文件夹，可训练 `pluno,bndno,dptno` 
  - **aii**：对 `vipno` 进行分组后训练的文件夹，可训练 `pluno,bndno,dptno`
  - **fpgrowth**：fpgrowth的算法包
  - **图表**：记录了a问所对应的所有生成的图表
- **b** ：`PrefixSpan` 算法对应的文件夹
  - **bi** ：对`uid`进行分组后训练的文件夹，可训练 `pluno,bndno,dptno`
  - **bii** ：对 `vipno` 进行分组后训练的文件夹，可训练 `pluno,bndno,dptno`
  - **prefix_span** ：`PrefixSpan`算法的具体实现
  - **图表** ：记录了b问所对应所有生成的图表
- **c** ：对上述挖掘生成的频繁项集进行结果的预测文件夹
  - **anticipate.py** ：a问和b问频繁集的预测结果分析代码
  - **图表** ：分析预测时生成的所有图表
- **data** ：含有`trade.csv` 和 `trade_new.csv`

**a,b,c** 三个文件夹下分别对应每道题的项目文档



#### 依赖说明

python：3.5.5

bokeh：0.12.15

numpy：1.12.1

pandas：0.22.0



#### 代码运行说明

1. 从hw2文件夹加载项目
2. 对于a问和b问来说，可以构建python文件加载`ai.py`、`aii.py` ,`bi.py` ,`bii.py` 下的类，运行 `run_algorithm` 函数。参数为
   - property：从`pluno,bndno,dptno`中选择
   - min_support：任选一个正整数做min support
   - file：1对应的是trade.csv文件，2对应的是trade_new.csv文件，可以分别加载对应的数据集
3. 对于c问来说，可以构建Anticipate类，然后运行anticipate()函数进行预测，show_information()会显示总体的预测信息。anticipate()函数的参数如下：
   - property:从`pluno,bndno,dptno`中选择
   - min_support: 任意选择一个正整数作为min_support
   - k: 生成预测的频繁集的个数
   - func：从`ai,aii,bi,bii` 中选择，对应前两问所拥有的run_algorithm函数
   - file：1对应的是trade.csv文件，2对应的是trade_new.csv文件，可以分别加载对应的数据集



#### 参考说明

代码参考：

​	fp-growth  参考代码 https://github.com/enaeseth/python-fp-growth

​	prefix-span 参考代码 https://github.com/rangeonnicolas/PrefixSpan

文献参考：

​	survey paper：http://www.philippe-fournier-viger.com/dspr-paper5.pdf

​	prefix span：http://www.philippe-fournier-viger.com/spmf/prefixspan.pdf

​	FP Growth：http://www.philippe-fournier-viger.com/spmf/fpgrowth_04.pdf