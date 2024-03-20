### RUN

```shell
python main.py
```


### Note

- 这个代码使用 GCN 实现 MUTAG 数据集上的图分类任务。

- 没有表示学习的过程，在 GCN model 中直接进行分类。

- 代码的数据处理过程十分值得参考：

  - 使用了 PyG 的 dataset 类
  - 使用了 PyG 的 DataLoader 方法
  - `global_mean_pool` 层由每个节点的特征得到每个图一个特征，用于下游的分类。




> 参考：https://zhuanlan.zhihu.com/p/435945714。这个帖子中还有优化的版本，我暂时还没看。