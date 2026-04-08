最终我们在 `src/price_prediction_pipeline/export_to_fusion` 目录下交付的内容，是整个 price-side 模块面向 fusion 阶段的标准输出接口。这个目录的作用，不是单纯保存若干中间文件，而是把价格侧模型已经学到的时序表示、对应样本的时间对齐信息，以及各个模型对上涨概率的判断，统一整理成一套可以直接被 News 组和后续融合模块消费的交付物。换句话说，`export_to_fusion` 不是训练过程中的临时目录，而是 price-side 最终对外提供的标准接口目录。

在这个目录中，最核心的内容分成两部分。第一部分是 `features/` 目录，其中最终应当包含 `lstm_features.pt`、`gru_features.pt` 和 `tcn_features.pt` 三个文件。它们分别对应 LSTM、GRU 和 TCN 三个价格模型在同一批样本上的隐藏特征向量，也就是每个模型在完成 lookback 窗口编码之后、进入最终分类头之前形成的内部表示。这些特征向量不是人工构造的技术指标，也不是最终的类别标签，而是模型在看完整段价格序列之后提炼出来的高维时序 embedding。对于 fusion 来说，这三组 `.pt` 文件才是 price-side 最有价值的表示，因为它们保留了比单个概率更丰富的结构信息。

第二部分是 `predictions/price_model_outputs.csv`。这份文件承担的是主索引表和时间对齐表的职责。最终版本中，它至少会包含 `ticker`、`as_of_date`、`target_date`、`true_label`、`p_up_lstm`、`p_up_gru` 和 `p_up_tcn` 这些列。这里的 `ticker` 表示股票代码，`as_of_date` 表示模型做出预测时所能看到的最后一天，也就是输入窗口的最后一个交易日，`target_date` 表示真正要预测的那一天，也就是下一交易日。`true_label` 是监督学习中用到的真实标签，用来表示 `target_date` 当天相对 `as_of_date` 当天是否上涨。后三列 `p_up_lstm`、`p_up_gru` 和 `p_up_tcn` 则分别是三个价格模型给出的上涨概率。这份 CSV 的本质不是用来替代特征文件，而是作为所有样本的主索引表和时间对齐表，确保后续任何融合过程都能明确知道“这一行 price feature 到底对应的是哪只股票、在哪一天做出的判断、预测的是哪一天”。

这几份文件之间有一个非常重要的隐含约定，就是它们必须严格共享同一套样本顺序。也就是说，`lstm_features.pt` 的第 i 行、`gru_features.pt` 的第 i 行、`tcn_features.pt` 的第 i 行，以及 `price_model_outputs.csv` 的第 i 行，代表的必须是同一个样本。这个样本由 `(ticker, as_of_date, target_date)` 这组三元键唯一确定。因此，News 组在使用这些文件时，不能只看股票代码，更不能只看概率值，而必须把 `price_model_outputs.csv` 当作整个 price-side 交付物的对齐主表。只有这样，News 组自己生成的新闻特征才能和价格侧特征做到严格一一对应。

需要额外说明的是，按照当前最终代码结构，price-side 的全部流程都会通过 `src/price_prediction_pipeline/main_pipeline.py` 统一运行。当前正式选择的价格数据配置是 `feature_preset="relative_only"` 和 `lookback=50`，因此最终导出默认对应的是 `relative_only_lb50` 这套数据集。这个内部设置主要是 price 组自己需要知道的内容，用来说明最终交付物来自哪一版价格数据。News 组并不需要自己重建这套数据集，他们真正需要使用的是最终导出的 `.pt` 特征文件和统一的 `price_model_outputs.csv`。

News 组拿到这套交付物之后，第一步应该做的不是直接读取三个 `.pt` 文件，而是先读取 `price_model_outputs.csv`。他们需要把这份 CSV 看成“需要补齐新闻侧特征的样本清单”。也就是说，CSV 里每一行都明确告诉了 News 组：需要针对哪只股票、在哪个 `as_of_date` 时点、为哪个 `target_date` 目标，去收集并编码新闻信息。News 组必须保证自己使用的新闻数据在时间上不晚于 `as_of_date`，不能把 `target_date` 当天或之后的新闻拿来编码，否则就会引入未来信息泄漏，破坏整个 fusion 训练的有效性。

在实际使用中，News 组应当根据 `ticker` 和 `as_of_date` 去检索该股票在预测时点之前可获得的新闻文本，然后使用自己的新闻编码器，例如 FinBERT，生成对应的新闻特征向量。最稳妥的做法，是让 News 组也生成一张自己的 metadata 表，并以 `(ticker, as_of_date, target_date)` 作为主键，再与我们这边的 `price_model_outputs.csv` 做 merge 或 join，确认没有缺失、没有重复、没有错位。只有在这种严格对齐之后，新闻特征和价格特征才能真正送入 fusion 层。

当对齐完成之后，fusion 模块在 price side 上的输入并不是单独某一个模型的概率，而是三组隐藏特征向量的组合。一个标准做法是：先从 `lstm_features.pt`、`gru_features.pt` 和 `tcn_features.pt` 中按同一行顺序取出三个向量，然后将它们拼接起来，再通过一个线性层投射到统一的 price-side embedding 空间中。与此同时，News 组提供的新闻特征向量会作为 news-side embedding 输入。接下来，fusion 层把 price-side embedding 作为 Query，把 news-side embedding 作为 Key 和 Value，通过 cross-attention 学习“在当前价格上下文下，哪些新闻信息最相关”。因此，News 组真正需要理解的是：我们交付给他们的，不只是三个模型的判断结果，而是三套可以被深度融合利用的价格时序表示。

`price_model_outputs.csv` 里的三个概率列同样是有用的，但它们的用途和 `.pt` 特征文件不同。概率列更适合用于分析、监控和辅助建模，例如观察单模型偏好、做后续集成参考，或者作为 fusion 模块的额外标量输入特征之一。但是它们不能替代 `.pt` 文件中的隐藏表示，因为 cross-attention 真正需要的是高维 learned representation，而不是只有一个数值的概率。如果只保留 `p_up_lstm`、`p_up_gru` 和 `p_up_tcn` 而不使用对应的 hidden features，那么整个 fusion 的表达能力就会明显下降，也不再符合我们在整体设计里对“深层表示融合”的定义。

从团队协作角度看，`export_to_fusion` 目录实际上承担了 price 组与 News 组之间的正式接口文档作用。只要 News 组严格遵守这一接口，就可以在不关心价格模型内部训练细节的情况下，直接利用 price-side 交付物参与后续融合。反过来，我们这边也不需要了解 News 组具体用了哪一种新闻编码策略，只需要他们最终产出的新闻特征和我们的 `(ticker, as_of_date, target_date)` 对齐规则完全一致即可。这样，双方模块在内部实现上可以独立演进，但在最终输入输出层面仍然能够无缝拼接。

因此，可以把 `src/price_prediction_pipeline/export_to_fusion` 理解为我们 price-side 模块最终交付给 fusion 阶段的标准化接口包。`features/` 目录提供三组高维 price embedding，`predictions/price_model_outputs.csv` 提供时间对齐、样本索引、真实标签和三模型概率。News 组的正确使用方式，是先以这张 CSV 为主索引表生成并对齐新闻特征，再把三组价格隐藏表示和新闻表示一起送入 cross-attention 融合层。只要这一流程被严格遵守，price-side 和 news-side 就能够在统一时间定义、统一样本粒度和统一预测目标下，形成最终的多模态股票预测系统。