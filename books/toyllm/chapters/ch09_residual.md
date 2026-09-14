# 残差连接（Residual Connections）

> **本章代码**：把注意力和 FFN 用一条旁路包起来，Decoder Block 至此完整。
>
> - **造哪个框**：`rc.png` 里那两个 `+Residual`——第一章说过，它在数据流向图上画不出来
> - **形状**：`(T, 576) → (T, 576)`。旁路不改变形状，主干也不改变形状，两者相加还是 `(T, 576)`
> - **骨架**：`h = x + Attn(norm(x))`，`y = h + FFN(norm(h))`
> - **引擎**：`ids → embed → [norm → attn → norm → ffn] × 30 → norm → lm_head → logits`
> - **要讲**：残差连接的基本概念；它和第五章 pre-norm / post-norm 的关系（norm 在旁路里还是旁路外）；以及 HC、mHC、AttnRes 几种改进
> - **对拍**：HC / mHC / AttnRes 都不是 SmolLM2 用的方案，**这部分没有可对拍的参考实现**，只讲原理不落代码


**一个图上看不见的东西：残差连接。** 图里写的是「Attention+Residual」和「FFN+Residual」，残差被折进框里了，没有画出那条绕过去的旁路。但 pre-norm 和 post-norm 的区别恰恰是"norm 放在旁路里还是旁路外"，讲这一节必须把那条旁路补出来：

```
h = x + Attention(Norm(x))
y = h + FFN(Norm(h))
```



## 残差连接



## HC&mHC



## AttnRes



## 本章小结

TODO

