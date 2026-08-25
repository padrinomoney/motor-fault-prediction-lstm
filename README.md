工厂电机故障预测系统
基于 LSTM 时序模型的电机振动故障检测项目，对比传统机器学习基线，实现网页端可视化故障预测工具。
项目简介
本项目通过模拟电机振动时序数据集，构建随机森林基线模型与 LSTM 深度学习模型，完成电机故障二分类任务；导出 ONNX 模型，基于 Gradio 搭建简易 Web 交互界面，实现输入振动数据即可实时判断电机运行状态。
项目结构
motor-fault-prediction-lstm/
├─ motor_fault.py               # 主程序代码
├─ motor_fault_model.onnx       # 导出的推理模型
├─ motor_fault_model.onnx.data  # 模型配套数据文件
├─ README.md                    # 项目说明文档
└─ screenshots/                  # 项目效果截图
   ├─ 1_随机森林基线结果.png
   ├─ 2_LSTM训练Loss曲线.png
   ├─ 3_模型评估指标与混淆矩阵.png
   └─ 4_Gradio网页预测界面.png
环境依赖
执行以下命令安装所需库：
pip install torch numpy pandas scikit-learn gradio matplotlib onnxscript
运行方式
安装全部依赖环境
运行主程序 motor_fault.py
复制终端输出的本地访问地址，在浏览器打开
输入 30 维振动传感器数据（逗号分隔），提交即可得到故障预测结果
模型效果
随机森林基线：准确率 1.00，故障召回率 1.00
LSTM 时序模型：测试准确率 1.00，故障召回率 1.00
Loss 曲线平稳下降，模型收敛效果优异，可精准区分正常与故障电机状态
技术亮点
采用时序神经网络 LSTM 处理振动序列数据，适配工业设备故障检测场景
传统机器学习模型作为基线对照，验证深度学习模型效果
模型导出为 ONNX 格式，便于部署落地
Gradio 快速搭建 Web 交互页面，无需前端开发即可实现可视化工具