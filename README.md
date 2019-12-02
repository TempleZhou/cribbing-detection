# 作弊检测

## 目录结构

```
├── README.md                
├── datasets    # 预处理好的数据集        
├── model       # 存放训练的模型         
├── model_web   # 存放训练的模型      
├── prepare     # 准备数据
├── training    # 训练模型的 Python 程序
└── webui       # 前端展示，加载训练好的模型
```

## 使用方法

### 准备数据

```bash
cd prepare
python build_imagenet_data.py 
```

产生的数据生成在 datasets 下面

### 训练模型

```bash
cd training
python mobilenet_train.py
```

转换模型供 js 脚本使用

```bash
tensorflowjs_converter  --input_format tf_frozen_model \
                        --output_format tfjs_graph_model \
                        --output_node_names='MobilenetV1/Predictions/Reshape_1'  \
                        model/xxx.pb model_web
```

因训练设备性能有限，训练模型仅使用了 ImageNet 的样例数据，未使用所有数据。

因此 model_web 下最终加载的模型为预训练好的模型。

### 提供模型下载服务

```bash
cd model_web
env FLASK_APP=tfjs-serving.py flask run
```

端口 5000

### 运行前端

> 保证存在 npm 运行环境并安装了 yarn

安装相关依赖

```bash
cd webui
yarn install
```

dev 方式运行 node 项目

```bash
yarn watch
```