# 机器学习论文作业

## 环境配置
我建议使用uv进行环境管理，通过pip安装uv并且设置环境变量后。
如果你们不用的话，就当这几个文件不存在(qaq)，我后面也可以通过gitignore忽略掉
有一个需要注意的是pytorch，这个可能大家的cuda环境不同，需要自己安装
```bash
# cmd或者powershell的路径需要在当前文件夹下
uv venv # 创建虚拟环境，会自动安装对应的python版本
.venv\Scripts\activate # 激活虚拟环境
# 请先安装pytorch，避免安装其他需要cuda的包的时候自动下载cpu的
uv pip install ... # 这里使用pytorch提供的链接就行了，
uv sync # 会同步依赖，建议每次pull后都执行此命令，保证环境的一致性

# 当然，也可以通过配置来指定下载的pytorch的版本，比如我用的就是
# uv sync --extra rocm
```

然后统一把数据集放在`data`文件夹下，在这个文件夹下就是`csv`和`image`文件夹
