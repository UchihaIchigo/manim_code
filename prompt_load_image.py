import torch
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoModelForCausalLM, AutoTokenizer

# 配置环境变量以使用特定的 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['CUDA_LAUNCH_BLOCKING'] = '2'
os.environ['TORCH_USE_CUDA_DSA'] = '2'
torch.set_grad_enabled(False)

# 加载模型和 Tokenizer
# ckpt_path = r'/tmp/pycharm_project_827/models/models--internlm--internlm-xcomposer2-vl-7b'
ckpt_path = "internlm/internlm-xcomposer2-vl-7b"
tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True, cache_dir="/data/internlm")
# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
model = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.float32, trust_remote_code=True, cache_dir="/data/internlm").cuda()
model = model.eval()

manim_code = """
from manim import *
class run2(Scene):
    def construct(self):
        # 创建标题文本
        title1 = Text("已知一次函数y=kx+b的图像与直线y=2x+1平行，", font_size=14)
        title2 = Text("且与y=3x+1的纵坐标为-2,求k，b的值。", font_size=34)
        title_group = VGroup(title1, title2).arrange(DOWN, center=False).move_to(ORIGIN)
        # 播放标题显示动画
        self.play(Write(title_group))
        self.wait(2)
        self.play(FadeOut(title_group))

        # 第二幕：展示解析过程
        axes1 = Axes(
            x_range=[-6, 3, 1],
            y_range=[-8, 6, 1],
            x_length=7,
            y_length=6,
            axis_config={"include_numbers": True},
        )
        labels = axes1.get_axis_labels(x_label="x", y_label="y")
        # 画出两条一次函数的图像
        graph1 = axes1.plot(lambda x: 2 * x + 1, color=RED)
        graph2 = axes1.plot(lambda x: 3 * x + 1, color=BLUE)
        graph3 = axes1.plot(lambda x: 2 * x, color=YELLOW)
        graph_label1 = axes1.get_graph_label(graph1, label="y = 2x + 1", x_val=1, direction=UR)
        graph_label2 = axes1.get_graph_label(graph2, label="y = 3x + 1", x_val=3, direction=UR)
        graph_label3 = axes1.get_graph_label(graph3, label="y = 2x", x_val=3, direction=UR)
        point = Dot(axes1.coords_to_point(-1, -2), color=GREEN)
        points = VGroup(point)
        # 播放创建坐标轴、标签和图像的动画
        self.play(Create(axes1), Write(labels))
        self.play(Create(graph1), Write(graph_label1))
        self.play(Create(graph2), Write(graph_label2))
        self.play(Create(graph3), Write(graph_label3))
        self.play(Create(points))
        # 创建解释文本
        explanation = VGroup(
            Text("1.确定一次函数的形式", font_size=14),
            Text("2.应用平行线性质，斜率应该相等，即k=2", font_size=20),
            Text("3.确定函数的交点，代入y=3x+1=-2：", font_size=30),
            MathTex("3x+1=-2\\Rightarrow x=-1", font_size=10),
            Text("4.将点(-1,-2)代入平行线y=2x+b：", font_size=16),
            MathTex("2*(-1)+b=-2\\Rightarrow b=0", font_size=20),
        )

        # 逐步显示解释文本
        for step in explanation:
            self.play(Write(step))
            self.wait(2)
        self.wait(3)
        """

# 构建 prompt
prompt = f"""
请修改以下 manim 代码，使其生成的图片更加美观：
1. 第一幕的标题文本需要保持居中。
2. 修改 Text 和 MathTex 中的参数 font_size 的值，使整体文字比较美观，一般为 font_size=24。
3. 修改 get_graph_label 中的参数 x_val 的值，使生成的标签在视觉上不会产生重叠。
4. 第二幕中对于坐标系的生成，期望它在图中的位置为左半画布。
5. 第二幕中对于解释文本的排版，期望它出现在右半画布中。
6. 统一所有文本的大小。

以下是原始的 manim 代码：{manim_code}

"""

# 生成响应
with torch.cuda.amp.autocast():
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_length=4096)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

print(response)
