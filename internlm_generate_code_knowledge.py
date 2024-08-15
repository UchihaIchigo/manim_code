import pandas as pd
import openai
import time
import torch
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# torch.cuda.set_device(1)  # 选择第二个GPU
os.environ['CUDA_VISIBLE_DEVICES']='3' #此处选择你要用的GPU序号 0，1，2，3
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致
os.environ['TORCH_USE_CUDA_DSA'] = '1'
from transformers import AutoTokenizer, AutoModelForCausalLM
ckpt_path = "internlm/internlm-xcomposer2-vl-7b"
# ckpt_path = "internlm/internlm-xcomposer2d5-7b"
tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True, cache_dir="/data/internlm")
# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
model = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.float32, trust_remote_code=True, cache_dir="/data/internlm").cuda()
model = model.eval()

# model_1 = AutoModelForCausalLM.from_pretrained(ckpt_path, device_map="cuda:4", torch_dtype=torch.float16, trust_remote_code=True, cache_dir="/data/internlm").cuda()
# model_2 = AutoModelForCausalLM.from_pretrained(ckpt_path, device_map="cuda:5", torch_dtype=torch.float16, trust_remote_code=True, cache_dir="/data/internlm").cuda()
# model_3 = AutoModelForCausalLM.from_pretrained(ckpt_path, device_map="cuda:6", torch_dtype=torch.float16, trust_remote_code=True, cache_dir="/data/internlm").cuda()
# model_1 = model_1.eval()
# model_2 = model_2.eval()
# model_3 = model_3.eval()


def generate_solution_from_problem(math_problem, model=model, attempt=0, max_retries=3):
    # 生成解题过程的提示
    prompt = f"""用中文一步一步给出以下数学题的详细解题过程。
                        示例：
                        数学题：求函数$f(x) = x^2 - 4x + 3$的顶点，并画出其图像。
                        解题过程: 1.Derive the Function: First, we find the derivative of $f(x)$, which is $f'(x) = 2x - 4$.
                            2.Find the Zero of the Derivative: Setting $f'(x) = 0$, we solve for $x$ and get $x = 2$. This means the function has an extremum at $x = 2$.
                            3.Compute the $y$ Coordinate of the Vertex: Substituting $x = 2$ into the original function, we get $f(2) = 2^2 - 4*2 + 3 = -1$. Thus, the vertex is at $(2, -1)$.
                            4.Determine the Type of the Vertex: Since the coefficient of the quadratic term is positive, we know this vertex represents the minimum point of the function, i.e., the lowest point on the graph.
                        数学题：{math_problem}
                        
                        解题过程：
                        """

    # # 调用OpenAI API生成解题过程
    # response = openai.ChatCompletion.create(
    #     model=model,
    #     messages=[
    #         {"role": "user", "content": prompt}
    #     ]
    # )
    # 生成响应
    with torch.cuda.amp.autocast():
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_length=1024)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        # 手动去除prompt_knowledge的内容
        response = response.replace(prompt, "").strip()
    print("---------------------------------------------------------------------------------------------------------")
    print("解题过程：\n",response)
    print("---------------------------------------------------------------------------------------------------------")
    # 返回生成的解题过程
    return response






# def generate_manim_code_from_solution(problem, solution, model=model, attempt=0, max_retries=3):
#     # 生成Manim代码的提示
#     prompt = f"""
#              根据下面的题目与解题过程,生成相应的Manim代码，来展示这个解题过程中的数学函数图像和解题步骤。展示按照顺序如下：
#                         1.每个视频展开的样式分为两幕，第一幕为展示题目，第二幕为展示解析。
#                         2.第一幕展示后先清除，再展示第二幕。
#                         3.第一幕的目标为展示题目信息，文字不能超过边界，且文字不能重叠。
#                         4.第二幕要求屏幕一分为二，左边为画图，右边为文字解析：
#                           4.1 首先在画布左半边展示画图内容，画图符合题目意思，且要素完整，图像内容只能在左半区。
#                           4.2 其次，在画布的右半边展示文字讲解内容，要求不能重叠，文字内容只能在右半区。
#                         5.文字与图像不能重叠
#
#                         可以根据示例manim代码对输入的解题过程进行模板嵌套：
#                         示例manim代码模板：
#     from manim import *
#     class SolveLinearFunction(Scene):
#         def construct(self):
#             # 第一幕 - 展示题目
#             question = Text("已知一次函数y=ax+b,求解x=c时y的值", font="思源黑体 CN Regular", font_size=24).scale(0.7)
#             self.play(Write(question))
#             self.wait(2)
#             # 进入第二幕
#             self.play(FadeOut(question))
#             # 第二幕画图部分（左边） - 绘制y=ax+b的函数图像
#             axes = Axes(
#                 x_range=[-3, 3, 1],
#                 y_range=[-3, 3, 1],
#                 x_length=6,
#                 y_length=6,
#             ).scale(0.5).to_edge(LEFT, buff=0.8)
#             labels = axes.get_axis_labels(x_label="x", y_label="y")
#             self.play(Create(axes), Write(labels))
#             line = axes.plot(lambda x: 2 * x + 1,color=BLUE)
#             graph_label = axes.get_graph_label(line, label="y = ax + b", x_val=1, direction=UR)
#             # self.play(Write(axes), run_time=3)
#             point_y_intercept = Dot(axes.coords_to_point(1, 3), color=RED)
#             points = VGroup(point_y_intercept)
#             self.play(Create(line), Write(graph_label))
#             self.play(Create(points))
#             # 第二幕解析部分（右边） - 描述解题过程
#             explanation = VGroup(
#                 Text("1.将已知的x值代入一次函数:", font_size=24),
#                 Text("y= ax + b", font_size=24),
#                 Text("2.求解y值:", font_size=24),
#                 Text("y= ac + b", font_size=24),
#             ).arrange(DOWN, aligned_edge=LEFT).scale(0.7).to_edge(RIGHT, buff=0.8)
#             for step in explanation:
#                 self.play(Write(step))
#                 self.wait(2)
#             self.wait(3)
#
#
#
#                         数学题目：{problem}
#                         解题过程：{solution}
#                 """
#
#     # # 调用OpenAI API生成Manim代码
#     # response = openai.ChatCompletion.create(
#     #     model=model,
#     #     messages=[
#     #         {"role": "user", "content": prompt}
#     #     ]
#     # )
#     # 生成响应
#     with torch.cuda.amp.autocast():
#         inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
#         output = model.generate(**inputs, max_length=4096)
#         response = tokenizer.decode(output[0], skip_special_tokens=True)
#
#     # 返回生成的Manim代码
#     return response
#
#
# # 新添加的一个prompt代码
# def generate_manim_code_from_problem_and_solution(math_problem, solution, is_physics=False, model=model, attempt=0,
#                                                   max_retries=3):
#     # Encode the problem and solution into UTF-8 strings
#     math_problem_utf8 = math_problem.encode('utf-8')
#     solution_utf8 = solution.encode('utf-8')
#
#     problem_type = "physics" if is_physics else "mathematics"
#     prompt = f"""
#     Given the following {problem_type} problem and its solution, create Manim code to visually represent both the solution process and the key concepts. The code should use Manim's capabilities to create any necessary diagrams or models directly in the animation, such as cylinders or cones for geometric problems, without relying on any external images. Here are specific guidelines:
#
#     1. Start with the problem statement at the top, aligning it to the upper edge of the animation.
#     2. Automatically include visual diagrams or models at each step where they help visualize the concept being explained.
#     3. Use next_to for positioning equations and explanations, maintaining proper spacing.
#     4. Set the font size to 24 for all text and math elements.
#     5. Clearly distinguish between the problem statement and the solution process.
#     6. Use smooth visual transitions to ensure viewer comprehension without haste.
#     7. Space out text and graphics evenly to avoid clutter.
#     8. Intuitively interpret solution steps to include relevant diagrams or graphics such as force diagrams in physics or geometric constructions in mathematics.
#     9. Ensure mathematical expressions and shapes are properly displayed.
#     10. Manage text overflow to keep all elements on-screen.
#
#     For physics problems:
#     - Automatically draw force diagrams, free-body diagrams, or motion paths where applicable.
#
#     For mathematics problems:
#     - Automatically generate graphs, geometric constructions, or other illustrative diagrams as needed to elucidate the solution.
#
#     Enhance clarity with:
#     - Concise text explanations alongside relevant visuals, split into multiple lines if necessary.
#     - Limit each line to 10 words and each page to 10 lines.
#
#     Example Solution: {solution_utf8}
#
#     Manim code requirements:
#     - Interpret and visualize crucial steps using diagrams.
#     - Include FadeIn and corresponding FadeOut animations to prevent overlapping.
#     """
#
#     # response = openai.ChatCompletion.create(
#     #     model=model,
#     #     messages=[{"role": "user", "content": prompt}]
#     # )
#     # code = response['choices'][0]['message']['content']
#     # 生成响应
#     with torch.cuda.amp.autocast():
#         inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
#         output = model.generate(**inputs, max_length=4096)
#         response = tokenizer.decode(output[0], skip_special_tokens=True)
#     return response
#     # if validate_code(code):
#     #     return code
#     # else:
#     #     raise ValueError("The generated code is not valid.")



# math_problem = "已知一次函数y=-2x+5,求其图像关于y轴对称的函数表达式"


# math_problem = "已知直线经过点(1,2)和(3,0)，求这条直线的解析式"
# math_problem = "若一次函数 y = kx + b 的函数值 y 随 x 的增大而减小，且图像与 y 轴的负半轴相交。求对 k 和 b 的符号判断"
# math_problem = "求函数y = -\\frac{3}{4}x + 3的坐标三角形的三条边长。"
# math_problem = "一次函数 y = kx + b 的图像与 y 轴的交点坐标是什么？"
# math_problem = "已知一次函数y=kx+b的图像与x轴的交点坐标是什么？"
# math_problem = "已知一次函数y = -\\frac{3}{4}x + b的坐标三角形周长为16，求三角形的面积。"
# math_problem = "如果一次函数y = kx + b的图像不经过第二象限,那么k和b的范围是多少"
# math_problem = "已知一次函数 y = kx + b 经过点 (0, 1) 和 (1, 0)，求该函数的解析式"
# math_problem = "已知一次函数 y = kx + b 经过点 (0, 4) 和 (1, 2)，求该函数的解析式"
# math_problem = "如果一次函数y = kx + b的图像与y轴的交点在y轴的负半轴上，求b的符号"
# math_problem = "已知一次函数y=ax+b,求解x=c时y的值"
# math_problem = "若直线y=ax+b经过第一、二、三象限，求解a*b的符号"
# math_problem = "若一次函数y=ax+b与直线y=3x+6垂直，且它们的交点是(-1,3)，求解a和b的值"
# math_problem = "某城市的人口y与时间t之间存在一次函数关系。已知在2010年人口为50万，在2015年人口为55万，求人口y与时间t的一次函数方程。"
# math_problem = "已知一次函数y=x+2，将该函数的图像向右平移3个单位，再向上平移2个单位，求平移后的函数表达式。"
# math_problem = "已知一次函数y=-3x+5，x的取值范围是[-2.3]时，函数的值域。"
# math_problem = "已知两个一次函数f(x)=3x-4和g(x)=2x+1，求组合函数h(x)=f(g(x))的表达式，并求当h(x)=10时x的值。"
# math_problem = "已知一次函数y=-2x+5,求其图像关于y轴对称的函数表达式。"
# math_problem = "已知一次函数y=kx+3的图像与直线y=2x+1平行，且它们的交点在y轴上的纵坐标为-3,求k的值。"
# math_problem = "已知一次函数f(x)=mx+c和g(x)=nx+d,且当x=1时，f(x)=g(x)，当x=2时，f(x)-g(x)=4.求m、n、c、d之间的关系。"
# math_problem = "已知一次函数y＝kx+b（k＜0）的图象经过点（x1，5）、（x2，﹣2），求x1和x2的大小关系"
# math_problem = "若点A（m，n）在y=2/3x+b的图象上，且2m﹣3n＞6，求b的取值范围。"
# math_problem = "已知直线y＝kx+b过点（2，2），并与x轴负半轴相交，若m＝3k+2b，求m的取值范围为。"
# math_problem = "已知函数y＝3x+1的图象经过点A（2/3，m），求关于x的不等式3x＜m﹣1的解集。"
# math_problem = "已知点（-1，y1），（4，y2）在一次函数y=3x-2的图象上，求y1，y2，0的大小关系。"
# math_problem = "已知一次函数y=(k-2)x+k不经过第三象限，求k的取值范围。"
# math_problem = "将一次函数y=kx+2的图象向下平移3个单位长度后经过点（-4，3），求k的值。"
# math_problem = "求函数y=(x^(1/2))/(x-2)中自变量x的取值范围。"
# math_problem = "已知点A(m,y1), B(m+1, y2)，都在直线y=2x-3上，求y2-y1的值。"
# math_problem = "若点P(a,b)在直线y=2x-1上，求代数式8-4a+2b的值。"
# math_problem = "一次函数y=x+m+2的图象不经过第二象限，求m的取值范围。"
# math_problem = "若一次函数y＝kx+2的图象，y随x的增大而增大，并与x轴、y轴所围成的三角形的面积为2，求k的值。"
# math_problem = "在平面直角坐标系中，已知直线y=kx+b与直线y=2x+2022平行，且与y轴交于点M(0,4)，与x轴的交点为N，求三角形MNO的面积。"
# math_problem = "把直线y=-x-3向上平移m个单位后，与直线y=2x+4的交点在第二象限，求m的取值范围。"
# math_problem = "无论m取任何实数，一次函数y=（m﹣1）x+m﹣3必过一定点，求定点的坐标。"
# math_problem = "已知一次函数y=kx+2k+3的图象不经过第三象限，求k的取值范围。"
# math_problem = "已知方程|x|=ax+1有一个负根但没有正根，求a的取值范围。"
# math_problem = "直线y＝kx﹣2与直线y＝x﹣1（1≤x≤4）有交点，求k的取值范围。"
# math_problem = "将直线先向上平移2个单位，再向右平移2个单位得到的直线l对应的一次函数的表达式为。"
math_problem = "问已知一次函数y=kx+b+1且k不等于0的图像经过第一、二、四象限，且过点（3，-2），求b关于k的代数表达式，以及k的取值范围是。"


solution = generate_solution_from_problem(math_problem)


# manim_code = generate_manim_code_from_solution(math_problem, solution)
# manim_code = generate_manim_code_from_problem_and_solution(math_problem, solution)
# manim_code = generate_manim_code_from_solution_ch(math_problem, solution)
#
# print("解题过程：", solution)
# print("Manim代码:", manim_code)


def prompt_question_number(problem, solution, model=model, attempt=0, max_retries=3):
    prompt_knowledge = f"""
                     对所给出的问题进行分析，并且输出它属于哪一个知识点类别，要求只输出最可能的一个序号就行。
                     1.点的坐标特征，如：点位于什么象限
                     2.点到坐标轴的距离，或者到直线的距离
                     3.点的几何变换，如：平移，对称，旋转
                     4.一次函数的图像和性质
                     5.一次函数与方程组的解
                     6.一次函数与不等式组
                     7.应用题
                     8.无需画图分析的代数计算题目

                     数学题目：{problem}
                     
            """

    # response = openai.ChatCompletion.create(
    #     model=model,
    #     messages=[{"role": "user", "content": prompt_knowledge}]
    # )
    # 生成响应
    with torch.cuda.amp.autocast():
        inputs = tokenizer(prompt_knowledge, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_length=1024)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        # 手动去除prompt_knowledge的内容
        response = response.replace(prompt_knowledge, "").strip()

    print("---------------------------------------------------------------------------------------------------------")
    print("选择类别：\n", response)
    print("---------------------------------------------------------------------------------------------------------")
    return response






prompt_config = []
p1 = """
         根据下面的题目与解题过程,生成相应的Manim代码，来展示这个解题过程中的数学函数图像和解题步骤。展示按照顺序如下：
                    1.每个视频展开的样式分为两幕，第一幕为展示题目，第二幕为展示解析。
                    2.第一幕的目标为展示题目信息，文字不能超过边界，且文字不能重叠。
                    3.第二幕要求分步展示解题过程：
                      3.1首先在画布左半边展示画图内容，画图符合题目意思，且要素完整，图像内容不能超过边界。
                      3.2其次，在画布的右半边展示文字讲解内容，要求不能重叠，文字内容不能超过边界。
                    4.文字与图像不能重叠

                    特别要求：需要展示象限的具体数学特征（第一象限、第二象限、第三象限、第四象限）
                            需要在画图中标注出点的位置

"""

p2 = """
         根据下面的题目与解题过程,生成相应的Manim代码，来展示这个解题过程中的数学函数图像和解题步骤。展示按照顺序如下：
                    1.每个视频展开的样式分为两幕，第一幕为展示题目，第二幕为展示解析。
                    2.第一幕的目标为展示题目信息，文字不能超过边界，且文字不能重叠。
                    3.第二幕要求分步展示解题过程：
                      3.1首先在画布左半边展示画图内容，画图符合题目意思，且要素完整，图像内容不能超过边界。
                      3.2其次，在画布的右半边展示文字讲解内容，要求不能重叠，文字内容不能超过边界。
                    4.文字与图像不能重叠

                    特别要求：展示出点在象限中的位置及数值
                            展示出点与两个轴之间的距离


"""

p3 = """
         根据下面的题目与解题过程,生成相应的Manim代码，来展示这个解题过程中的数学函数图像和解题步骤。展示按照顺序如下：
                    1.每个视频展开的样式分为两幕，第一幕为展示题目，第二幕为展示解析。
                    2.第一幕的目标为展示题目信息，文字不能超过边界，且文字不能重叠。
                    3.第二幕要求分步展示解题过程：
                      3.1首先在画布左半边展示画图内容，画图符合题目意思，且要素完整，图像内容不能超过边界。
                      3.2其次，在画布的右半边展示文字讲解内容，要求不能重叠，文字内容不能超过边界。
                    4.文字与图像不能重叠

                    特别要求：展示几何图形各个顶点的取值，标注出各个顶点的位置以及命名，标注出顶点变化前后的取值


"""

p4 = """
         根据下面的题目与解题过程,生成相应的Manim代码，来展示这个解题过程中的数学函数图像和解题步骤。展示按照顺序如下：
                    1.每个视频展开的样式分为两幕，第一幕为展示题目，第二幕为展示解析。
                    2.第一幕的目标为展示题目信息，文字不能超过边界，且文字不能重叠。
                    3.第二幕要求分步展示解题过程：
                      3.1首先在画布左半边展示画图内容，画图符合题目意思，且要素完整，图像内容不能超过边界。
                      3.2其次，在画布的右半边展示文字讲解内容，要求不能重叠，文字内容不能超过边界。
                    4.文字与图像不能重叠

                    特别要求：展示几何图形各个顶点的取值，标注出各个顶点的位置以及命名，标注出顶点变化前后的取值


"""

p5 = """
         根据下面的题目与解题过程,生成相应的Manim代码，来展示这个解题过程中的数学函数图像和解题步骤。展示按照顺序如下：
                    1.每个视频展开的样式分为两幕，第一幕为展示题目，第二幕为展示解析。
                    2.第一幕的目标为展示题目信息，文字不能超过边界，且文字不能重叠。
                    3.第二幕要求分步展示解题过程：
                      3.1首先在画布左半边展示画图内容，画图符合题目意思，且要素完整，图像内容不能超过边界。
                      3.2其次，在画布的右半边展示文字讲解内容，要求不能重叠，文字内容不能超过边界。
                    4.文字与图像不能重叠
                    特别要求：展示坐标轴及单位，展示直线方程信息，展示直线以及直线的变换
    """

p6 = """
         根据下面的题目与解题过程,生成相应的Manim代码，来展示这个解题过程中的数学函数图像和解题步骤。展示按照顺序如下：
                    1.每个视频展开的样式分为两幕，第一幕为展示题目，第二幕为展示解析。
                    2.第一幕的目标为展示题目信息，文字不能超过边界，且文字不能重叠。
                    3.第二幕要求分步展示解题过程：
                      3.1首先在画布左半边展示画图内容，画图符合题目意思，且要素完整，图像内容不能超过边界。
                      3.2其次，在画布的右半边展示文字讲解内容，要求不能重叠，文字内容不能超过边界。
                    4.文字与图像不能重叠

                    特别要求：展示一次函数图像与坐标轴的交点来确认方程组的解，交点的横坐标是x值，交点的纵坐标是y值


"""

p7 = """
         根据下面的题目与解题过程,生成相应的Manim代码，来展示这个解题过程中的数学函数图像和解题步骤。展示按照顺序如下：
                    1.每个视频展开的样式分为两幕，第一幕为展示题目，第二幕为展示解析。
                    2.第一幕的目标为展示题目信息，文字不能超过边界，且文字不能重叠。
                    3.第二幕要求分步展示解题过程：
                      3.1首先在画布左半边展示画图内容，画图符合题目意思，且要素完整，图像内容不能超过边界。
                      3.2其次，在画布的右半边展示文字讲解内容，要求不能重叠，文字内容不能超过边界。
                    4.文字与图像不能重叠

                    特别要求：显示直角坐标系及对应的单位、取值
                            显示一次函数的直线（一般是2条直线）
                            结合题意重点突出图像区域


"""

p8 = """
         根据下面的题目与解题过程,生成相应的Manim代码，来展示这个解题过程中的数学函数图像和解题步骤。展示按照顺序如下：
                    1.每个视频展开的样式分为两幕，第一幕为展示题目，第二幕为展示解析。
                    2.第一幕的目标为展示题目信息，文字不能超过边界，且文字不能重叠。
                    3.第二幕要求分步展示解题过程：
                      3.1首先在画布左半边展示画图内容，画图符合题目意思，且要素完整，图像内容不能超过边界。
                      3.2其次，在画布的右半边展示文字讲解内容，要求不能重叠，文字内容不能超过边界。
                    4.文字与图像不能重叠

                    特别要求：对于应用题、代数计算为主的题目以及概念与定义理解类型的题目，可以不绘制图像，仅显示文字解读



"""

# p1 = """
#          According to the following questions and problem-solving process, generate the corresponding Manim code to display the mathematical function image and problem-solving steps in this problem-solving process. The display is in the following order:
# 1. Each video is divided into two scenes, the first scene is to display the question, and the second scene is to display the analysis.
# 2. The goal of the first scene is to display the question information. The text cannot exceed the boundary and the text cannot overlap.
# 3. The second scene requires a step-by-step display of the problem-solving process:
# 3.1 First, display the drawing content on the left half of the canvas. The drawing is consistent with the meaning of the question, and the elements are complete. The image content cannot exceed the boundary.
# 3.2 Secondly, display the text explanation content on the right half of the canvas. It is required that it cannot overlap and the text content cannot exceed the boundary.
# 4. The text and image cannot overlap
#
# Special requirements: It is necessary to display the specific mathematical characteristics of the quadrant (first quadrant, second quadrant, third quadrant, fourth quadrant)
# It is necessary to mark the position of the point in the drawing
#
# """
#
# p2 = """
#          According to the following questions and problem-solving process, generate the corresponding Manim code to display the mathematical function image and problem-solving steps in this problem-solving process. The display is in the following order:
# 1. Each video is divided into two scenes, the first scene is to display the question, and the second scene is to display the analysis.
# 2. The goal of the first scene is to display the question information. The text cannot exceed the boundary and the text cannot overlap.
# 3. The second scene requires a step-by-step display of the problem-solving process:
# 3.1 First, display the drawing content on the left half of the canvas. The drawing is consistent with the meaning of the question, and the elements are complete. The image content cannot exceed the boundary.
# 3.2 Secondly, display the text explanation content on the right half of the canvas. It is required that it cannot overlap and the text content cannot exceed the boundary.
# 4. The text and image cannot overlap
#
# Special requirements: Display the position and value of the point in the quadrant
# Display the distance between the point and the two axes
#
#
# """
#
# p3 = """
#          According to the following questions and problem-solving process, generate the corresponding Manim code to display the mathematical function image and problem-solving steps in this problem-solving process. The display is in the following order:
# 1. Each video is divided into two scenes, the first scene is to display the question, and the second scene is to display the analysis.
# 2. The goal of the first scene is to display the question information. The text cannot exceed the boundary and the text cannot overlap.
# 3. The second scene requires a step-by-step display of the problem-solving process:
# 3.1 First, display the drawing content on the left half of the canvas. The drawing is consistent with the meaning of the question, and the elements are complete. The image content cannot exceed the boundary.
# 3.2 Secondly, display the text explanation content on the right half of the canvas. It is required that it cannot overlap and the text content cannot exceed the boundary.
# 4. The text and image cannot overlap
#
# Special requirements: Display the values of each vertex of the geometric figure, mark the position and name of each vertex, and mark the values before and after the vertex changes
#
#
# """
#
# p4 = """
#          According to the following questions and problem-solving process, generate the corresponding Manim code to display the mathematical function image and problem-solving steps in this problem-solving process. The display is in the following order:
# 1. Each video is divided into two scenes, the first scene is to display the question, and the second scene is to display the analysis.
# 2. The goal of the first scene is to display the question information. The text cannot exceed the boundary and the text cannot overlap.
# 3. The second scene requires a step-by-step display of the problem-solving process:
# 3.1 First, display the drawing content on the left half of the canvas. The drawing is consistent with the meaning of the question, and the elements are complete. The image content cannot exceed the boundary.
# 3.2 Secondly, display the text explanation content on the right half of the canvas. It is required that it cannot overlap and the text content cannot exceed the boundary.
# 4. The text and image cannot overlap
#
# Special requirements: Display the values of each vertex of the geometric figure, mark the position and name of each vertex, and mark the values before and after the vertex changes
#
#
# """
#
# p5 = """
#          According to the following questions and problem-solving process, generate the corresponding Manim code to display the mathematical function image and problem-solving steps in this problem-solving process. The display is in the following order:
# 1. Each video is divided into two scenes, the first scene is to display the question, and the second scene is to display the analysis.
# 2. The goal of the first scene is to display the question information. The text cannot exceed the boundary and the text cannot overlap.
# 3. The second scene requires a step-by-step display of the problem-solving process:
# 3.1 First, display the drawing content on the left half of the canvas. The drawing is consistent with the meaning of the question, and the elements are complete. The image content cannot exceed the boundary.
# 3.2 Secondly, display the text explanation content on the right half of the canvas. It is required that it cannot overlap and the text content cannot exceed the boundary.
# 4. The text and image cannot overlap
# Special requirements: display the coordinate axis and unit, display the straight line equation information, display the straight line and the transformation of the straight line
#     """
#
# p6 = """
#          According to the following questions and problem-solving process, generate the corresponding Manim code to display the mathematical function image and problem-solving steps in this problem-solving process. The display is in the following order:
# 1. Each video is divided into two scenes, the first scene is to display the question, and the second scene is to display the analysis.
# 2. The goal of the first scene is to display the question information. The text cannot exceed the boundary and the text cannot overlap.
# 3. The second scene requires a step-by-step display of the problem-solving process:
# 3.1 First, display the drawing content on the left half of the canvas. The drawing is consistent with the meaning of the question, and the elements are complete. The image content cannot exceed the boundary.
# 3.2 Secondly, display the text explanation content on the right half of the canvas. It is required that it cannot overlap and the text content cannot exceed the boundary.
# 4. The text and the image cannot overlap
#
# Special requirements: Display the intersection of the function image and the coordinate axis to confirm the solution of the equation system. The horizontal coordinate of the intersection is the x value, and the vertical coordinate of the intersection is the y value
#
#
# """
#
# p7 = """
#          According to the following questions and problem-solving process, generate the corresponding Manim code to display the mathematical function image and problem-solving steps in this problem-solving process. The display is in the following order:
# 1. Each video is divided into two scenes, the first scene is to display the question, and the second scene is to display the analysis.
# 2. The goal of the first scene is to display the question information. The text cannot exceed the boundary and the text cannot overlap.
# 3. The second scene requires a step-by-step display of the problem-solving process:
# 3.1 First, display the drawing content on the left half of the canvas. The drawing is in line with the meaning of the question, and the elements are complete. The image content cannot exceed the boundary.
# 3.2 Secondly, display the text explanation content on the right half of the canvas. It is required that it cannot overlap and the text content cannot exceed the boundary.
# 4. The text and image cannot overlap
#
# Special requirements: Display the rectangular coordinate system and the corresponding units and values
# Display the straight line of the linear function (usually 2 straight lines)
# Combine the meaning of the question to highlight the image area
#
#
# """
#
# p8 = """
#          According to the following questions and problem-solving process, generate the corresponding Manim code to display the mathematical function image and problem-solving steps in this problem-solving process. The display is in the following order:
# 1. Each video is divided into two scenes, the first scene is to display the question, and the second scene is to display the analysis.
# 2. The goal of the first scene is to display the question information. The text cannot exceed the boundary and the text cannot overlap.
# 3. The second scene requires a step-by-step display of the problem-solving process:
# 3.1 First, display the drawing content on the left half of the canvas. The drawing is consistent with the meaning of the question, and the elements are complete. The image content cannot exceed the boundary.
# 3.2 Secondly, display the text explanation content on the right half of the canvas. It is required that it cannot overlap and the text content cannot exceed the boundary.
# 4. The text and image cannot overlap
#
# Special requirements: For application questions, questions mainly based on algebraic calculations, and questions of concept and definition understanding, you can not draw images, only display text interpretation
#
#
#
# """
prompt_config = [p1, p2, p3, p4, p5, p6, p7, p8]
# print(prompt_config)


# example_manim = """
# from manim import *
# class SolveCompositeFunction(Scene):
#     def construct(self):
#         # 创建标题文本
#         title1 = Text("已知两个一次函数f(x)=3x-4和g(x)=2x+1，", font_size=24)
#         title2 = Text("求组合函数h(x)=f(g(x))的表达式，并求当h(x)=10时x的值。", font_size=24)
#         title_group = VGroup(title1, title2).arrange(DOWN, center=False).move_to(ORIGIN)
#
#         # 播放标题显示动画
#         self.play(Write(title_group))
#         self.wait(2)
#         self.play(FadeOut(title_group))
#
#         axes = Axes(
#           x_range=[***, ***, ***],
#           y_range=[***, ***, ***],  # 调整比例
#           x_length=10,
#           y_length=10,
#             axis_config={"color": WHITE},
#         ).scale(0.5).to_edge(LEFT, buff=0.5)
#         labels = axes.get_axis_labels(x_label="x", y_label="y")
#         self.play(Create(axes), Write(labels))
#
#         f_line = axes.plot(lambda x: 3*x-4, color=BLUE)
#         g_line = axes.plot(lambda x: 2*x+1, color=RED)
#         f_label = axes.get_graph_label(f_line, label='y=3x-4', x_val=1)
#         g_label = axes.get_graph_label(g_line, label='g(x)', x_val=1)
#         fg_line = axes.plot(lambda x: 6*x-1, color=YELLOW)
#         fg_label = axes.get_graph_label(fg_line, label='f(g(x))', x_val=1)
#         #播放创建坐标轴、标签和图像的动画
#
#         self.play(Create(f_line), Write(f_label))
#         self.play(Create(g_line), Write(g_label))
#         self.play(Create(fg_line), Write(fg_label))
#
#
#         #[supplement]
#
#
#         # 创建解释文本
#         explanation = VGroup(
#             Text("1. f(x)=3x-4函数如蓝线所示:", font_size=24),
#             Text("2. g(x)=2x+1函数如红线所示", font_size=24),
#             Text("3. h(x)=f(g(x))=6x-1函数如黄线所示", font_size=24),
#             Text("4. 求当h(x)=10时：", font_size=24),
#             MathTex("6x-1 = 10 \\Rightarrow x=\\frac{11}{6}", font_size=24),
#         ).arrange(DOWN, aligned_edge=LEFT).scale(0.7).to_edge(RIGHT, buff=0.5)
#
#         # 逐步显示解释文本
#         for step in explanation:
#             self.play(Write(step))
#             self.wait(2)
#         self.wait(3)
# """


# example_manim = """
# from manim import *
# class SolveCompositeFunction(Scene):
#     def construct(self):
#         # 创建标题文本
#         title1 = Text("已知两个一次函数f(x)=3x-4和g(x)=2x+1，", font_size=24)
#         title2 = Text("求组合函数h(x)=f(g(x))的表达式，并求当h(x)=10时x的值。", font_size=24)
#         title_group = VGroup(title1, title2).arrange(DOWN, center=False).move_to(ORIGIN)
#
#         # 播放标题显示动画
#         self.play(Write(title_group))
#         self.wait(2)
#         self.play(FadeOut(title_group))
#
#         axes = Axes(
#           x_range=[***, ***, ***],
#           y_range=[***, ***, ***],  # 调整比例
#           x_length=10,
#           y_length=10,
#             axis_config={"color": WHITE},
#         ).scale(0.5).to_edge(LEFT, buff=0.5)
#         labels = axes.get_axis_labels(x_label="x", y_label="y")
#         self.play(Create(axes), Write(labels))
#
#
#         #根据解题需求选择合适的<>内容并生成代码，可以不选择生成
#         <
#         创建直线
#         f_line = axes.plot(lambda x: 3*x-4, color=BLUE)
#         g_line = axes.plot(lambda x: 2*x+1, color=RED)
#         f_label = axes.get_graph_label(f_line, label='y=3x-4', x_val=1)
#         g_label = axes.get_graph_label(g_line, label='g(x)', x_val=1)
#         fg_line = axes.plot(lambda x: 6*x-1, color=YELLOW)
#         fg_label = axes.get_graph_label(fg_line, label='f(g(x))', x_val=1)
#         self.play(Create(f_line), Write(f_label))
#         self.play(Create(g_line), Write(g_label))
#         self.play(Create(fg_line), Write(fg_label))
#         >
#
#         <
#         创建直线的交点，对未知数取适当的数
#         point_1 = Dot(axes.coords_to_point(***, ***), color=RED)
#         point_2 = Dot(axes.coords_to_point(***, ***), color=RED)
#         label_1 = MathTex("(***, ***)").move_to(axes.coords_to_point(***, ***) + UP * 0.5 + LEFT * 0.5)
#         label_2 = MathTex("(***, ***)").move_to(axes.coords_to_point(***, ***) + DOWN * 0.5 + LEFT * 0.5)
#         point = VGroup(point_1, point_2, label_1, label_2)
#         self.play(Create(point))
#         >
#
#         <
#         创建图形（原理是把点连成线）
#         point_origin = Dot(axes.coords_to_point(***, ***), color=RED)
#             label_origin = MathTex("***").next_to(point_origin, DR, buff=0.1)
#             point_x_intercept = Dot(axes.coords_to_point(***, ***), color=RED)
#             label_x_intercept = MathTex("A(***, ***)").next_to(point_x_intercept, DL, buff=0.1)
#             point_y_intercept = Dot(axes.coords_to_point(***, ***), color=RED)
#             label_y_intercept = MathTex("B(***, ***)").next_to(point_y_intercept, UR, buff=0.1)
#             points = VGroup(point_origin, label_origin, point_x_intercept, label_x_intercept, point_y_intercept,
#                             label_y_intercept)
#             triangle = Polygon(
#                 axes.coords_to_point(***, ***),
#                 axes.coords_to_point(***, ***),
#                 axes.coords_to_point(***, ***),
#                 color=YELLOW
#             )
#             self.play(Create(points))
#             self.play(Create(triangle))
#             ]
#         >
#
#
#         # 创建解释文本
#         explanation = VGroup(
#             Text("1. f(x)=3x-4函数如蓝线所示:", font_size=24),
#             Text("2. g(x)=2x+1函数如红线所示", font_size=24),
#             Text("3. h(x)=f(g(x))=6x-1函数如黄线所示", font_size=24),
#             Text("4. 求当h(x)=10时：", font_size=24),
#             MathTex("6x-1 = 10 \\Rightarrow x=\\frac{11}{6}", font_size=24),
#         ).arrange(DOWN, aligned_edge=LEFT).scale(0.7).to_edge(RIGHT, buff=0.5)
#         for step in explanation:
#             self.play(Write(step))
#             self.wait(2)
#         self.wait(3)
# """


example_manim = """
示例题目：
已知两个一次函数f(x)=3x-4和g(x)=2x+1，求组合函数h(x)=f(g(x))的表达式，并求当h(x)=10时x的值。

示例解题过程：
1. f(x)=3x-4函数如蓝线所示:
2. g(x)=2x+1函数如红线所示
3. 两条直线的交点为(5, 11)
4. h(x)=f(g(x))=6x-1函数如黄线所示"
5. 求当h(x)=10时：
   6x-1 = 10 所以 x=11/6

示例代码：
from manim import *
class SolveCompositeFunction(Scene):
    def construct(self):
        # 创建标题文本
        title1 = Text("已知两个一次函数f(x)=3x-4和g(x)=2x+1，", font_size=24)
        title2 = Text("求组合函数h(x)=f(g(x))的表达式，并求当h(x)=10时x的值。", font_size=24)
        title_group = VGroup(title1, title2).arrange(DOWN, center=False).move_to(ORIGIN)
        # 播放标题显示动画
        self.play(Write(title_group))
        self.wait(2)
        self.play(FadeOut(title_group))

        axes = Axes(
          x_range=[-2, 4, 1], y_range=[-6, 18, 1]
        ).scale(0.5).to_edge(LEFT, buff=0.5)
        labels = axes.get_axis_labels(x_label="x", y_label="y")
        
        #创建解析过程中出现的每一个点
        point_1 = Dot(axes.coords_to_point(5, 11), color=RED)
        label_1 = MathTex("(5, 11)").move_to(axes.coords_to_point(5, 11) + UP * 0.5 + LEFT * 0.5)
        point = VGroup(point_1, label_1)
        self.play(Create(point))
        
        #创建解析过程中出现的每一条直线
        f_graph = axes.plot(lambda x: 3*x-4, color=BLUE)
        g_graph = axes.plot(lambda x: 2*x+1, color=RED)
        f_label = axes.get_graph_label(f_graph, label='f(x)')
        g_label = axes.get_graph_label(g_graph, label='g(x)')
        fg_graph = axes.plot(lambda x: 6*x-1, color=YELLOW)
        fg_label = axes.get_graph_label(fg_graph, label='f(g(x))')
        # 播放创建坐标轴、标签和图像的动画
        self.play(Create(axes), Write(labels))
        self.play(Create(f_graph), Write(f_label))
        self.play(Create(g_graph), Write(g_label))
        self.play(Create(fg_graph), Write(fg_label))
        
        
        
        # 创建解释文本
        explanation = VGroup(
            Text("1. f(x)=3x-4函数如蓝线所示:", font_size=24),
            Text("2. g(x)=2x+1函数如红线所示", font_size=24),
            Text("3. h(x)=f(g(x))=6x-1函数如黄线所示", font_size=24),
            Text("4. 求当h(x)=10时：", font_size=24),
            MathTex("6x-1 = 10 \\Rightarrow x=\\frac{11}{6}", font_size=24),
        ).arrange(DOWN, aligned_edge=LEFT).scale(0.7).to_edge(RIGHT, buff=0.5)

        # 逐步显示解释文本
        for step in explanation:
            self.play(Write(step))
            self.wait(2)
        self.wait(3)
"""

# example_manim = """
# from manim import *
# class SolveCompositeFunction(Scene):
#     def construct(self):
#         #In the first scene, create the mathematics question, using commas and periods to separate the lines that appear
#         title1 = Text("<mathematics question>，", font_size=24)
#         title2 = Text("<Display the text in math problems line by line>。", font_size=24)
#         title_group = VGroup(title1, title2).arrange(DOWN, center=False).move_to(ORIGIN)
#         self.play(Write(title_group))
#         self.wait(2)
#         self.play(FadeOut(title_group))
#
#         #Act 2: Create problem-solving animation and text
#         #To create the axis, you must have
#         axes = Axes(
#           x_range=[***, ***, ***],
#           y_range=[***, ***, ***],
#           x_length=10,
#           y_length=10,
#             axis_config={"color": WHITE},
#         ).scale(0.5).to_edge(LEFT, buff=0.5)
#         labels = axes.get_axis_labels(x_label="x", y_label="y")
#         self.play(Create(axes), Write(labels))
#
#         #Please select the appropriate annotated code below to assist in the generation of animation (note that each annotated code can be referenced up to 4 times. If it is not necessary, you can choose not to create it)：
#         [
#         <Create Line>
#         f_line = axes.plot(lambda x: a*x-4, color=BLUE)
#         g_line = axes.plot(lambda x: <Function Expression>, color=RED)
#         f_label = axes.get_graph_label(f_line, label='y=3x-4', x_val=1)
#         g_label = axes.get_graph_label(g_line, label='<Function Expression>', x_val=1)
#         self.play(Create(f_line), Write(f_label))
#         self.play(Create(g_line), Write(g_label))
#
#         <Create Intersection of straight lines, take appropriate number for unknown number>
#         point_1 = Dot(axes.coords_to_point(***, ***), color=RED)
#         point_2 = Dot(axes.coords_to_point(***, ***), color=RED)
#         label_1 = MathTex("(***, ***)").move_to(axes.coords_to_point(***, ***) + UP * 0.5 + LEFT * 0.5)
#         label_2 = MathTex("(***, ***)").move_to(axes.coords_to_point(***, ***) + DOWN * 0.5 + LEFT * 0.5)
#         point = VGroup(point_1, point_2, label_1, label_2)
#         self.play(Create(point))
#
#         <Create graphics (the principle is to connect points into lines)>
#         point_origin = Dot(axes.coords_to_point(***, ***), color=RED)
#             label_origin = MathTex("***").next_to(point_origin, DR, buff=0.1)
#             point_x_intercept = Dot(axes.coords_to_point(***, ***), color=RED)
#             label_x_intercept = MathTex("A(***, ***)").next_to(point_x_intercept, DL, buff=0.1)
#             point_y_intercept = Dot(axes.coords_to_point(***, ***), color=RED)
#             label_y_intercept = MathTex("B(***, ***)").next_to(point_y_intercept, UR, buff=0.1)
#             points = VGroup(point_origin, label_origin, point_x_intercept, label_x_intercept, point_y_intercept,
#                             label_y_intercept)
#             triangle = Polygon(
#                 axes.coords_to_point(***, ***),
#                 axes.coords_to_point(***, ***),
#                 axes.coords_to_point(***, ***),
#                 color=YELLOW
#             )
#             self.play(Create(points))
#             self.play(Create(triangle))
#             ]
#
#
#         #Create the explanation text. This step must be filled in with the problem-solving process <Display the text of the problem-solving process line by line>. It can only appear once, and the number of lines displayed is divided by commas and periods.
#         #A maximum of 30 Chinese characters appear in each line
#         explanation = VGroup(
#             Text("1. <Display the text in the problem-solving process line by line>", font_size=24),
#             Text("2. <Display the text in the problem-solving process line by line>", font_size=24),
#             Text("3. <Display the text in the problem-solving process line by line>", font_size=24),
#             MathTex("<If there is a formula, it will be shown on the right：>6x-1 = 10 \\Rightarrow x=\\frac{11}{6}", font_size=24),
#         ).arrange(DOWN, aligned_edge=LEFT).scale(0.7).to_edge(RIGHT, buff=0.5)
#
#         for step in explanation:
#             self.play(Write(step))
#             self.wait(2)
#         self.wait(3)
# """


def generate_manim_code_from_solution_ch(problem, solution, model=model, attempt=0, max_retries=3, prompt_config=prompt_config, example_manim=example_manim):
    num = prompt_question_number(problem, solution, model=model, attempt=0, max_retries=3)
    # if "[UNUSED_TOKEN_145]" in num:
    #     num = num.replace("[UNUSED_TOKEN_145]", "")
    print("num=",num[0])
    print("num_lenth = ",len(num))
    print("num_type=",type(num))
    number = int(num[0])

    manim_supplement = """
    你可以从以下代码中挑选合适的代码对原本生成的[supplement]内容进行补充：
            ### 1. 绘制相交的点（选择性生成）
            point_1 = Dot(axes.coords_to_point(***, ***), color=RED)
            point_2 = Dot(axes.coords_to_point(***, ***), color=RED)
            label_1 = MathTex("(***, ***)").move_to(axes.coords_to_point(***, ***) + UP * 0.5 + LEFT * 0.5)  
            label_2 = MathTex("(***, ***)").move_to(axes.coords_to_point(***, ***) + DOWN * 0.5 + LEFT * 0.5)  
            point = VGroup(point_1, point_2, label_1, label_2)
            self.play(Create(point))

            ### 2. 绘制相交的点，绘制几何图形（选择性生成）
            point_origin = Dot(axes.coords_to_point(***, ***), color=RED)
            label_origin = MathTex("***").next_to(point_origin, DR, buff=0.1)
            point_x_intercept = Dot(axes.coords_to_point(***, ***), color=RED)
            label_x_intercept = MathTex("A(***, ***)").next_to(point_x_intercept, DL, buff=0.1)
            point_y_intercept = Dot(axes.coords_to_point(***, ***), color=RED)
            label_y_intercept = MathTex("B(***, ***)").next_to(point_y_intercept, UR, buff=0.1)
            points = VGroup(point_origin, label_origin, point_x_intercept, label_x_intercept, point_y_intercept,
                            label_y_intercept)
            triangle = Polygon(
                axes.coords_to_point(***, ***),
                axes.coords_to_point(***, ***),
                axes.coords_to_point(***, ***),
                color=YELLOW
            )
            self.play(Create(points))
            self.play(Create(triangle))
            
            从下面选择合适的补充代码并替换[supplement]：{manim_supplement}
    """

    # 生成Manim代码的提示
    prompt = f"""
    你是一个给学生讲解解题过程的数学老师。
    接下来你需要根据给定的数学题目和解题过程，以视频化形式为学生展开生动的解析，解析视频过程由manim代码生成。
    所有要求如下，请给出正确的代码：
                        {prompt_config[number - 1]}
                        可以根据示例manim代码对输入的解题过程进行模板嵌套：
                        示例流程：{example_manim}
                        数学题目：{problem}
                        解题过程：{solution}请将解题过程中出现的所有函数表达式，在manim代码画图时展示出来
                        





                        解题过程solution中每一行最多不超过25个字，若超出了25个字，请分段展示

                        最终结果只需要在response中展示manim代码即可



                """
    # prompt = f"""
    #                         {prompt_config[number - 1]}
    #                         mathematics question：{problem}
    #                         Problem Solving Process：{solution}Please display all function expressions that appear in the problem-solving process when drawing the manim code.
    #                         You can nest templates for the input problem-solving process based on the example manim code:
    #                         Example manim code template:{example_manim}
    #
    #
    #
    #
    #
    #                         The maximum length of each line in the solution is 25 characters. If it exceeds 25 characters, please display it in sections.
    #
    #                         The final result only needs to display the manim code in the response
    #
    #
    #
    #                 """

    # # 调用OpenAI API生成Manim代码
    # response = openai.ChatCompletion.create(
    #     model=model,
    #     messages=[
    #         {"role": "user", "content": prompt}
    #     ]
    # )
    # # 返回生成的Manim代码
    # return response['choices'][0]['message']['content']
    # 生成响应
    with torch.cuda.amp.autocast():
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_length=4096)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        # 手动去除prompt_knowledge的内容
        response = response.replace(prompt, "").strip()
    # print(response)
    return response


#
manim_code = generate_manim_code_from_solution_ch(math_problem, solution)

# print("解题过程：", solution)
# print("Manim代码:", manim_code)
print("---------------------------------------------------------------------------------------------------------")
print("Manim代码:\n", manim_code)
print("---------------------------------------------------------------------------------------------------------")