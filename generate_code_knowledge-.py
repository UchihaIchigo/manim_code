import pandas as pd
import openai
import time

# 设置API密钥
openai.api_key = ""
openai.api_base = "https://api.chatanywhere.tech/v1"


def generate_solution_from_problem(math_problem, model="gpt-4", attempt=0, max_retries=3):
    try:
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

        # 调用OpenAI API生成解题过程
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # 返回生成的解题过程
        return response['choices'][0]['message']['content']

    except (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError, OSError) as e:
        print(f"An error occurred: {e}.")
        if attempt < max_retries:
            print(f"Retrying... Attempt {attempt + 1} of {max_retries}")
            time.sleep(5)  # 等待5秒再重试
            return generate_solution_from_problem(math_problem, model=model, attempt=attempt + 1,
                                                  max_retries=max_retries)
        else:
            print("Max retries reached. Moving to the next word.")
            return "Error: Max retries reached."

    except Exception as e:
        print(f"An unexpected error occurred: {e}. Moving to the next word.")
        return "Error: An unexpected error occurred."


def generate_manim_code_from_solution(problem, solution, model="gpt-4", attempt=0, max_retries=3):
    try:
        # 生成Manim代码的提示
        prompt = f"""
         根据下面的题目与解题过程,生成相应的Manim代码，来展示这个解题过程中的数学函数图像和解题步骤。展示按照顺序如下：
                    1.每个视频展开的样式分为两幕，第一幕为展示题目，第二幕为展示解析。
                    2.第一幕展示后先清除，再展示第二幕。
                    3.第一幕的目标为展示题目信息，文字不能超过边界，且文字不能重叠。
                    4.第二幕要求屏幕一分为二，左边为画图，右边为文字解析：
                      4.1 首先在画布左半边展示画图内容，画图符合题目意思，且要素完整，图像内容只能在左半区。
                      4.2 其次，在画布的右半边展示文字讲解内容，要求不能重叠，文字内容只能在右半区。
                    5.文字与图像不能重叠
                    
                    可以根据示例manim代码对输入的解题过程进行模板嵌套：
                    manim代码：
from manim import *
class SolveLinearFunction(Scene):
    def construct(self):
        # 第一幕 - 展示题目
        question = Text("已知一次函数y=ax+b,求解x=c时y的值", font="思源黑体 CN Regular", font_size=24).scale(0.7)
        self.play(Write(question))
        self.wait(2)
        # 进入第二幕
        self.play(FadeOut(question))
        # 第二幕画图部分（左边） - 绘制y=ax+b的函数图像
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=6,
            y_length=6,
        ).scale(0.5).to_edge(LEFT, buff=0.8)
        labels = axes.get_axis_labels(x_label="x", y_label="y")
        self.play(Create(axes), Write(labels))
        line = axes.plot(lambda x: 2 * x + 1,color=BLUE)
        graph_label = axes.get_graph_label(line, label="y = ax + b", x_val=1, direction=UR)
        # self.play(Write(axes), run_time=3)
        point_y_intercept = Dot(axes.coords_to_point(1, 3), color=RED)
        points = VGroup(point_y_intercept)
        self.play(Create(line), Write(graph_label))
        self.play(Create(points))
        # 第二幕解析部分（右边） - 描述解题过程
        explanation = VGroup(
            Text("1.将已知的x值代入一次函数:", font_size=24),
            Text("y= ax + b", font_size=24),
            Text("2.求解y值:", font_size=24),
            Text("y= ac + b", font_size=24),
        ).arrange(DOWN, aligned_edge=LEFT).scale(0.7).to_edge(RIGHT, buff=0.8)
        for step in explanation:
            self.play(Write(step))
            self.wait(2)
        self.wait(3)
                    

                    数学题目：{problem}
                    解题过程：{solution}
            """



        # 调用OpenAI API生成Manim代码
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # 返回生成的Manim代码
        return response['choices'][0]['message']['content']

    except (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError, OSError) as e:
        print(f"An error occurred: {e}.")
        if attempt < max_retries:
            print(f"Retrying... Attempt {attempt + 1} of {max_retries}")
            time.sleep(5)  # 等待5秒再重试
            return generate_manim_code_from_solution(solution, model=model, attempt=attempt + 1,
                                                     max_retries=max_retries)
        else:
            print("Max retries reached. Moving to the next word.")
            return "Error: Max retries reached."

    except Exception as e:
        print(f"An unexpected error occurred: {e}. Moving to the next word.")
        return "Error: An unexpected error occurred."


#新添加的一个prompt代码
def generate_manim_code_from_problem_and_solution(math_problem, solution, is_physics=False, model="gpt-4", attempt=0, max_retries=3):
    try:
        # Encode the problem and solution into UTF-8 strings
        math_problem_utf8 = math_problem.encode('utf-8')
        solution_utf8 = solution.encode('utf-8')

        problem_type = "physics" if is_physics else "mathematics"
        prompt = f"""
Given the following {problem_type} problem and its solution, create Manim code to visually represent both the solution process and the key concepts. The code should use Manim's capabilities to create any necessary diagrams or models directly in the animation, such as cylinders or cones for geometric problems, without relying on any external images. Here are specific guidelines:

1. Start with the problem statement at the top, aligning it to the upper edge of the animation.
2. Automatically include visual diagrams or models at each step where they help visualize the concept being explained.
3. Use next_to for positioning equations and explanations, maintaining proper spacing.
4. Set the font size to 24 for all text and math elements.
5. Clearly distinguish between the problem statement and the solution process.
6. Use smooth visual transitions to ensure viewer comprehension without haste.
7. Space out text and graphics evenly to avoid clutter.
8. Intuitively interpret solution steps to include relevant diagrams or graphics such as force diagrams in physics or geometric constructions in mathematics.
9. Ensure mathematical expressions and shapes are properly displayed.
10. Manage text overflow to keep all elements on-screen.

For physics problems:
- Automatically draw force diagrams, free-body diagrams, or motion paths where applicable.

For mathematics problems:
- Automatically generate graphs, geometric constructions, or other illustrative diagrams as needed to elucidate the solution.

Enhance clarity with:
- Concise text explanations alongside relevant visuals, split into multiple lines if necessary.
- Limit each line to 10 words and each page to 10 lines.

Example Solution: {solution_utf8}

Manim code requirements:
- Interpret and visualize crucial steps using diagrams.
- Include FadeIn and corresponding FadeOut animations to prevent overlapping.
"""

        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        code = response['choices'][0]['message']['content']
        return code
        # if validate_code(code):
        #     return code
        # else:
        #     raise ValueError("The generated code is not valid.")
    except Exception as e:
        if attempt < max_retries:
            print(f"Retrying to generate code... Attempt {attempt + 1}")
            time.sleep(2)  # short delay before retrying
            return generate_manim_code_from_problem_and_solution(math_problem, solution, is_physics=is_physics, model=model, attempt=attempt + 1, max_retries=max_retries)
        else:
            return f"Error: {e}"



math_problem = "已知一次函数y=-2x+5,求其图像关于y轴对称的函数表达式"
solution = generate_solution_from_problem(math_problem)
# manim_code = generate_manim_code_from_solution(math_problem, solution)
# manim_code = generate_manim_code_from_problem_and_solution(math_problem, solution)
# manim_code = generate_manim_code_from_solution_ch(math_problem, solution)
#
# print("解题过程：", solution)
# print("Manim代码:", manim_code)





















def prompt_question_number(problem, solution, model="gpt-4", attempt=0, max_retries=3):
    try:
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

        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt_knowledge}]
        )
        print(response)
        return response



    except (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError, OSError) as e:
        print(f"An error occurred: {e}.")
        if attempt < max_retries:
            print(f"Retrying... Attempt {attempt + 1} of {max_retries}")
            time.sleep(5)  # 等待5秒再重试
            return generate_manim_code_from_solution(solution, model=model, attempt=attempt + 1,
                                                     max_retries=max_retries)
        else:
            print("Max retries reached. Moving to the next word.")
            return "Error: Max retries reached."

    except Exception as e:
        print(f"An unexpected error occurred: {e}. Moving to the next word.")
        return "Error: An unexpected error occurred."









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
prompt_config = [p1, p2, p3, p4, p5, p6, p7, p8]
print(prompt_config)


def generate_manim_code_from_solution_ch(problem, solution, model="gpt-4", attempt=0, max_retries=3, prompt_config=[]):
    num = prompt_question_number(problem, solution, model="gpt-4", attempt=0, max_retries=3)
    try:
        # 生成Manim代码的提示
        prompt = f"""
                    {prompt_config[num-1]}

                    数学题目：{problem}
                    解题过程：{solution}
            """

        # 调用OpenAI API生成Manim代码
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # 返回生成的Manim代码
        return response['choices'][0]['message']['content']

    except (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError, OSError) as e:
        print(f"An error occurred: {e}.")
        if attempt < max_retries:
            print(f"Retrying... Attempt {attempt + 1} of {max_retries}")
            time.sleep(5)  # 等待5秒再重试
            return generate_manim_code_from_solution(solution, model=model, attempt=attempt + 1,
                                                     max_retries=max_retries)
        else:
            print("Max retries reached. Moving to the next word.")
            return "Error: Max retries reached."

    except Exception as e:
        print(f"An unexpected error occurred: {e}. Moving to the next word.")
        return "Error: An unexpected error occurred."



manim_code = generate_manim_code_from_solution_ch(math_problem, solution)

print("解题过程：", solution)
print("Manim代码:", manim_code)