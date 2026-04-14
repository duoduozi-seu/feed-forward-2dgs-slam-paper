"""验证单栏正文版收敛曲线 demo 脚本是否生成目标图片。"""

import subprocess
import sys
import unittest
from pathlib import Path


class DrawSingleColumnConvergenceDemoTest(unittest.TestCase):
    """覆盖正文版图 4 demo 的单图输出行为。"""

    def test_script_generates_single_column_demo_figure(self):
        """运行脚本后，应生成一张适合正文排版的单栏收敛图。"""
        # 通过测试文件位置定位分析目录，避免依赖外部工作目录。
        analysis_dir = Path(__file__).resolve().parent
        script_path = analysis_dir / "draw_single_column_convergence_demo.py"
        output_path = analysis_dir / "figs" / "ablation_curves_single_column_demo.png"

        # 先删除旧文件，确保测试检查的是本次运行结果。
        if output_path.exists():
            output_path.unlink()

        # 运行目标脚本，模拟本地终端中的真实出图流程。
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(analysis_dir),
            check=True,
        )

        # 要求目标文件存在且非空，避免脚本只创建空壳文件。
        self.assertTrue(output_path.exists(), f"缺少输出图片: {output_path.name}")
        self.assertGreater(output_path.stat().st_size, 0, f"输出图片为空: {output_path.name}")


if __name__ == "__main__":
    unittest.main()
