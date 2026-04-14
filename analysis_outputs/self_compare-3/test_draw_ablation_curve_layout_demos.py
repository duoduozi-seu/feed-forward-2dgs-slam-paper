"""验证图 4 两种版式 demo 脚本是否生成目标图片。"""

import subprocess
import sys
import unittest
from pathlib import Path


class DrawAblationCurveLayoutDemosTest(unittest.TestCase):
    """覆盖正文版与补充材料版两种图 4 demo 的输出行为。"""

    def test_script_generates_paper_and_supplementary_demos(self):
        """运行脚本后，应同时生成正文版和补充材料版两张图片。"""
        # 通过测试文件定位分析目录，避免依赖外部工作目录。
        analysis_dir = Path(__file__).resolve().parent
        script_path = analysis_dir / "draw_ablation_curve_layout_demos.py"
        figs_dir = analysis_dir / "figs"

        expected_paths = [
            figs_dir / "ablation_curves_paper_compact_demo.png",
            figs_dir / "ablation_curves_supplementary_demo.png",
        ]

        # 先删除旧输出，确保测试只检查本次运行结果。
        for expected_path in expected_paths:
            if expected_path.exists():
                expected_path.unlink()

        # 直接运行目标脚本，模拟本地终端出图流程。
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(analysis_dir),
            check=True,
        )

        # 两张图片都必须存在且非空，避免出现空壳文件。
        for expected_path in expected_paths:
            self.assertTrue(expected_path.exists(), f"缺少输出图片: {expected_path.name}")
            self.assertGreater(expected_path.stat().st_size, 0, f"输出图片为空: {expected_path.name}")


if __name__ == "__main__":
    unittest.main()
