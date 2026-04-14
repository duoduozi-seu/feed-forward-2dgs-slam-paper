"""验证汇总均值收敛曲线绘图脚本是否生成三张目标图片。"""

import subprocess
import sys
import unittest
from pathlib import Path


class DrawConvergenceCurvesByTrainTimesFigsTest(unittest.TestCase):
    """覆盖用户要求的三张均值收敛曲线图输出。"""

    def test_script_generates_three_mean_metric_figures_in_figs_directory(self):
        """运行脚本后，应在 figs 目录下生成三张均值指标图。"""
        # 通过测试文件位置定位脚本目录，避免依赖外部工作目录。
        analysis_dir = Path(__file__).resolve().parent
        script_path = analysis_dir / "draw_convergence_curves_by_train_times_figs.py"
        figs_dir = analysis_dir / "figs"

        # 本次需求只要求输出 3 张汇总均值图。
        expected_paths = [
            figs_dir / "psnr_convergence_top30.png",
            figs_dir / "ssim_convergence_top30.png",
            figs_dir / "lpips_convergence_top30.png",
        ]

        # 先删除旧的目标文件，保证测试结论来自本次脚本运行。
        for expected_path in expected_paths:
            if expected_path.exists():
                expected_path.unlink()

        # 直接运行脚本，模拟用户在本地终端执行出图。
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(analysis_dir),
            check=True,
        )

        # 要求三张图都存在，且文件大小大于 0，避免生成空文件。
        for expected_path in expected_paths:
            self.assertTrue(expected_path.exists(), f"缺少输出图片: {expected_path.name}")
            self.assertGreater(expected_path.stat().st_size, 0, f"输出图片为空: {expected_path.name}")


if __name__ == "__main__":
    unittest.main()
