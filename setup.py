import os
import subprocess
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from pathlib import Path

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        import torch
        
        # 1. Define Paths
        root_dir = Path(__file__).parent.absolute()
        build_dir = root_dir / "build" 
        build_dir.mkdir(parents=True, exist_ok=True)
        ext_fullpath = Path(self.get_ext_fullpath(ext.name))
        ext_output_dir = ext_fullpath.parent
        ext_output_dir.mkdir(parents=True, exist_ok=True)
        ext_output_name = ext_fullpath.stem

        # 2. Prepare CMake Arguments
        cmake_args = [
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
            f"-DPYTHON_EXECUTABLE={os.sys.executable}",
            f"-DATTENTION_VARIANTS_OUTPUT_DIR={ext_output_dir}",
            f"-DATTENTION_VARIANTS_OUTPUT_NAME={ext_output_name}",
            "-DBUILD_LEGACY_PYCUATTENTION=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        # 3. Execute Build
        subprocess.check_call(["cmake", "--fresh", ext.sourcedir] + cmake_args, cwd=build_dir)
        subprocess.check_call(["cmake", "--build", ".", "--parallel"], cwd=build_dir)
        print(f"--- Built structured extension into {ext_fullpath}")

setup(
    name="Attention_Variants",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=[
        CMakeExtension("attention_variants.backends.cuda.attention_variants_cuda")
    ],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
