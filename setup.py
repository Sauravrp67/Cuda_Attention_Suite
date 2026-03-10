import os
import subprocess
from setuptools import setup, Extension
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
        # build_temp = Path(self.build_temp)
        # Standardize the output to your root 'build' folder for consistency
        build_dir = root_dir / "build" 
        build_dir.mkdir(parents=True, exist_ok=True)
        
        # The internal package destination
        # Maps 'PyCuAttention.kernels.cuda_attn_backend' to 'PyCuAttention/kernels/bin'
        target_bin_dir = root_dir / "PyCuAttention" / "kernels" / "bin"
        target_bin_dir.mkdir(parents=True, exist_ok=True)

        # 2. Prepare CMake Arguments
        cmake_args = [
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
            f"-DPYTHON_EXECUTABLE={os.sys.executable}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={build_dir}/lib",
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        # 3. Execute Build
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_dir)
        subprocess.check_call(["cmake", "--build", ".", "--parallel"], cwd=build_dir)

        # 4. Create the Symlink (The "No Redundancy" behavior)
        # This mirrors your build.sh logic within the pip install flow
        binary_name = "cuda_attn_backend.so"
        source_path = build_dir / "lib" / binary_name
        link_path = target_bin_dir / binary_name

        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        
        # Creating relative symlink for portability within the same machine
        os.symlink(source_path, link_path)
        print(f"--- Symlinked {source_path} -> {link_path}")

setup(
    name="Attention_Variants",
    version="0.1.0",
    packages=["PyCuAttention"],
    # Ensure the name matches what you import
    ext_modules=[CMakeExtension("PyCuAttention.kernels.cuda_attn_backend")],
    cmdclass={"build_ext": CMakeBuild},
)