import os

from setuptools import find_packages, setup

NAME = "SAM-2"
VERSION = "1.0"
DESCRIPTION = "SAM 2: Segment Anything in Images and Videos"

REQUIRED_PACKAGES = [
    "hydra-core>=1.3.2",
    "iopath>=0.1.10",
]

BUILD_CUDA = os.getenv("SAM2_BUILD_CUDA", "1") == "1"
BUILD_ALLOW_ERRORS = os.getenv("SAM2_BUILD_ALLOW_ERRORS", "1") == "1"

CUDA_ERROR_MSG = (
    "{}\n\n"
    "Failed to build the SAM 2 CUDA extension due to the error above. "
    "You can still use SAM 2 and it's OK to ignore the error above, although some "
    "post-processing functionality may be limited (which doesn't affect the results in most cases).\n"
)


def get_extensions():
    if not BUILD_CUDA:
        return []

    try:
        from torch.utils.cpp_extension import CUDAExtension, CUDA_HOME
        import subprocess, re

        srcs = ["sam2/csrc/connected_components.cu"]
        compile_args = {
            "cxx": [],
            "nvcc": [
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
                "-gencode=arch=compute_70,code=sm_70",
                "-gencode=arch=compute_75,code=sm_75",
                "-gencode=arch=compute_80,code=sm_80",
                "-gencode=arch=compute_86,code=sm_86",
            ],
        }
        cuda_version = None
        try:
            nvcc_output = subprocess.check_output(
                [os.path.join(CUDA_HOME, "bin", "nvcc"), "--version"]
            ).decode("utf-8")
            match = re.search(r"release (\d+\.\d+)", nvcc_output)
            if match:
                cuda_version = float(match.group(1))
                print(f"Detected CUDA version: {cuda_version}")
        except Exception as e:
            print(f"Warning: Could not detect CUDA version: {e}")
        if cuda_version is not None and cuda_version >= 12.8:
            compile_args["nvcc"].append(
                "-gencode=arch=compute_120,code=sm_120"
            )

        ext_modules = [
            CUDAExtension("sam2._C", srcs, extra_compile_args=compile_args)
        ]
    except Exception as e:
        if BUILD_ALLOW_ERRORS:
            print(CUDA_ERROR_MSG.format(e))
            ext_modules = []
        else:
            raise e

    return ext_modules


try:
    from torch.utils.cpp_extension import BuildExtension

    class BuildExtensionIgnoreErrors(BuildExtension):
        def finalize_options(self):
            try:
                super().finalize_options()
            except Exception as e:
                print(CUDA_ERROR_MSG.format(e))
                self.extensions = []

        def build_extensions(self):
            try:
                super().build_extensions()
            except Exception as e:
                print(CUDA_ERROR_MSG.format(e))
                self.extensions = []

        def get_ext_filename(self, ext_name):
            try:
                return super().get_ext_filename(ext_name)
            except Exception as e:
                print(CUDA_ERROR_MSG.format(e))
                self.extensions = []
                return "_C.so"

    cmdclass = {
        "build_ext": (
            BuildExtensionIgnoreErrors.with_options(no_python_abi_suffix=True)
            if BUILD_ALLOW_ERRORS
            else BuildExtension.with_options(no_python_abi_suffix=True)
        )
    }
except Exception as e:
    cmdclass = {}
    if BUILD_ALLOW_ERRORS:
        print(CUDA_ERROR_MSG.format(e))
    else:
        raise e

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    packages=find_packages(include=["sam2", "sam2.*"]),
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES,
    python_requires=">=3.10.0",
    ext_modules=get_extensions(),
    cmdclass=cmdclass,
)
