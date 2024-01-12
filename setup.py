import setuptools
from torch.utils.cpp_extension import BuildExtension

setuptools.setup(
    # Meta-data
    name="skyeye",
    author="Nikhil Gosala",
    author_email="gosalan@cs.uni-freiburg.de",
    description="SkyEye: Self-Supervised Bird's-Eye-View Semantic Mapping Using Monocular Frontal View Images",
    #long_description=long_description,
    #long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],

    # Requirements
    python_requires=">=3, <4",

    # Package description
    packages=[
        "skyeye",
        "skyeye.algos",
        "skyeye.config",
        "skyeye.data",
        "skyeye.models",
        "skyeye.modules",
        "skyeye.modules.heads",
        "skyeye.utils",
        "skyeye.utils.parallel",
    ],
    ext_modules=[],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
)
