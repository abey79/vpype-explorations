from setuptools import find_packages, setup

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="vpype-explorations",
    version="0.1.0",
    description="",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Antoine Beyeler",
    url="https://github.com/abey79/vpype-explorations/",
    license=license,
    packages=["vpype_explorations"],
    install_requires=[
        "axi @ git+https://github.com/fogleman/axi",
        "click",
        "vpype[all]>=1.10",
        "shapely",
        "numpy",
        "scipy",
        "scikit-image",
        "opencv-python-headless",
    ],
    entry_points="""
            [vpype.plugins]
            alien=vpype_explorations.alien:alien
            fracture=vpype_explorations.fracture:fracture
            variablewidth=vpype_explorations.variablewidth:variablewidth
            mdgrid=vpype_explorations.mdgrid:mdgrid
            msimage=vpype_explorations.moduleset:msimage
            msrandom=vpype_explorations.moduleset:msrandom
            msfingerprint=vpype_explorations.moduleset:msfingerprint
            mstiles=vpype_explorations.moduleset:mstiles
            fake3d=vpype_explorations.fake3d:fake3d
            spiro=vpype_explorations.spiro:spiro
            poly=vpype_explorations.poly:poly
            fill=vpype_explorations.fill:fill
            cfill=vpype_explorations.fill:cfill
            whlfarris=vpype_explorations.wheels:whlfarris
            whlrandom=vpype_explorations.wheels:whlrandom
            whlboard=vpype_explorations.wheels:whlboard
            circles=vpype_explorations.oldcircles:circles
            holes=vpype_explorations.oldcircles:holes
            circlecrop=vpype_explorations.circlecrop:circlecrop
        """,
)
