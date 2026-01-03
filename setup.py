from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT= "-e ."
# -e . in req will automatically trigger setup.py file
def get_requirements(file_path:str) ->List[str]:

    """
    This function will return the list of requirements
    """
    requirements=[]
    with open(file_path, "r") as file_object:
        requirements= file_object.readlines()
        requirements= [line.replace("\n", "") for line in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name= "mlproject",
    version= "0.0.1",
    author= "Saksham",
    author_email= "mittal.saksham2001@gmail.com",
    packages= find_packages(),
    install_requires= get_requirements("requirements.txt")
)