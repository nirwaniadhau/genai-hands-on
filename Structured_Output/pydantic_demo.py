from pydantic import BaseModel

class Student(BaseModel):
    name:str

new_student={'name':'Nirwani'}

student_obj=Student(**new_student)
print(student_obj)
print(type(student_obj))