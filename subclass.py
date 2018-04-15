#coding=UTF-8

class SchoolMember:
	'''代表学校里的成员'''
	def __init__(self, name, age):
		self.name = name
		self.age = age
		print('(Initialized SchoolMember: {})'.format(self.name))

	def tell(self):
		'''告诉我细节'''
		print('Name:"{}"  Age:"{}"'.format(self.name, self.age), end="  ")

class Teacher(SchoolMember):
	'''代表一位老师'''
	def __init__(self, name, age, salary):
		SchoolMember.__init__(self, name, age)
		self.salary = salary
		print('(Initialized Teacher: {})'.format(self.name))

	def tell(self):
		SchoolMember.tell(self)
		print('Salary: "{:d}"'.format(self.salary))

class Student(SchoolMember):
	"""代表一位学生"""
	def __init__(self, name, age, grades):
		SchoolMember.__init__(self, name, age)
		self.grades = grades
		print('(Initialized Student: {})'.format(self.name))

	def tell(self):
		SchoolMember.tell(self)
		print('Grades: "{}"'.format(self.grades))

t = Teacher('Dr.Qian', 24, 300000)
s1 = Student('Tao   Mamba', 24, 'S+')
s2 = Student('Wei   Mamba', 24, 'S+')
s3 = Student('Jiang Mamba', 24, 'S+')



print()

members= [t, s1, s2, s3]

for member in members:
	member.tell()


		
	


