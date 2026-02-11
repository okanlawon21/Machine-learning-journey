from sklearn.externals.array_api_compat.torch import subtract
from sklearn.gaussian_process.kernels import Product

songs = ['mr money', 'wait for you', 'Essence', 'blessings', 'favorite song']

songs.append("Ye")
songs.append("Ojuelegba")

songs.pop(2)

print("First song:", songs[0])
print("Last song:", songs[-1])


print(len(songs))

print("Buga" in songs)

songs.sort()

print(songs)


numbers =[10, 25, 30, 15, 40, 20, 35]

print(max(numbers))
print(min(numbers))
print(sum(numbers))

Above_20 = [num for num in numbers if num > 20]
print(Above_20)

print(sum(numbers)/len(numbers))

doubled_numbers = [num * 2 for num in numbers]
print(doubled_numbers)


numbers.reverse()
print(numbers)

student = {
    'Name': 'Christopher',
    "Age" : 25,
    "City" : 'Lagos',
    "grade" : "80"
}

print('Name:', student['Name'])

student["grade"] = 95

student["phone"] = '07073546789'

student.pop('City')

print("Does age exist?", 'Age' in student)

print(student.keys())

print(student.values())

for key, value in student.items():
    print(key, ":", value)




countries = {
   "Nigeria": "Abuja",
    "Ghana": "Accra",
    "France": "Paris",
    "Germany": "Berlin",
    "United Kingdom": "England",
}

countries["Japan"] = "Tokyo"
countries["Italy"] = "Rome"

print(countries["Nigeria"])

for country, capital in countries.items():
    print(capital)

capitals = list(countries.values())
print(capitals)

print("USA" in countries)

print(len(countries))

Name = 'Adeola'
Age = 25
City = 'London'

print(f'My Name is {Name}, I am {Age} years old from {City}')

X = 20
y = 30
print(X + y)
print(subtract(X, y))
print(Product(X, y))
print(X/y)
print(X % 5)
print (X ** y)


a=10
b=20
a, b = b, a
print(a, b)

int_var = 10
float_var = 3.5
str_var = "Python"
bool_var = True

print(type(int_var))
print(type(float_var))
print(type(str_var))
print(type(bool_var))

print(int_var + float_var)


num_str = "123"
num_int = int(num_str)
print(num_int)


num_float = 45.7
print(int(num_float))


