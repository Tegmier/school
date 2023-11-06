################## if statement #######################
print("################## if statement #######################")
a = 10
if a > 5:
    print("Bigger than 5")
print("done")

your_hobbies = ['sleeping', 'eating', 'singing', 'punching']
if 'sleeping' in your_hobbies:
    print("lazzy!")
elif 'eating' in your_hobbies:
    print('time to lose weight!')
else:
    print('great, I do not know what to say!')
print('done')

attendance_rate = 10
exam_score = 80
if attendance_rate > 50 & exam_score >= 60 :
    print("passed!")
else:
    print("you let your big mother down!")

################## for statement #######################
print("################## for statement #######################")
list_of_sports = ['tennis', 'basketball', 'football']
for sport in list_of_sports:
    print(sport)
print('done')

a_string = 'I like Tokyo!'
for x in a_string:
    print(x)
print('done')

user = {'name': 'alice', 
        'age' : 30, 
        'adress' : 'tokyo'}
for key in user:
    print(user[key])
print('done')

################## Break and Continue #######################
print("################## Break and Continue #######################")
colors = ['red', 'blue', 'green', 'yellow', 'white']
for color in colors:
    if color == 'green':
        break
    print(color)
print('done')

for color in colors:
    if color == 'green':
        continue
    print(color)
print('done')

################## While loop #######################
print('################## While loop #######################')
x =1 
while x < 5:
    print(x)
    x += 1
print('done')

################## else statement #######################
print("################## else statement #######################")
for i in ['red', 'blue', 'green']:
    if i == 'green':
        break
    print(i)
else:
    print('yellow')

################## Python Fuctions #######################
def my_function():
    print("This is my function!")
my_function()
print('done')

def greeting(f_name, l_name):
    print("Hello", f_name, l_name)
first_name = 'Peter'
last_name = 'Parker'
greeting(first_name, last_name)

def greeting2(f_name, l_name):
    salute = "Hello " + f_name + ' ' + l_name
    return salute
message = greeting2(first_name, last_name)
print(message)

def sum_and_multiply (a,b):
    return a+b, a*b
print(sum_and_multiply(2,3))

def comic_listing (*titles):
    print(titles)
comic_listing("Dragon Ball")
comic_listing("Dragon Ball", "Naruto")

def greeting3(**full_name):
    print(full_name)
greeting3(A = 'A', B = 'B', C= 'C')