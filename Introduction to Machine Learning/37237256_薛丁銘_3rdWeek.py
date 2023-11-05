################## Dictionaries Part 1 #######################
print('################## Dictionaries Part 1 #######################')
SLASH = '----------------------------------------'
user = {'name': 'alice', 'age' : 30, 'adress' : 'tokyo'}
print(user)

user = {'name': 'alice', 
        'age' : 30, 
        'adress' : 'tokyo'}

print(user.keys())
print(user.values())
print(list(user.keys())[1])
user['hobbies'] = 'playing'
print(user.keys())

print(SLASH)

user.pop('age')
print(user.items())
user['age'] = 50
new_info = {'age' : 60}
user.update(new_info)
print(user.items())

################## Dictionaries Part 2 #######################

print('################## Dictionaries Part 2 #######################')

my_car = {
    'brand' : 'bmw',
    'type' : 'series 3',
    'year' : 2020
}
your_car = my_car
your_car['type'] = 'X2'
print(my_car['type'])

print(SLASH)

my_comics = {
    'comic_1' : {
        'title' : 'A',
        'year' : 1999
    },
    'comic_2' : {
        'title' : 'B',
        'year' : 2000
    },
    'comic_3' : {
        'title' : 'C',
        'year' : 2001
    }
}

print(my_comics['comic_1']['title'])
################## Dictionaries #######################