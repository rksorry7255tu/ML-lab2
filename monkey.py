# Global variable
i = 0

def Monkey_go_box(x, y):
    global i
    i += 1
    print('Step:', i, 'Monkey slave', x, 'go to', y)

def Monkey_move_box(x, y):
    global i
    i += 1
    print('Step:', i, 'Monkey take the box from', x, 'deliver to', y)

def Monkey_on_box():
    global i
    i += 1
    print('Step:', i, 'Monkey climbs up the box')

def Monkey_get_banana():
    global i
    i += 1
    print('Step:', i, 'Monkey picked a banana')

# Read the input operating parameters
codeIn = input("Enter the locations of monkey, banana, and box (separated by spaces): ")
codeInList = codeIn.split()

# The operating parameters indicate the locations of monkey, banana, and box respectively.
monkey = codeInList[0]
banana = codeInList[1]
box = codeInList[2]

print('The steps are as follows:')
# Please use the least steps to complete the monkey picking banana task
Monkey_go_box(monkey, box)
Monkey_move_box(box, banana)
Monkey_on_box()
Monkey_get_banana()
