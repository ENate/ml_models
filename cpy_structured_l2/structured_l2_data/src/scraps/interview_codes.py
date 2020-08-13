def value_in_array(arr, val):
    if val in arr:
        print(val)


def func_swap_string(my_string2):
    my_new_string = my_string2[::-1]
    print(my_new_string)


def display_name(my_string):
    reverse_x = my_string[::-1]
    for k in reverse_x:
        print(k)


if __name__ == '__main__':
    value_in_array([1, 3, 5], 3)
    func_swap_string('ABCD')
    display_name('ABCD')

