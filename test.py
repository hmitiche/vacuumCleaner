my_dict = {
    'a': 1,
    'b': 2,
    'c': 3,
    'x': 2,

}
xx = my_dict.items()
print(xx)
#print(next(xx))

value_to_find = 2
key = next((k for k, v in my_dict.items() if v == value_to_find), None)

print(key)  # Output: 'b'

# reverse a dictionary
# only when keys are unique 

rev_dict = {v:k for k,v in my_dict.items()}
print(f"reverse dictionary {rev_dict}")

# Print without newline
print("Hello", "World", sep=", ", end="!")