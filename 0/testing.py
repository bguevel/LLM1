import re



#current regex doesnt capture colons commas, periods, paranthesees, !, ? quotations as well, need to keep time colon
my_string = "hey this is my string, it's cool this is context-free. \"I wonder if this works (hey its me).\""

my_string1 = re.sub(r"[^-'\w\s]", "", my_string)
#my_grammar = re.sub(r'[,."()!?]', '', my_string)
#my_grammar = re.findall(r'[,."()!?]', my_string)

print("these are the grammar elements")
print(re.findall(r'[,.\"()!?]', my_string))

print(my_string)


print("my tokens")
tokens = re.findall(r"\w+(?:'\w+)?|[,.\"()!?-]", my_string)
print(tokens)
#print(my_string.split(","))

#print(my_string.split(" "))