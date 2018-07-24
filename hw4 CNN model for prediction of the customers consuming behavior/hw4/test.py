# suqare = lambda x: x * x
# is_odd = lambda x : x % 2 is 1
#
# def func(s, map_fn, filter):
#     return [map_fn(i) for i in s if filter(i)]
#
# print(func([1,2,3,4,5], suqare, is_odd))

l = []
for i in range(1, 6):
    l.append(i)
print(l)

a = [[1,2], [3,4], [5, 6], [7, 8]]

print([(k, i) for (k, i) in a if i % 2 is 0])

print({})
